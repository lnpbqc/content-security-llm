from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT / "forensics" / "layer_scan_results_20260811"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="汇总完整Layer扫描JSON并生成Layer–CRS曲线。"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="可重复指定案例；不传时读取所有完整span结果。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="显式允许覆盖图和汇总CSV。",
    )
    parser.add_argument(
        "--single-case-clear",
        action="store_true",
        help=(
            "为单个--case-id生成只包含CRS曲线、"
            "poison/flip/clean阈值和关键层标注的清晰图。"
        ),
    )
    parser.add_argument(
        "--single-output",
        type=Path,
        default=None,
        help="单案例清晰图的输出路径。",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON顶层必须是对象：{path}")
    return value


def select_paths(
    input_dir: Path,
    case_ids: list[str] | None,
) -> list[Path]:
    if case_ids:
        paths = [
            input_dir
            / f"{case_id}.layer_scan.candidate_prediction_span.json"
            for case_id in case_ids
        ]
    else:
        paths = sorted(
            input_dir.glob(
                "*.layer_scan.candidate_prediction_span.json"
            )
        )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "缺少完整span扫描结果："
            + ", ".join(str(path) for path in missing)
        )
    if not paths:
        raise FileNotFoundError("没有找到完整span扫描JSON。")
    return paths


def validate_result(
    result: dict[str, Any],
    path: Path,
) -> None:
    if result.get("status") != "completed":
        raise RuntimeError(f"结果未完成：{path}")
    if result.get("intervention", {}).get("patch_scope") != (
        "candidate_prediction_span"
    ):
        raise RuntimeError(f"不是prediction-span结果：{path}")
    if not result.get("technical_checks", {}).get("technical_pass"):
        raise RuntimeError(f"技术检查未通过：{path}")

    layer_results = result.get("layer_results", [])
    layer_indices = [item.get("layer_index") for item in layer_results]
    if layer_indices != list(range(36)):
        raise RuntimeError(
            f"结果不是0–35层完整扫描：{path}"
        )


def first_margin_flip(result: dict[str, Any]) -> int | None:
    return next(
        (
            item["layer_index"]
            for item in result["layer_results"]
            if item["patched_margin"] > 0
        ),
        None,
    )


def write_summary_csv(
    path: Path,
    results: list[dict[str, Any]],
) -> None:
    fieldnames = [
        "case_id",
        "gold_answer",
        "poison_target_answer",
        "gold_token_count",
        "poison_token_count",
        "clean_margin",
        "poison_margin",
        "first_meaningful_recovery_layer",
        "first_positive_margin_layer",
        "max_crs_layer",
        "max_crs",
        "top3_layers",
        "technical_pass",
    ]
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            scoring = result["scoring"]
            clean = scoring["clean_baseline"]
            summary = result["summary"]
            writer.writerow(
                {
                    "case_id": result["case_id"],
                    "gold_answer": scoring["gold_answer"],
                    "poison_target_answer": scoring[
                        "poison_target_answer"
                    ],
                    "gold_token_count": len(
                        clean["gold_score"]["candidate_token_ids"]
                    ),
                    "poison_token_count": len(
                        clean["poison_score"]["candidate_token_ids"]
                    ),
                    "clean_margin": clean["margin"],
                    "poison_margin": scoring["poison_baseline"][
                        "margin"
                    ],
                    "first_meaningful_recovery_layer": summary[
                        "first_meaningful_recovery_layer"
                    ],
                    "first_positive_margin_layer": first_margin_flip(
                        result
                    ),
                    "max_crs_layer": summary["max_crs_layer"],
                    "max_crs": summary["max_crs"],
                    "top3_layers": ";".join(
                        f"{item['layer_index']}:{item['crs']:.6f}"
                        for item in summary["top3_layers"]
                    ),
                    "technical_pass": result["technical_checks"][
                        "technical_pass"
                    ],
                }
            )
    temporary_path.replace(path)


def write_curve(
    path: Path,
    results: list[dict[str, Any]],
) -> None:
    figure, axis = plt.subplots(figsize=(11, 6.5), dpi=160)
    for result in results:
        layers = [
            item["layer_index"] for item in result["layer_results"]
        ]
        crs_values = [item["crs"] for item in result["layer_results"]]
        axis.plot(
            layers,
            crs_values,
            marker="o",
            markersize=3.2,
            linewidth=1.8,
            label=result["case_id"],
        )

    axis.axhline(0.0, color="#555555", linewidth=1.0)
    axis.axhline(
        0.05,
        color="#cc7a00",
        linestyle="--",
        linewidth=1.2,
        label="Meaningful recovery threshold (CRS=0.05)",
    )
    axis.set_title(
        "Qwen3-4B Layer-wise Causal Recovery\n"
        "Clean to Poison Candidate-Prediction-Span Patch"
    )
    axis.set_xlabel("Decoder layer index")
    axis.set_ylabel("Causal recovery score (CRS)")
    axis.set_xticks(range(0, 36, 2))
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper left", frameon=True)
    figure.tight_layout()

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    figure.savefig(
        temporary_path,
        format="png",
        bbox_inches="tight",
    )
    plt.close(figure)
    temporary_path.replace(path)


def write_single_case_clear_curve(
    path: Path,
    result: dict[str, Any],
) -> None:
    """绘制单一CRS曲线，避免与patched margin重复叠加。"""

    layers = [
        int(item["layer_index"]) for item in result["layer_results"]
    ]
    crs_values = [
        float(item["crs"]) for item in result["layer_results"]
    ]
    clean_margin = float(
        result["scoring"]["clean_baseline"]["margin"]
    )
    poison_margin = float(
        result["scoring"]["poison_baseline"]["margin"]
    )
    denominator = clean_margin - poison_margin
    if denominator <= 0:
        raise RuntimeError("clean margin必须高于poison margin。")
    flip_crs = -poison_margin / denominator

    summary = result["summary"]
    first_meaningful = summary["first_meaningful_recovery_layer"]
    first_flip = first_margin_flip(result)
    max_layer = int(summary["max_crs_layer"])
    max_crs = float(summary["max_crs"])
    crs_by_layer = dict(zip(layers, crs_values, strict=True))

    figure, axis = plt.subplots(figsize=(12, 6.8), dpi=180)
    axis.plot(
        layers,
        crs_values,
        color="#2563eb",
        marker="o",
        markersize=4.2,
        linewidth=2.4,
        label="Causal Recovery Score (CRS)",
        zorder=4,
    )
    axis.axhline(
        0.0,
        color="#64748b",
        linewidth=1.2,
        label="Poison baseline (CRS=0)",
        zorder=2,
    )
    axis.axhline(
        flip_crs,
        color="#f59e0b",
        linestyle="--",
        linewidth=1.7,
        label=f"Candidate flip threshold (CRS={flip_crs:.3f})",
        zorder=2,
    )
    axis.axhline(
        1.0,
        color="#16a34a",
        linestyle="--",
        linewidth=1.5,
        label="Clean baseline (CRS=1)",
        zorder=2,
    )

    lower = min(min(crs_values) - 0.035, -0.05)
    upper = max(max(crs_values) + 0.08, 1.08)
    axis.axhspan(lower, flip_crs, color="#ef4444", alpha=0.035)
    axis.axhspan(flip_crs, upper, color="#22c55e", alpha=0.035)

    highlights = [
        (
            first_meaningful,
            "First meaningful recovery\n"
            f"L{first_meaningful}, CRS={crs_by_layer[first_meaningful]:.3f}",
            (-108, 34),
            "#7c3aed",
        ),
        (
            first_flip,
            "First candidate flip\n"
            f"L{first_flip}, CRS={crs_by_layer[first_flip]:.3f}",
            (18, -58),
            "#f59e0b",
        ),
        (
            max_layer,
            "Maximum recovery\n"
            f"L{max_layer}, CRS={max_crs:.3f}",
            (-142, -52),
            "#16a34a",
        ),
    ]
    used_layers: set[int] = set()
    for layer_index, text, offset, color in highlights:
        if layer_index is None or layer_index in used_layers:
            continue
        used_layers.add(layer_index)
        axis.scatter(
            [layer_index],
            [crs_by_layer[layer_index]],
            s=72,
            color=color,
            edgecolor="white",
            linewidth=1.2,
            zorder=5,
        )
        axis.annotate(
            text,
            xy=(layer_index, crs_by_layer[layer_index]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9.5,
            color="#111827",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": color,
                "alpha": 0.94,
            },
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "linewidth": 1.1,
            },
            zorder=6,
        )

    axis.set_title(
        f"{result['case_id']}: 36-layer Causal Recovery\n"
        "CRS-only view (clean-to-poison activation patch)",
        fontsize=15,
    )
    axis.set_xlabel("Decoder layer index", fontsize=11)
    axis.set_ylabel("Causal Recovery Score (CRS)", fontsize=11)
    axis.set_xlim(-0.8, 35.8)
    axis.set_ylim(lower, upper)
    axis.set_xticks(range(0, 36, 2))
    axis.grid(True, alpha=0.22)
    axis.legend(loc="upper left", frameon=True, fontsize=9.5)
    axis.text(
        0.99,
        0.02,
        "Below flip threshold: poison candidate preferred\n"
        "Above flip threshold: gold candidate preferred",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
        color="#475569",
    )
    figure.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.stem + ".tmp" + path.suffix)
    figure.savefig(temporary_path, format="png", bbox_inches="tight")
    plt.close(figure)
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    paths = select_paths(input_dir, args.case_ids)
    results = [read_json(path) for path in paths]
    for result, path in zip(results, paths, strict=True):
        validate_result(result, path)

    results.sort(key=lambda item: item["case_id"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.single_case_clear:
        if len(results) != 1:
            raise ValueError(
                "--single-case-clear必须且只能指定一个--case-id。"
            )
        single_path = (
            args.single_output.expanduser().resolve()
            if args.single_output is not None
            else output_dir
            / f"{results[0]['case_id']}.layer_trace_crs_clear.png"
        )
        if single_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"输出已存在，本次不覆盖：{single_path}"
            )
        write_single_case_clear_curve(single_path, results[0])
        print(f"已生成单案例CRS清晰图：{single_path}")
        print(
            "关键层："
            f"first_meaningful="
            f"{results[0]['summary']['first_meaningful_recovery_layer']}, "
            f"first_margin_positive={first_margin_flip(results[0])}, "
            f"max_crs_layer={results[0]['summary']['max_crs_layer']}"
        )
        return

    curve_path = output_dir / "layer_scan_crs_three_cases.png"
    summary_path = output_dir / "layer_scan_three_cases_summary.csv"
    existing = [
        path for path in (curve_path, summary_path) if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "输出已存在，本次不覆盖："
            + ", ".join(str(path) for path in existing)
        )

    write_curve(curve_path, results)
    write_summary_csv(summary_path, results)

    print(f"已汇总案例数：{len(results)}")
    for result in results:
        print(
            f"  {result['case_id']}: "
            f"first_meaningful="
            f"{result['summary']['first_meaningful_recovery_layer']}, "
            f"first_margin_positive={first_margin_flip(result)}, "
            f"max_crs_layer={result['summary']['max_crs_layer']}"
        )
    print(f"曲线图：{curve_path}")
    print(f"汇总CSV：{summary_path}")


if __name__ == "__main__":
    main()
