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
    PROJECT_ROOT / "forensics" / "activation_difference_results_20260812"
)
DEFAULT_CASE_IDS = (
    "CASE_HIS_028_V01",
    "CASE_MAR_031_V01",
    "CASE_NAT_030_V02",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="绘制三个案例的36层clean-poison activation差异对比图。"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR
    )
    parser.add_argument("--case-ids", nargs="+", default=DEFAULT_CASE_IDS)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="默认写入input-dir/activation_difference_three_cases.png。",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="默认写入input-dir/activation_difference_three_cases_summary.csv。",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_result(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("schema_version") != "clean-poison-activation-difference-v1":
        raise ValueError(f"非预期activation差异结果：{path}")
    if not result.get("technical_checks", {}).get("technical_pass"):
        raise RuntimeError(f"技术检查未通过：{path}")
    return result


def write_summary(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "max_normalized_l2_layer",
        "max_normalized_l2",
        "max_cosine_distance_layer",
        "max_cosine_distance",
        "largest_normalized_l2_jump_layer",
        "largest_normalized_l2_jump",
        "max_crs_layer",
        "max_crs",
        "max_difference_and_max_crs_same_layer",
        "pearson_normalized_l2_vs_crs",
    ]
    rows = [
        {"case_id": result["case_id"], **result["summary"]}
        for result in results
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else input_dir / "activation_difference_three_cases.png"
    )
    summary_path = (
        args.summary_csv.expanduser().resolve()
        if args.summary_csv is not None
        else input_dir / "activation_difference_three_cases_summary.csv"
    )
    existing = [
        path for path in (output_path, summary_path) if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "输出已存在，本次不覆盖："
            + ", ".join(str(path) for path in existing)
        )

    results = [
        read_result(input_dir / f"{case_id}.activation_difference.json")
        for case_id in args.case_ids
    ]
    colors = ["#2563eb", "#f97316", "#7c3aed", "#0891b2"]
    figure, axes = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True, constrained_layout=True
    )
    for color, result in zip(colors, results, strict=False):
        rows = result["layer_results"]
        layers = [row["layer_index"] for row in rows]
        axes[0].plot(
            layers,
            [row["normalized_l2"] for row in rows],
            color=color,
            marker="o",
            markersize=2.5,
            label=result["case_id"],
        )
        axes[1].plot(
            layers,
            [row["cosine_distance"] for row in rows],
            color=color,
            marker="o",
            markersize=2.5,
            label=result["case_id"],
        )

    axes[0].set_title(
        "36-layer clean-poison activation difference comparison"
    )
    axes[0].set_ylabel("Normalized L2")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")
    axes[1].set_xlabel("Layer index")
    axes[1].set_ylabel("Cosine distance")
    axes[1].set_xticks(range(0, 36, 2))
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        output_path.stem + ".tmp" + output_path.suffix
    )
    figure.savefig(temporary_path, dpi=180)
    plt.close(figure)
    temporary_path.replace(output_path)
    write_summary(summary_path, results)

    print("三案例activation差异对比图已生成：")
    print(output_path)
    print("汇总CSV已生成：")
    print(summary_path)
    for result in results:
        summary = result["summary"]
        print(
            f"{result['case_id']}: "
            f"max_diff=L{summary['max_normalized_l2_layer']}, "
            f"max_crs=L{summary['max_crs_layer']}, "
            f"same={summary['max_difference_and_max_crs_same_layer']}"
        )


if __name__ == "__main__":
    main()
