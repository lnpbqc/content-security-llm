from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any


def _normalize_omp_num_threads() -> None:
    raw_value = os.environ.get("OMP_NUM_THREADS", "").strip()
    if not raw_value.isdigit() or int(raw_value) <= 0:
        os.environ["OMP_NUM_THREADS"] = "1"


_normalize_omp_num_threads()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as functional
from nnsight import LanguageModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.model_runtime import MODEL_PATH, load_llm
from forensics.activation_patch_scoring import (
    _build_prompt_token_ids,
    _make_inputs,
)
from forensics.layer_scan import (
    count_registered_hooks,
    load_case_inputs,
    query_gpu_processes,
    read_json,
    sha256_file,
    validate_case_id,
    write_json_atomic,
)


DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT / "forensics" / "neural_trace_inputs_20260811"
)
DEFAULT_LAYER_SCAN_DIR = (
    PROJECT_ROOT / "forensics" / "layer_scan_results_20260811"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "forensics" / "activation_difference_results_20260812"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "比较clean/poison Prompt在36个Decoder Layer最后一个Prompt "
            "token位置的activation差异，并关联已有CRS。"
        )
    )
    parser.add_argument("--case-id", required=True)
    parser.add_argument(
        "--input-root", type=Path, default=DEFAULT_INPUT_ROOT
    )
    parser.add_argument(
        "--layer-scan-dir", type=Path, default=DEFAULT_LAYER_SCAN_DIR
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="允许GPU已有其他计算进程时继续（不推荐）。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="显式允许覆盖同名输出。",
    )
    return parser.parse_args()


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "layer_index",
        "layer_path",
        "clean_norm",
        "poison_norm",
        "l2_difference",
        "normalized_l2",
        "relative_to_clean_l2",
        "mean_abs_difference",
        "max_abs_difference",
        "cosine_similarity",
        "cosine_distance",
        "normalized_l2_jump",
        "crs",
        "causal_patched_margin",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def capture_all_layer_activations(
    traced_model: LanguageModel,
    base_model: torch.nn.Module,
    tokenizer,
    prompt: str,
) -> dict[str, Any]:
    token_ids = _build_prompt_token_ids(tokenizer, prompt)
    input_ids, attention_mask = _make_inputs(base_model, token_ids)
    layer_count = len(base_model.model.layers)

    saved_activations = []
    with torch.no_grad():
        with traced_model.trace(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ):
            for layer_index in range(layer_count):
                saved_activations.append(
                    traced_model.model.layers[layer_index]
                    .output[:, -1, :]
                    .detach()
                    .float()
                    .cpu()
                    .clone()
                    .save()
                )

    activations = [
        activation.reshape(-1).contiguous()
        for activation in saved_activations
    ]
    return {
        "token_ids": token_ids,
        "prompt_token_count": len(token_ids),
        "last_prompt_token_id": token_ids[-1],
        "activations": activations,
        "activation_shapes": [
            list(activation.shape) for activation in activations
        ],
    }


def pearson_correlation(
    values_a: list[float], values_b: list[float]
) -> float | None:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return None
    tensor_a = torch.tensor(values_a, dtype=torch.float64)
    tensor_b = torch.tensor(values_b, dtype=torch.float64)
    centered_a = tensor_a - tensor_a.mean()
    centered_b = tensor_b - tensor_b.mean()
    denominator = torch.linalg.vector_norm(
        centered_a
    ) * torch.linalg.vector_norm(centered_b)
    if denominator.item() <= 1e-12:
        return None
    return float((centered_a @ centered_b / denominator).item())


def plot_case(
    path: Path,
    case_id: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    layers = [row["layer_index"] for row in rows]
    normalized_l2 = [row["normalized_l2"] for row in rows]
    cosine_distance = [row["cosine_distance"] for row in rows]
    crs = [row["crs"] for row in rows]

    figure, axes = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, constrained_layout=True
    )
    axes[0].plot(
        layers,
        normalized_l2,
        color="#2563eb",
        marker="o",
        markersize=3,
        label="Normalized L2",
    )
    axes[0].plot(
        layers,
        cosine_distance,
        color="#f97316",
        marker="s",
        markersize=3,
        label="Cosine distance",
    )
    axes[0].axvline(
        summary["max_normalized_l2_layer"],
        color="#2563eb",
        linestyle="--",
        alpha=0.55,
        label=(
            "Max activation difference: L"
            f"{summary['max_normalized_l2_layer']}"
        ),
    )
    axes[0].set_ylabel("Activation difference")
    axes[0].set_title(
        f"{case_id}: clean-poison activation difference"
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(
        layers,
        crs,
        color="#16a34a",
        marker="o",
        markersize=3,
        label="Causal Recovery Score (CRS)",
    )
    axes[1].axhline(0.0, color="#64748b", linewidth=1)
    axes[1].axhline(
        1.0,
        color="#16a34a",
        linewidth=1,
        linestyle=":",
        alpha=0.7,
    )
    axes[1].axvline(
        summary["max_crs_layer"],
        color="#16a34a",
        linestyle="--",
        alpha=0.55,
        label=f"Max CRS: L{summary['max_crs_layer']}",
    )
    axes[1].set_xlabel("Layer index")
    axes[1].set_ylabel("CRS")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    axes[1].set_xticks(range(0, len(layers), 2))

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.stem + ".tmp" + path.suffix)
    figure.savefig(temporary_path, dpi=180)
    plt.close(figure)
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    validate_case_id(args.case_id)
    case_inputs = load_case_inputs(args.input_root, args.case_id)

    layer_scan_path = (
        args.layer_scan_dir.expanduser().resolve()
        / (
            f"{args.case_id}.layer_scan."
            "candidate_prediction_span.json"
        )
    )
    if not layer_scan_path.is_file():
        raise FileNotFoundError(f"缺少36层CRS结果：{layer_scan_path}")
    layer_scan = read_json(layer_scan_path)
    if not layer_scan.get("technical_checks", {}).get("technical_pass"):
        raise RuntimeError("关联的36层CRS扫描未通过技术检查。")
    if not layer_scan.get("intervention", {}).get("all_layers_scanned"):
        raise RuntimeError("关联的CRS结果不是完整36层扫描。")

    output_dir = args.output_dir.expanduser().resolve()
    json_path = output_dir / f"{args.case_id}.activation_difference.json"
    csv_path = output_dir / f"{args.case_id}.activation_difference.csv"
    png_path = output_dir / f"{args.case_id}.activation_difference.png"
    existing_paths = [
        path for path in (json_path, csv_path, png_path) if path.exists()
    ]
    if existing_paths and not args.overwrite:
        raise FileExistsError(
            "输出已存在，本次不覆盖："
            + ", ".join(str(path) for path in existing_paths)
        )

    gpu_processes = query_gpu_processes()
    if gpu_processes and not args.allow_busy_gpu:
        raise RuntimeError(
            "检测到其他GPU计算进程，为避免影响队友，"
            "本次停止。\n" + "\n".join(gpu_processes)
        )

    metadata = case_inputs["metadata"]
    token_alignment = metadata["token_alignment"]
    if not token_alignment.get("token_counts_equal"):
        raise RuntimeError("clean/poison Prompt token数不同，不能直接同位置比较。")
    if not token_alignment.get("last_prompt_token_equal"):
        raise RuntimeError("最后一个Prompt token不同，不能直接同位置比较。")

    base_model = None
    traced_model = None
    try:
        print("开始加载团队统一的Qwen3-4B……")
        base_model, tokenizer = load_llm()
        base_model.eval()
        layer_count = len(base_model.model.layers)
        if layer_count != len(layer_scan["layer_results"]):
            raise RuntimeError(
                f"模型层数{layer_count}与CRS结果数"
                f"{len(layer_scan['layer_results'])}不一致。"
            )

        hooks_before_wrap = count_registered_hooks(base_model)
        traced_model = LanguageModel(base_model, tokenizer=tokenizer)
        hooks_before_capture = count_registered_hooks(base_model)

        print(f"案例：{args.case_id}")
        print(f"模型层数：{layer_count}")
        print("比较位置：最后一个Prompt token")
        print("[1/3] clean forward：同时保存36层activation……")
        clean_capture = capture_all_layer_activations(
            traced_model,
            base_model,
            tokenizer,
            case_inputs["clean_prompt"],
        )
        print("[2/3] poison forward：同时保存36层activation……")
        poison_capture = capture_all_layer_activations(
            traced_model,
            base_model,
            tokenizer,
            case_inputs["poison_prompt"],
        )

        if clean_capture["token_ids"] != token_alignment["clean_token_ids"]:
            raise RuntimeError("clean Prompt token序列与对齐记录不一致。")
        if poison_capture["token_ids"] != token_alignment["poison_token_ids"]:
            raise RuntimeError("poison Prompt token序列与对齐记录不一致。")

        print("[3/3] 计算逐层差异并关联已有CRS……")
        causal_by_layer = {
            int(item["layer_index"]): item
            for item in layer_scan["layer_results"]
        }
        rows: list[dict[str, Any]] = []
        previous_normalized_l2: float | None = None
        for layer_index, (clean_activation, poison_activation) in enumerate(
            zip(
                clean_capture["activations"],
                poison_capture["activations"],
                strict=True,
            )
        ):
            if clean_activation.shape != poison_activation.shape:
                raise RuntimeError(f"layer={layer_index}的activation形状不一致。")
            difference = clean_activation - poison_activation
            clean_norm = float(torch.linalg.vector_norm(clean_activation).item())
            poison_norm = float(torch.linalg.vector_norm(poison_activation).item())
            l2_difference = float(torch.linalg.vector_norm(difference).item())
            normalized_l2 = l2_difference / (
                clean_norm + poison_norm + 1e-12
            )
            relative_to_clean_l2 = l2_difference / (clean_norm + 1e-12)
            cosine_similarity = float(
                functional.cosine_similarity(
                    clean_activation.unsqueeze(0),
                    poison_activation.unsqueeze(0),
                    dim=-1,
                ).item()
            )
            cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
            normalized_l2_jump = (
                0.0
                if previous_normalized_l2 is None
                else normalized_l2 - previous_normalized_l2
            )
            previous_normalized_l2 = normalized_l2
            causal = causal_by_layer[layer_index]
            row = {
                "layer_index": layer_index,
                "layer_path": f"model.layers.{layer_index}",
                "clean_norm": clean_norm,
                "poison_norm": poison_norm,
                "l2_difference": l2_difference,
                "normalized_l2": normalized_l2,
                "relative_to_clean_l2": relative_to_clean_l2,
                "mean_abs_difference": float(difference.abs().mean().item()),
                "max_abs_difference": float(difference.abs().max().item()),
                "cosine_similarity": cosine_similarity,
                "cosine_distance": 1.0 - cosine_similarity,
                "normalized_l2_jump": normalized_l2_jump,
                "crs": float(causal["crs"]),
                "causal_patched_margin": float(causal["patched_margin"]),
            }
            if not all(
                math.isfinite(float(value))
                for key, value in row.items()
                if key not in {"layer_path"}
            ):
                raise RuntimeError(f"layer={layer_index}出现非有限数值。")
            rows.append(row)
            print(
                f"  layer={layer_index:02d} "
                f"normalized_l2={normalized_l2:.6f} "
                f"cosine_distance={1.0 - cosine_similarity:.6f} "
                f"CRS={float(causal['crs']):+.6f}"
            )

        hooks_after_capture = count_registered_hooks(base_model)
        max_normalized_l2 = max(rows, key=lambda item: item["normalized_l2"])
        max_cosine_distance = max(
            rows, key=lambda item: item["cosine_distance"]
        )
        largest_jump = max(rows[1:], key=lambda item: item["normalized_l2_jump"])
        max_crs = max(rows, key=lambda item: item["crs"])
        correlation = pearson_correlation(
            [row["normalized_l2"] for row in rows],
            [row["crs"] for row in rows],
        )
        summary = {
            "max_normalized_l2_layer": max_normalized_l2["layer_index"],
            "max_normalized_l2": max_normalized_l2["normalized_l2"],
            "max_cosine_distance_layer": max_cosine_distance["layer_index"],
            "max_cosine_distance": max_cosine_distance["cosine_distance"],
            "largest_normalized_l2_jump_layer": largest_jump["layer_index"],
            "largest_normalized_l2_jump": largest_jump[
                "normalized_l2_jump"
            ],
            "max_crs_layer": max_crs["layer_index"],
            "max_crs": max_crs["crs"],
            "max_difference_and_max_crs_same_layer": (
                max_normalized_l2["layer_index"] == max_crs["layer_index"]
            ),
            "pearson_normalized_l2_vs_crs": correlation,
        }
        technical_checks = {
            "alignment_valid": metadata["validation"]["alignment_valid"],
            "token_counts_equal": token_alignment["token_counts_equal"],
            "last_prompt_token_equal": token_alignment[
                "last_prompt_token_equal"
            ],
            "captured_layer_count": len(rows),
            "expected_layer_count": layer_count,
            "all_activation_shapes_equal_hidden_size": all(
                shape == [int(base_model.config.hidden_size)]
                for shape in clean_capture["activation_shapes"]
                + poison_capture["activation_shapes"]
            ),
            "all_values_finite": True,
            "hooks_before_wrap": hooks_before_wrap,
            "hooks_before_capture": hooks_before_capture,
            "hooks_after_capture": hooks_after_capture,
            "hook_counts_restored": hooks_before_capture
            == hooks_after_capture,
        }
        technical_checks["technical_pass"] = all(
            [
                technical_checks["alignment_valid"],
                technical_checks["token_counts_equal"],
                technical_checks["last_prompt_token_equal"],
                technical_checks["captured_layer_count"]
                == technical_checks["expected_layer_count"],
                technical_checks[
                    "all_activation_shapes_equal_hidden_size"
                ],
                technical_checks["all_values_finite"],
                technical_checks["hook_counts_restored"],
            ]
        )

        result = {
            "schema_version": "clean-poison-activation-difference-v1",
            "status": "completed",
            "generated_at": datetime.now().astimezone().isoformat(),
            "case_id": args.case_id,
            "purpose": (
                "Observational clean-poison activation difference; "
                "not a causal intervention result."
            ),
            "source": {
                "token_alignment": str(case_inputs["metadata_path"]),
                "token_alignment_sha256": sha256_file(
                    case_inputs["metadata_path"]
                ),
                "clean_prompt": str(case_inputs["clean_path"]),
                "clean_prompt_sha256": sha256_file(case_inputs["clean_path"]),
                "poison_prompt": str(case_inputs["poison_path"]),
                "poison_prompt_sha256": sha256_file(
                    case_inputs["poison_path"]
                ),
                "candidate_scoring": str(
                    case_inputs["candidate_scoring_path"]
                ),
                "candidate_scoring_sha256": case_inputs[
                    "candidate_scoring_sha256"
                ],
                "causal_layer_scan": str(layer_scan_path),
                "causal_layer_scan_sha256": sha256_file(layer_scan_path),
            },
            "environment": {
                "model_path": MODEL_PATH,
                "device": str(
                    base_model.get_input_embeddings().weight.device
                ),
                "dtype": str(next(base_model.parameters()).dtype),
                "torch": version("torch"),
                "transformers": version("transformers"),
                "nnsight": version("nnsight"),
                "omp_num_threads": os.environ["OMP_NUM_THREADS"],
            },
            "measurement": {
                "tensor": "decoder_layer_output",
                "token_position": -1,
                "token_position_description": "last_prompt_token",
                "forward_passes": 2,
                "intervention_applied": False,
                "l2_definition": "norm(clean_activation - poison_activation)",
                "normalized_l2_definition": (
                    "l2_difference / (clean_norm + poison_norm + epsilon)"
                ),
                "cosine_distance_definition": (
                    "1 - cosine_similarity(clean_activation, poison_activation)"
                ),
                "linked_causal_metric": "CRS from candidate_prediction_span layer scan",
            },
            "token_alignment": {
                "clean_prompt_token_count": clean_capture[
                    "prompt_token_count"
                ],
                "poison_prompt_token_count": poison_capture[
                    "prompt_token_count"
                ],
                "last_prompt_token_index": token_alignment[
                    "last_prompt_token_index"
                ],
                "clean_last_prompt_token_id": clean_capture[
                    "last_prompt_token_id"
                ],
                "poison_last_prompt_token_id": poison_capture[
                    "last_prompt_token_id"
                ],
                "positionwise_difference_count": token_alignment[
                    "positionwise_difference_count"
                ],
                "positionwise_differences": token_alignment[
                    "positionwise_differences"
                ],
            },
            "layer_results": rows,
            "summary": summary,
            "technical_checks": technical_checks,
        }

        write_json_atomic(json_path, result)
        write_csv_atomic(csv_path, rows)
        plot_case(png_path, args.case_id, rows, summary)

        print("\n===== activation差异扫描汇总 =====")
        print(
            "归一化L2差异最大层："
            f"L{summary['max_normalized_l2_layer']} "
            f"({summary['max_normalized_l2']:.6f})"
        )
        print(
            "余弦距离最大层："
            f"L{summary['max_cosine_distance_layer']} "
            f"({summary['max_cosine_distance']:.6f})"
        )
        print(
            "层间差异最大跃升点："
            f"L{summary['largest_normalized_l2_jump_layer']} "
            f"({summary['largest_normalized_l2_jump']:+.6f})"
        )
        print(
            f"因果CRS最大层：L{summary['max_crs_layer']} "
            f"({summary['max_crs']:+.6f})"
        )
        print(
            "最大差异层与最大CRS层相同："
            f"{summary['max_difference_and_max_crs_same_layer']}"
        )
        print(
            "normalized L2与CRS的Pearson相关："
            f"{summary['pearson_normalized_l2_vs_crs']}"
        )
        print(f"技术检查通过：{technical_checks['technical_pass']}")
        print(f"JSON结果：{json_path}")
        print(f"CSV结果：{csv_path}")
        print(f"图片结果：{png_path}")

    finally:
        if traced_model is not None:
            del traced_model
        if base_model is not None:
            del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(
                "CUDA缓存已清理；当前进程allocated memory："
                f"{torch.cuda.memory_allocated()} bytes"
            )


if __name__ == "__main__":
    main()
