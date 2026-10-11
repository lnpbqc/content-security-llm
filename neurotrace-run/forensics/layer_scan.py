from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import re
import subprocess
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

import torch
from nnsight import LanguageModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.candidate_scoring import score_candidate_answer
from core.model_runtime import MODEL_PATH, load_llm
from forensics.activation_patch_scoring import (
    capture_candidate_prediction_activations,
    capture_prompt_activation,
    compare_candidate_scores,
    score_answer_pair_with_prediction_span_patch,
    score_answer_pair_with_patch,
)


DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT / "forensics" / "neural_trace_inputs_20260811"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "forensics" / "layer_scan_results_20260811"
)
CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "对一个已对齐的clean/poison案例执行Decoder Layer"
            "逐层activation patch扫描。"
        )
    )
    parser.add_argument(
        "--case-id",
        required=True,
        help="案例ID，例如CASE_NAT_030_V02。",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="prepare_neural_trace_inputs.py的输出根目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="扫描JSON和CSV输出目录。",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="扫描层：all、0-35、0,1,18或0-3,18。默认all。",
    )
    parser.add_argument(
        "--patch-scope",
        choices=("last_prompt_token", "candidate_prediction_span"),
        default="last_prompt_token",
        help=(
            "干预范围。默认仅Prompt最后一个token；"
            "多token答案可选候选预测位置整段。"
        ),
    )
    parser.add_argument(
        "--meaningful-crs-threshold",
        type=float,
        default=0.05,
        help="“明显恢复”的CRS阈值，默认0.05。",
    )
    parser.add_argument(
        "--score-tolerance",
        type=float,
        default=1e-5,
        help="NNsight无patch评分与统一评分的容差。",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON顶层必须是对象：{path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary_path.replace(path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "layer_index",
        "layer_path",
        "patched_margin",
        "margin_delta",
        "crs",
        "moves_toward_gold",
        "preferred_candidate",
        "gold_mean_logprob",
        "poison_mean_logprob",
        "clean_activation_norm",
        "clean_gold_activation_norm",
        "clean_poison_activation_norm",
        "gold_patch_verified",
        "poison_patch_verified",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def query_gpu_processes() -> list[str]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "nvidia-smi进程检查失败："
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    return [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


def count_registered_hooks(model: torch.nn.Module) -> dict[str, int]:
    forward_hooks = 0
    forward_pre_hooks = 0
    backward_hooks = 0
    for module in model.modules():
        forward_hooks += len(module._forward_hooks)
        forward_pre_hooks += len(module._forward_pre_hooks)
        backward_hooks += len(module._backward_hooks)
    return {
        "forward_hooks": forward_hooks,
        "forward_pre_hooks": forward_pre_hooks,
        "backward_hooks": backward_hooks,
    }


def parse_layer_selection(spec: str, layer_count: int) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(layer_count))

    selected: set[int] = set()
    for part in spec.split(","):
        item = part.strip()
        if not item:
            raise ValueError("--layers中存在空项。")
        if "-" in item:
            start_text, end_text = item.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"层范围起点大于终点：{item}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(item))

    invalid = sorted(
        index for index in selected if not 0 <= index < layer_count
    )
    if invalid:
        raise ValueError(
            f"层索引越界：{invalid}；模型共{layer_count}层。"
        )
    if not selected:
        raise ValueError("未选择任何层。")
    return sorted(selected)


def validate_case_id(case_id: str) -> None:
    if not CASE_ID_PATTERN.fullmatch(case_id):
        raise ValueError(
            "case-id只能包含字母、数字、下划线和短横线。"
        )


def load_case_inputs(
    input_root: Path,
    case_id: str,
) -> dict[str, Any]:
    case_dir = input_root.expanduser().resolve() / case_id
    metadata_path = case_dir / "token_alignment.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"缺少对齐元数据：{metadata_path}")

    metadata = read_json(metadata_path)
    if metadata.get("case_id") != case_id:
        raise ValueError("token_alignment.json的case_id不匹配。")
    if not metadata.get("validation", {}).get("alignment_valid"):
        raise RuntimeError("该案例未通过Prompt/token对齐检查。")

    prompt_files = metadata["prompt_files"]
    clean_path = case_dir / prompt_files["clean_prompt"]
    poison_path = case_dir / prompt_files["poison_prompt"]
    for path in (clean_path, poison_path):
        if not path.is_file():
            raise FileNotFoundError(f"缺少Prompt文件：{path}")

    actual_clean_sha = sha256_file(clean_path)
    actual_poison_sha = sha256_file(poison_path)
    if actual_clean_sha != prompt_files["clean_prompt_sha256"]:
        raise RuntimeError("clean_prompt.txt的SHA256与对齐记录不一致。")
    if actual_poison_sha != prompt_files["poison_prompt_sha256"]:
        raise RuntimeError("poison_prompt.txt的SHA256与对齐记录不一致。")

    scoring_path = PROJECT_ROOT / "core" / "candidate_scoring.py"
    expected_scoring_sha = metadata["source_files"][
        "candidate_scoring_sha256"
    ]
    actual_scoring_sha = sha256_file(scoring_path)
    if actual_scoring_sha != expected_scoring_sha:
        raise RuntimeError(
            "candidate_scoring.py已在3.1对齐后变化，"
            "请先重新生成对齐输入。"
        )

    return {
        "case_dir": case_dir,
        "metadata_path": metadata_path,
        "metadata": metadata,
        "clean_path": clean_path,
        "poison_path": poison_path,
        "clean_prompt": clean_path.read_text(encoding="utf-8"),
        "poison_prompt": poison_path.read_text(encoding="utf-8"),
        "candidate_scoring_path": scoring_path,
        "candidate_scoring_sha256": actual_scoring_sha,
    }


def compact_score(score: dict[str, Any]) -> dict[str, Any]:
    return {
        "scoring_method": score["scoring_method"],
        "candidate_answer": score["candidate_answer"],
        "candidate_token_ids": score["candidate_token_ids"],
        "candidate_tokens": score["candidate_tokens"],
        "token_logprobs": score["token_logprobs"],
        "sum_logprob": score["sum_logprob"],
        "mean_logprob": score["mean_logprob"],
    }


def score_pair_core(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    gold_answer: str,
    poison_answer: str,
) -> dict[str, Any]:
    gold_score = score_candidate_answer(
        model,
        tokenizer,
        prompt,
        gold_answer,
    )
    poison_score = score_candidate_answer(
        model,
        tokenizer,
        prompt,
        poison_answer,
    )
    return {
        "gold_score": compact_score(gold_score),
        "poison_score": compact_score(poison_score),
        "margin": (
            gold_score["mean_logprob"] - poison_score["mean_logprob"]
        ),
    }


def verify_nnsight_scoring(
    traced_model: LanguageModel,
    tokenizer,
    layer_index: int,
    clean_prompt: str,
    poison_prompt: str,
    gold_answer: str,
    poison_answer: str,
    clean_baseline: dict[str, Any],
    poison_baseline: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    clean_nnsight = score_answer_pair_with_patch(
        traced_model,
        tokenizer,
        clean_prompt,
        gold_answer,
        poison_answer,
        layer_index=layer_index,
    )
    poison_nnsight = score_answer_pair_with_patch(
        traced_model,
        tokenizer,
        poison_prompt,
        gold_answer,
        poison_answer,
        layer_index=layer_index,
    )

    checks = {
        "layer_index_used_for_unpatched_trace": layer_index,
        "clean_gold": compare_candidate_scores(
            clean_baseline["gold_score"],
            clean_nnsight["gold_score"],
            absolute_tolerance=tolerance,
        ),
        "clean_poison": compare_candidate_scores(
            clean_baseline["poison_score"],
            clean_nnsight["poison_score"],
            absolute_tolerance=tolerance,
        ),
        "poison_gold": compare_candidate_scores(
            poison_baseline["gold_score"],
            poison_nnsight["gold_score"],
            absolute_tolerance=tolerance,
        ),
        "poison_poison": compare_candidate_scores(
            poison_baseline["poison_score"],
            poison_nnsight["poison_score"],
            absolute_tolerance=tolerance,
        ),
    }
    checks["all_match"] = all(
        item["all_match"]
        for key, item in checks.items()
        if key != "layer_index_used_for_unpatched_trace"
    )
    return checks


def compact_patched_candidate(score: dict[str, Any]) -> dict[str, Any]:
    patch = score["patch"]
    return {
        "candidate_answer": score["candidate_answer"],
        "candidate_token_ids": score["candidate_token_ids"],
        "candidate_tokens": score["candidate_tokens"],
        "token_logprobs": score["token_logprobs"],
        "sum_logprob": score["sum_logprob"],
        "mean_logprob": score["mean_logprob"],
        "replacement_verified": patch["replacement_verified"],
        "applied_replacement_max_abs_difference": patch[
            "applied_replacement_max_abs_difference"
        ],
    }


def main() -> None:
    args = parse_args()
    validate_case_id(args.case_id)
    if args.meaningful_crs_threshold < 0:
        raise ValueError("--meaningful-crs-threshold不能为负数。")
    if args.score_tolerance < 0:
        raise ValueError("--score-tolerance不能为负数。")

    case_inputs = load_case_inputs(args.input_root, args.case_id)
    output_dir = args.output_dir.expanduser().resolve()
    output_stem = f"{args.case_id}.layer_scan"
    if args.patch_scope != "last_prompt_token":
        output_stem += f".{args.patch_scope}"
    json_path = output_dir / f"{output_stem}.json"
    csv_path = output_dir / f"{output_stem}.csv"
    existing_paths = [path for path in (json_path, csv_path) if path.exists()]
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
    gold_answer = metadata["gold_answer"]
    poison_answer = metadata["poison_target_answer"]
    clean_prompt = case_inputs["clean_prompt"]
    poison_prompt = case_inputs["poison_prompt"]

    base_model = None
    traced_model = None
    try:
        print("开始加载团队统一的Qwen3-4B……")
        base_model, tokenizer = load_llm()
        base_model.eval()
        layer_count = len(base_model.model.layers)
        selected_layers = parse_layer_selection(args.layers, layer_count)

        print(f"案例：{args.case_id}")
        print(f"模型层数：{layer_count}")
        print(f"本次扫描层：{selected_layers}")
        print(f"Patch范围：{args.patch_scope}")
        print(f"gold答案：{gold_answer!r}")
        print(f"poison答案：{poison_answer!r}")

        print("\n[1/3] 用统一candidate_scoring.py计算baseline……")
        clean_baseline = score_pair_core(
            base_model,
            tokenizer,
            clean_prompt,
            gold_answer,
            poison_answer,
        )
        poison_baseline = score_pair_core(
            base_model,
            tokenizer,
            poison_prompt,
            gold_answer,
            poison_answer,
        )
        denominator = clean_baseline["margin"] - poison_baseline["margin"]
        if abs(denominator) <= 1e-8:
            raise RuntimeError(
                "clean margin与poison margin几乎相同，无法稳定计算CRS。"
            )
        if denominator <= 0:
            raise RuntimeError(
                "clean margin没有高于poison margin，不满足层扫描前置条件。"
            )

        hooks_before_wrap = count_registered_hooks(base_model)
        traced_model = LanguageModel(base_model, tokenizer=tokenizer)
        hooks_before_scan = count_registered_hooks(base_model)

        print("[2/3] 核对NNsight无干预评分与统一评分……")
        scoring_consistency = verify_nnsight_scoring(
            traced_model,
            tokenizer,
            selected_layers[0],
            clean_prompt,
            poison_prompt,
            gold_answer,
            poison_answer,
            clean_baseline,
            poison_baseline,
            args.score_tolerance,
        )
        if not scoring_consistency["all_match"]:
            raise RuntimeError(
                "NNsight无patch评分与统一candidate_scoring.py不一致，"
                "已在真正扫描前停止。"
            )

        print("[3/3] 开始逐层clean→poison activation patch……")
        layer_results: list[dict[str, Any]] = []
        for ordinal, layer_index in enumerate(selected_layers, start=1):
            if args.patch_scope == "last_prompt_token":
                clean_capture = capture_prompt_activation(
                    traced_model,
                    tokenizer,
                    clean_prompt,
                    layer_index,
                    prompt_token_position=-1,
                )
                patched_pair = score_answer_pair_with_patch(
                    traced_model,
                    tokenizer,
                    poison_prompt,
                    gold_answer,
                    poison_answer,
                    layer_index=layer_index,
                    replacement_activation=clean_capture["activation"],
                    prompt_token_position=-1,
                )
                clean_activation_norm = clean_capture[
                    "activation_norm"
                ]
                clean_gold_activation_norm = clean_activation_norm
                clean_poison_activation_norm = clean_activation_norm
                position_metadata = {
                    "token_position": -1,
                    "resolved_clean_prompt_token_position": (
                        clean_capture[
                            "resolved_prompt_token_position"
                        ]
                    ),
                }
            else:
                clean_gold_capture = (
                    capture_candidate_prediction_activations(
                        traced_model,
                        tokenizer,
                        clean_prompt,
                        gold_answer,
                        layer_index,
                    )
                )
                clean_poison_capture = (
                    capture_candidate_prediction_activations(
                        traced_model,
                        tokenizer,
                        clean_prompt,
                        poison_answer,
                        layer_index,
                    )
                )
                patched_pair = (
                    score_answer_pair_with_prediction_span_patch(
                        traced_model,
                        tokenizer,
                        poison_prompt,
                        gold_answer,
                        poison_answer,
                        layer_index=layer_index,
                        gold_replacement_activations=(
                            clean_gold_capture["activations"]
                        ),
                        poison_replacement_activations=(
                            clean_poison_capture["activations"]
                        ),
                    )
                )
                clean_activation_norm = None
                clean_gold_activation_norm = clean_gold_capture[
                    "activation_norm"
                ]
                clean_poison_activation_norm = clean_poison_capture[
                    "activation_norm"
                ]
                position_metadata = {
                    "gold_prediction_span": [
                        clean_gold_capture["prediction_start_position"],
                        clean_gold_capture[
                            "prediction_end_position_exclusive"
                        ],
                    ],
                    "poison_prediction_span": [
                        clean_poison_capture[
                            "prediction_start_position"
                        ],
                        clean_poison_capture[
                            "prediction_end_position_exclusive"
                        ],
                    ],
                }

            patched_margin = float(patched_pair["margin"])
            margin_delta = patched_margin - poison_baseline["margin"]
            crs = margin_delta / denominator
            gold_patched = compact_patched_candidate(
                patched_pair["gold_score"]
            )
            poison_patched = compact_patched_candidate(
                patched_pair["poison_score"]
            )
            patch_verified = (
                gold_patched["replacement_verified"]
                and poison_patched["replacement_verified"]
            )

            row = {
                "layer_index": layer_index,
                "layer_path": f"model.layers.{layer_index}",
                "patch_scope": args.patch_scope,
                **position_metadata,
                "clean_activation_norm": clean_activation_norm,
                "clean_gold_activation_norm": (
                    clean_gold_activation_norm
                ),
                "clean_poison_activation_norm": (
                    clean_poison_activation_norm
                ),
                "gold_score": gold_patched,
                "poison_score": poison_patched,
                "patched_margin": patched_margin,
                "margin_delta": margin_delta,
                "crs": crs,
                "moves_toward_gold": margin_delta > 0,
                "preferred_candidate": (
                    "gold" if patched_margin > 0 else "poison"
                ),
                "patch_verified": patch_verified,
            }
            layer_results.append(row)
            print(
                f"  [{ordinal:02d}/{len(selected_layers):02d}] "
                f"layer={layer_index:02d} "
                f"patched_margin={patched_margin:+.6f} "
                f"delta={margin_delta:+.6f} CRS={crs:+.6f} "
                f"verified={patch_verified}"
            )
            if args.patch_scope == "last_prompt_token":
                del clean_capture
            else:
                del clean_gold_capture, clean_poison_capture
            del patched_pair

        hooks_after_scan = count_registered_hooks(base_model)
        hooks_restored = hooks_after_scan == hooks_before_scan
        all_patches_verified = all(
            item["patch_verified"] for item in layer_results
        )

        ranked = sorted(
            layer_results,
            key=lambda item: item["crs"],
            reverse=True,
        )
        top3 = [
            {
                "rank": rank,
                "layer_index": item["layer_index"],
                "layer_path": item["layer_path"],
                "patched_margin": item["patched_margin"],
                "margin_delta": item["margin_delta"],
                "crs": item["crs"],
                "preferred_candidate": item["preferred_candidate"],
            }
            for rank, item in enumerate(ranked[:3], start=1)
        ]
        positive_layers = [
            item for item in layer_results if item["margin_delta"] > 0
        ]
        meaningful_layers = [
            item
            for item in layer_results
            if item["crs"] >= args.meaningful_crs_threshold
        ]

        result = {
            "schema_version": "nnsight-layer-scan-v2",
            "status": "completed",
            "generated_at": datetime.now().astimezone().isoformat(),
            "case_id": args.case_id,
            "source": {
                "input_root": str(args.input_root.expanduser().resolve()),
                "token_alignment": str(case_inputs["metadata_path"]),
                "token_alignment_sha256": sha256_file(
                    case_inputs["metadata_path"]
                ),
                "clean_prompt": str(case_inputs["clean_path"]),
                "clean_prompt_sha256": sha256_file(
                    case_inputs["clean_path"]
                ),
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
            },
            "review_snapshot": metadata.get("review_snapshot"),
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
            "intervention": {
                "library": "nnsight",
                "mode": "local_trace",
                "patch_direction": "clean_to_poison",
                "tensor": "decoder_layer_output",
                "patch_scope": args.patch_scope,
                "token_position": (
                    -1
                    if args.patch_scope == "last_prompt_token"
                    else None
                ),
                "token_position_description": (
                    "last_prompt_token"
                    if args.patch_scope == "last_prompt_token"
                    else "candidate_prediction_positions"
                ),
                "selected_layers": selected_layers,
                "model_layer_count": layer_count,
                "all_layers_scanned": selected_layers
                == list(range(layer_count)),
            },
            "scoring": {
                "method": "teacher_forcing_candidate_only",
                "margin_definition": (
                    "gold_mean_logprob - poison_mean_logprob"
                ),
                "crs_definition": (
                    "(patched_margin - poison_margin) / "
                    "(clean_margin - poison_margin)"
                ),
                "gold_answer": gold_answer,
                "poison_target_answer": poison_answer,
                "clean_baseline": clean_baseline,
                "poison_baseline": poison_baseline,
                "clean_minus_poison_margin": denominator,
            },
            "preflight_checks": {
                "alignment_valid": metadata["validation"][
                    "alignment_valid"
                ],
                "clean_margin_above_poison_margin": denominator > 0,
                "nnsight_matches_unified_scoring": scoring_consistency,
            },
            "layer_results": layer_results,
            "summary": {
                "meaningful_crs_threshold": (
                    args.meaningful_crs_threshold
                ),
                "top3_layers": top3,
                "first_positive_recovery_layer": (
                    positive_layers[0]["layer_index"]
                    if positive_layers
                    else None
                ),
                "first_meaningful_recovery_layer": (
                    meaningful_layers[0]["layer_index"]
                    if meaningful_layers
                    else None
                ),
                "max_crs_layer": ranked[0]["layer_index"],
                "max_crs": ranked[0]["crs"],
                "positive_recovery_layer_count": len(positive_layers),
                "meaningful_recovery_layer_count": len(
                    meaningful_layers
                ),
            },
            "technical_checks": {
                "all_patches_verified": all_patches_verified,
                "hooks_before_wrap": hooks_before_wrap,
                "hooks_before_scan": hooks_before_scan,
                "hooks_after_scan": hooks_after_scan,
                "hook_counts_restored": hooks_restored,
                "technical_pass": (
                    scoring_consistency["all_match"]
                    and all_patches_verified
                    and hooks_restored
                ),
            },
        }

        csv_rows = [
            {
                "layer_index": item["layer_index"],
                "layer_path": item["layer_path"],
                "patched_margin": item["patched_margin"],
                "margin_delta": item["margin_delta"],
                "crs": item["crs"],
                "moves_toward_gold": item["moves_toward_gold"],
                "preferred_candidate": item["preferred_candidate"],
                "gold_mean_logprob": item["gold_score"][
                    "mean_logprob"
                ],
                "poison_mean_logprob": item["poison_score"][
                    "mean_logprob"
                ],
                "clean_activation_norm": item[
                    "clean_activation_norm"
                ],
                "clean_gold_activation_norm": item[
                    "clean_gold_activation_norm"
                ],
                "clean_poison_activation_norm": item[
                    "clean_poison_activation_norm"
                ],
                "gold_patch_verified": item["gold_score"][
                    "replacement_verified"
                ],
                "poison_patch_verified": item["poison_score"][
                    "replacement_verified"
                ],
            }
            for item in layer_results
        ]
        write_json_atomic(json_path, result)
        write_csv_atomic(csv_path, csv_rows)

        print("\n===== 36层扫描汇总 =====")
        print(f"clean margin：{clean_baseline['margin']:+.6f}")
        print(f"poison margin：{poison_baseline['margin']:+.6f}")
        print(f"clean-poison gap：{denominator:+.6f}")
        print("恢复效果Top-3：")
        for item in top3:
            print(
                f"  #{item['rank']} layer={item['layer_index']:02d} "
                f"CRS={item['crs']:+.6f} "
                f"delta={item['margin_delta']:+.6f} "
                f"patched_margin={item['patched_margin']:+.6f}"
            )
        print(
            "首个明显恢复层："
            f"{result['summary']['first_meaningful_recovery_layer']}"
        )
        print(f"技术检查通过：{result['technical_checks']['technical_pass']}")
        print(f"JSON结果：{json_path}")
        print(f"CSV结果：{csv_path}")

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
