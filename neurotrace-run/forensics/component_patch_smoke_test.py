from __future__ import annotations

import argparse
import gc
import json
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

import torch
from nnsight import LanguageModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.model_runtime import MODEL_PATH, load_llm
from forensics.activation_patch_scoring import (
    capture_candidate_prediction_activations,
    score_answer_pair_with_prediction_span_patch,
)
from forensics.component_patch_scoring import (
    SUPPORTED_COMPONENTS,
    capture_component_prediction_activations,
    score_answer_pair_with_component_patch,
)
from forensics.layer_scan import (
    DEFAULT_INPUT_ROOT,
    compact_patched_candidate,
    count_registered_hooks,
    load_case_inputs,
    query_gpu_processes,
    read_json,
    score_pair_core,
    sha256_file,
    validate_case_id,
    verify_nnsight_scoring,
    write_json_atomic,
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "forensics" / "component_patch_results_20260812"
)
DEFAULT_LAYER_SCAN_DIR = (
    PROJECT_ROOT / "forensics" / "layer_scan_results_20260811"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "对一个案例、一个Layer执行整层/Attention/MLP "
            "candidate-prediction-span patch Smoke Test。"
        )
    )
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--layer-index", required=True, type=int)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
    )
    parser.add_argument(
        "--layer-scan-dir",
        type=Path,
        default=DEFAULT_LAYER_SCAN_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--score-tolerance",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--dominance-crs-gap",
        type=float,
        default=0.01,
        help="Attention/MLP CRS差距小于该值时标记unclear。",
    )
    parser.add_argument(
        "--dominance-min-crs",
        type=float,
        default=0.05,
        help="两组件最大CRS低于该值时标记unclear。",
    )
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def compact_component_score(score: dict[str, Any]) -> dict[str, Any]:
    patch = score["patch"]
    return {
        "candidate_answer": score["candidate_answer"],
        "candidate_token_ids": score["candidate_token_ids"],
        "candidate_tokens": score["candidate_tokens"],
        "token_logprobs": score["token_logprobs"],
        "sum_logprob": score["sum_logprob"],
        "mean_logprob": score["mean_logprob"],
        "component": patch["component"],
        "component_path": patch["component_path"],
        "component_output_selector": patch[
            "component_output_selector"
        ],
        "replacement_shape": patch["replacement_shape"],
        "replacement_verified": patch["replacement_verified"],
        "applied_replacement_max_abs_difference": patch[
            "applied_replacement_max_abs_difference"
        ],
    }


def build_patch_result(
    patched_pair: dict[str, Any],
    poison_margin: float,
    denominator: float,
    *,
    capture_norms: dict[str, float],
) -> dict[str, Any]:
    patched_margin = float(patched_pair["margin"])
    margin_delta = patched_margin - poison_margin
    crs = margin_delta / denominator
    gold_score = patched_pair["gold_score"]
    poison_score = patched_pair["poison_score"]
    gold_verified = gold_score["patch"]["replacement_verified"]
    poison_verified = poison_score["patch"]["replacement_verified"]
    return {
        "patched_margin": patched_margin,
        "margin_delta": margin_delta,
        "crs": crs,
        "moves_toward_gold": margin_delta > 0,
        "patched_preferred_candidate": (
            "gold" if patched_margin > 0 else "poison"
        ),
        "patched_answer": None,
        "actual_generation_performed": False,
        "actual_generation_note": (
            "Smoke Test只执行teacher-forcing候选评分，"
            "未将候选偏好冒充为自由生成答案。"
        ),
        "gold_score": compact_component_score(gold_score)
        if "component" in gold_score["patch"]
        else compact_patched_candidate(gold_score),
        "poison_score": compact_component_score(poison_score)
        if "component" in poison_score["patch"]
        else compact_patched_candidate(poison_score),
        "clean_capture_norms": capture_norms,
        "replacement_verified": gold_verified and poison_verified,
    }


def select_dominant_component(
    attention_crs: float,
    mlp_crs: float,
    gap: float,
    minimum_crs: float = 0.05,
) -> str:
    if max(attention_crs, mlp_crs) < minimum_crs:
        return "unclear"
    if abs(attention_crs - mlp_crs) < gap:
        return "unclear"
    return "attention" if attention_crs > mlp_crs else "mlp"


def load_layer_reference(
    directory: Path,
    case_id: str,
    layer_index: int,
) -> dict[str, Any]:
    path = (
        directory.expanduser().resolve()
        / f"{case_id}.layer_scan.candidate_prediction_span.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"缺少正式Layer扫描结果：{path}")
    result = read_json(path)
    layer_result = next(
        (
            item
            for item in result["layer_results"]
            if item["layer_index"] == layer_index
        ),
        None,
    )
    if layer_result is None:
        raise RuntimeError(
            f"正式Layer扫描中没有Layer {layer_index}。"
        )
    return {
        "path": path,
        "sha256": sha256_file(path),
        "result": result,
        "layer_result": layer_result,
    }


def main() -> None:
    args = parse_args()
    validate_case_id(args.case_id)
    if args.score_tolerance < 0:
        raise ValueError("--score-tolerance不能为负数。")
    if args.dominance_crs_gap < 0:
        raise ValueError("--dominance-crs-gap不能为负数。")
    if args.dominance_min_crs < 0:
        raise ValueError("--dominance-min-crs不能为负数。")

    case_inputs = load_case_inputs(args.input_root, args.case_id)
    layer_reference = load_layer_reference(
        args.layer_scan_dir,
        args.case_id,
        args.layer_index,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_path = (
        output_dir
        / f"{args.case_id}.layer{args.layer_index}.component_smoke.json"
    )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"结果已存在，本次不覆盖：{output_path}"
        )

    gpu_processes = query_gpu_processes()
    if gpu_processes and not args.allow_busy_gpu:
        raise RuntimeError(
            "检测到其他GPU计算进程，本次停止。\n"
            + "\n".join(gpu_processes)
        )

    metadata = case_inputs["metadata"]
    clean_prompt = case_inputs["clean_prompt"]
    poison_prompt = case_inputs["poison_prompt"]
    gold_answer = metadata["gold_answer"]
    poison_answer = metadata["poison_target_answer"]

    base_model = None
    traced_model = None
    try:
        print("开始加载团队统一的Qwen3-4B……")
        base_model, tokenizer = load_llm()
        base_model.eval()
        layer_count = len(base_model.model.layers)
        if not 0 <= args.layer_index < layer_count:
            raise ValueError(
                f"layer-index必须位于0到{layer_count - 1}之间。"
            )

        print(f"案例：{args.case_id}")
        print(f"层：model.layers.{args.layer_index}")
        print(f"gold答案：{gold_answer!r}")
        print(f"poison答案：{poison_answer!r}")

        print("\n[1/4] 计算统一baseline……")
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
        if denominator <= 1e-8:
            raise RuntimeError("clean/poison margin不满足CRS前置条件。")

        hooks_before_wrap = count_registered_hooks(base_model)
        traced_model = LanguageModel(base_model, tokenizer=tokenizer)
        hooks_before_traces = count_registered_hooks(base_model)

        print("[2/4] 核对NNsight无patch评分……")
        scoring_consistency = verify_nnsight_scoring(
            traced_model,
            tokenizer,
            args.layer_index,
            clean_prompt,
            poison_prompt,
            gold_answer,
            poison_answer,
            clean_baseline,
            poison_baseline,
            args.score_tolerance,
        )
        if not scoring_consistency["all_match"]:
            raise RuntimeError("NNsight评分与统一评分不一致。")

        print("[3/4] 重跑整层prediction-span patch作为参照……")
        full_gold_capture = capture_candidate_prediction_activations(
            traced_model,
            tokenizer,
            clean_prompt,
            gold_answer,
            args.layer_index,
        )
        full_poison_capture = capture_candidate_prediction_activations(
            traced_model,
            tokenizer,
            clean_prompt,
            poison_answer,
            args.layer_index,
        )
        full_pair = score_answer_pair_with_prediction_span_patch(
            traced_model,
            tokenizer,
            poison_prompt,
            gold_answer,
            poison_answer,
            layer_index=args.layer_index,
            gold_replacement_activations=full_gold_capture[
                "activations"
            ],
            poison_replacement_activations=full_poison_capture[
                "activations"
            ],
        )
        full_result = build_patch_result(
            full_pair,
            poison_baseline["margin"],
            denominator,
            capture_norms={
                "gold": full_gold_capture["activation_norm"],
                "poison": full_poison_capture["activation_norm"],
            },
        )

        reference_margin = float(
            layer_reference["layer_result"]["patched_margin"]
        )
        reference_margin_difference = abs(
            full_result["patched_margin"] - reference_margin
        )
        layer_reference_matches = (
            reference_margin_difference <= args.score_tolerance
        )
        if not layer_reference_matches:
            raise RuntimeError("整层Smoke重跑与正式Layer扫描不一致。")

        print("[4/4] 分别执行Attention和MLP patch……")
        component_results: dict[str, Any] = {}
        for component in SUPPORTED_COMPONENTS:
            print(f"  组件：{component}")
            gold_capture = capture_component_prediction_activations(
                traced_model,
                tokenizer,
                clean_prompt,
                gold_answer,
                layer_index=args.layer_index,
                component=component,
            )
            poison_capture = capture_component_prediction_activations(
                traced_model,
                tokenizer,
                clean_prompt,
                poison_answer,
                layer_index=args.layer_index,
                component=component,
            )
            patched_pair = score_answer_pair_with_component_patch(
                traced_model,
                tokenizer,
                poison_prompt,
                gold_answer,
                poison_answer,
                layer_index=args.layer_index,
                component=component,
                gold_replacement_activations=gold_capture[
                    "activations"
                ],
                poison_replacement_activations=poison_capture[
                    "activations"
                ],
            )
            component_result = build_patch_result(
                patched_pair,
                poison_baseline["margin"],
                denominator,
                capture_norms={
                    "gold": gold_capture["activation_norm"],
                    "poison": poison_capture["activation_norm"],
                },
            )
            component_result.update(
                {
                    "component": component,
                    "component_path": gold_capture[
                        "component_path"
                    ],
                    "component_output_selector": gold_capture[
                        "component_output_selector"
                    ],
                    "gold_activation_shape": gold_capture[
                        "activation_shape"
                    ],
                    "poison_activation_shape": poison_capture[
                        "activation_shape"
                    ],
                }
            )
            component_results[component] = component_result
            print(
                f"    margin={component_result['patched_margin']:+.6f} "
                f"delta={component_result['margin_delta']:+.6f} "
                f"CRS={component_result['crs']:+.6f} "
                f"verified={component_result['replacement_verified']}"
            )
            del gold_capture, poison_capture, patched_pair

        hooks_after_traces = count_registered_hooks(base_model)
        hooks_restored = hooks_after_traces == hooks_before_traces
        replacements_verified = (
            full_result["replacement_verified"]
            and all(
                item["replacement_verified"]
                for item in component_results.values()
            )
        )
        dominant_component = select_dominant_component(
            component_results["self_attn"]["crs"],
            component_results["mlp"]["crs"],
            args.dominance_crs_gap,
            args.dominance_min_crs,
        )
        technical_pass = (
            scoring_consistency["all_match"]
            and layer_reference_matches
            and replacements_verified
            and hooks_restored
        )

        result = {
            "schema_version": "nnsight-component-patch-smoke-v2",
            "status": "completed",
            "generated_at": datetime.now().astimezone().isoformat(),
            "case_id": args.case_id,
            "source": {
                "token_alignment": str(case_inputs["metadata_path"]),
                "token_alignment_sha256": sha256_file(
                    case_inputs["metadata_path"]
                ),
                "candidate_scoring": str(
                    case_inputs["candidate_scoring_path"]
                ),
                "candidate_scoring_sha256": case_inputs[
                    "candidate_scoring_sha256"
                ],
                "formal_layer_scan": str(layer_reference["path"]),
                "formal_layer_scan_sha256": layer_reference["sha256"],
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
            "intervention": {
                "layer_index": args.layer_index,
                "layer_path": f"model.layers.{args.layer_index}",
                "patch_scope": "candidate_prediction_span",
                "patch_direction": "clean_to_poison",
                "components": list(SUPPORTED_COMPONENTS),
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
            "full_layer_reference": {
                "rerun": full_result,
                "formal_patched_margin": reference_margin,
                "patched_margin_abs_difference": (
                    reference_margin_difference
                ),
                "matches_formal_layer_scan": layer_reference_matches,
            },
            "component_results": component_results,
            "summary": {
                "layer_crs": full_result["crs"],
                "attention_crs": component_results["self_attn"]["crs"],
                "mlp_crs": component_results["mlp"]["crs"],
                "dominance_crs_gap": args.dominance_crs_gap,
                "dominance_min_crs": args.dominance_min_crs,
                "dominant_component": dominant_component,
                "attention_plus_mlp_is_not_assumed_additive": True,
            },
            "technical_checks": {
                "nnsight_matches_unified_scoring": (
                    scoring_consistency
                ),
                "full_layer_matches_formal_scan": (
                    layer_reference_matches
                ),
                "all_replacements_verified": replacements_verified,
                "hooks_before_wrap": hooks_before_wrap,
                "hooks_before_traces": hooks_before_traces,
                "hooks_after_traces": hooks_after_traces,
                "hook_counts_restored": hooks_restored,
                "technical_pass": technical_pass,
            },
        }
        write_json_atomic(output_path, result)

        print("\n===== Component Smoke Test汇总 =====")
        print(f"clean margin：{clean_baseline['margin']:+.6f}")
        print(f"poison margin：{poison_baseline['margin']:+.6f}")
        print(f"Layer CRS：{full_result['crs']:+.6f}")
        print(
            "Attention CRS："
            f"{component_results['self_attn']['crs']:+.6f}"
        )
        print(f"MLP CRS：{component_results['mlp']['crs']:+.6f}")
        print(f"主导组件：{dominant_component}")
        print(f"技术检查通过：{technical_pass}")
        print(f"结果：{output_path}")

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
