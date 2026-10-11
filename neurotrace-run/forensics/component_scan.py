from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
from collections import Counter
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

from core.model_runtime import MODEL_PATH, generate_answer, load_llm
from forensics.component_patch_scoring import (
    SUPPORTED_COMPONENTS,
    capture_component_prediction_activations,
    score_answer_pair_with_component_patch,
)
from forensics.activation_patch_scoring import (
    capture_candidate_prediction_activations,
)
from forensics.component_patch_smoke_test import (
    build_patch_result,
    select_dominant_component,
)
from forensics.document_trace import answer_matches, compact_generation
from forensics.layer_scan import (
    DEFAULT_INPUT_ROOT,
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
from forensics.patched_generation import (
    generate_answer_with_prompt_last_token_patch,
)


DEFAULT_LAYER_SCAN_DIR = (
    PROJECT_ROOT / "forensics" / "layer_scan_results_20260811"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "forensics" / "component_scan_results_20260812"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "对一个正式案例的Layer CRS Top-3及额外指定层"
            "执行Attention/MLP组件patch。"
        )
    )
    parser.add_argument("--case-id", required=True)
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
        "--top-k",
        type=int,
        default=3,
        help="从正式Layer扫描结果中选取CRS Top-K，默认3。",
    )
    parser.add_argument(
        "--include-layer",
        type=int,
        action="append",
        default=[],
        help="额外纳入的Layer，可重复传入。",
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
    )
    parser.add_argument(
        "--dominance-min-crs",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--generate-patched-answers",
        action="store_true",
        help=(
            "对整层、Attention和MLP执行Prompt最后一个token的"
            "真实自由生成patch，并写入patched_answer。"
        ),
    )
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_formal_layer_scan(
    directory: Path,
    case_id: str,
) -> dict[str, Any]:
    path = (
        directory.expanduser().resolve()
        / f"{case_id}.layer_scan.candidate_prediction_span.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"缺少正式Layer扫描结果：{path}")
    result = read_json(path)
    if result.get("case_id") != case_id:
        raise ValueError("正式Layer扫描的case_id不匹配。")
    if not result.get("technical_checks", {}).get("technical_pass"):
        raise RuntimeError("正式Layer扫描未通过技术检查。")
    if not result.get("intervention", {}).get("all_layers_scanned"):
        raise RuntimeError("正式Layer扫描不是0–35完整扫描。")
    return {
        "path": path,
        "sha256": sha256_file(path),
        "result": result,
    }


def select_layers(
    layer_scan: dict[str, Any],
    top_k: int,
    extra_layers: list[int],
    layer_count: int,
) -> tuple[list[int], dict[int, list[str]]]:
    if top_k <= 0:
        raise ValueError("--top-k必须大于0。")
    ranked = sorted(
        layer_scan["layer_results"],
        key=lambda item: item["crs"],
        reverse=True,
    )
    if top_k > len(ranked):
        raise ValueError("--top-k超过Layer结果数量。")

    reasons: dict[int, list[str]] = {}
    for rank, item in enumerate(ranked[:top_k], start=1):
        layer_index = int(item["layer_index"])
        reasons.setdefault(layer_index, []).append(
            f"layer_crs_top_{rank}"
        )
    for layer_index in extra_layers:
        if not 0 <= layer_index < layer_count:
            raise ValueError(
                f"--include-layer {layer_index}越界；"
                f"模型共{layer_count}层。"
            )
        reasons.setdefault(layer_index, []).append("explicit_extra_layer")
    return sorted(reasons), reasons


def write_csv_atomic(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fieldnames = [
        "case_id",
        "layer_index",
        "layer_path",
        "selection_reason",
        "layer_patched_margin",
        "layer_crs",
        "attention_patched_margin",
        "attention_crs",
        "attention_preferred_candidate",
        "attention_patched_answer",
        "attention_answer_matches_gold",
        "attention_answer_matches_poison",
        "mlp_patched_margin",
        "mlp_crs",
        "mlp_preferred_candidate",
        "mlp_patched_answer",
        "mlp_answer_matches_gold",
        "mlp_answer_matches_poison",
        "layer_patched_answer",
        "layer_answer_matches_gold",
        "layer_answer_matches_poison",
        "dominant_component",
        "all_replacements_verified",
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
    formal_layer = load_formal_layer_scan(
        args.layer_scan_dir,
        args.case_id,
    )
    output_dir = args.output_dir.expanduser().resolve()
    json_path = output_dir / f"{args.case_id}.component_scan.json"
    csv_path = output_dir / f"{args.case_id}.component_scan.csv"
    existing = [path for path in (json_path, csv_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "输出已存在，本次不覆盖："
            + ", ".join(str(path) for path in existing)
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
    layer_scan_result = formal_layer["result"]
    formal_by_layer = {
        int(item["layer_index"]): item
        for item in layer_scan_result["layer_results"]
    }

    base_model = None
    traced_model = None
    try:
        print("开始加载团队统一的Qwen3-4B……")
        base_model, tokenizer = load_llm()
        base_model.eval()
        layer_count = len(base_model.model.layers)
        selected_layers, selection_reasons = select_layers(
            layer_scan_result,
            args.top_k,
            args.include_layer,
            layer_count,
        )
        print(f"案例：{args.case_id}")
        print(f"选中层：{selected_layers}")
        for layer_index in selected_layers:
            print(
                f"  Layer {layer_index}: "
                + ", ".join(selection_reasons[layer_index])
            )

        baseline_generation = None
        if args.generate_patched_answers:
            print("\n[0/4] 生成clean/poison未干预答案……")
            clean_generation = generate_answer(
                base_model,
                tokenizer,
                clean_prompt,
            )
            poison_generation = generate_answer(
                base_model,
                tokenizer,
                poison_prompt,
            )
            baseline_generation = {
                "clean": compact_generation(clean_generation),
                "poison": compact_generation(poison_generation),
            }
            print(
                "  clean actual answer="
                f"{clean_generation['model_answer']!r}"
            )
            print(
                "  poison actual answer="
                f"{poison_generation['model_answer']!r}"
            )

        step_total = 4 if args.generate_patched_answers else 3
        print(f"\n[1/{step_total}] 计算统一baseline……")
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

        print(f"[2/{step_total}] 核对NNsight无patch评分……")
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
            raise RuntimeError("NNsight评分与统一评分不一致。")

        print(
            f"[3/{step_total}] 执行各选中层的"
            "Attention/MLP patch……"
        )
        layer_results: list[dict[str, Any]] = []
        for ordinal, layer_index in enumerate(selected_layers, start=1):
            formal_result = formal_by_layer[layer_index]
            print(
                f"\n  [{ordinal}/{len(selected_layers)}] "
                f"Layer {layer_index} "
                f"(整层CRS={formal_result['crs']:+.6f})"
            )
            component_results: dict[str, Any] = {}
            for component in SUPPORTED_COMPONENTS:
                gold_capture = capture_component_prediction_activations(
                    traced_model,
                    tokenizer,
                    clean_prompt,
                    gold_answer,
                    layer_index=layer_index,
                    component=component,
                )
                poison_capture = capture_component_prediction_activations(
                    traced_model,
                    tokenizer,
                    clean_prompt,
                    poison_answer,
                    layer_index=layer_index,
                    component=component,
                )
                patched_pair = score_answer_pair_with_component_patch(
                    traced_model,
                    tokenizer,
                    poison_prompt,
                    gold_answer,
                    poison_answer,
                    layer_index=layer_index,
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
                if args.generate_patched_answers:
                    generation_result = (
                        generate_answer_with_prompt_last_token_patch(
                            base_model,
                            tokenizer,
                            poison_prompt,
                            layer_index=layer_index,
                            target=component,
                            clean_last_token_activation=gold_capture[
                                "activations"
                            ][:, 0:1, :],
                        )
                    )
                    patched_answer = generation_result[
                        "model_answer"
                    ]
                    component_result.update(
                        {
                            "patched_answer": patched_answer,
                            "actual_generation_performed": True,
                            "actual_generation_note": (
                                "已执行真实贪心自由生成；生成patch只作用于"
                                "Prompt最后一个token，不使用未来候选token。"
                            ),
                            "actual_generation": generation_result,
                            "patched_answer_matches_gold": (
                                answer_matches(
                                    patched_answer,
                                    gold_answer,
                                    metadata.get(
                                        "gold_answer_aliases"
                                    ),
                                )
                            ),
                            "patched_answer_matches_poison": (
                                answer_matches(
                                    patched_answer,
                                    poison_answer,
                                    metadata.get(
                                        "poison_answer_aliases"
                                    ),
                                )
                            ),
                        }
                    )
                component_results[component] = component_result
                print(
                    f"    {component}: "
                    f"margin={component_result['patched_margin']:+.6f} "
                    f"CRS={component_result['crs']:+.6f} "
                    f"verified="
                    f"{component_result['replacement_verified']}"
                )
                if args.generate_patched_answers:
                    print(
                        "      actual answer="
                        f"{component_result['patched_answer']!r}"
                    )
                del gold_capture, poison_capture, patched_pair

            dominant_component = select_dominant_component(
                component_results["self_attn"]["crs"],
                component_results["mlp"]["crs"],
                args.dominance_crs_gap,
                args.dominance_min_crs,
            )
            layer_generation = None
            if args.generate_patched_answers:
                layer_capture = (
                    capture_candidate_prediction_activations(
                        traced_model,
                        tokenizer,
                        clean_prompt,
                        gold_answer,
                        layer_index,
                    )
                )
                layer_generation_raw = (
                    generate_answer_with_prompt_last_token_patch(
                        base_model,
                        tokenizer,
                        poison_prompt,
                        layer_index=layer_index,
                        target="layer",
                        clean_last_token_activation=layer_capture[
                            "activations"
                        ][:, 0:1, :],
                    )
                )
                layer_answer = layer_generation_raw["model_answer"]
                layer_generation = {
                    "patched_answer": layer_answer,
                    "actual_generation_performed": True,
                    "patched_answer_matches_gold": answer_matches(
                        layer_answer,
                        gold_answer,
                        metadata.get("gold_answer_aliases"),
                    ),
                    "patched_answer_matches_poison": answer_matches(
                        layer_answer,
                        poison_answer,
                        metadata.get("poison_answer_aliases"),
                    ),
                    "generation": layer_generation_raw,
                }
                print(
                    "    full layer actual answer="
                    f"{layer_answer!r}"
                )
                del layer_capture

            layer_results.append(
                {
                    "layer_index": layer_index,
                    "layer_path": f"model.layers.{layer_index}",
                    "selection_reason": selection_reasons[layer_index],
                    "full_layer_reference": {
                        "patched_margin": formal_result[
                            "patched_margin"
                        ],
                        "margin_delta": formal_result["margin_delta"],
                        "crs": formal_result["crs"],
                        "patch_verified": formal_result[
                            "patch_verified"
                        ],
                        "generation": layer_generation,
                    },
                    "component_results": component_results,
                    "summary": {
                        "layer_crs": formal_result["crs"],
                        "attention_crs": component_results[
                            "self_attn"
                        ]["crs"],
                        "mlp_crs": component_results["mlp"]["crs"],
                        "dominant_component": dominant_component,
                        "attention_plus_mlp_is_not_assumed_additive": (
                            True
                        ),
                    },
                }
            )

        if args.generate_patched_answers:
            print("[4/4] 已完成所有真实patched自由生成。")

        hooks_after_traces = count_registered_hooks(base_model)
        hooks_restored = hooks_after_traces == hooks_before_traces
        all_replacements_verified = all(
            component_result["replacement_verified"]
            for layer_result in layer_results
            for component_result in layer_result[
                "component_results"
            ].values()
        )
        all_layer_references_verified = all(
            item["full_layer_reference"]["patch_verified"]
            for item in layer_results
        )
        all_generation_patches_verified = (
            not args.generate_patched_answers
            or all(
                item["full_layer_reference"]["generation"]
                ["generation"]["patch"]["replacement_verified"]
                and all(
                    component_result["actual_generation"]
                    ["patch"]["replacement_verified"]
                    for component_result in item[
                        "component_results"
                    ].values()
                )
                for item in layer_results
            )
        )
        technical_pass = (
            scoring_consistency["all_match"]
            and all_replacements_verified
            and all_layer_references_verified
            and all_generation_patches_verified
            and hooks_restored
        )
        dominant_counts = Counter(
            item["summary"]["dominant_component"]
            for item in layer_results
        )

        result = {
            "schema_version": (
                "nnsight-component-scan-v3"
                if args.generate_patched_answers
                else "nnsight-component-scan-v2"
            ),
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
                "formal_layer_scan": str(formal_layer["path"]),
                "formal_layer_scan_sha256": formal_layer["sha256"],
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
            "selection": {
                "top_k": args.top_k,
                "explicit_extra_layers": args.include_layer,
                "dominance_crs_gap": args.dominance_crs_gap,
                "dominance_min_crs": args.dominance_min_crs,
                "selected_layers": selected_layers,
                "selection_reasons": {
                    str(key): value
                    for key, value in selection_reasons.items()
                },
            },
            "intervention": {
                "patch_scope": "candidate_prediction_span",
                "patch_direction": "clean_to_poison",
                "components": list(SUPPORTED_COMPONENTS),
                "actual_generation_performed": (
                    args.generate_patched_answers
                ),
                "generation_patch_scope": (
                    "generation_prompt_last_token"
                    if args.generate_patched_answers
                    else None
                ),
                "generation_does_not_use_future_candidate_tokens": (
                    True if args.generate_patched_answers else None
                ),
            },
            "baseline_generation": baseline_generation,
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
            "layer_results": layer_results,
            "summary": {
                "selected_layer_count": len(selected_layers),
                "dominant_component_counts": dict(dominant_counts),
                "attention_dominant_layers": [
                    item["layer_index"]
                    for item in layer_results
                    if item["summary"]["dominant_component"]
                    == "attention"
                ],
                "mlp_dominant_layers": [
                    item["layer_index"]
                    for item in layer_results
                    if item["summary"]["dominant_component"] == "mlp"
                ],
                "unclear_layers": [
                    item["layer_index"]
                    for item in layer_results
                    if item["summary"]["dominant_component"]
                    == "unclear"
                ],
            },
            "technical_checks": {
                "nnsight_matches_unified_scoring": (
                    scoring_consistency
                ),
                "all_component_replacements_verified": (
                    all_replacements_verified
                ),
                "all_layer_references_verified": (
                    all_layer_references_verified
                ),
                "all_generation_patches_verified": (
                    all_generation_patches_verified
                ),
                "hooks_before_wrap": hooks_before_wrap,
                "hooks_before_traces": hooks_before_traces,
                "hooks_after_traces": hooks_after_traces,
                "hook_counts_restored": hooks_restored,
                "technical_pass": technical_pass,
            },
        }
        write_json_atomic(json_path, result)

        csv_rows = []
        for item in layer_results:
            attention = item["component_results"]["self_attn"]
            mlp = item["component_results"]["mlp"]
            csv_rows.append(
                {
                    "case_id": args.case_id,
                    "layer_index": item["layer_index"],
                    "layer_path": item["layer_path"],
                    "selection_reason": ";".join(
                        item["selection_reason"]
                    ),
                    "layer_patched_margin": item[
                        "full_layer_reference"
                    ]["patched_margin"],
                    "layer_crs": item["summary"]["layer_crs"],
                    "attention_patched_margin": attention[
                        "patched_margin"
                    ],
                    "attention_crs": attention["crs"],
                    "attention_preferred_candidate": attention[
                        "patched_preferred_candidate"
                    ],
                    "attention_patched_answer": attention[
                        "patched_answer"
                    ],
                    "attention_answer_matches_gold": attention.get(
                        "patched_answer_matches_gold"
                    ),
                    "attention_answer_matches_poison": attention.get(
                        "patched_answer_matches_poison"
                    ),
                    "mlp_patched_margin": mlp["patched_margin"],
                    "mlp_crs": mlp["crs"],
                    "mlp_preferred_candidate": mlp[
                        "patched_preferred_candidate"
                    ],
                    "mlp_patched_answer": mlp["patched_answer"],
                    "mlp_answer_matches_gold": mlp.get(
                        "patched_answer_matches_gold"
                    ),
                    "mlp_answer_matches_poison": mlp.get(
                        "patched_answer_matches_poison"
                    ),
                    "layer_patched_answer": (
                        item["full_layer_reference"]["generation"]
                        ["patched_answer"]
                        if item["full_layer_reference"]["generation"]
                        else None
                    ),
                    "layer_answer_matches_gold": (
                        item["full_layer_reference"]["generation"]
                        ["patched_answer_matches_gold"]
                        if item["full_layer_reference"]["generation"]
                        else None
                    ),
                    "layer_answer_matches_poison": (
                        item["full_layer_reference"]["generation"]
                        ["patched_answer_matches_poison"]
                        if item["full_layer_reference"]["generation"]
                        else None
                    ),
                    "dominant_component": item["summary"][
                        "dominant_component"
                    ],
                    "all_replacements_verified": (
                        attention["replacement_verified"]
                        and mlp["replacement_verified"]
                    ),
                }
            )
        write_csv_atomic(csv_path, csv_rows)

        print("\n===== Component Scan汇总 =====")
        for item in layer_results:
            summary = item["summary"]
            print(
                f"Layer {item['layer_index']:02d}: "
                f"layer={summary['layer_crs']:+.6f}, "
                f"attention={summary['attention_crs']:+.6f}, "
                f"mlp={summary['mlp_crs']:+.6f}, "
                f"dominant={summary['dominant_component']}"
            )
        print(f"技术检查通过：{technical_pass}")
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
