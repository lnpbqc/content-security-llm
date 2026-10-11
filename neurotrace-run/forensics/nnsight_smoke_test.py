from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any


# 必须在导入torch前修正，否则libgomp会读取到非法值并发出警告。
omp_num_threads = os.environ.get("OMP_NUM_THREADS", "")
if not omp_num_threads.isdigit() or int(omp_num_threads) <= 0:
    os.environ["OMP_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from nnsight import LanguageModel

from core.model_runtime import load_llm


CASE_ID = "CASE_NAT_030_V02"
DEFAULT_LAYER_INDEX = 18
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "forensics" / "layer_patch_smoke_test.json"
)
E1_PATH = (
    PROJECT_ROOT
    / "results"
    / "e0_e4_pilot20"
    / CASE_ID
    / "E1.json"
)
E2_PATH = (
    PROJECT_ROOT
    / "results"
    / "e0_e4_pilot20"
    / CASE_ID
    / "E2.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "使用NNsight完成单案例、单层、单token的activation patch "
            "Smoke Test。"
        )
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=DEFAULT_LAYER_INDEX,
        help="需要干预的Decoder Layer编号，默认18。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="结果JSON路径。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已经存在的结果文件。默认禁止覆盖。",
    )
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="即使检测到其他GPU计算进程也继续。协作环境中不建议使用。",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"结果文件不存在：{path}")

    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)

    if not isinstance(value, dict):
        raise TypeError(f"结果文件顶层必须是对象：{path}")

    return value


def validate_record(
    record: dict[str, Any],
    path: Path,
    expected_experiment_id: str,
) -> None:
    expected_values = {
        "case_id": CASE_ID,
        "experiment_id": expected_experiment_id,
        "status": "success",
    }

    for field, expected in expected_values.items():
        actual = record.get(field)
        if actual != expected:
            raise ValueError(
                f"{path}中的{field}不符合预期："
                f"expected={expected!r}, actual={actual!r}"
            )

    if not isinstance(record.get("rag_prompt"), str):
        raise TypeError(f"{path}缺少有效rag_prompt。")

    input_token_ids = record.get("input_token_ids")
    if not isinstance(input_token_ids, list) or not input_token_ids:
        raise TypeError(f"{path}缺少有效input_token_ids。")


def build_input_ids(
    tokenizer,
    record: dict[str, Any],
    source_path: Path,
) -> list[int]:
    messages = [
        {
            "role": "user",
            "content": record["rag_prompt"],
        }
    ]
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    rebuilt_ids = tokenizer(
        full_prompt,
        add_special_tokens=False,
    )["input_ids"]
    stored_ids = record["input_token_ids"]

    if rebuilt_ids != stored_ids:
        raise RuntimeError(
            f"当前tokenizer重建的Prompt token与{source_path}记录不一致。"
        )

    return rebuilt_ids


def get_single_token_id(tokenizer, answer: str, label: str) -> tuple[int, str]:
    token_ids = tokenizer(
        answer,
        add_special_tokens=False,
    )["input_ids"]

    if len(token_ids) != 1:
        raise RuntimeError(
            f"Smoke Test要求{label}为单token，但得到{len(token_ids)}个token："
            f"answer={answer!r}, token_ids={token_ids}"
        )

    token_id = int(token_ids[0])
    token = tokenizer.convert_ids_to_tokens([token_id])[0]
    return token_id, token


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
            "无法检查GPU进程：" + completed.stderr.strip()
        )

    return [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


def count_registered_hooks(model: torch.nn.Module) -> dict[str, int]:
    counts = {
        "forward_hooks": 0,
        "forward_pre_hooks": 0,
        "backward_hooks": 0,
    }

    for module in model.modules():
        counts["forward_hooks"] += len(module._forward_hooks)
        counts["forward_pre_hooks"] += len(module._forward_pre_hooks)
        counts["backward_hooks"] += len(module._backward_hooks)

    return counts


def make_model_inputs(
    token_ids: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def trace_clean(
    model: LanguageModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        with model.trace(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ):
            clean_activation = (
                model.model.layers[layer_index]
                .output[:, -1, :]
                .detach()
                .clone()
                .save()
            )
            clean_logits = (
                model.lm_head.output[:, -1, :]
                .detach()
                .float()
                .cpu()
                .clone()
                .save()
            )

    return clean_activation, clean_logits


def trace_baseline(
    model: LanguageModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        with model.trace(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ):
            activation = (
                model.model.layers[layer_index]
                .output[:, -1, :]
                .detach()
                .clone()
                .save()
            )
            logits = (
                model.lm_head.output[:, -1, :]
                .detach()
                .float()
                .cpu()
                .clone()
                .save()
            )

    return activation, logits


def trace_patched(
    model: LanguageModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_index: int,
    clean_activation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        with model.trace(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ):
            layer_output = model.model.layers[layer_index].output
            layer_output[:, -1, :] = clean_activation
            patched_activation = (
                model.model.layers[layer_index]
                .output[:, -1, :]
                .detach()
                .clone()
                .save()
            )
            patched_logits = (
                model.lm_head.output[:, -1, :]
                .detach()
                .float()
                .cpu()
                .clone()
                .save()
            )

    return patched_activation, patched_logits


def score_logits(
    next_token_logits: torch.Tensor,
    gold_token_id: int,
    poison_token_id: int,
) -> dict[str, float]:
    if next_token_logits.ndim != 2 or next_token_logits.shape[0] != 1:
        raise RuntimeError(
            "next-token logits形状不符合预期："
            f"{tuple(next_token_logits.shape)}"
        )

    log_probs = torch.log_softmax(
        next_token_logits.float(),
        dim=-1,
    )
    gold_logprob = float(log_probs[0, gold_token_id].item())
    poison_logprob = float(log_probs[0, poison_token_id].item())

    return {
        "gold_mean_logprob": gold_logprob,
        "poison_mean_logprob": poison_logprob,
        "margin": gold_logprob - poison_logprob,
    }


def tensor_norm(tensor: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(tensor.float()).item())


def max_abs_difference(
    left: torch.Tensor,
    right: torch.Tensor,
) -> float:
    if left.shape != right.shape:
        raise RuntimeError(
            f"比较张量形状不一致：{tuple(left.shape)} vs {tuple(right.shape)}"
        )
    return float((left.float() - right.float()).abs().max().item())


def write_json_atomic(
    output_path: Path,
    result: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    temporary_path.replace(output_path)


def main() -> None:
    args = parse_args()
    output_path = args.output.expanduser().resolve()

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"结果文件已经存在：{output_path}。"
            "如确认需要覆盖，请显式添加--overwrite。"
        )

    gpu_processes = query_gpu_processes()
    if gpu_processes and not args.allow_busy_gpu:
        raise RuntimeError(
            "检测到其他GPU计算进程，为避免影响队友，本次停止。\n"
            + "\n".join(gpu_processes)
        )

    clean_record = read_json(E1_PATH)
    poison_record = read_json(E2_PATH)
    validate_record(clean_record, E1_PATH, "E1")
    validate_record(poison_record, E2_PATH, "E2")

    if clean_record["gold_answer"] != poison_record["gold_answer"]:
        raise ValueError("E1/E2的gold_answer不一致。")
    if (
        clean_record["poison_target_answer"]
        != poison_record["poison_target_answer"]
    ):
        raise ValueError("E1/E2的poison_target_answer不一致。")

    base_model = None
    traced_model = None

    try:
        print("开始加载团队统一的Qwen3-4B……")
        base_model, tokenizer = load_llm()
        base_model.eval()

        layer_count = len(base_model.model.layers)
        if not 0 <= args.layer_index < layer_count:
            raise ValueError(
                f"layer-index必须位于0到{layer_count - 1}之间，"
                f"当前为{args.layer_index}。"
            )

        clean_token_ids = build_input_ids(
            tokenizer,
            clean_record,
            E1_PATH,
        )
        poison_token_ids = build_input_ids(
            tokenizer,
            poison_record,
            E2_PATH,
        )

        if len(clean_token_ids) != len(poison_token_ids):
            raise RuntimeError(
                "Smoke Test要求clean和poison Prompt等长，"
                f"当前为{len(clean_token_ids)}和{len(poison_token_ids)}。"
            )

        gold_answer = clean_record["gold_answer"]
        poison_answer = clean_record["poison_target_answer"]
        gold_token_id, gold_token = get_single_token_id(
            tokenizer,
            gold_answer,
            "gold_answer",
        )
        poison_token_id, poison_token = get_single_token_id(
            tokenizer,
            poison_answer,
            "poison_target_answer",
        )

        input_device = base_model.get_input_embeddings().weight.device
        clean_input_ids, clean_attention_mask = make_model_inputs(
            clean_token_ids,
            input_device,
        )
        poison_input_ids, poison_attention_mask = make_model_inputs(
            poison_token_ids,
            input_device,
        )

        hooks_before_wrap = count_registered_hooks(base_model)
        traced_model = LanguageModel(
            base_model,
            tokenizer=tokenizer,
        )
        hooks_before_traces = count_registered_hooks(base_model)

        print(f"模型层数：{layer_count}")
        print(f"干预层：model.layers.{args.layer_index}")
        print(f"Prompt token数：{len(clean_token_ids)}")
        print(
            f"候选token：gold={gold_answer!r}/{gold_token_id}, "
            f"poison={poison_answer!r}/{poison_token_id}"
        )

        print("\n[1/4] clean trace：保存activation并计算baseline……")
        clean_activation, clean_logits = trace_clean(
            traced_model,
            clean_input_ids,
            clean_attention_mask,
            args.layer_index,
        )

        print("[2/4] poison trace：计算未干预baseline……")
        poison_activation, poison_logits = trace_baseline(
            traced_model,
            poison_input_ids,
            poison_attention_mask,
            args.layer_index,
        )

        print("[3/4] patched poison trace：注入clean activation……")
        patched_activation, patched_logits = trace_patched(
            traced_model,
            poison_input_ids,
            poison_attention_mask,
            args.layer_index,
            clean_activation,
        )

        print("[4/4] cleanup trace：退出干预后重跑原始poison……")
        _post_cleanup_activation, post_cleanup_logits = trace_baseline(
            traced_model,
            poison_input_ids,
            poison_attention_mask,
            args.layer_index,
        )

        hooks_after_traces = count_registered_hooks(base_model)

        clean_scores = score_logits(
            clean_logits,
            gold_token_id,
            poison_token_id,
        )
        poison_scores = score_logits(
            poison_logits,
            gold_token_id,
            poison_token_id,
        )
        patched_scores = score_logits(
            patched_logits,
            gold_token_id,
            poison_token_id,
        )
        post_cleanup_scores = score_logits(
            post_cleanup_logits,
            gold_token_id,
            poison_token_id,
        )

        activation_patch_max_diff = max_abs_difference(
            patched_activation,
            clean_activation,
        )
        cleanup_logits_max_diff = max_abs_difference(
            post_cleanup_logits,
            poison_logits,
        )
        clean_poison_activation_diff_norm = tensor_norm(
            clean_activation - poison_activation
        )
        patched_margin_delta = (
            patched_scores["margin"] - poison_scores["margin"]
        )

        patch_matches_clean = activation_patch_max_diff <= 1e-6
        cleanup_logits_match = cleanup_logits_max_diff <= 1e-6
        hook_counts_restored = hooks_after_traces == hooks_before_traces
        scoring_precondition_pass = (
            clean_scores["margin"] > poison_scores["margin"]
        )
        causal_effect_detected = abs(patched_margin_delta) > 1e-6
        patched_toward_gold = patched_margin_delta > 0
        technical_smoke_pass = (
            patch_matches_clean
            and cleanup_logits_match
            and hook_counts_restored
        )

        result = {
            "schema_version": "nnsight-layer-patch-smoke-v1",
            "status": "completed",
            "generated_at": datetime.now().astimezone().isoformat(),
            "case_id": CASE_ID,
            "source_files": {
                "clean": str(E1_PATH),
                "poison": str(E2_PATH),
            },
            "environment": {
                "nnsight": version("nnsight"),
                "torch": version("torch"),
                "transformers": version("transformers"),
                "model_path": "/data/models/Qwen3-4B",
                "device": str(input_device),
                "dtype": str(next(base_model.parameters()).dtype),
                "omp_num_threads": os.environ["OMP_NUM_THREADS"],
            },
            "tokenization": {
                "prompt_token_count": len(clean_token_ids),
                "gold_answer": gold_answer,
                "gold_token_id": gold_token_id,
                "gold_token": gold_token,
                "poison_target_answer": poison_answer,
                "poison_token_id": poison_token_id,
                "poison_token": poison_token,
                "prompts_are_equal_length": True,
                "stored_prompt_tokens_match_current_tokenizer": True,
            },
            "intervention": {
                "library": "nnsight",
                "mode": "local_trace",
                "remote": False,
                "layer_index": args.layer_index,
                "layer_path": f"model.layers.{args.layer_index}",
                "tensor": "decoder_layer_output",
                "token_position": -1,
                "token_position_description": "last_prompt_token",
                "patch_direction": "clean_E1_to_poison_E2",
                "use_cache": False,
            },
            "scoring": {
                "method": "teacher_forcing_next_token_logprob_single_token",
                "margin_definition": (
                    "gold_mean_logprob - poison_mean_logprob"
                ),
                "clean": clean_scores,
                "poison_baseline": poison_scores,
                "patched_poison": patched_scores,
                "post_cleanup_poison": post_cleanup_scores,
                "clean_minus_poison_margin": (
                    clean_scores["margin"] - poison_scores["margin"]
                ),
                "patched_minus_poison_margin": patched_margin_delta,
            },
            "activation_checks": {
                "clean_activation_norm": tensor_norm(clean_activation),
                "poison_activation_norm": tensor_norm(poison_activation),
                "clean_poison_difference_norm": (
                    clean_poison_activation_diff_norm
                ),
                "patched_clean_max_abs_difference": (
                    activation_patch_max_diff
                ),
                "patch_matches_clean": patch_matches_clean,
            },
            "cleanup_checks": {
                "poison_post_cleanup_logits_max_abs_difference": (
                    cleanup_logits_max_diff
                ),
                "post_cleanup_logits_match_baseline": cleanup_logits_match,
                "hooks_before_wrap": hooks_before_wrap,
                "hooks_before_traces": hooks_before_traces,
                "hooks_after_traces": hooks_after_traces,
                "hook_counts_restored": hook_counts_restored,
            },
            "conclusions": {
                "scoring_precondition_pass": scoring_precondition_pass,
                "technical_smoke_pass": technical_smoke_pass,
                "causal_effect_detected_at_selected_layer": (
                    causal_effect_detected
                ),
                "patched_margin_moves_toward_gold": patched_toward_gold,
                "ready_for_layer_scan": (
                    scoring_precondition_pass and technical_smoke_pass
                ),
            },
        }

        write_json_atomic(output_path, result)

        print("\n===== NNsight Smoke Test 汇总 =====")
        print(f"clean margin：{clean_scores['margin']:.6f}")
        print(f"poison margin：{poison_scores['margin']:.6f}")
        print(f"patched margin：{patched_scores['margin']:.6f}")
        print(f"patch margin变化：{patched_margin_delta:.6f}")
        print(f"activation替换验证：{patch_matches_clean}")
        print(f"退出trace后logits恢复：{cleanup_logits_match}")
        print(f"Hook数量恢复：{hook_counts_restored}")
        print(f"技术Smoke Test通过：{technical_smoke_pass}")
        print(f"patched margin朝gold移动：{patched_toward_gold}")
        print(f"结果已保存：{output_path}")

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
