from __future__ import annotations

from typing import Any, Literal

import torch

from core.candidate_scoring import _build_chat_prompt
from core.model_runtime import generate_answer


PatchTarget = Literal["layer", "self_attn", "mlp"]
SUPPORTED_GENERATION_TARGETS: tuple[PatchTarget, ...] = (
    "layer",
    "self_attn",
    "mlp",
)


def _validate_target(target: str) -> PatchTarget:
    if target not in SUPPORTED_GENERATION_TARGETS:
        raise ValueError(
            f"target必须是{SUPPORTED_GENERATION_TARGETS}之一，"
            f"当前为{target!r}。"
        )
    return target


def _validate_layer_index(
    model: torch.nn.Module,
    layer_index: int,
) -> int:
    layer_count = len(model.model.layers)
    if not 0 <= layer_index < layer_count:
        raise ValueError(
            f"layer_index={layer_index}越界；模型共{layer_count}层。"
        )
    return layer_count


def _target_module(
    model: torch.nn.Module,
    layer_index: int,
    target: PatchTarget,
) -> torch.nn.Module:
    layer = model.model.layers[layer_index]
    if target == "layer":
        return layer
    if target == "self_attn":
        return layer.self_attn
    return layer.mlp


def _target_path(layer_index: int, target: PatchTarget) -> str:
    if target == "layer":
        return f"model.layers.{layer_index}"
    return f"model.layers.{layer_index}.{target}"


def _output_tensor(output: Any, target: PatchTarget) -> torch.Tensor:
    if target == "self_attn":
        if not isinstance(output, tuple) or not output:
            raise TypeError("self_attn forward输出不是预期的tuple。")
        tensor = output[0]
    else:
        tensor = output
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{target} forward输出中未找到Tensor。")
    if tensor.ndim != 3:
        raise ValueError(
            f"{target}输出应为[batch, sequence, hidden]，"
            f"当前shape={tuple(tensor.shape)}。"
        )
    return tensor


def _replace_output_tensor(
    output: Any,
    patched_tensor: torch.Tensor,
    target: PatchTarget,
) -> Any:
    if target == "self_attn":
        return (patched_tensor, *output[1:])
    return patched_tensor


def _prepare_last_token_activation(
    replacement_activation: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(replacement_activation, torch.Tensor):
        raise TypeError("replacement_activation必须是Tensor。")
    if replacement_activation.ndim == 2:
        replacement_activation = replacement_activation.unsqueeze(1)
    if replacement_activation.ndim != 3:
        raise ValueError(
            "replacement_activation必须为[batch, 1, hidden]或"
            "[batch, hidden]。"
        )
    if replacement_activation.shape[0] != 1:
        raise ValueError("自由生成patch当前只支持batch_size=1。")
    if replacement_activation.shape[1] != 1:
        raise ValueError("自由生成patch必须只提供一个Prompt token。")
    return replacement_activation.detach().clone()


def generate_answer_with_prompt_last_token_patch(
    model: torch.nn.Module,
    tokenizer,
    poison_prompt: str,
    *,
    layer_index: int,
    target: PatchTarget,
    clean_last_token_activation: torch.Tensor,
) -> dict[str, Any]:
    """在生成prefill阶段替换最后一个Prompt token并自由生成。

    该函数不使用未来gold/poison token。Hook只在第一次完整Prompt前向时
    应用一次，随后模型按团队统一的确定性配置自行生成。
    """

    if not isinstance(poison_prompt, str) or not poison_prompt.strip():
        raise ValueError("poison_prompt不能为空。")
    target = _validate_target(target)
    layer_count = _validate_layer_index(model, layer_index)
    replacement = _prepare_last_token_activation(
        clean_last_token_activation
    )

    full_prompt = _build_chat_prompt(tokenizer, poison_prompt)
    prompt_token_ids = tokenizer.encode(
        full_prompt,
        add_special_tokens=False,
    )
    prompt_token_count = len(prompt_token_ids)
    if prompt_token_count <= 0:
        raise RuntimeError("生成Prompt token数为0。")

    module = _target_module(model, layer_index, target)
    state: dict[str, Any] = {
        "forward_call_count": 0,
        "patch_application_count": 0,
        "observed_sequence_lengths": [],
        "original_last_token_norm": None,
        "applied_max_abs_difference": None,
    }

    def patch_hook(_module, _inputs, output):
        state["forward_call_count"] += 1
        tensor = _output_tensor(output, target)
        sequence_length = int(tensor.shape[1])
        state["observed_sequence_lengths"].append(sequence_length)

        if state["patch_application_count"] > 0:
            return output
        if sequence_length != prompt_token_count:
            return output

        prepared = replacement.to(
            device=tensor.device,
            dtype=tensor.dtype,
        )
        if prepared.shape[-1] != tensor.shape[-1]:
            raise ValueError(
                "replacement hidden size与模块输出不一致："
                f"replacement={prepared.shape[-1]}，"
                f"output={tensor.shape[-1]}。"
            )

        original_slice = tensor[:, -1:, :].detach().float()
        patched_tensor = tensor.clone()
        patched_tensor[:, -1:, :] = prepared
        applied_slice = patched_tensor[:, -1:, :].detach().float()

        state["patch_application_count"] += 1
        state["original_last_token_norm"] = float(
            torch.linalg.vector_norm(original_slice).item()
        )
        state["applied_max_abs_difference"] = float(
            (
                applied_slice
                - prepared.detach().float()
            )
            .abs()
            .max()
            .item()
        )
        return _replace_output_tensor(output, patched_tensor, target)

    handle = module.register_forward_hook(patch_hook)
    try:
        generation = generate_answer(model, tokenizer, poison_prompt)
    finally:
        handle.remove()

    patch_applied = state["patch_application_count"] == 1
    max_difference = state["applied_max_abs_difference"]
    replacement_verified = (
        patch_applied
        and max_difference is not None
        and max_difference <= 1e-6
    )
    if not patch_applied:
        raise RuntimeError(
            "自由生成期间未找到完整Prompt prefill输出，patch未应用。"
        )
    if not replacement_verified:
        raise RuntimeError("自由生成activation替换验证失败。")

    return {
        "model_answer": generation["model_answer"],
        "input_token_count": generation["input_token_count"],
        "output_token_count": generation["output_token_count"],
        "output_token_ids": generation["output_token_ids"],
        "generation_config": generation["generation_config"],
        "patch": {
            "applied": patch_applied,
            "replacement_verified": replacement_verified,
            "patch_scope": "generation_prompt_last_token",
            "patch_direction": "clean_to_poison",
            "target": target,
            "target_path": _target_path(layer_index, target),
            "layer_index": layer_index,
            "layer_count": layer_count,
            "prompt_token_count": prompt_token_count,
            "prompt_token_position": prompt_token_count - 1,
            "replacement_shape": list(replacement.shape),
            "forward_call_count": state["forward_call_count"],
            "patch_application_count": state[
                "patch_application_count"
            ],
            "observed_sequence_lengths": state[
                "observed_sequence_lengths"
            ],
            "original_last_token_norm": state[
                "original_last_token_norm"
            ],
            "clean_replacement_norm": float(
                torch.linalg.vector_norm(replacement.float()).item()
            ),
            "applied_replacement_max_abs_difference": max_difference,
        },
    }

