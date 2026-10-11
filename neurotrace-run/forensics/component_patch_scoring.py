from __future__ import annotations

from typing import Any, Literal

import torch
from nnsight import LanguageModel

from core.candidate_scoring import (
    _build_chat_prompt,
    _check_context_length,
    _tokenize_prompt_and_candidate,
)
from forensics.activation_patch_scoring import (
    _get_base_model,
    _make_inputs,
    _prepare_replacement_activation_span,
    _validate_layer_index,
    _validate_text,
    compute_candidate_logprobs,
)


ComponentName = Literal["self_attn", "mlp"]
SUPPORTED_COMPONENTS: tuple[ComponentName, ...] = (
    "self_attn",
    "mlp",
)


def _validate_component(component: str) -> ComponentName:
    if component not in SUPPORTED_COMPONENTS:
        raise ValueError(
            f"component必须是{SUPPORTED_COMPONENTS}之一，"
            f"当前为{component!r}。"
        )
    return component


def _component_path(layer_index: int, component: ComponentName) -> str:
    return f"model.layers.{layer_index}.{component}"


def _component_output_proxy(
    model: LanguageModel,
    layer_index: int,
    component: ComponentName,
):
    layer = model.model.layers[layer_index]
    if component == "self_attn":
        return layer.self_attn.output[0]
    return layer.mlp.output


def _build_scoring_inputs(
    base_model: torch.nn.Module,
    tokenizer,
    prompt: str,
    candidate_answer: str,
) -> dict[str, Any]:
    _validate_text(prompt, "prompt")
    _validate_text(candidate_answer, "candidate_answer")

    full_prompt = _build_chat_prompt(tokenizer, prompt)
    (
        prompt_token_ids,
        candidate_token_ids,
        input_token_ids,
    ) = _tokenize_prompt_and_candidate(
        tokenizer,
        full_prompt,
        candidate_answer,
    )
    _check_context_length(base_model, len(input_token_ids))

    start_position = len(prompt_token_ids) - 1
    end_position = start_position + len(candidate_token_ids)
    input_ids, attention_mask = _make_inputs(
        base_model,
        input_token_ids,
    )
    return {
        "prompt_token_ids": prompt_token_ids,
        "candidate_token_ids": candidate_token_ids,
        "input_token_ids": input_token_ids,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "start_position": start_position,
        "end_position": end_position,
    }


def capture_component_prediction_activations(
    model: LanguageModel,
    tokenizer,
    prompt: str,
    candidate_answer: str,
    *,
    layer_index: int,
    component: ComponentName,
) -> dict[str, Any]:
    """保存clean运行中负责预测候选token的组件输出。"""

    base_model = _get_base_model(model)
    layer_count = _validate_layer_index(base_model, layer_index)
    component = _validate_component(component)
    inputs = _build_scoring_inputs(
        base_model,
        tokenizer,
        prompt,
        candidate_answer,
    )
    start_position = inputs["start_position"]
    end_position = inputs["end_position"]

    with torch.no_grad():
        with model.trace(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        ):
            component_output = _component_output_proxy(
                model,
                layer_index,
                component,
            )
            activations = (
                component_output[:, start_position:end_position, :]
                .detach()
                .clone()
                .save()
            )

    candidate_token_ids = inputs["candidate_token_ids"]
    return {
        "activations": activations,
        "component": component,
        "component_path": _component_path(layer_index, component),
        "component_output_selector": (
            "output[0]" if component == "self_attn" else "output"
        ),
        "layer_index": layer_index,
        "layer_count": layer_count,
        "prompt_token_count": len(inputs["prompt_token_ids"]),
        "candidate_token_count": len(candidate_token_ids),
        "candidate_token_ids": candidate_token_ids,
        "candidate_tokens": tokenizer.convert_ids_to_tokens(
            candidate_token_ids
        ),
        "prediction_start_position": start_position,
        "prediction_end_position_exclusive": end_position,
        "activation_shape": list(activations.shape),
        "activation_norm": float(
            torch.linalg.vector_norm(activations.float()).item()
        ),
    }


def score_candidate_answer_with_component_patch(
    model: LanguageModel,
    tokenizer,
    prompt: str,
    candidate_answer: str,
    *,
    layer_index: int,
    component: ComponentName,
    replacement_activations: torch.Tensor,
) -> dict[str, Any]:
    """替换指定组件的候选预测位置输出并重新评分。"""

    base_model = _get_base_model(model)
    layer_count = _validate_layer_index(base_model, layer_index)
    component = _validate_component(component)
    inputs = _build_scoring_inputs(
        base_model,
        tokenizer,
        prompt,
        candidate_answer,
    )
    candidate_token_ids = inputs["candidate_token_ids"]
    start_position = inputs["start_position"]
    end_position = inputs["end_position"]
    prepared_activations = _prepare_replacement_activation_span(
        replacement_activations,
        base_model,
        len(candidate_token_ids),
    )

    with torch.no_grad():
        with model.trace(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        ):
            component_output = _component_output_proxy(
                model,
                layer_index,
                component,
            )
            component_output[:, start_position:end_position, :] = (
                prepared_activations
            )
            applied_activations = (
                component_output[:, start_position:end_position, :]
                .detach()
                .clone()
                .save()
            )
            candidate_logits = (
                model.lm_head.output[
                    :,
                    start_position:end_position,
                    :,
                ]
                .detach()
                .float()
                .cpu()
                .clone()
                .save()
            )

    score_values = compute_candidate_logprobs(
        candidate_logits,
        candidate_token_ids,
    )
    max_difference = float(
        (
            applied_activations.float()
            - prepared_activations.float()
        )
        .abs()
        .max()
        .item()
    )

    return {
        "scoring_method": "teacher_forcing_candidate_only",
        "execution_backend": "nnsight_local_trace",
        "candidate_answer": candidate_answer,
        "prompt_token_count": len(inputs["prompt_token_ids"]),
        "candidate_token_count": len(candidate_token_ids),
        "candidate_token_ids": candidate_token_ids,
        "candidate_tokens": tokenizer.convert_ids_to_tokens(
            candidate_token_ids
        ),
        "token_logprobs": score_values["token_logprobs"],
        "sum_logprob": score_values["sum_logprob"],
        "mean_logprob": score_values["mean_logprob"],
        "patch": {
            "applied": True,
            "patch_scope": "candidate_prediction_span",
            "component": component,
            "component_path": _component_path(
                layer_index,
                component,
            ),
            "component_output_selector": (
                "output[0]"
                if component == "self_attn"
                else "output"
            ),
            "layer_index": layer_index,
            "layer_count": layer_count,
            "prediction_start_position": start_position,
            "prediction_end_position_exclusive": end_position,
            "replacement_shape": list(prepared_activations.shape),
            "applied_replacement_max_abs_difference": max_difference,
            "replacement_verified": max_difference <= 1e-6,
        },
    }


def score_answer_pair_with_component_patch(
    model: LanguageModel,
    tokenizer,
    prompt: str,
    gold_answer: str,
    poison_target_answer: str,
    *,
    layer_index: int,
    component: ComponentName,
    gold_replacement_activations: torch.Tensor,
    poison_replacement_activations: torch.Tensor,
) -> dict[str, Any]:
    """使用各自clean组件输出对gold/poison候选评分。"""

    gold_score = score_candidate_answer_with_component_patch(
        model,
        tokenizer,
        prompt,
        gold_answer,
        layer_index=layer_index,
        component=component,
        replacement_activations=gold_replacement_activations,
    )
    poison_score = score_candidate_answer_with_component_patch(
        model,
        tokenizer,
        prompt,
        poison_target_answer,
        layer_index=layer_index,
        component=component,
        replacement_activations=poison_replacement_activations,
    )
    return {
        "margin_definition": (
            "gold_mean_logprob - poison_mean_logprob"
        ),
        "component": component,
        "gold_score": gold_score,
        "poison_score": poison_score,
        "margin": (
            gold_score["mean_logprob"]
            - poison_score["mean_logprob"]
        ),
        "patch_applied": True,
        "patch_scope": "candidate_prediction_span",
    }
