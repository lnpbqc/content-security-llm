from __future__ import annotations

from typing import Any

import torch
from nnsight import LanguageModel

from core.candidate_scoring import (
    _build_chat_prompt,
    _check_context_length,
    _tokenize_prompt_and_candidate,
)


def _validate_text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}不能为空。")


def _get_base_model(model: LanguageModel) -> torch.nn.Module:
    if not isinstance(model, LanguageModel):
        raise TypeError(
            "model必须是nnsight.LanguageModel实例。"
        )

    base_model = model._module
    if not isinstance(base_model, torch.nn.Module):
        raise TypeError("NNsight包装器没有有效的底层PyTorch模型。")

    if base_model.training:
        raise RuntimeError(
            "底层模型当前处于训练模式，请先调用model.eval()。"
        )

    return base_model


def _build_prompt_token_ids(tokenizer, prompt: str) -> list[int]:
    _validate_text(prompt, "prompt")

    full_prompt = _build_chat_prompt(tokenizer, prompt)
    prompt_token_ids = tokenizer(
        full_prompt,
        add_special_tokens=False,
    )["input_ids"]

    if not prompt_token_ids:
        raise RuntimeError("Prompt没有生成有效token。")

    return [int(token_id) for token_id in prompt_token_ids]


def _resolve_prompt_token_position(
    position: int,
    prompt_token_count: int,
) -> int:
    if not isinstance(position, int):
        raise TypeError("prompt_token_position必须是整数。")

    resolved_position = position
    if resolved_position < 0:
        resolved_position += prompt_token_count

    if not 0 <= resolved_position < prompt_token_count:
        raise IndexError(
            "Prompt token位置越界："
            f"position={position}, prompt_token_count={prompt_token_count}"
        )

    return resolved_position


def _validate_layer_index(
    base_model: torch.nn.Module,
    layer_index: int,
) -> int:
    if not isinstance(layer_index, int):
        raise TypeError("layer_index必须是整数。")

    if not hasattr(base_model, "model") or not hasattr(
        base_model.model,
        "layers",
    ):
        raise AttributeError(
            "底层模型没有model.layers，无法执行Layer patch。"
        )

    layer_count = len(base_model.model.layers)
    if not 0 <= layer_index < layer_count:
        raise IndexError(
            f"layer_index必须位于0到{layer_count - 1}之间，"
            f"当前为{layer_index}。"
        )

    return layer_count


def _prepare_replacement_activation(
    activation: torch.Tensor,
    base_model: torch.nn.Module,
) -> torch.Tensor:
    if not isinstance(activation, torch.Tensor):
        raise TypeError("replacement_activation必须是torch.Tensor。")

    prepared = activation.detach()
    if prepared.ndim == 1:
        prepared = prepared.unsqueeze(0)

    if prepared.ndim != 2 or prepared.shape[0] != 1:
        raise ValueError(
            "replacement_activation形状必须为[hidden_size]或"
            f"[1, hidden_size]，当前为{tuple(prepared.shape)}。"
        )

    hidden_size = int(base_model.config.hidden_size)
    if prepared.shape[1] != hidden_size:
        raise ValueError(
            "replacement_activation的hidden_size不匹配："
            f"expected={hidden_size}, actual={prepared.shape[1]}"
        )

    input_embeddings = base_model.get_input_embeddings().weight
    return prepared.to(
        device=input_embeddings.device,
        dtype=input_embeddings.dtype,
    )


def _prepare_replacement_activation_span(
    activation: torch.Tensor,
    base_model: torch.nn.Module,
    expected_length: int,
) -> torch.Tensor:
    if not isinstance(activation, torch.Tensor):
        raise TypeError("replacement_activations必须是torch.Tensor。")

    prepared = activation.detach()
    if prepared.ndim == 2:
        prepared = prepared.unsqueeze(0)

    if prepared.ndim != 3 or prepared.shape[0] != 1:
        raise ValueError(
            "replacement_activations形状必须为[length, hidden_size]"
            f"或[1, length, hidden_size]，当前为{tuple(prepared.shape)}。"
        )

    if prepared.shape[1] != expected_length:
        raise ValueError(
            "replacement_activations的序列长度不匹配："
            f"expected={expected_length}, actual={prepared.shape[1]}"
        )

    hidden_size = int(base_model.config.hidden_size)
    if prepared.shape[2] != hidden_size:
        raise ValueError(
            "replacement_activations的hidden_size不匹配："
            f"expected={hidden_size}, actual={prepared.shape[2]}"
        )

    input_embeddings = base_model.get_input_embeddings().weight
    return prepared.to(
        device=input_embeddings.device,
        dtype=input_embeddings.dtype,
    )


def _make_inputs(
    base_model: torch.nn.Module,
    token_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    input_device = base_model.get_input_embeddings().weight.device
    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=input_device,
    )
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def compute_candidate_logprobs(
    candidate_logits: torch.Tensor,
    candidate_token_ids: list[int],
) -> dict[str, Any]:
    """根据teacher-forcing logits计算候选答案逐token分数。"""

    if not isinstance(candidate_logits, torch.Tensor):
        raise TypeError("candidate_logits必须是torch.Tensor。")

    if candidate_logits.ndim == 3:
        if candidate_logits.shape[0] != 1:
            raise ValueError("只支持batch_size=1。")
        candidate_logits = candidate_logits[0]

    if candidate_logits.ndim != 2:
        raise ValueError(
            "candidate_logits形状必须为[candidate_length, vocab_size]"
            f"或[1, candidate_length, vocab_size]，当前为"
            f"{tuple(candidate_logits.shape)}。"
        )

    if candidate_logits.shape[0] != len(candidate_token_ids):
        raise ValueError(
            "候选logits数量与token数量不一致："
            f"logits={candidate_logits.shape[0]}, "
            f"tokens={len(candidate_token_ids)}"
        )

    labels = torch.tensor(
        candidate_token_ids,
        dtype=torch.long,
        device=candidate_logits.device,
    )
    log_probs = torch.log_softmax(
        candidate_logits.float(),
        dim=-1,
    )
    token_logprobs_tensor = (
        log_probs
        .gather(
            dim=-1,
            index=labels.unsqueeze(-1),
        )
        .squeeze(-1)
    )

    return {
        "token_logprobs": (
            token_logprobs_tensor.detach().cpu().tolist()
        ),
        "sum_logprob": float(token_logprobs_tensor.sum().item()),
        "mean_logprob": float(token_logprobs_tensor.mean().item()),
    }


def capture_prompt_activation(
    model: LanguageModel,
    tokenizer,
    prompt: str,
    layer_index: int,
    prompt_token_position: int = -1,
) -> dict[str, Any]:
    """
    保存Prompt在指定Decoder Layer、指定Prompt token位置的输出。

    返回的activation仍位于模型设备上，供后续NNsight patch直接使用。
    默认位置-1表示Prompt最后一个token，而不是整段候选输入的最后一个token。
    """

    base_model = _get_base_model(model)
    layer_count = _validate_layer_index(base_model, layer_index)
    prompt_token_ids = _build_prompt_token_ids(tokenizer, prompt)
    resolved_position = _resolve_prompt_token_position(
        prompt_token_position,
        len(prompt_token_ids),
    )
    input_ids, attention_mask = _make_inputs(
        base_model,
        prompt_token_ids,
    )

    with torch.no_grad():
        with model.trace(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ):
            activation = (
                model.model.layers[layer_index]
                .output[:, resolved_position, :]
                .detach()
                .clone()
                .save()
            )

    return {
        "activation": activation,
        "layer_index": layer_index,
        "layer_count": layer_count,
        "layer_path": f"model.layers.{layer_index}",
        "prompt_token_position": prompt_token_position,
        "resolved_prompt_token_position": resolved_position,
        "prompt_token_count": len(prompt_token_ids),
        "prompt_token_ids": prompt_token_ids,
        "activation_shape": list(activation.shape),
        "activation_norm": float(
            torch.linalg.vector_norm(activation.float()).item()
        ),
    }


def score_candidate_answer_with_patch(
    model: LanguageModel,
    tokenizer,
    prompt: str,
    candidate_answer: str,
    *,
    layer_index: int,
    replacement_activation: torch.Tensor | None = None,
    prompt_token_position: int = -1,
) -> dict[str, Any]:
    """
    使用NNsight和teacher forcing评分任意长度候选答案。

    replacement_activation=None时只执行NNsight trace，不做干预，
    可用于和core/candidate_scoring.py核对评分口径。

    有replacement_activation时，在指定Decoder Layer的指定Prompt token
    位置注入activation。候选答案无论包含多少token，patch位置始终按
    Prompt长度解析，不会误放到候选答案末尾。
    """

    base_model = _get_base_model(model)
    layer_count = _validate_layer_index(base_model, layer_index)
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
    prompt_length = len(prompt_token_ids)
    candidate_length = len(candidate_token_ids)
    _check_context_length(base_model, len(input_token_ids))
    resolved_position = _resolve_prompt_token_position(
        prompt_token_position,
        prompt_length,
    )

    input_ids, attention_mask = _make_inputs(
        base_model,
        input_token_ids,
    )

    start_position = prompt_length - 1
    end_position = start_position + candidate_length
    patch_applied = replacement_activation is not None
    prepared_activation = None

    if patch_applied:
        prepared_activation = _prepare_replacement_activation(
            replacement_activation,
            base_model,
        )

    with torch.no_grad():
        with model.trace(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ):
            if patch_applied:
                layer_output = model.model.layers[layer_index].output
                layer_output[:, resolved_position, :] = prepared_activation
                applied_activation = (
                    model.model.layers[layer_index]
                    .output[:, resolved_position, :]
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

    result: dict[str, Any] = {
        "scoring_method": "teacher_forcing_candidate_only",
        "execution_backend": "nnsight_local_trace",
        "candidate_answer": candidate_answer,
        "prompt_token_count": prompt_length,
        "candidate_token_count": candidate_length,
        "prompt_token_ids": prompt_token_ids,
        "candidate_token_ids": candidate_token_ids,
        "candidate_tokens": tokenizer.convert_ids_to_tokens(
            candidate_token_ids
        ),
        "input_token_ids": input_token_ids,
        "token_logprobs": score_values["token_logprobs"],
        "sum_logprob": score_values["sum_logprob"],
        "mean_logprob": score_values["mean_logprob"],
        "patch": {
            "applied": patch_applied,
            "patch_scope": "last_prompt_token",
            "layer_index": layer_index,
            "layer_count": layer_count,
            "layer_path": f"model.layers.{layer_index}",
            "prompt_token_position": prompt_token_position,
            "resolved_prompt_token_position": resolved_position,
            "candidate_logit_start_position": start_position,
            "candidate_logit_end_position_exclusive": end_position,
        },
    }

    if patch_applied:
        max_difference = float(
            (
                applied_activation.float()
                - prepared_activation.float()
            )
            .abs()
            .max()
            .item()
        )
        result["patch"].update(
            {
                "replacement_shape": list(prepared_activation.shape),
                "applied_replacement_max_abs_difference": max_difference,
                "replacement_verified": max_difference <= 1e-6,
            }
        )

    return result


def score_answer_pair_with_patch(
    model: LanguageModel,
    tokenizer,
    prompt: str,
    gold_answer: str,
    poison_target_answer: str,
    *,
    layer_index: int,
    replacement_activation: torch.Tensor | None = None,
    prompt_token_position: int = -1,
) -> dict[str, Any]:
    """分别评分gold/poison答案并返回统一margin。"""

    gold_score = score_candidate_answer_with_patch(
        model,
        tokenizer,
        prompt,
        gold_answer,
        layer_index=layer_index,
        replacement_activation=replacement_activation,
        prompt_token_position=prompt_token_position,
    )
    poison_score = score_candidate_answer_with_patch(
        model,
        tokenizer,
        prompt,
        poison_target_answer,
        layer_index=layer_index,
        replacement_activation=replacement_activation,
        prompt_token_position=prompt_token_position,
    )

    return {
        "margin_definition": (
            "gold_mean_logprob - poison_mean_logprob"
        ),
        "gold_score": gold_score,
        "poison_score": poison_score,
        "margin": (
            gold_score["mean_logprob"]
            - poison_score["mean_logprob"]
        ),
        "patch_applied": replacement_activation is not None,
    }


def capture_candidate_prediction_activations(
    model: LanguageModel,
    tokenizer,
    prompt: str,
    candidate_answer: str,
    layer_index: int,
) -> dict[str, Any]:
    """保存负责预测候选答案各token的一段Layer输出。"""

    base_model = _get_base_model(model)
    layer_count = _validate_layer_index(base_model, layer_index)
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

    prompt_length = len(prompt_token_ids)
    candidate_length = len(candidate_token_ids)
    start_position = prompt_length - 1
    end_position = start_position + candidate_length
    input_ids, attention_mask = _make_inputs(
        base_model,
        input_token_ids,
    )

    with torch.no_grad():
        with model.trace(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ):
            activations = (
                model.model.layers[layer_index]
                .output[:, start_position:end_position, :]
                .detach()
                .clone()
                .save()
            )

    return {
        "activations": activations,
        "layer_index": layer_index,
        "layer_count": layer_count,
        "layer_path": f"model.layers.{layer_index}",
        "patch_scope": "candidate_prediction_span",
        "prompt_token_count": prompt_length,
        "candidate_token_count": candidate_length,
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


def score_candidate_answer_with_prediction_span_patch(
    model: LanguageModel,
    tokenizer,
    prompt: str,
    candidate_answer: str,
    *,
    layer_index: int,
    replacement_activations: torch.Tensor,
) -> dict[str, Any]:
    """
    替换负责预测候选答案各token的整段Layer输出并评分。

    这是多token答案的一维Layer扫描扩展，不扫描全部Prompt token。
    """

    base_model = _get_base_model(model)
    layer_count = _validate_layer_index(base_model, layer_index)
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

    prompt_length = len(prompt_token_ids)
    candidate_length = len(candidate_token_ids)
    start_position = prompt_length - 1
    end_position = start_position + candidate_length
    prepared_activations = _prepare_replacement_activation_span(
        replacement_activations,
        base_model,
        candidate_length,
    )
    input_ids, attention_mask = _make_inputs(
        base_model,
        input_token_ids,
    )

    with torch.no_grad():
        with model.trace(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ):
            layer_output = model.model.layers[layer_index].output
            layer_output[:, start_position:end_position, :] = (
                prepared_activations
            )
            applied_activations = (
                model.model.layers[layer_index]
                .output[:, start_position:end_position, :]
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
        "prompt_token_count": prompt_length,
        "candidate_token_count": candidate_length,
        "prompt_token_ids": prompt_token_ids,
        "candidate_token_ids": candidate_token_ids,
        "candidate_tokens": tokenizer.convert_ids_to_tokens(
            candidate_token_ids
        ),
        "input_token_ids": input_token_ids,
        "token_logprobs": score_values["token_logprobs"],
        "sum_logprob": score_values["sum_logprob"],
        "mean_logprob": score_values["mean_logprob"],
        "patch": {
            "applied": True,
            "patch_scope": "candidate_prediction_span",
            "layer_index": layer_index,
            "layer_count": layer_count,
            "layer_path": f"model.layers.{layer_index}",
            "prediction_start_position": start_position,
            "prediction_end_position_exclusive": end_position,
            "replacement_shape": list(prepared_activations.shape),
            "applied_replacement_max_abs_difference": max_difference,
            "replacement_verified": max_difference <= 1e-6,
        },
    }


def score_answer_pair_with_prediction_span_patch(
    model: LanguageModel,
    tokenizer,
    prompt: str,
    gold_answer: str,
    poison_target_answer: str,
    *,
    layer_index: int,
    gold_replacement_activations: torch.Tensor,
    poison_replacement_activations: torch.Tensor,
) -> dict[str, Any]:
    """使用各自的clean预测位置表示评分gold/poison答案。"""

    gold_score = score_candidate_answer_with_prediction_span_patch(
        model,
        tokenizer,
        prompt,
        gold_answer,
        layer_index=layer_index,
        replacement_activations=gold_replacement_activations,
    )
    poison_score = score_candidate_answer_with_prediction_span_patch(
        model,
        tokenizer,
        prompt,
        poison_target_answer,
        layer_index=layer_index,
        replacement_activations=poison_replacement_activations,
    )

    return {
        "margin_definition": (
            "gold_mean_logprob - poison_mean_logprob"
        ),
        "gold_score": gold_score,
        "poison_score": poison_score,
        "margin": (
            gold_score["mean_logprob"]
            - poison_score["mean_logprob"]
        ),
        "patch_applied": True,
        "patch_scope": "candidate_prediction_span",
    }


def compare_candidate_scores(
    reference_score: dict[str, Any],
    nnsight_score: dict[str, Any],
    *,
    absolute_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """比较普通评分与NNsight无patch评分是否使用相同口径。"""

    if absolute_tolerance < 0:
        raise ValueError("absolute_tolerance不能为负数。")

    reference_ids = reference_score.get("candidate_token_ids")
    nnsight_ids = nnsight_score.get("candidate_token_ids")
    token_ids_match = reference_ids == nnsight_ids

    reference_logprobs = reference_score.get("token_logprobs", [])
    nnsight_logprobs = nnsight_score.get("token_logprobs", [])
    token_count_match = len(reference_logprobs) == len(nnsight_logprobs)

    if token_count_match and reference_logprobs:
        token_logprob_max_abs_difference = max(
            abs(float(left) - float(right))
            for left, right in zip(
                reference_logprobs,
                nnsight_logprobs,
                strict=True,
            )
        )
    elif token_count_match:
        token_logprob_max_abs_difference = 0.0
    else:
        token_logprob_max_abs_difference = float("inf")

    sum_difference = abs(
        float(reference_score["sum_logprob"])
        - float(nnsight_score["sum_logprob"])
    )
    mean_difference = abs(
        float(reference_score["mean_logprob"])
        - float(nnsight_score["mean_logprob"])
    )
    sum_logprob_tolerance = absolute_tolerance * max(
        1,
        len(reference_logprobs),
    )

    all_match = (
        token_ids_match
        and token_count_match
        and token_logprob_max_abs_difference <= absolute_tolerance
        and sum_difference <= sum_logprob_tolerance
        and mean_difference <= absolute_tolerance
    )

    return {
        "absolute_tolerance": absolute_tolerance,
        "sum_logprob_tolerance": sum_logprob_tolerance,
        "token_ids_match": token_ids_match,
        "token_count_match": token_count_match,
        "token_logprob_max_abs_difference": (
            token_logprob_max_abs_difference
        ),
        "sum_logprob_abs_difference": sum_difference,
        "mean_logprob_abs_difference": mean_difference,
        "all_match": all_match,
    }
