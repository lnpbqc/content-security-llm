from __future__ import annotations

from typing import Any

import torch


def _build_chat_prompt(tokenizer, prompt: str) -> str:
    """使用与core/model_runtime.py生成阶段相同的聊天模板。"""
    messages = [{"role": "user", "content": prompt}]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _tokenize_prompt_and_candidate(
    tokenizer,
    full_prompt: str,
    candidate_answer: str,
) -> tuple[list[int], list[int], list[int]]:
    """
    在同一个完整字符串中确定候选答案token，避免Prompt/答案边界处分词漂移。

    返回(prompt_token_ids, candidate_token_ids, input_token_ids)。
    """
    prompt_token_ids = tokenizer(
        full_prompt,
        add_special_tokens=False,
    )["input_ids"]
    input_token_ids = tokenizer(
        full_prompt + candidate_answer,
        add_special_tokens=False,
    )["input_ids"]

    if not prompt_token_ids:
        raise RuntimeError("Prompt没有生成有效token。")

    if input_token_ids[: len(prompt_token_ids)] != prompt_token_ids:
        raise RuntimeError(
            "Prompt与候选答案拼接后发生边界分词变化。"
            "请检查聊天模板末尾和candidate_answer开头是否缺少空格或换行。"
        )

    candidate_token_ids = input_token_ids[len(prompt_token_ids) :]

    if not candidate_token_ids:
        raise RuntimeError("候选答案没有生成有效token。")

    return prompt_token_ids, candidate_token_ids, input_token_ids


def _model_input_device(model) -> torch.device:
    """取得输入Embedding所在设备，兼容单GPU及device_map加载。"""
    embeddings = model.get_input_embeddings()

    if embeddings is None or not hasattr(embeddings, "weight"):
        raise RuntimeError("无法确定模型输入Embedding设备。")

    return embeddings.weight.device


def _check_context_length(model, input_length: int) -> None:
    max_positions = getattr(
        getattr(model, "config", None),
        "max_position_embeddings",
        None,
    )

    if (
        isinstance(max_positions, int)
        and max_positions > 0
        and input_length > max_positions
    ):
        raise ValueError(
            f"输入token数{input_length}超过模型上下文长度{max_positions}。"
        )


def score_candidate_answer(
    model,
    tokenizer,
    prompt: str,
    candidate_answer: str,
) -> dict[str, Any]:
    """
    使用teacher forcing计算候选答案的逐token log probability。

    参数prompt必须是core/prompt_builder.py生成的原始RAG Prompt，
    不能传已经套过聊天模板的full_prompt，否则会重复套模板。

    本函数：
    - 关闭thinking并复用生成阶段的聊天模板；
    - 只对候选答案token计分；
    - 不生成答案、不训练模型、不修改模型参数；
    - 不自动追加EOS，比较的是候选答案文本本身；
    - 返回逐token、sum和mean log probability。
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt不能为空。")

    if (
        not isinstance(candidate_answer, str)
        or not candidate_answer.strip()
    ):
        raise ValueError("candidate_answer不能为空。")

    if model.training:
        raise RuntimeError(
            "模型当前处于训练模式，请先调用model.eval()。"
        )

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

    _check_context_length(model, len(input_token_ids))
    input_device = _model_input_device(model)
    input_ids = torch.tensor(
        [input_token_ids],
        dtype=torch.long,
        device=input_device,
    )
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

    if not hasattr(outputs, "logits"):
        raise RuntimeError("模型前向输出中没有logits。")

    prompt_length = len(prompt_token_ids)
    candidate_length = len(candidate_token_ids)

    # 因果语言模型的logits位置比目标token位置提前一位：
    # Prompt最后一个token预测候选答案第一个token。
    start_position = prompt_length - 1
    end_position = start_position + candidate_length
    candidate_logits = outputs.logits[
        0,
        start_position:end_position,
        :,
    ]
    candidate_labels = input_ids[
        0,
        prompt_length : prompt_length + candidate_length,
    ]

    if candidate_logits.shape[0] != candidate_labels.shape[0]:
        raise RuntimeError(
            "候选答案logits数量与token数量不一致："
            f"logits={candidate_logits.shape[0]}，"
            f"labels={candidate_labels.shape[0]}。"
        )

    # BF16 logits先转换为float32再计算log_softmax，减少数值误差。
    candidate_log_probs = torch.log_softmax(
        candidate_logits.float(),
        dim=-1,
    )
    token_logprobs_tensor = candidate_log_probs.gather(
        dim=-1,
        index=candidate_labels.unsqueeze(-1),
    ).squeeze(-1)

    if not torch.isfinite(token_logprobs_tensor).all():
        raise RuntimeError("候选答案的token logprob出现NaN或Inf。")

    token_logprobs = (
        token_logprobs_tensor.detach().cpu().tolist()
    )
    candidate_tokens = tokenizer.convert_ids_to_tokens(
        candidate_token_ids
    )

    if len(token_logprobs) != candidate_length:
        raise RuntimeError("逐token logprob数量不正确。")

    token_details = [
        {
            "candidate_position": position,
            "token_id": int(token_id),
            "token": token,
            "logprob": float(logprob),
        }
        for position, (token_id, token, logprob) in enumerate(
            zip(
                candidate_token_ids,
                candidate_tokens,
                token_logprobs,
            )
        )
    ]

    return {
        "scoring_method": "teacher_forcing_candidate_only",
        "chat_template": {
            "enable_thinking": False,
            "add_generation_prompt": True,
            "add_special_tokens_after_template": False,
        },
        "candidate_answer": candidate_answer,
        "append_eos": False,
        "full_prompt": full_prompt,
        "prompt_token_count": prompt_length,
        "candidate_token_count": candidate_length,
        "input_token_count": len(input_token_ids),
        "prompt_token_ids": [int(item) for item in prompt_token_ids],
        "candidate_token_ids": [
            int(item) for item in candidate_token_ids
        ],
        "candidate_tokens": candidate_tokens,
        "input_token_ids": [int(item) for item in input_token_ids],
        "token_logprobs": [
            float(item) for item in token_logprobs
        ],
        "token_details": token_details,
        "sum_logprob": float(token_logprobs_tensor.sum().item()),
        "mean_logprob": float(token_logprobs_tensor.mean().item()),
    }
