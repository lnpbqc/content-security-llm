import json
from pathlib import Path


MODEL_PATH = str(Path(__file__).resolve().parents[1] / "models" / "Qwen3-4B")


def load_llm():
    """加载Qwen3-4B和tokenizer。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，无法运行正式实验。")

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前GPU不支持BF16。")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    model.eval()
    return model, tokenizer


def generate_answer(
    model,
    tokenizer,
    prompt: str,
) -> dict:
    """
    按老师规定的固定参数生成答案。

    - batch_size=1
    - enable_thinking=False
    - do_sample=False
    - max_new_tokens=64
    """
    import torch

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt不能为空。")

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    model_inputs = tokenizer(
        [full_prompt],
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)

    input_length = model_inputs["input_ids"].shape[1]

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            do_sample=False,
            max_new_tokens=64,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    output_ids = generated_ids[0, input_length:]

    model_answer = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
    ).strip()

    return {
        "full_prompt": full_prompt,
        "model_answer": model_answer,
        "input_token_ids": (
            model_inputs["input_ids"][0]
            .detach()
            .cpu()
            .tolist()
        ),
        "output_token_ids": (
            output_ids.detach().cpu().tolist()
        ),
        "input_token_count": int(input_length),
        "output_token_count": int(output_ids.shape[0]),
        "generation_config": {
            "batch_size": 1,
            "enable_thinking": False,
            "do_sample": False,
            "max_new_tokens": 64,
            "torch_dtype": "bfloat16",
        },
    }


def main():
    """最小生成测试。"""
    print("开始加载Qwen3-4B模型……")

    model, tokenizer = load_llm()

    print("模型加载成功")
    print("模型设备：", next(model.parameters()).device)
    print("模型精度：", next(model.parameters()).dtype)
    print("模型层数：", len(model.model.layers))

    result = generate_answer(
        model,
        tokenizer,
        "请只输出最终简短答案：中国的首都是哪里？",
    )

    print("\n===== 生成测试 =====")
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
