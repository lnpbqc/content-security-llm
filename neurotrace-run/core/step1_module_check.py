from pathlib import Path

from core.model_runtime import load_llm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "forensics" / "model_module_tree.txt"


def main() -> None:
    """加载团队统一模型，并导出可供 Hook 使用的模块路径。"""
    model, _tokenizer = load_llm()

    module_rows = [
        f"{name}  ->  {type(module)}"
        for name, module in model.named_modules()
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        "\n".join(module_rows) + "\n",
        encoding="utf-8",
    )

    module_names = {name for name, _module in model.named_modules()}
    required_paths = {
        "model.layers": "model.layers",
        "self_attn": "model.layers.0.self_attn",
        "mlp": "model.layers.0.mlp",
        "o_proj": "model.layers.0.self_attn.o_proj",
        "down_proj": "model.layers.0.mlp.down_proj",
    }
    missing = [
        label
        for label, path in required_paths.items()
        if path not in module_names
    ]

    print(f"模块树已经输出到：{OUTPUT_PATH}")
    print(f"模型层数：{len(model.model.layers)}")

    if missing:
        raise RuntimeError(
            "以下关键模块路径未找到：" + ", ".join(missing)
        )

    print("关键模块路径检查通过：layers、self_attn、mlp、o_proj、down_proj")


if __name__ == "__main__":
    main()
