import argparse
import json
from collections import defaultdict
from pathlib import Path
from random import Random

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import PROJECT_ROOT, Settings
from db.database import Database
from service.governance_data import BusinessGovernanceData

DATASET_KEY = "qwen-pollution-60"
SEED = 42
MODEL_DIR = PROJECT_ROOT.parent / "本地训练" / "models" / "Qwen2.5-0.5B-Instruct"
INSTRUCTION = (
    "判断正文与给定主题标签是否匹配。正文主题与标签明显冲突时输出「污染」。"
    "主题匹配、合理近义或属于合理的上下位类别时输出「未污染」。只输出这两个标签之一。"
)


def load_samples(version):
    database = Database(Settings.load())
    with database.connect() as connection:
        datasets = connection.execute(
            "SELECT id FROM datasets "
            "WHERE JSON_UNQUOTE(JSON_EXTRACT(metadata_json, '$.dataset_key')) = %s",
            (DATASET_KEY,),
        ).fetchall()

    if len(datasets) != 1:
        raise ValueError("请给 Qwen 的训练集设置唯一的 dataset_key：qwen-pollution-60")

    return BusinessGovernanceData(database).samples({
        "dataset_id": datasets[0]["id"],
        "version_id": version,
    })


def encode_sample(tokenizer, sample):
    data = sample["metadata"]
    text = data.get("text")
    topic = data.get("topic_label")
    gold = data.get("reference_is_anomaly")
    if not text or not topic or type(gold) is not bool:
        raise ValueError(f"样本 {sample['id']} 缺少正文、主题标签或污染标注")

    prompt = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": f"主题标签：{topic}\n正文：{text}"},
    ]
    answer = {"role": "assistant", "content": "污染" if gold else "未污染"}
    prefix = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template([*prompt, answer], tokenize=True)
    if full[:len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError(f"样本 {sample['id']} 的回答无法与提示对齐")

    inputs = torch.tensor([full])
    labels = inputs.clone()
    labels[:, :len(prefix)] = -100
    return inputs, labels


def split_samples(samples):
    groups = defaultdict(list)
    for sample in samples:
        data = sample["metadata"]
        groups[(data.get("topic_label"), data.get("reference_is_anomaly"))].append(sample)

    rng = Random(SEED)
    training, validation = [], []
    for group in groups.values():
        rng.shuffle(group)
        holdout = min(len(group) - 1, max(1, len(group) // 6))
        validation.extend(group[:holdout])
        training.extend(group[holdout:])

    rng.shuffle(training)
    if not validation:
        if len(training) < 2:
            raise ValueError("至少需要两条样本才能划分训练集和验证集")
        validation.append(training.pop())
    return training, validation


def sample_loss(model, inputs, labels):
    loss = model(input_ids=inputs, labels=labels, use_cache=False).loss
    if not torch.isfinite(loss).item():
        raise RuntimeError("模型损失出现非有限值")
    return loss


@torch.inference_mode()
def validation_loss(model, examples, cancel_path):
    model.eval()
    total_loss = 0.0
    for inputs, labels in examples:
        if cancel_path.exists():
            return None
        total_loss += sample_loss(model, inputs, labels).item()
    return total_loss / len(examples)


def report(events, event):
    events.write(json.dumps(event, ensure_ascii=False) + "\n")
    events.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    config = json.loads(parser.parse_args().config.read_text(encoding="utf-8"))
    parameters = config["parameters"]
    samples = load_samples(Path(config["data_dir"]).name)
    training, validation = split_samples(samples)
    print(f"Training samples: {len(training)} | Validation samples: {len(validation)}", flush=True)
    output_dir = Path(config["output_dir"])
    cancel_path = Path(config["cancel_path"])

    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    train_examples = [encode_sample(tokenizer, sample) for sample in training]
    validation_examples = [encode_sample(tokenizer, sample) for sample in validation]
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, use_safetensors=True, dtype=torch.float32
    )
    model = get_peft_model(base, LoraConfig(
        task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"], r=8, lora_alpha=16
    ))
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=parameters["learning_rate"],
    )

    epochs = parameters["epochs"]
    total_steps = epochs * len(train_examples)
    total_samples_per_epoch = len(train_examples) + len(validation_examples)
    step = 0
    with Path(config["events_path"]).open("a", encoding="utf-8") as events:
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            for inputs, labels in train_examples:
                if cancel_path.exists():
                    return

                optimizer.zero_grad(set_to_none=True)
                loss = sample_loss(model, inputs, labels)
                loss.backward()
                optimizer.step()

                step += 1
                epoch_loss += loss.item()
                print(f"Step {step}/{total_steps} | loss {loss.item():.4f}", flush=True)
                completed = step + (epoch - 1) * len(validation_examples)
                report(events, {
                    "type": "progress", "progress": 100 * completed / (epochs * total_samples_per_epoch)
                })

            train_loss = epoch_loss / len(train_examples)
            report(events, {
                "type": "metric", "phase": "train", "epoch": epoch, "step": step,
                "metrics": {"loss": train_loss},
            })
            valid_loss = validation_loss(model, validation_examples, cancel_path)
            if valid_loss is None or cancel_path.exists():
                return

            checkpoint_dir = output_dir if epoch == epochs else output_dir / f"checkpoint-{epoch:02d}"
            model.save_pretrained(checkpoint_dir, safe_serialization=True)
            for filename in ("adapter_model.safetensors", "adapter_config.json"):
                relative_path = (checkpoint_dir / filename).relative_to(output_dir).as_posix()
                report(events, {"type": "artifact", "filename": relative_path, "kind": "checkpoint"})

            report(events, {
                "type": "metric", "phase": "validation", "epoch": epoch, "step": step,
                "metrics": {"loss": valid_loss, "train_loss": train_loss},
            })
            report(events, {"type": "progress", "progress": 100 * epoch / epochs})
            print(f"Epoch {epoch}/{epochs} | train loss {train_loss:.4f} | validation loss {valid_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
