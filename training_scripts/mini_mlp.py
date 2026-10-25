"""迷你真实训练示例：合成二维数据的 MLP 二分类。

用途是演示"脚本任务→事件→产物"完整链路（本地 CPU 秒级完成），
不是课题算法成果；正式训练在 A100 上按各自登记的脚本执行。
所有 loss、accuracy 均来自真实前向与反向计算，checkpoint 为真实 state_dict。
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
config = json.loads(Path(parser.parse_args().config).read_text(encoding="utf-8"))

params = config["parameters"]
events_path = Path(config["events_path"])
output_dir = Path(config["output_dir"])
cancel_path = Path(config["cancel_path"])
BATCH = 64


def emit(event: dict) -> None:
    with events_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()


def make_data(samples: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    half = samples // 2
    inner = torch.randn(half, 2, generator=generator) * 0.6 + torch.tensor([1.6, 1.6])
    outer = torch.randn(samples - half, 2, generator=generator) * 0.6 + torch.tensor([-1.6, -1.6])
    features = torch.cat([inner, outer])
    labels = torch.cat([torch.zeros(half, dtype=torch.long), torch.ones(samples - half, dtype=torch.long)])
    shuffle = torch.randperm(features.shape[0], generator=generator)
    return features[shuffle], labels[shuffle]


model = nn.Sequential(
    nn.Linear(2, int(params["hidden_size"])),
    nn.Tanh(),
    nn.Linear(int(params["hidden_size"]), 2),
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=float(params["learning_rate"]))
torch.manual_seed(int(params["seed"]))
train_x, train_y = make_data(int(params["samples"]), int(params["seed"]))
valid_x, valid_y = make_data(256, int(params["seed"]) + 10_000)
epochs = int(params["epochs"])
steps_per_epoch = max(1, math.ceil(train_x.shape[0] / BATCH))
best_loss = float("inf")
started = time.time()
print(f"mini-mlp 开始：{epochs} 轮 × {steps_per_epoch} 批，样本 {train_x.shape[0]}", flush=True)

for epoch in range(1, epochs + 1):
    model.train()
    permutation = torch.randperm(train_x.shape[0])
    epoch_loss = 0.0
    for step, index in enumerate(permutation.split(BATCH), start=1):
        if cancel_path.exists():
            print("收到取消请求，提前退出", flush=True)
            raise SystemExit(130)
        optimizer.zero_grad()
        logits = model(train_x[index])
        loss = criterion(logits, train_y[index])
        loss.backward()
        optimizer.step()
        epoch_loss += float(loss) * index.numel()
        emit({"type": "metric", "phase": "train", "epoch": epoch, "step": step,
              "metrics": {"loss": round(float(loss), 6),
                          "learning_rate": float(params["learning_rate"])}})
    train_loss = epoch_loss / train_x.shape[0]
    model.eval()
    with torch.no_grad():
        valid_logits = model(valid_x)
        valid_loss = float(criterion(valid_logits, valid_y))
        accuracy = float((valid_logits.argmax(dim=1) == valid_y).float().mean())
    emit({"type": "metric", "phase": "validation", "epoch": epoch,
          "metrics": {"loss": round(valid_loss, 6), "train_loss": round(train_loss, 6),
                      "accuracy": round(accuracy, 4)}})
    emit({"type": "progress", "progress": round(epoch * 100 / epochs, 1)})
    state = {"model_state_dict": model.state_dict(), "epoch": epoch,
             "validation_loss": valid_loss}
    torch.save(state, output_dir / "last.pt")
    emit({"type": "artifact", "filename": "last.pt", "kind": "checkpoint"})
    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(state, output_dir / "best.pt")
        emit({"type": "artifact", "filename": "best.pt", "kind": "model"})

metadata = {"script_id": config["script_id"], "parameters": params,
            "torch_version": torch.__version__, "seconds": round(time.time() - started, 2),
            "best_validation_loss": round(best_loss, 6)}
(output_dir / "parameters.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
emit({"type": "artifact", "filename": "parameters.json", "kind": "report"})
print(f"mini-mlp 完成：best_validation_loss={best_loss:.4f}，用时 {time.time() - started:.1f}s", flush=True)
