"""独立 FashionMNIST 训练示例：python fashion_cnn.py --config config.json。

灰度 28×28 十分类，约 8 MB 数据、两层卷积：CPU 上分钟级就能跑完一轮真实训练，
用于把"真实图片样本 → 真实训练 → 真实权重 → 真实推理"整条链路在演示现场走完。
数据与 CIFAR 示例分目录缓存，下载失败直接报错，不做合成数据兜底。
"""

import argparse
import json
import platform
import random
import signal
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

LABEL_COUNT = 10
LABELS = [
    {"index": 0, "zh": "T恤/上衣", "en": "T-shirt/top"},
    {"index": 1, "zh": "长裤", "en": "Trouser"},
    {"index": 2, "zh": "套头衫", "en": "Pullover"},
    {"index": 3, "zh": "连衣裙", "en": "Dress"},
    {"index": 4, "zh": "外套", "en": "Coat"},
    {"index": 5, "zh": "凉鞋", "en": "Sandal"},
    {"index": 6, "zh": "衬衫", "en": "Shirt"},
    {"index": 7, "zh": "运动鞋", "en": "Sneaker"},
    {"index": 8, "zh": "手提包", "en": "Bag"},
    {"index": 9, "zh": "短靴", "en": "Ankle boot"},
]
# FashionMNIST 官方训练集逐像素均值/方差，写死以保证同一份权重推理口径一致。
MEAN, STD = 0.2860, 0.3536


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, LABEL_COUNT),
        )

    def forward(self, images):
        return self.layers(images)


def stratified_indices(targets, count, seed):
    """固定种子按类等额抽样，保证每次运行的训练/验证子集可复现。"""
    generator = np.random.default_rng(seed)
    targets = np.asarray(targets)
    chosen = []
    for label in range(LABEL_COUNT):
        pool = generator.permutation(np.flatnonzero(targets == label))
        quota = count // LABEL_COUNT + (label < count % LABEL_COUNT)
        if quota > len(pool):
            raise ValueError("每类配额 {} 超过该类样本量 {}".format(quota, len(pool)))
        chosen.extend(pool[:quota].tolist())
    return sorted(chosen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    params = config["parameters"]
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    events_path = Path(config["events_path"])
    events_path.parent.mkdir(parents=True, exist_ok=True)
    cancel_path = Path(config["cancel_path"])
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)

    def check_cancel():
        if stopped or cancel_path.exists():
            print("任务已请求停止", flush=True)
            raise SystemExit(0)

    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if params["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但本机 GPU/CUDA 不可用；不会自动切换 CPU")
    device = torch.device(params["device"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    # 小型示例不使用 DataLoader 子进程，便于独立运行和及时取消。
    torch.set_num_threads(2)
    with events_path.open("a", encoding="utf-8") as stream:
        def emit(event):
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()

        def artifact(filename, kind):
            emit({"type": "artifact", "filename": filename, "kind": kind})

        def save_json(filename, value):
            (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
            artifact(filename, "metadata")

        environment = {"python": platform.python_version(), "torch": torch.__version__,
                       "device": str(device), "cuda": torch.version.cuda,
                       "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
        save_json("environment.json", environment)
        save_json("parameters.json", params)
        save_json("labels.json", LABELS)
        print("环境：{}".format(json.dumps(environment, ensure_ascii=False)), flush=True)
        check_cancel()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((MEAN,), (STD,)),
        ])
        print("准备 FashionMNIST 数据；首次运行需要下载约 8 MB。", flush=True)
        train_set = datasets.FashionMNIST(root=config["data_dir"], train=True, transform=transform,
                                          download=config["inputs"]["download"])
        validation_set = datasets.FashionMNIST(root=config["data_dir"], train=False, transform=transform,
                                               download=config["inputs"]["download"])
        train_indices = stratified_indices(train_set.targets, params["train_samples"], seed)
        validation_indices = stratified_indices(validation_set.targets, params["validation_samples"], seed)
        save_json("split.json", {"seed": seed, "train_indices": train_indices,
                                 "validation_indices": validation_indices,
                                 "train_source": "official_train", "validation_source": "official_test"})
        train_loader = DataLoader(Subset(train_set, train_indices), batch_size=params["batch_size"],
                                  shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
        validation_loader = DataLoader(Subset(validation_set, validation_indices),
                                       batch_size=params["batch_size"], num_workers=0)
        model = SmallCNN().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"])
        criterion = nn.CrossEntropyLoss()
        best, step = float("inf"), 0
        total_steps = params["epochs"] * len(train_loader)
        epochs_summary = []
        for epoch in range(1, params["epochs"] + 1):
            model.train()
            loss_sum, count = 0.0, 0
            for images, targets in train_loader:
                check_cancel()
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), targets)
                if not torch.isfinite(loss):
                    raise ValueError("训练 loss 不是有限数值")
                loss.backward()
                optimizer.step()
                value, size = loss.item(), targets.shape[0]
                loss_sum += value * size
                count += size
                step += 1
                emit({"type": "metric", "phase": "train", "epoch": epoch, "step": step,
                      "metrics": {"loss": value, "learning_rate": params["learning_rate"]}})
                emit({"type": "progress", "progress": min(99.9, 100 * step / total_steps)})
            model.eval()
            actual, predicted, validation_sum, validation_count = [], [], 0.0, 0
            with torch.no_grad():
                for images, targets in validation_loader:
                    check_cancel()
                    images, targets = images.to(device), targets.to(device)
                    outputs = model(images)
                    validation_sum += criterion(outputs, targets).item() * targets.shape[0]
                    validation_count += targets.shape[0]
                    actual.extend(targets.cpu().tolist())
                    predicted.extend(outputs.argmax(dim=1).cpu().tolist())
            precision, recall, f1, _ = precision_recall_fscore_support(
                actual, predicted, labels=list(range(LABEL_COUNT)), average="macro", zero_division=0)
            values = {"loss": validation_sum / validation_count, "train_loss": loss_sum / count,
                      "accuracy": float(accuracy_score(actual, predicted)), "precision": float(precision),
                      "recall": float(recall), "f1": float(f1)}
            emit({"type": "metric", "phase": "validation", "epoch": epoch, "step": step, "metrics": values})
            print("epoch {}/{}：{}".format(epoch, params["epochs"], json.dumps(values)), flush=True)
            epochs_summary.append({"epoch": epoch, **values})
            filenames = ["last.pt", "best.pt"] if values["loss"] < best else ["last.pt"]
            for filename in filenames:
                temporary = output / (filename + ".tmp")
                torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                            "labels": [label["en"] for label in LABELS],
                            "normalization": {"mean": MEAN, "std": STD},
                            "epoch": epoch, "validation_loss": values["loss"]}, temporary)
                temporary.replace(output / filename)
                artifact(filename, "weights")
            best = min(best, values["loss"])
        check_cancel()
        save_json("validation.json", {
            "epochs": epochs_summary,
            "labels": LABELS,
            "confusion_matrix": confusion_matrix(actual, predicted, labels=list(range(LABEL_COUNT))).tolist(),
            "validation_size": len(actual),
        })
        emit({"type": "progress", "progress": 100})


if __name__ == "__main__":
    main()
