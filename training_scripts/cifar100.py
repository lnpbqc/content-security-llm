"""独立 CIFAR-100 训练示例：python cifar100.py --config config.json。"""

import argparse
import json
import platform
import random
import signal
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(128, 100),
        )

    def forward(self, images):
        return self.layers(images)


def stratified_split(targets, train_count, validation_count, seed):
    """固定种子按类分配配额；训练与验证来自同一训练集且不重叠。"""
    generator = np.random.default_rng(seed)
    targets = np.asarray(targets)
    train, validation = [], []
    for label in range(100):
        indices = generator.permutation(np.flatnonzero(targets == label))
        ntrain = train_count // 100 + (label < train_count % 100)
        nval = validation_count // 100 + (label >= 100 - validation_count % 100)
        if ntrain < 1 or nval < 1 or ntrain + nval > len(indices):
            raise ValueError("每类训练和验证配额必须非空且不超过该类样本量")
        train.extend(indices[:ntrain].tolist())
        validation.extend(indices[ntrain:ntrain + nval].tolist())
    return train, validation


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
        print("环境：{}".format(json.dumps(environment, ensure_ascii=False)), flush=True)
        check_cancel()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        print("准备 CIFAR-100 数据；首次运行可能需要下载。", flush=True)
        dataset = datasets.CIFAR100(root=config["data_dir"], train=True, transform=transform,
                                    download=config["inputs"]["download"])
        train_indices, validation_indices = stratified_split(
            dataset.targets, params["train_samples"], params["validation_samples"], seed)
        save_json("split.json", {"seed": seed, "train_indices": train_indices,
                                 "validation_indices": validation_indices, "source_split": "official_train"})
        train_loader = DataLoader(Subset(dataset, train_indices), batch_size=params["batch_size"], shuffle=True,
                                  generator=torch.Generator().manual_seed(seed), num_workers=0)
        validation_loader = DataLoader(Subset(dataset, validation_indices), batch_size=params["batch_size"], num_workers=0)
        model = SmallCNN().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"])
        criterion = nn.CrossEntropyLoss()
        best, step = float("inf"), 0
        total_steps = params["epochs"] * len(train_loader)
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
                actual, predicted, labels=list(range(100)), average="macro", zero_division=0)
            values = {"loss": validation_sum / validation_count, "train_loss": loss_sum / count,
                      "accuracy": float(accuracy_score(actual, predicted)), "precision": float(precision),
                      "recall": float(recall), "f1": float(f1)}
            emit({"type": "metric", "phase": "validation", "epoch": epoch, "step": step, "metrics": values})
            print("epoch {}/{}：{}".format(epoch, params["epochs"], json.dumps(values)), flush=True)
            filenames = ["last.pt", "best.pt"] if values["loss"] < best else ["last.pt"]
            for filename in filenames:
                temporary = output / (filename + ".tmp")
                torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, temporary)
                temporary.replace(output / filename)
                artifact(filename, "weights")
            best = min(best, values["loss"])
        check_cancel()
        emit({"type": "progress", "progress": 100})


if __name__ == "__main__":
    main()
