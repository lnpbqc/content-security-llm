"""在训练子进程内执行可信代码，由系统控制 PyTorch 训练循环。"""

import importlib.util
import math
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, Subset, default_collate

from schemas.training import METRIC_NAMES, TrainingCreate


class TrainingCanceled(Exception):
    pass


def map_tensors(value, transform):
    if isinstance(value, torch.Tensor):
        return transform(value)
    if isinstance(value, dict) and value:
        return {key: map_tensors(item, transform) for key, item in value.items()}
    raise ValueError("数据必须为 Tensor 或非空 Tensor 字典")


def check_shape(value, spec, path):
    if isinstance(spec.get("dtype"), str) and isinstance(spec.get("shape"), list):
        if not isinstance(value, torch.Tensor):
            raise ValueError("{} 必须为 Tensor".format(path))
        if str(value.dtype) != "torch." + spec["dtype"]:
            raise ValueError("{} dtype 应为 {}，实际为 {}".format(path, spec["dtype"], value.dtype))
        shape = spec["shape"]
        if len(shape) != value.ndim or any(want is not None and want != got
                                         for want, got in zip(shape, value.shape)):
            raise ValueError("{} shape 应为 {}，实际为 {}".format(path, shape, list(value.shape)))
    else:
        if not isinstance(value, dict) or set(value) != set(spec):
            raise ValueError("{} 字典字段与 contract 不一致".format(path))
        for key in spec:
            check_shape(value[key], spec[key], path + "." + key)


def batch_size(inputs):
    sizes = [tensor.shape[0] for tensor in inputs.values() if tensor.ndim > 0]
    if len(sizes) != len(inputs) or len(set(sizes)) != 1 or sizes[0] < 1:
        raise ValueError("inputs 的每个张量必须具有相同且非空的 batch 维度")
    return sizes[0]


def validate_builtin(outputs, targets, config):
    if not isinstance(outputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise ValueError("内置 loss/指标要求 outputs 和 targets 为 Tensor；字典请使用自定义计算")
    if not outputs.is_floating_point() or not torch.isfinite(outputs).all():
        raise ValueError("模型输出必须为有限的浮点张量")
    if config["task_type"] == "multiclass":
        if outputs.ndim < 2 or outputs.shape[1] < 2:
            raise ValueError("多分类 logits 必须为 [B, C, ...]，且 C >= 2")
        if targets.dtype != torch.int64 or targets.shape != (outputs.shape[0], *outputs.shape[2:]):
            raise ValueError("多分类标签必须为 int64，shape 为 [B, ...]")
        valid = targets != config["loss_params"].get("ignore_index", -100)
        if not valid.any() or ((targets[valid] < 0) | (targets[valid] >= outputs.shape[1])).any():
            raise ValueError("多分类标签超出类别范围或 batch 全为忽略标签")
    else:
        if not targets.is_floating_point() or targets.shape != outputs.shape or not torch.isfinite(targets).all():
            raise ValueError("二分类、多标签及回归的标签须为有限浮点数且与输出 shape 一致")
        if config["task_type"] in ("binary", "multilabel"):
            if not ((targets == 0) | (targets == 1)).all():
                raise ValueError("分类标签只能为 0 或 1")
            if config["task_type"] == "binary" and not (
                outputs.ndim == 1 or (outputs.ndim == 2 and outputs.shape[1] == 1)
            ):
                raise ValueError("二分类输出必须为 [B] 或 [B, 1]")
            if config["task_type"] == "multilabel" and outputs.ndim != 2:
                raise ValueError("多标签输出必须为 [B, C]")


def classification_metrics(batches, config):
    """汇总整轮预测再计算，不对 batch 的 precision/F1 求平均。"""
    actual, predicted = [], []
    labels = None
    for batch in batches:
        output, target = batch["outputs"], batch["targets"]
        validate_builtin(output, target, config)
        if config["task_type"] == "multiclass":
            labels = np.arange(output.shape[1])
            prediction = output.argmax(dim=1)
            valid = target != config["loss_params"].get("ignore_index", -100)
            actual.append(target[valid].numpy())
            predicted.append(prediction[valid].numpy())
        else:
            prediction = (output.sigmoid() >= config["threshold"]).to(torch.int64)
            if config["task_type"] == "binary":
                prediction, target = prediction.reshape(-1), target.reshape(-1)
                labels = [0, 1]
            actual.append(target.to(torch.int64).numpy())
            predicted.append(prediction.numpy())
    y_true, y_pred = np.concatenate(actual), np.concatenate(predicted)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=config["average"],
        pos_label=config["pos_label"], zero_division=0,
    )
    return {"precision": float(precision), "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1), "recall": float(recall)}


def select_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        # 可用性标志不保证当前显卡能执行所安装 wheel 的内核。
        probe = torch.ones(2, device="cuda", requires_grad=True)
        probe.square().sum().backward()
        torch.cuda.synchronize()
        return torch.device("cuda")
    except Exception:
        if requested == "cuda":
            raise
        return torch.device("cpu")


def load_module(path):
    spec = importlib.util.spec_from_file_location("training_plugin", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ("build_model", "build_dataset"):
        if not callable(getattr(module, name, None)):
            raise ValueError("Python 模块缺少函数 {}".format(name))
    return module


def split_indices(dataset, config, module):
    count = len(dataset)
    if count < 2:
        raise ValueError("数据集至少需要两个样本用于训练和验证")
    custom = getattr(module, "split_dataset", None)
    if custom is not None:
        train, validation = custom(dataset, {"seed": config["seed"],
                                            "validation_ratio": config["validation_ratio"]})
        train, validation = list(train), list(validation)
    else:
        validation_count = min(count - 1, max(1, math.floor(count * config["validation_ratio"])))
        indices = torch.randperm(count, generator=torch.Generator().manual_seed(config["seed"])).tolist()
        validation, train = indices[:validation_count], indices[validation_count:]
    for indices in (train, validation):
        if not indices or any(type(i) is not int or i < 0 or i >= count for i in indices):
            raise ValueError("划分索引必须非空，且为数据集范围内的整数")
        if len(set(indices)) != len(indices):
            raise ValueError("划分索引不能重复")
    if set(train) & set(validation):
        raise ValueError("训练集和验证集不能重叠")
    return train, validation


def save_json(path, value):
    import json
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


def run_training(config, records, directory: Path, emit, canceled):
    """emit 上报进度；canceled 在 batch 边界协作停止。"""
    config = TrainingCreate.model_validate(config).model_dump()
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    device = select_device(config["device"])
    runtime = {"python": platform.python_version(), "torch": torch.__version__,
               "device": str(device), "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    save_json(directory / "environment.json", runtime)
    emit({"type": "runtime", "metadata": runtime})
    emit({"type": "artifact", "filename": "environment.json", "kind": "metadata"})
    module = load_module(directory / "source.py")
    for function, required in (("compute_loss", config["loss"] == "custom"),
                               ("compute_metrics", config["metrics"] == "custom")):
        if required and not callable(getattr(module, function, None)):
            raise ValueError("Python 模块缺少函数 {}".format(function))
    dataset = module.build_dataset(records, config["data_params"])
    if not isinstance(dataset, Dataset):
        raise ValueError("build_dataset 必须返回 torch.utils.data.Dataset")
    train_indices, validation_indices = split_indices(dataset, config, module)
    split = {"train_indices": train_indices, "validation_indices": validation_indices,
             "seed": config["seed"], "validation_ratio": config["validation_ratio"]}
    save_json(directory / "split.json", split)
    emit({"type": "artifact", "filename": "split.json", "kind": "metadata"})
    collate = getattr(module, "collate_batch", default_collate)
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=config["batch_size"],
                              shuffle=True, collate_fn=collate, num_workers=0,
                              generator=torch.Generator().manual_seed(config["seed"]))
    val_loader = DataLoader(Subset(dataset, validation_indices), batch_size=config["batch_size"],
                            collate_fn=collate, num_workers=0)
    model = module.build_model(config["model_params"])
    if not isinstance(model, torch.nn.Module):
        raise ValueError("build_model 必须返回 torch.nn.Module")
    model.to(device)
    optimizer = getattr(torch.optim, config["optimizer"])(
        model.parameters(), lr=config["learning_rate"], **config["optimizer_params"]
    )
    scheduler = (getattr(torch.optim.lr_scheduler, config["scheduler"])(
        optimizer, **config["scheduler_params"]) if config["scheduler"] else None)
    if config["loss"] == "custom":
        criterion = lambda output, target: module.compute_loss(output, target, config["loss_params"])
    else:
        loss_params = dict(config["loss_params"])
        output_spec = config["contract"]["outputs"]
        loss_dtype = getattr(torch, output_spec["dtype"]) if isinstance(output_spec.get("dtype"), str) else torch.float32
        for key in ("weight", "pos_weight"):
            if key in loss_params:
                loss_params[key] = torch.tensor(loss_params[key], dtype=loss_dtype, device=device)
        criterion = getattr(torch.nn, config["loss"])(**loss_params).to(device)
    step, best = 0, math.inf
    total_steps = len(train_loader) * config["epochs"]

    def check_cancel():
        if canceled():
            raise TrainingCanceled()

    def forward(batch, training):
        check_cancel()
        if not isinstance(batch, dict) or set(batch) != {"inputs", "targets"}:
            raise ValueError("batch 必须包含且仅包含 inputs、targets")
        check_shape(batch["inputs"], config["contract"]["inputs"], "inputs")
        check_shape(batch["targets"], config["contract"]["targets"], "targets")
        size = batch_size(batch["inputs"])
        inputs = map_tensors(batch["inputs"], lambda value: value.to(device))
        targets = map_tensors(batch["targets"], lambda value: value.to(device))
        for target in targets.values() if isinstance(targets, dict) else [targets]:
            if target.ndim == 0 or target.shape[0] != size:
                raise ValueError("targets batch 维度必须与 inputs 一致")
        output = model(**inputs)
        check_shape(output, config["contract"]["outputs"], "outputs")
        if config["loss"] != "custom":
            validate_builtin(output, targets, config)
        loss = criterion(output, targets)
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("loss 必须为有限的标量 Tensor")
        if training and not loss.requires_grad:
            raise ValueError("训练 loss 必须保留梯度链路")
        return size, output, targets, loss

    for epoch in range(1, config["epochs"] + 1):
        check_cancel()
        model.train()
        train_sum, train_count = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            size, _, _, loss = forward(batch, True)
            learning_rate = optimizer.param_groups[0]["lr"]
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not gradients:
                raise ValueError("loss 没有连接到模型参数的梯度链路")
            if any(not torch.isfinite(gradient).all() for gradient in gradients):
                raise ValueError("模型梯度包含非有限值")
            optimizer.step()
            step += 1
            value = loss.item()
            train_sum += value * size
            train_count += size
            emit({"type": "metric", "phase": "train", "epoch": epoch, "step": step,
                  "progress": min(99.9, 100 * step / total_steps),
                  "metrics": {"loss": value, "learning_rate": learning_rate}})
        model.eval()
        val_sum, val_count, metric_batches = 0.0, 0, []
        with torch.no_grad():
            for batch in val_loader:
                size, output, targets, loss = forward(batch, False)
                val_sum += loss.item() * size
                val_count += size
                if config["metrics"] != "none":
                    metric_batches.append({"outputs": map_tensors(output, lambda t: t.detach().cpu()),
                                           "targets": map_tensors(targets, lambda t: t.detach().cpu())})
        values = dict.fromkeys(METRIC_NAMES)
        if config["metrics"] == "custom":
            custom_values = module.compute_metrics(metric_batches, config["metric_params"])
            if not isinstance(custom_values, dict) or any(
                not isinstance(key, str) or not key or isinstance(value, bool)
                or not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value))
                for key, value in custom_values.items()
            ):
                raise ValueError("compute_metrics 必须返回有限数值的字典")
            reserved = {"loss", "train_loss", "learning_rate"}
            if reserved & custom_values.keys():
                raise ValueError("自定义指标不能覆盖 loss、train_loss、learning_rate")
            values.update({key: float(value) for key, value in custom_values.items()})
        elif config["metrics"] == "auto":
            values.update(classification_metrics(metric_batches, config))
        validation_loss = val_sum / val_count
        values.update(loss=validation_loss, train_loss=train_sum / train_count,
                      learning_rate=optimizer.param_groups[0]["lr"])
        check_cancel()
        emit({"type": "metric", "phase": "validation", "epoch": epoch, "step": step,
              "progress": min(99.9, 100 * step / total_steps), "metrics": values})
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        for filename in (["last.pt", "best.pt"] if validation_loss < best else ["last.pt"]):
            target = directory / filename
            temporary = directory / (filename + ".tmp")
            torch.save(state, temporary)
            temporary.replace(target)
            emit({"type": "artifact", "filename": filename, "kind": "weights",
                  "epoch": epoch, "loss": validation_loss})
        best = min(best, validation_loss)
        if scheduler is not None:
            if config["scheduler"] == "ReduceLROnPlateau":
                scheduler.step(validation_loss)
            else:
                scheduler.step()
    check_cancel()
