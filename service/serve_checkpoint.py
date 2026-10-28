"""通用训练产物推理服务：给任意 outputs 目录里的 best.pt 起一个端口。

启动（在 content-security-llm 目录下）：
    python service/serve_checkpoint.py --weights "db/training-runs/<任务ID>/outputs" --port 8020
    python service/serve_checkpoint.py --weights "db/training-runs/<任务ID>/outputs/best.pt" --port 8021
不给 --weights 时挑 db/training-runs 下最新的 best.pt。

识别本项目 training_scripts 的三种真实产物，结构不认识就明确报错退出，不做猜测式加载：
    fashion-cnn  含 state_dict/labels/normalization 的 bundle，1×28×28 灰度，归一化参数取自权重文件
    mini-mlp     model_state_dict，Linear→Tanh→Linear，特征维度与隐藏层从权重形状读出
    cifar100     裸 state_dict（layers.*），3×32×32 彩色，归一化沿用训练脚本的 CIFAR 常量

接口：
    GET  /healthz  权重身份（路径、sha256、架构、输入规格、类别、归一化、训练轮次/验证损失）
    POST /predict  {"input": [...], "top_k": 3}
                   图像模型 input 为按 (C,H,W) 展平的 0-255 像素；特征模型为同口径特征值。
                   返回真实 softmax 概率，不做任何编造。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# 与 training_scripts/cifar100.py 的 transforms.Normalize 参数一致；
# CIFAR 产物是裸 state_dict，没有自描述归一化字段，只能沿用训练时的常量。
CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD = (0.2675, 0.2565, 0.2761)

_SCRIPT_CACHE: dict[str, object] = {}


def _import_training_script(filename: str):
    """按文件路径加载 training_scripts 下的模型定义；这些脚本都有 __main__ 守卫，导入不会触发训练。"""
    if filename not in _SCRIPT_CACHE:
        path = REPO_ROOT / "training_scripts" / filename
        spec = importlib.util.spec_from_file_location(f"serve_{Path(filename).stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SCRIPT_CACHE[filename] = module
    return _SCRIPT_CACHE[filename]


def resolve_weights(target: Path) -> Path:
    """文件直接用；目录按 best.pt → outputs/best.pt 找，找不到再递归挑最新的 best.pt。"""
    if target.is_file():
        return target
    if not target.is_dir():
        raise FileNotFoundError(f"权重路径不存在：{target}")
    for candidate in (target / "best.pt", target / "outputs" / "best.pt",
                      target / "last.pt", target / "outputs" / "last.pt"):
        if candidate.is_file():
            return candidate
    found = [path for path in target.rglob("best.pt") if path.is_file()]
    if found:
        return max(found, key=lambda path: path.stat().st_mtime)
    raise FileNotFoundError(f"{target} 下没有 best.pt / last.pt")


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _sibling_metadata(weights: Path, name: str):
    """产物元数据可能在权重同目录（outputs/），也可能在任务目录（老任务的权重直接放任务根）。"""
    for directory in (weights.parent, weights.parent.parent, weights.parent / "outputs"):
        path = directory / name
        if path.is_file():
            value = _read_json(path)
            if value is not None:
                return value
    return None


def _extract_state(checkpoint) -> tuple[dict, dict]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        return checkpoint["state_dict"], checkpoint
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model_state_dict"), dict):
        return checkpoint["model_state_dict"], checkpoint
    if isinstance(checkpoint, dict) and checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint, {}
    raise ValueError("checkpoint 结构无法识别：既不是自描述 bundle，也不是裸 state_dict")


def _output_features(state: dict) -> int:
    matrices = [value for value in state.values() if torch.is_tensor(value) and value.ndim == 2]
    if not matrices:
        raise ValueError("state_dict 里没有线性层权重，无法确定输出类别数")
    return int(matrices[-1].shape[0])


def _labels_from(count: int, labels_file, bundle_labels) -> list[dict]:
    source = labels_file if isinstance(labels_file, list) and labels_file else bundle_labels
    if source:
        entries = []
        for position, item in enumerate(source):
            if isinstance(item, dict):
                index = int(item.get("index", position))
                en = str(item.get("en") or item.get("label") or f"class-{index}")
                zh = str(item.get("zh") or en)
            else:
                index, en, zh = position, str(item), str(item)
            entries.append({"index": index, "label": en, "labelZh": zh})
        if len(entries) != count:
            raise ValueError(f"标签有 {len(entries)} 类，权重输出 {count} 类，不匹配")
        return entries
    return [{"index": index, "label": f"class-{index}", "labelZh": f"class-{index}"} for index in range(count)]


@dataclass
class LoadedModel:
    model: nn.Module
    architecture: str
    input_kind: str
    input_shape: tuple[int, ...]
    labels: list[dict]
    normalization: dict | None = None
    epoch: int | None = None
    validation_loss: float | None = None


def load_weights(weights: Path) -> LoadedModel:
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    state, bundle = _extract_state(checkpoint)
    parameters = _sibling_metadata(weights, "parameters.json")
    parameters = parameters if isinstance(parameters, dict) else {}
    labels_file = _sibling_metadata(weights, "labels.json")
    output_count = _output_features(state)

    conv_weights = [value for value in state.values() if torch.is_tensor(value) and value.ndim == 4]
    if conv_weights:
        channels = int(conv_weights[0].shape[1])
        if channels == 1:
            if not {"labels", "normalization"} <= set(bundle):
                raise ValueError("1 通道卷积权重缺少 labels/normalization 自描述字段，不是 fashion-cnn 训练产物")
            script = _import_training_script("fashion_cnn.py")
            model = script.SmallCNN()
            normalization = {"mean": [float(bundle["normalization"]["mean"])],
                             "std": [float(bundle["normalization"]["std"])]}
            input_shape = (1, 28, 28)
            architecture = f"fashion-cnn SmallCNN(1×28×28→{output_count})"
            labels = _labels_from(output_count, labels_file, bundle["labels"])
            epoch, validation_loss = bundle.get("epoch"), bundle.get("validation_loss")
        elif channels == 3:
            script = _import_training_script("cifar100.py")
            model = script.SmallCNN()
            normalization = {"mean": list(CIFAR_MEAN), "std": list(CIFAR_STD)}
            input_shape = (3, 32, 32)
            architecture = f"cifar100 SmallCNN(3×32×32→{output_count})"
            labels = _labels_from(output_count, labels_file, None)
            epoch, validation_loss = None, None
        else:
            raise ValueError(f"未登记的训练脚本产物：首层卷积输入通道 {channels}，只认识 1（fashion-cnn）和 3（cifar100）")
    elif set(state) == {"0.weight", "0.bias", "2.weight", "2.bias"}:
        script_id = parameters.get("script_id")
        if script_id not in (None, "mini-mlp"):
            raise ValueError(f"parameters.json 登记为 {script_id}，不是 mini-mlp")
        hidden, features = (int(size) for size in state["0.weight"].shape)
        model = nn.Sequential(nn.Linear(features, hidden), nn.Tanh(), nn.Linear(hidden, output_count))
        normalization = None
        input_shape = (features,)
        architecture = f"mini-mlp MLP({features}→{hidden}→Tanh→{output_count})"
        labels = _labels_from(output_count, labels_file, None)
        epoch, validation_loss = bundle.get("epoch"), bundle.get("validation_loss")
    else:
        preview = ", ".join(sorted(state)[:8]) + ("…" if len(state) > 8 else "")
        raise ValueError(f"未登记的训练产物结构：{preview}；只认识 fashion-cnn / mini-mlp / cifar100")

    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise ValueError(f"权重结构与 {architecture} 不一致：{exc}") from exc
    model.eval()
    return LoadedModel(
        model=model,
        architecture=architecture,
        input_kind="image" if conv_weights else "features",
        input_shape=input_shape,
        labels=labels,
        normalization=normalization,
        epoch=int(epoch) if epoch is not None else None,
        validation_loss=round(float(validation_loss), 6) if validation_loss is not None else None,
    )


class ServeState:
    def __init__(self) -> None:
        self.loaded: LoadedModel | None = None
        self.weights_path: str | None = None
        self.weights_sha256: str | None = None
        self.started_at = time.monotonic()

    def load(self, weights: Path) -> LoadedModel:
        loaded = load_weights(weights)
        self.loaded = loaded
        self.weights_path = str(weights)
        self.weights_sha256 = sha256(weights.read_bytes()).hexdigest()
        return loaded

    @property
    def ready(self) -> bool:
        return self.loaded is not None


state = ServeState()


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: list[float]
    top_k: int | None = Field(default=None, ge=1, le=1000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 直接 `uvicorn service.serve_checkpoint:app` 时没有命令行参数，这里挑最新产物。
    if not state.ready:
        try:
            state.load(resolve_weights(REPO_ROOT / "db" / "training-runs"))
        except Exception as exc:  # noqa: BLE001
            print(f"未能自动加载权重：{exc}")
    yield


app = FastAPI(title="训练产物通用推理服务", version="0.1.0", lifespan=lifespan)


def _input_spec(loaded: LoadedModel) -> dict:
    if loaded.input_kind == "image":
        channels, height, width = loaded.input_shape
        return {"kind": "image", "shape": [channels, height, width], "valueRange": [0, 255],
                "note": "按 (C,H,W) 展平的 0-255 像素；服务内部 /255 后按训练归一化口径处理"}
    return {"kind": "features", "size": loaded.input_shape[0],
            "note": "与训练脚本同口径的特征向量"}


def _identity() -> dict:
    loaded = state.loaded
    assert loaded is not None
    return {
        "service": "checkpoint-server",
        "architecture": loaded.architecture,
        "weights": os.path.basename(state.weights_path or ""),
        "weightsPath": state.weights_path,
        "weightsSha256": state.weights_sha256,
        "input": _input_spec(loaded),
        "classes": len(loaded.labels),
        "labels": loaded.labels,
        "normalization": loaded.normalization,
        "trainedEpoch": loaded.epoch,
        "validationLoss": loaded.validation_loss,
        "torch": torch.__version__,
        "device": "cpu",
    }


@app.get("/health")
@app.get("/healthz")
def healthz():
    if not state.ready:
        raise HTTPException(status_code=503, detail="权重未加载，服务不可用")
    return {"status": "ok", "uptimeSeconds": round(time.monotonic() - state.started_at, 1), **_identity()}


@app.post("/predict")
def predict(request: PredictRequest):
    if not state.ready:
        raise HTTPException(status_code=503, detail="权重未加载，服务不可用")
    loaded = state.loaded
    assert loaded is not None
    expected = 1
    for size in loaded.input_shape:
        expected *= size
    if len(request.input) != expected:
        shape = "×".join(str(size) for size in loaded.input_shape)
        raise HTTPException(status_code=422, detail=f"input 需要 {expected} 个数（{shape}），收到 {len(request.input)}")
    if request.top_k is not None and request.top_k > len(loaded.labels):
        raise HTTPException(status_code=422, detail=f"top_k 最大 {len(loaded.labels)}")
    if any(value != value or abs(value) == float("inf") for value in request.input):
        raise HTTPException(status_code=422, detail="input 必须是有限数值")

    tensor = torch.tensor(request.input, dtype=torch.float32)
    if loaded.input_kind == "image":
        if any(value < 0 or value > 255 for value in request.input):
            raise HTTPException(status_code=422, detail="图像模型的 input 必须是 0-255 的像素值")
        tensor = tensor.div_(255.0).view(1, *loaded.input_shape)
        mean = torch.tensor(loaded.normalization["mean"], dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(loaded.normalization["std"], dtype=torch.float32).view(-1, 1, 1)
        tensor = tensor.sub_(mean).div_(std)
    else:
        tensor = tensor.view(1, *loaded.input_shape)

    began = time.perf_counter()
    with torch.no_grad():
        probabilities = torch.softmax(loaded.model(tensor), dim=1)[0]
    elapsed_ms = round((time.perf_counter() - began) * 1000, 2)
    ranked = torch.topk(probabilities, k=min(request.top_k or 3, len(loaded.labels)))
    by_index = {item["index"]: item for item in loaded.labels}
    ranking = []
    for index, probability in zip(ranked.indices.tolist(), ranked.values.tolist()):
        label = by_index.get(index, {"label": f"class-{index}", "labelZh": f"class-{index}"})
        ranking.append({"index": index, "label": label["label"], "labelZh": label["labelZh"],
                        "probability": round(float(probability), 6)})

    top = ranking[0]
    return {"index": top["index"], "label": top["label"], "labelZh": top["labelZh"],
            "confidence": top["probability"], "ranking": ranking, "elapsedMs": elapsed_ms,
            "architecture": loaded.architecture, "weights": os.path.basename(state.weights_path or ""),
            "weightsSha256": state.weights_sha256}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        help="best.pt 文件，或 outputs / 任务目录；缺省挑 db/training-runs 下最新 best.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    args = parser.parse_args()

    target = args.weights if args.weights is not None else REPO_ROOT / "db" / "training-runs"
    try:
        weights = resolve_weights(target)
        loaded = state.load(weights)
    except Exception as exc:  # noqa: BLE001
        print(f"启动失败：{exc}")
        return 1

    print(f"已加载权重：{weights}")
    print(f"架构 {loaded.architecture}，输入 {_input_spec(loaded)}")
    print(f"sha256 {state.weights_sha256[:16]}…，监听 http://{args.host}:{args.port}")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
