"""FashionMNIST CNN 真实推理服务（#2.3）：加载训练产物 best.pt 做真前向，不编造类别。

启动（在 content-security-llm 目录下）：
    python service/fashion_serve.py --port 8010
    python service/fashion_serve.py --weights "db/training-runs/<任务ID>/best.pt" --port 8010
不带 --weights 时按修改时间从新到旧扫描 db/training-runs/*/best.pt，取第一个"自描述权重"
（含 state_dict/labels/normalization 三个键，即本仓库训练脚本的产物）。

接口：
    GET  /healthz  服务与权重身份（文件名、sha256、torch 版本、设备、已运行秒数）
    POST /predict  {"pixels": [784 个 0-255 整数], "top_k": 3}
                   -> {"label","labelZh","confidence","ranking","elapsedMs","normalization","weights"}

诚实约束：权重缺失或像素不合法直接 503/422，返回真实错误；归一化口径写在权重文件里，
推理与训练同一份参数，不会出现线上线下两套预处理。服务默认只绑 127.0.0.1，无需令牌
（演示用本机服务；对外部署请放到反向代理后面并自行加鉴权）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import asynccontextmanager
from hashlib import sha256
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


def _load_training_script():
    """训练脚本目录 training_scripts/ 与本仓库 service/training_scripts.py 同名，
    直接按文件路径加载，保证 `python service/fashion_serve.py` 与 `-m` 两种起法都拿到同一份模型定义。"""
    import importlib.util

    path = REPO_ROOT / "training_scripts" / "fashion_cnn.py"
    spec = importlib.util.spec_from_file_location("fashion_cnn_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_fashion = _load_training_script()
LABELS, SmallCNN = _fashion.LABELS, _fashion.SmallCNN

PIXEL_COUNT = 28 * 28


def _candidate_weights(explicit: Path | None) -> list[Path]:
    if explicit is not None:
        return [explicit]
    runs = REPO_ROOT / "db" / "training-runs"
    # worker 把产物写在 <任务ID>/outputs/ 下，老任务可能直接落在任务目录，两种都找。
    candidates = [path for pattern in ("*/best.pt", "*/outputs/best.pt", "*/last.pt",
                                       "*/outputs/last.pt") for path in runs.glob(pattern)]
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def load_bundle(path: Path) -> dict:
    """读取并校验自描述权重；非本仓库训练脚本的产物直接拒绝，不做"猜测式加载"。"""
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if not {"state_dict", "labels", "normalization"} <= set(bundle):
        raise ValueError("权重文件缺少 labels/normalization 自描述信息，不是本服务可用的训练产物")
    return bundle


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pixels: list[int] = Field(min_length=PIXEL_COUNT, max_length=PIXEL_COUNT)
    top_k: int = Field(default=3, ge=1, le=10)


class ServeState:
    def __init__(self) -> None:
        self.model: SmallCNN | None = None
        self.normalization: dict | None = None
        self.weights_path: str | None = None
        self.weights_sha256: str | None = None
        self.started_at = time.monotonic()

    def load(self, path: Path) -> None:
        bundle = load_bundle(path)
        model = SmallCNN()
        model.load_state_dict(bundle["state_dict"])
        model.eval()
        self.model = model
        self.normalization = bundle["normalization"]
        self.weights_path = str(path)
        self.weights_sha256 = sha256(path.read_bytes()).hexdigest()

    @property
    def ready(self) -> bool:
        return self.model is not None


state = ServeState()


def load_first(paths) -> Path | None:
    """按顺序尝试加载，返回真正可用的权重路径。"""
    for candidate in paths:
        if not candidate.exists():
            print(f"跳过不存在的权重：{candidate}")
            continue
        try:
            state.load(candidate)
        except Exception as exc:  # noqa: BLE001
            print(f"权重不可用：{candidate} -> {exc}")
            continue
        print(f"已加载权重：{candidate}（sha256 {state.weights_sha256[:16]}…）")
        return candidate
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 直接 `uvicorn service.fashion_serve:app` 时没有命令行参数，这里自动取最新自描述权重。
    if not state.ready:
        load_first(_candidate_weights(None))
    yield


app = FastAPI(title="FashionMNIST CNN 推理服务", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    if not state.ready:
        raise HTTPException(status_code=503, detail="权重未加载，服务不可用")
    return {
        "status": "ok",
        "service": "fashion-cnn",
        "weights": os.path.basename(state.weights_path or ""),
        "weights_sha256": state.weights_sha256,
        "torch": torch.__version__,
        "device": "cpu",
        "classes": len(LABELS),
        "uptimeSeconds": round(time.monotonic() - state.started_at, 1),
    }


@app.post("/predict")
def predict(request: PredictRequest):
    if not state.ready:
        raise HTTPException(status_code=503, detail="权重未加载，服务不可用")
    if any(pixel < 0 or pixel > 255 for pixel in request.pixels):
        raise HTTPException(status_code=422, detail="像素值必须是 0-255 的灰度整数")
    mean = float(state.normalization["mean"])
    std = float(state.normalization["std"])
    tensor = torch.tensor(request.pixels, dtype=torch.float32).div_(255.0).sub_(mean).div_(std)
    began = time.perf_counter()
    with torch.no_grad():
        logits = state.model(tensor.view(1, 1, 28, 28))
        probabilities = torch.softmax(logits, dim=1)[0]
    elapsed_ms = round((time.perf_counter() - began) * 1000, 2)
    ranked = torch.topk(probabilities, k=request.top_k)
    ranking = [{
        "index": int(index),
        "label": LABELS[int(index)]["en"],
        "labelZh": LABELS[int(index)]["zh"],
        "probability": round(float(probability), 6),
    } for index, probability in zip(ranked.indices, ranked.values)]
    top = ranking[0]
    return {
        "label": top["label"],
        "labelZh": top["labelZh"],
        "index": top["index"],
        "confidence": top["probability"],
        "ranking": ranking,
        "elapsedMs": elapsed_ms,
        "normalization": {"mean": mean, "std": std},
        "weights": os.path.basename(state.weights_path or ""),
        "weightsSha256": state.weights_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, help="训练产物 best.pt；缺省自动取最新的自描述权重")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()

    if args.weights is not None and load_first([args.weights]) is None:
        print("指定的权重不可用；请确认路径，或不带 --weights 让服务自动挑选训练产物")
        return 1
    # 未指定权重时交给启动钩子：优先 db/training-runs 里最新的自描述产物。

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
