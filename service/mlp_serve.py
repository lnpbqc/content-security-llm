"""迷你 MLP 真实推理服务：加载脚本训练产物 best.pt 做真前向，不编造概率。

启动（在 content-security-llm 目录下）：
    python service/mlp_serve.py --port 8011
    python service/mlp_serve.py --weights "db/training-runs/<任务ID>/outputs/best.pt" --port 8011
不带 --weights 时按修改时间从新到旧扫描 db/training-runs，只挑 mini-mlp 的产物：
checkpoint 必须含 model_state_dict，且同目录 parameters.json 的 script_id 为 mini-mlp；
CNN 等其它脚本的权重结构不同，会被明确拒绝而不是猜着加载。

接口：
    GET  /healthz  服务与权重身份（文件名、sha256、网络结构、训练轮次、验证损失、已运行秒数）
    POST /predict  {"features": [x, y], "top_k": 2}
                   -> {"index","label","confidence","ranking","elapsedMs","weights",...}

诚实约束：权重未加载返回 503；输入不是两个有限数返回 422；概率是 softmax 后的真实前向结果。
类别名沿用训练脚本的合成口径（内簇=class-0、外簇=class-1），不赋予业务含义。
服务默认只绑 127.0.0.1，无需令牌（本机演示服务；对外部署请自行加反向代理与鉴权）。
"""
from __future__ import annotations

import argparse
import json
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
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

SERVICE_NAME = "mini-mlp"
# 训练脚本 mini_mlp.py 的特征宽度与类别数；换脚本时要一起改。
FEATURE_COUNT = 2
CLASS_COUNT = 2
CLASS_LABELS = {0: "class-0", 1: "class-1"}
DATA_NOTE = "合成二维两类：内簇中心 (1.6, 1.6) 为 class-0，外簇中心 (-1.6, -1.6) 为 class-1"


def build_model(hidden_size: int) -> nn.Sequential:
    """与 training_scripts/mini_mlp.py 完全一致的结构：Linear→Tanh→Linear。"""
    return nn.Sequential(
        nn.Linear(FEATURE_COUNT, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, CLASS_COUNT),
    )


def _candidate_weights(explicit: Path | None) -> list[Path]:
    if explicit is not None:
        return [explicit]
    runs = REPO_ROOT / "db" / "training-runs"
    patterns = ("*/outputs/best.pt", "*/best.pt", "*/outputs/last.pt", "*/last.pt")
    candidates = [path for pattern in patterns for path in runs.glob(pattern)]
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def _script_id(weights: Path) -> str | None:
    """读取产物同目录 parameters.json 里的 script_id；没有该文件时返回 None。"""
    for sibling in (weights.parent / "parameters.json", weights.parent.parent / "parameters.json"):
        if not sibling.exists():
            continue
        try:
            return json.loads(sibling.read_text(encoding="utf-8")).get("script_id")
        except (ValueError, OSError):
            return None
    return None


def load_bundle(path: Path) -> dict:
    """校验这是 mini-mlp 的产物；结构不符直接拒绝，避免把别的权重当本服务产物加载。"""
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    state = bundle.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint 缺少 model_state_dict，不是 mini-mlp 训练产物")
    if _script_id(path) not in (None, SERVICE_NAME):
        raise ValueError("该产物属于其它训练脚本，mini-mlp 服务不加载")
    expected = {"0.weight", "0.bias", "2.weight", "2.bias"}
    if set(state) != expected:
        raise ValueError("权重键与 Linear→Tanh→Linear 结构不符：{}".format(sorted(state)))
    if tuple(state["0.weight"].shape) != (int(state["0.weight"].shape[0]), FEATURE_COUNT):
        raise ValueError("输入维度不是 {} 维，与 mini-mlp 脚本不一致".format(FEATURE_COUNT))
    if int(state["2.weight"].shape[0]) != CLASS_COUNT:
        raise ValueError("类别数不是 {}，与 mini-mlp 脚本不一致".format(CLASS_COUNT))
    return bundle


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: list[float] = Field(min_length=FEATURE_COUNT, max_length=FEATURE_COUNT)
    top_k: int = Field(default=CLASS_COUNT, ge=1, le=CLASS_COUNT)


class ServeState:
    def __init__(self) -> None:
        self.model: nn.Sequential | None = None
        self.weights_path: str | None = None
        self.weights_sha256: str | None = None
        self.hidden_size = 0
        self.epoch = 0
        self.validation_loss: float | None = None
        self.started_at = time.monotonic()

    def load(self, path: Path) -> None:
        bundle = load_bundle(path)
        state = bundle["model_state_dict"]
        hidden = int(state["0.weight"].shape[0])
        model = build_model(hidden)
        model.load_state_dict(state)
        model.eval()
        self.model = model
        self.hidden_size = hidden
        self.epoch = int(bundle.get("epoch") or 0)
        loss = bundle.get("validation_loss")
        self.validation_loss = None if loss is None else round(float(loss), 6)
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
        print(f"已加载权重：{candidate}（sha256 {state.weights_sha256[:16]}…，隐藏层 {state.hidden_size}）")
        return candidate
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 直接 `uvicorn service.mlp_serve:app` 时没有命令行参数，这里自动挑最新的 mini-mlp 产物。
    if not state.ready:
        load_first(_candidate_weights(None))
    yield


app = FastAPI(title="迷你 MLP 推理服务", version="0.1.0", lifespan=lifespan)


def _identity() -> dict:
    return {
        "service": SERVICE_NAME,
        "weights": os.path.basename(state.weights_path or ""),
        "weights_sha256": state.weights_sha256,
        "architecture": "Linear(2,{})→Tanh→Linear({},2)".format(state.hidden_size, state.hidden_size),
        "trainedEpoch": state.epoch,
        "validationLoss": state.validation_loss,
        "torch": torch.__version__,
        "device": "cpu",
    }


@app.get("/healthz")
def healthz():
    if not state.ready:
        raise HTTPException(status_code=503, detail="权重未加载，服务不可用")
    return {"status": "ok", "uptimeSeconds": round(time.monotonic() - state.started_at, 1),
            "classes": CLASS_LABELS, "dataNote": DATA_NOTE, **_identity()}


@app.post("/predict")
def predict(request: PredictRequest):
    if not state.ready:
        raise HTTPException(status_code=503, detail="权重未加载，服务不可用")
    if any(value != value or abs(value) == float("inf") for value in request.features):
        raise HTTPException(status_code=422, detail="特征值必须是有限实数")
    tensor = torch.tensor(request.features, dtype=torch.float32)
    began = time.perf_counter()
    with torch.no_grad():
        probabilities = torch.softmax(state.model(tensor.view(1, FEATURE_COUNT)), dim=1)[0]
    elapsed_ms = round((time.perf_counter() - began) * 1000, 2)
    ranked = torch.topk(probabilities, k=request.top_k)
    ranking = [{
        "index": int(index),
        "label": CLASS_LABELS[int(index)],
        "probability": round(float(probability), 6),
    } for index, probability in zip(ranked.indices, ranked.values)]
    top = ranking[0]
    return {"index": top["index"], "label": top["label"], "confidence": top["probability"],
            "ranking": ranking, "elapsedMs": elapsed_ms, "features": request.features,
            **_identity()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, help="mini-mlp 训练产物 best.pt；缺省自动挑选最新产物")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()

    if args.weights is not None and load_first([args.weights]) is None:
        print("指定的权重不可用；请确认它是 mini-mlp 训练产物，或不带 --weights 让服务自动挑选")
        return 1

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
