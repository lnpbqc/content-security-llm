"""FashionMNIST 推理服务（#2.3）：权重自描述校验、真实前向返回与非法输入报错。"""
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

import service.fashion_serve as fashion_serve
from training_scripts.fashion_cnn import SmallCNN


@pytest.fixture()
def client(tmp_path: Path):
    """每个用例独立：先重置服务状态，再用随机权重走一遍真实加载。"""
    original = fashion_serve.state
    fashion_serve.state = fashion_serve.ServeState()
    bundle = {
        "state_dict": {key: value.detach().cpu() for key, value in SmallCNN().state_dict().items()},
        "labels": ["T-shirt/top"],
        "normalization": {"mean": 0.286, "std": 0.3536},
        "epoch": 1,
        "validation_loss": 1.0,
    }
    weights = tmp_path / "best.pt"
    torch.save(bundle, weights)
    try:
        yield TestClient(fashion_serve.app), weights
    finally:
        fashion_serve.state = original


def flat(value: int) -> list[int]:
    return [value] * 784


def test_healthz_refuses_until_weights_loaded(client):
    http, _ = client
    assert http.get("/healthz").status_code == 503
    assert http.post("/predict", json={"pixels": flat(7)}).status_code == 503


def test_load_rejects_plain_state_dict_without_self_description(client, tmp_path):
    http, _ = client
    bare = tmp_path / "cifar-style.pt"
    torch.save({key: value for key, value in SmallCNN().state_dict().items()}, bare)
    with pytest.raises(ValueError, match="自描述"):
        fashion_serve.load_bundle(bare)


def test_predict_returns_real_forward_probabilities(client):
    http, weights = client
    assert fashion_serve.load_first([weights]) == weights
    health = http.get("/healthz").json()
    assert health["service"] == "fashion-cnn" and len(health["weights_sha256"]) == 64

    response = http.post("/predict", json={"pixels": flat(200), "top_k": 3})
    body = response.json()
    assert response.status_code == 200
    assert {"label", "labelZh", "index", "confidence", "ranking", "elapsedMs",
            "normalization", "weights"} <= set(body)
    assert len(body["ranking"]) == 3
    assert body["ranking"][0]["probability"] >= body["ranking"][-1]["probability"]
    assert all(0.0 <= item["probability"] <= 1.0 for item in body["ranking"])
    assert body["normalization"] == {"mean": 0.286, "std": 0.3536}


def test_predict_rejects_illegal_pixels(client):
    http, weights = client
    fashion_serve.load_first([weights])
    assert http.post("/predict", json={"pixels": flat(300)}).status_code == 422
    assert http.post("/predict", json={"pixels": flat(1)[:100]}).status_code == 422
    assert http.post("/predict", json={"pixels": flat(1), "top_k": 20}).status_code == 422
    assert http.post("/predict", json={"pixels": flat(1), "extra": 1}).status_code == 422
