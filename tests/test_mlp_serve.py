"""迷你 MLP 推理服务：只加载本脚本产物、真实前向返回概率、非法输入明确报错。"""
import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

import service.mlp_serve as mlp_serve


def bundle(hidden: int = 8, epoch: int = 3, loss: float = 0.42) -> dict:
    return {"model_state_dict": {key: value.detach().cpu()
                                 for key, value in mlp_serve.build_model(hidden).state_dict().items()},
            "epoch": epoch, "validation_loss": loss}


@pytest.fixture()
def client(tmp_path: Path):
    """每个用例独立：重置服务状态，并用随机初始化权重走一遍真实加载。"""
    original = mlp_serve.state
    mlp_serve.state = mlp_serve.ServeState()
    weights = tmp_path / "best.pt"
    torch.save(bundle(), weights)
    try:
        yield TestClient(mlp_serve.app), weights
    finally:
        mlp_serve.state = original


def test_healthz_refuses_until_weights_loaded(client):
    http, _ = client
    assert http.get("/healthz").status_code == 503
    assert http.post("/predict", json={"features": [1.0, 1.0]}).status_code == 503


def test_load_reports_trained_identity(client):
    http, weights = client
    assert mlp_serve.load_first([weights]) == weights
    health = http.get("/healthz").json()
    assert health["service"] == "mini-mlp" and len(health["weights_sha256"]) == 64
    assert health["trainedEpoch"] == 3 and health["validationLoss"] == 0.42
    assert health["architecture"] == "Linear(2,8)→Tanh→Linear(8,2)"


def test_load_rejects_other_scripts_weights(client, tmp_path):
    http, _ = client
    cnn_style = tmp_path / "cnn.pt"
    torch.save({"state_dict": {}, "labels": [], "normalization": {}}, cnn_style)
    with pytest.raises(ValueError, match="model_state_dict"):
        mlp_serve.load_bundle(cnn_style)

    other = tmp_path / "other.pt"
    torch.save(bundle(), other)
    (tmp_path / "parameters.json").write_text(json.dumps({"script_id": "fashion-cnn"}), encoding="utf-8")
    with pytest.raises(ValueError, match="其它训练脚本"):
        mlp_serve.load_bundle(other)


def test_load_rejects_mismatched_architecture(client, tmp_path):
    http, _ = client
    deep = tmp_path / "deep.pt"
    state = mlp_serve.build_model(4).state_dict()
    state["4.weight"] = torch.zeros(2, 2)
    torch.save({"model_state_dict": state, "epoch": 1}, deep)
    with pytest.raises(ValueError, match="结构不符"):
        mlp_serve.load_bundle(deep)


def test_predict_returns_real_forward_probabilities(client):
    http, weights = client
    mlp_serve.load_first([weights])
    body = http.post("/predict", json={"features": [1.6, 1.6], "top_k": 2}).json()
    assert http.post("/predict", json={"features": [1.6, 1.6]}).status_code == 200
    assert len(body["ranking"]) == 2
    assert body["ranking"][0]["probability"] >= body["ranking"][1]["probability"]
    assert abs(sum(item["probability"] for item in body["ranking"]) - 1.0) < 1e-4
    assert body["confidence"] == body["ranking"][0]["probability"]
    assert body["elapsedMs"] >= 0 and body["weights_sha256"]


def test_predict_rejects_illegal_features(client):
    http, weights = client
    mlp_serve.load_first([weights])
    assert http.post("/predict", json={"features": [1.0]}).status_code == 422
    assert http.post("/predict", json={"features": [1.0, 2.0, 3.0]}).status_code == 422
    assert http.post("/predict", json={"features": [1.0, 2.0], "top_k": 5}).status_code == 422
    assert http.post("/predict", json={"features": [1.0, 2.0], "extra": 1}).status_code == 422
