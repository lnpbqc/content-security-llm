"""通用推理服务：识别三种训练产物、拒绝未知结构、预测返回真实 softmax。"""
import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

import service.serve_checkpoint as serve


@pytest.fixture(autouse=True)
def reset_state():
    original = serve.state
    serve.state = serve.ServeState()
    yield
    serve.state = original


def fashion_bundle() -> dict:
    script = serve._import_training_script("fashion_cnn.py")
    model = script.SmallCNN()
    return {"state_dict": model.state_dict(),
            "labels": [label["en"] for label in script.LABELS],
            "normalization": {"mean": script.MEAN, "std": script.STD},
            "epoch": 2, "validation_loss": 0.5}


def mini_mlp_bundle() -> dict:
    model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
    return {"model_state_dict": model.state_dict(), "epoch": 1, "validation_loss": 0.3}


def test_resolve_prefers_best_in_outputs(tmp_path):
    outputs = tmp_path / "run" / "outputs"
    outputs.mkdir(parents=True)
    best = outputs / "best.pt"
    torch.save({"x": torch.zeros(1)}, best)
    torch.save({"x": torch.ones(1)}, outputs / "last.pt")
    assert serve.resolve_weights(outputs) == best
    assert serve.resolve_weights(tmp_path / "run") == best


def test_fashion_bundle_loads_and_predicts(tmp_path):
    weights = tmp_path / "outputs" / "best.pt"
    weights.parent.mkdir()
    torch.save(fashion_bundle(), weights)
    (weights.parent / "labels.json").write_text(
        json.dumps([{"index": index, "zh": f"类{index}", "en": f"class-{index}"} for index in range(10)]),
        encoding="utf-8")

    loaded = serve.state.load(weights)
    assert loaded.input_shape == (1, 28, 28) and loaded.input_kind == "image"
    assert loaded.normalization == {"mean": [0.286], "std": [0.3536]}
    assert loaded.labels[0]["labelZh"] == "类0"

    http = TestClient(serve.app)
    health = http.get("/healthz").json()
    assert health["architecture"].startswith("fashion-cnn")
    assert health["input"]["shape"] == [1, 28, 28] and len(health["weightsSha256"]) == 64

    body = http.post("/predict", json={"input": [128] * 784, "top_k": 10}).json()
    assert len(body["ranking"]) == 10
    assert abs(sum(item["probability"] for item in body["ranking"]) - 1.0) < 1e-4
    assert body["confidence"] == body["ranking"][0]["probability"]


def test_mini_mlp_bundle_loads_and_rejects_other_script(tmp_path):
    weights = tmp_path / "best.pt"
    torch.save(mini_mlp_bundle(), weights)
    parameters = tmp_path / "parameters.json"
    parameters.write_text(json.dumps({"script_id": "mini-mlp"}), encoding="utf-8")

    loaded = serve.state.load(weights)
    assert loaded.input_shape == (2,) and loaded.input_kind == "features"
    assert loaded.architecture == "mini-mlp MLP(2→4→Tanh→2)"

    http = TestClient(serve.app)
    body = http.post("/predict", json={"input": [1.5, -1.5]}).json()
    assert len(body["ranking"]) == 2
    assert abs(sum(item["probability"] for item in body["ranking"]) - 1.0) < 1e-4

    parameters.write_text(json.dumps({"script_id": "fashion-cnn"}), encoding="utf-8")
    with pytest.raises(ValueError, match="mini-mlp"):
        serve.load_weights(weights)


def test_cifar_plain_state_dict_is_recognized(tmp_path):
    script = serve._import_training_script("cifar100.py")
    weights = tmp_path / "best.pt"
    torch.save(script.SmallCNN().state_dict(), weights)

    loaded = serve.load_weights(weights)
    assert loaded.input_shape == (3, 32, 32)
    assert len(loaded.labels) == 100
    assert loaded.architecture == "cifar100 SmallCNN(3×32×32→100)"


def test_unknown_structure_rejected(tmp_path):
    weights = tmp_path / "best.pt"
    torch.save({"foo": torch.zeros(2, 2)}, weights)
    with pytest.raises(ValueError, match="未登记"):
        serve.load_weights(weights)


def test_predict_rejects_bad_input(tmp_path):
    weights = tmp_path / "best.pt"
    torch.save(mini_mlp_bundle(), weights)
    serve.state.load(weights)
    http = TestClient(serve.app)
    assert http.post("/predict", json={"input": [1.0]}).status_code == 422
    assert http.post("/predict", json={"input": [1.0, 2.0, 3.0]}).status_code == 422
    assert http.post("/predict", json={"input": [1.0, 2.0], "top_k": 5}).status_code == 422
    assert http.post("/predict", json={"input": [1.0, 2.0], "extra": 1}).status_code == 422
