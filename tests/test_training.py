"""验证真实训练、后台进程、HTTP 契约和已知预测的指标口径。"""

import json
import math
import os
import subprocess
import sys
import threading
import time
from datetime import timedelta
from hashlib import sha256
from types import SimpleNamespace

import pytest
import torch
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from pydantic import ValidationError

from config import Settings
from db.database import Database
from db.training import TrainingRepository
from main import create_app
from schemas.training import TrainingCreate
from service.training_engine import (classification_metrics, load_module, run_training,
                                     select_device, split_indices)
from service.training_examples import SOURCE, example_request
from service.training_worker import TrainingWorker, worker_lock
from utils.time import utc_now


BASE = "/api/v1/training"
HEADERS = {"Authorization": "Bearer " + "a" * 32}
OTHER = {"Authorization": "Bearer " + "b" * 32}
RECORDS = [{"id": str(i), "payload": {"features": [-1.0, -0.5] if i % 2 else [1.0, 0.5],
                                       "label": i % 2}} for i in range(20)]


def config(**overrides):
    return TrainingCreate.model_validate({**example_request(), "device": "cpu", "epochs": 3,
                                         "learning_rate": 0.1, **overrides}).model_dump()


def train(directory, values=None, canceled=lambda: False):
    directory.mkdir(parents=True, exist_ok=True)
    values = values or config()
    (directory / "source.py").write_text(values["python_source"], encoding="utf-8")
    events = []
    run_training(values, RECORDS, directory, events.append, canceled)
    return events


@pytest.fixture
def training_client(tmp_path):
    settings = Settings(tmp_path / "app.sqlite3", "s" * 32, Fernet.generate_key())
    app = create_app(settings)
    with TestClient(app) as client:
        for label in ("a", "b"):
            app.state.auth_service.create_token("s" * 32, label * 32, utc_now() + timedelta(days=1))
        yield app, client


class RecordsSource:
    def __init__(self):
        self.calls = 0

    def samples(self, scope):
        self.calls += 1
        if scope["version_id"] != "v1.0.0":
            raise ValueError("数据集版本不匹配")
        return [{"id": row["id"], "metadata": row["payload"]} for row in RECORDS]


def create(client, **overrides):
    response = client.post(BASE + "/tasks", headers=HEADERS, json=config(**overrides))
    assert response.status_code == 202, response.text
    return response.json()["data"]


def test_defaults_parameter_override_and_no_http_execution(training_client, tmp_path):
    app, client = training_client
    defaults = client.get(BASE + "/defaults", headers=HEADERS).json()["data"]
    assert defaults["defaults"]["epochs"] == 10
    assert defaults["defaults"]["batch_size"] == 8
    assert defaults["defaults"]["learning_rate"] == 0.0002
    sentinel = tmp_path / "http-must-not-execute"
    source = "from pathlib import Path\nPath({!r}).touch()\n".format(str(sentinel)) + SOURCE
    task = create(client, python_source=source, optimizer="SGD", optimizer_params={"momentum": 0.9},
                  batch_size=4, scheduler="StepLR")
    assert task["status"] == "pending" and task["loss"] == []
    assert task["optimizer"] == "SGD" and task["batch_size"] == 4
    assert task["config"]["scheduler_params"] == {"step_size": 1}
    assert "python_source" not in task and not sentinel.exists()
    assert task["latest_metrics"]["f1"] is None
    app.state.database.initialize()
    assert client.get(BASE + "/tasks/" + task["id"], headers=HEADERS).json()["data"]["status"] == "pending"


@pytest.mark.parametrize("override", [
    {"epochs": 0}, {"learning_rate": -1}, {"validation_ratio": 1},
    {"optimizer_params": {"lr": 1}}, {"python_source": "def broken("},
    {"task_type": "custom", "loss": "MSELoss"}, {"loss_params": {"reduction": "sum"}},
    {"average": "binary"}, {"model_params": {"nested": [math.inf]}},
    {"contract": {"inputs": {}, "targets": {"dtype": "int64", "shape": [None]},
                  "outputs": {"dtype": "float32", "shape": [None, 2]}}},
])
def test_invalid_requests_are_rejected_without_tasks(override, training_client):
    _, client = training_client
    payload = {**example_request(), **override}
    # 非有限 JSON 以原始正文发送，避免 HTTP 客户端提前拒绝。
    response = client.post(BASE + "/tasks", headers={**HEADERS, "Content-Type": "application/json"},
                           content=json.dumps(payload))
    assert response.status_code == 422
    assert client.get(BASE + "/tasks", headers=HEADERS).json()["data"]["total"] == 0


def test_auth_pagination_cancel_pending_and_isolation(training_client):
    app, client = training_client
    tasks = [create(client, name="任务 {}".format(i)) for i in range(3)]
    assert client.get(BASE + "/defaults").status_code == 401
    page = client.get(BASE + "/tasks?page=2&page_size=2", headers=HEADERS).json()["data"]
    assert page["total"] == 3 and len(page["items"]) == 1
    assert client.get(BASE + "/tasks", headers=OTHER).json()["data"]["total"] == 0
    task_id = tasks[0]["id"]
    for suffix in ("", "/metrics", "/artifacts", "/artifacts/unknown/download"):
        assert client.get(BASE + "/tasks/" + task_id + suffix, headers=OTHER).status_code == 404
    assert client.post(BASE + "/tasks/" + task_id + "/cancel", headers=OTHER).status_code == 404
    canceled = client.post(BASE + "/tasks/" + task_id + "/cancel", headers=HEADERS).json()["data"]
    assert canceled["status"] == "canceled"
    assert client.post(BASE + "/tasks/" + task_id + "/cancel", headers=HEADERS).json()["data"]["status"] == "canceled"
    assert app.state.training_service.repository.claim()["id"] != task_id


def test_cpu_training_updates_weights_reduces_loss_and_records_whole_epochs(tmp_path):
    events = train(tmp_path)
    history = [event for event in events if event["type"] == "metric" and event["phase"] == "validation"]
    steps = [event for event in events if event["type"] == "metric" and event["phase"] == "train"]
    assert len(steps) == 6 and len(history) == 3
    assert history[-1]["metrics"]["train_loss"] < history[0]["metrics"]["train_loss"]
    assert all(0 <= history[-1]["metrics"][name] <= 1 for name in ("precision", "accuracy", "f1", "recall"))
    split = json.loads((tmp_path / "split.json").read_text())
    assert len(split["train_indices"]) == 16 and len(split["validation_indices"]) == 4
    assert not set(split["train_indices"]) & set(split["validation_indices"])
    torch.manual_seed(42)
    initial = load_module(tmp_path / "source.py").build_model({}).state_dict()
    saved = torch.load(tmp_path / "last.pt", weights_only=True)
    assert not torch.equal(initial["linear.weight"], saved["linear.weight"])
    restored = load_module(tmp_path / "source.py").build_model({})
    restored.load_state_dict(torch.load(tmp_path / "best.pt", weights_only=True))
    assert restored(x=torch.ones(1, 2)).shape == (1, 2)


def test_weighted_epoch_loss_handles_partial_batches(tmp_path):
    source = SOURCE + '''
def compute_loss(outputs, targets, loss_params):
    return outputs.sum() * 0 + targets.float().mean()
def split_dataset(dataset, split_params):
    return list(range(13)), list(range(13, 20))
'''
    events = train(tmp_path, config(python_source=source, loss="custom", metrics="none", epochs=1))
    history = [e for e in events if e["type"] == "metric" and e["phase"] == "validation"]
    assert history[0]["metrics"]["train_loss"] == pytest.approx(6 / 13)
    assert history[0]["metrics"]["loss"] == pytest.approx(4 / 7)


def test_custom_loss_metrics_split_and_prints(tmp_path):
    source = SOURCE + '''
def split_dataset(dataset, split_params):
    return list(range(12)), list(range(12, 20))
def compute_loss(outputs, targets, loss_params):
    return torch.nn.functional.cross_entropy(outputs, targets) * loss_params["scale"]
def compute_metrics(validation_batches, metric_params):
    assert all(b["outputs"].device.type == "cpu" for b in validation_batches)
    assert sum(len(b["targets"]) for b in validation_batches) == 8
    return {"custom_score": metric_params["score"]}
'''
    events = train(tmp_path, config(python_source=source, loss="custom", metrics="custom",
                                   loss_params={"scale": 2}, metric_params={"score": 0.25}))
    last = [e for e in events if e["type"] == "metric"][-1]["metrics"]
    assert last["custom_score"] == 0.25 and last["f1"] is None
    assert len(json.loads((tmp_path / "split.json").read_text())["validation_indices"]) == 8


@pytest.mark.parametrize("indices", [([0, 1], [1, 2]), ([], [1]), ([0, 0], [1]),
                                      ([0], [20]), ([True], [1]), ([0.0], [1])])
def test_invalid_custom_split(indices):
    module = SimpleNamespace(split_dataset=lambda dataset, params: indices)
    with pytest.raises(ValueError):
        split_indices(RECORDS, config(), module)


@pytest.mark.parametrize("source, message", [
    (SOURCE.replace('dtype=torch.float32)', 'dtype=torch.float64)'), "dtype"),
    (SOURCE.replace('self.linear(x)', 'self.linear(x)[:, :1]'), "shape"),
    (SOURCE + '\ndef compute_loss(outputs, targets, params):\n    return outputs.mean().detach()\n', "梯度"),
    (SOURCE + '\ndef compute_loss(outputs, targets, params):\n    return torch.tensor(1., requires_grad=True)\n', "梯度"),
    (SOURCE + '\ndef compute_loss(outputs, targets, params):\n    return outputs.mean() * float("nan")\n', "有限"),
    (SOURCE + '\ndef compute_loss(outputs, targets, params):\n    return outputs\n', "标量"),
])
def test_shape_and_loss_failures(source, message, tmp_path):
    values = config(python_source=source, loss="custom" if "compute_loss" in source else "auto")
    with pytest.raises(ValueError, match=message):
        train(tmp_path, values)
    assert not (tmp_path / "last.pt").exists()


def test_missing_custom_hooks_and_nonfinite_metrics(tmp_path):
    with pytest.raises(ValueError, match="compute_loss"):
        train(tmp_path / "missing", config(loss="custom"))
    source = SOURCE + '\ndef compute_metrics(batches, params):\n    return {"f1": float("inf")}\n'
    with pytest.raises(ValueError, match="有限"):
        train(tmp_path / "nonfinite", config(python_source=source, metrics="custom"))


def test_float64_model_with_external_class_weights(tmp_path):
    source = SOURCE.replace('dtype=torch.float32)', 'dtype=torch.float64)').replace(
        'return Classifier(model_params.get("input_size", 2), model_params.get("classes", 2))',
        'return Classifier(model_params.get("input_size", 2), model_params.get("classes", 2)).double()')
    contract = {"inputs": {"x": {"dtype": "float64", "shape": [None, 2]}},
                "targets": {"dtype": "int64", "shape": [None]},
                "outputs": {"dtype": "float64", "shape": [None, 2]}}
    events = train(tmp_path, config(python_source=source, contract=contract, loss_params={"weight": [1, 2]}, epochs=1))
    assert math.isfinite([e for e in events if e["type"] == "metric"][-1]["metrics"]["loss"])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_known_binary_metrics_whole_dataset_and_zero_division(dtype):
    values = config(task_type="binary")
    batches = [{"outputs": torch.tensor([10.0], dtype=dtype), "targets": torch.tensor([1.0], dtype=dtype)},
               {"outputs": torch.tensor([10.0, -10.0, -10.0], dtype=dtype),
                "targets": torch.tensor([0.0, 1.0, 0.0], dtype=dtype)}]
    assert classification_metrics(batches, values) == {"precision": 0.5, "recall": 0.5, "accuracy": 0.5, "f1": 0.5}
    assert classification_metrics([{"outputs": torch.tensor([-10.0, -10.0]),
                                    "targets": torch.tensor([0.0, 0.0])}], values)["f1"] == 0
    positive_zero = config(task_type="binary", pos_label=0, threshold=0.7)
    assert classification_metrics(batches, positive_zero)["precision"] == 0.5


@pytest.mark.parametrize("average,expected", [("macro", 1/6), ("micro", 1/3), ("weighted", 1/6)])
def test_known_multiclass_average_includes_all_output_classes(average, expected):
    batch = {"outputs": torch.tensor([[10., 0., 0.]] * 3), "targets": torch.tensor([0, 1, 2])}
    values = classification_metrics([batch], config(average=average))
    assert values["f1"] == pytest.approx(expected)


def test_multilabel_accuracy_is_exact_match():
    batch = {"outputs": torch.tensor([[10., -10.], [-10., 10.]]),
             "targets": torch.tensor([[1., 0.], [1., 1.]])}
    values = classification_metrics([batch], config(task_type="multilabel"))
    assert values["accuracy"] == 0.5
    assert values["f1"] == pytest.approx((2/3 + 1) / 2)


@pytest.mark.parametrize("task_type", ["binary", "regression", "multilabel"])
def test_builtin_training_task_types(task_type, tmp_path):
    multilabel = task_type == "multilabel"
    source = SOURCE.replace('torch.tensor(payload["label"], dtype=torch.int64)',
                            'torch.tensor([payload["label"], 1 - payload["label"]], dtype=torch.float32)'
                            if multilabel else 'torch.tensor([payload["label"]], dtype=torch.float32)')
    classes = 2 if multilabel else 1
    contract = {"inputs": {"x": {"dtype": "float32", "shape": [None, 2]}},
                "targets": {"dtype": "float32", "shape": [None, classes]},
                "outputs": {"dtype": "float32", "shape": [None, classes]}}
    events = train(tmp_path, config(task_type=task_type, python_source=source, contract=contract,
                                   model_params={"classes": classes}, epochs=1))
    last = [e for e in events if e["type"] == "metric"][-1]["metrics"]
    assert math.isfinite(last["loss"])
    assert (last["f1"] is None) == (task_type == "regression")


def test_dictionary_targets_and_outputs_with_custom_hooks(tmp_path):
    source = SOURCE.replace('"targets": torch.tensor(payload["label"], dtype=torch.int64)',
                            '"targets": {"y": torch.tensor(payload["label"], dtype=torch.int64)}')
    source = source.replace('return self.linear(x)', 'return {"scores": self.linear(x)}')
    source += '''
def compute_loss(outputs, targets, params):
    return torch.nn.functional.cross_entropy(outputs["scores"], targets["y"])
def compute_metrics(batches, params):
    assert batches[0]["targets"]["y"].device.type == "cpu"
    return {"custom_score": 0.5}
'''
    contract = {"inputs": {"x": {"dtype": "float32", "shape": [None, 2]}},
                "targets": {"y": {"dtype": "int64", "shape": [None]}},
                "outputs": {"scores": {"dtype": "float32", "shape": [None, 2]}}}
    events = train(tmp_path, config(task_type="custom", python_source=source, metrics="custom", contract=contract))
    assert [e for e in events if e["type"] == "metric"][-1]["metrics"]["custom_score"] == 0.5


@pytest.mark.parametrize("scheduler,params", [("StepLR", {"step_size": 1, "gamma": 0.5}),
                                              ("CosineAnnealingLR", {}),
                                              ("ReduceLROnPlateau", {})])
def test_learning_rate_schedulers(scheduler, params, tmp_path):
    events = train(tmp_path, config(scheduler=scheduler, scheduler_params=params, epochs=2))
    steps = [e for e in events if e["type"] == "metric" and e["phase"] == "train"]
    assert steps[0]["metrics"]["learning_rate"] == 0.1
    if scheduler != "ReduceLROnPlateau":
        assert steps[-1]["metrics"]["learning_rate"] < 0.1


def test_worker_real_subprocess_persists_metrics_downloads_and_no_query_mysql(training_client):
    app, client = training_client
    source_code = 'print("插件日志")\n' + SOURCE + '\nimport time\ndef compute_loss(outputs, targets, params):\n    time.sleep(0.1)\n    return torch.nn.functional.cross_entropy(outputs, targets)\n'
    task = create(client, python_source=source_code, loss="custom")
    source = RecordsSource()
    repository = app.state.training_service.repository
    seen_running = []

    def observe():
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            current = repository.get(task["id"])
            if current["status"] == "running" and repository.metrics(task["id"])["items"]:
                seen_running.append(True)
                return
            if current["status"] in ("succeeded", "failed"):
                return
            time.sleep(0.01)

    # 避免微型网络太快完成而无法观察中间结果。
    observer = threading.Thread(target=observe)
    observer.start()
    assert TrainingWorker(repository, source).run_once()
    observer.join()
    public = client.get(BASE + "/tasks/" + task["id"], headers=HEADERS).json()["data"]
    assert public["status"] == "succeeded", public["error_message"]
    assert seen_running and len(public["loss"]) == 3 and public["progress"] == 100
    assert public["runtime"]["python"].startswith("3.11")
    assert public["latest_metrics"]["f1"] is not None
    first = client.get(BASE + "/tasks/" + task["id"] + "/metrics?limit=2", headers=HEADERS).json()["data"]
    assert len(first["items"]) == 2 and first["has_more"]
    second = client.get(BASE + "/tasks/" + task["id"] + "/metrics", headers=HEADERS,
                        params={"after_id": first["next_after_id"]}).json()["data"]
    assert all(row["id"] > first["next_after_id"] for row in second["items"])
    artifacts = client.get(BASE + "/tasks/" + task["id"] + "/artifacts", headers=HEADERS).json()["data"]
    assert {a["filename"] for a in artifacts} >= {"source.py", "data.json", "config.json", "split.json",
                                                "environment.json", "last.pt", "best.pt", "stderr.log"}
    for artifact in artifacts:
        path = BASE + "/tasks/" + task["id"] + "/artifacts/" + artifact["id"] + "/download"
        response = client.get(path, headers=HEADERS)
        assert response.status_code == 200 and len(response.content) == artifact["size"]
        assert sha256(response.content).hexdigest() == artifact["sha256"]
        assert client.get(path, headers=OTHER).status_code == 404
    assert source.calls == 1
    assert "插件日志" in (repository.directory(task["id"]) / "stderr.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("override,error", [
    ({"version_id": "old"}, "版本"),
    ({"python_source": "import nonexistent_training_dependency\n" + SOURCE}, "ModuleNotFoundError"),
    ({"python_source": SOURCE.replace('torch.tensor(payload["label"], dtype=torch.int64)',
                                      'torch.tensor(999, dtype=torch.int64)')}, "类别范围"),
])
def test_worker_failures_preserve_error_without_fake_results(override, error, training_client):
    app, client = training_client
    task = create(client, **override)
    assert TrainingWorker(app.state.training_service.repository, RecordsSource()).run_once()
    public = client.get(BASE + "/tasks/" + task["id"], headers=HEADERS).json()["data"]
    assert public["status"] == "failed" and error in public["error_message"]
    assert public["loss"] == [] and public["latest_metrics"]["f1"] is None
    assert not public["checkpoints"]


@pytest.mark.parametrize("timeout,source", [
    (30, SOURCE + '\nimport time\ndef compute_loss(outputs, targets, params):\n    time.sleep(0.05)\n    return torch.nn.functional.cross_entropy(outputs, targets)\n'),
    (0.1, SOURCE.replace('return Classifier(model_params.get("input_size", 2), model_params.get("classes", 2))',
                         'import time\n    time.sleep(60)\n    return Classifier(2, 2)')),
])
def test_cooperative_and_forced_cancel(timeout, source, training_client):
    app, client = training_client
    task = create(client, python_source=source, epochs=100, loss="custom" if "compute_loss" in source else "auto")
    repository = app.state.training_service.repository
    worker = threading.Thread(target=TrainingWorker(repository, RecordsSource(), cancel_timeout=timeout).run_once)
    worker.start()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and not (repository.directory(task["id"]) / "environment.json").exists():
        time.sleep(0.02)
    assert worker.is_alive()
    response = client.post(BASE + "/tasks/" + task["id"] + "/cancel", headers=HEADERS)
    assert response.json()["data"]["status"] in ("canceling", "canceled")
    worker.join(timeout=40)
    assert not worker.is_alive()
    assert repository.get(task["id"])["status"] == "canceled"


def test_worker_single_instance_and_restart_does_not_requeue(training_client):
    app, client = training_client
    repository = app.state.training_service.repository
    task = create(client)
    repository.claim()
    repository.record_metric(task["id"], {"phase": "train", "epoch": 1, "step": 1,
                                          "progress": 5, "metrics": {"loss": 0.7}})
    with worker_lock(repository.root):
        with pytest.raises(RuntimeError, match="单实例"):
            with worker_lock(repository.root):
                pass
    restarted = TrainingRepository(Database(app.state.database.path))
    restarted.recover()
    assert restarted.get(task["id"])["status"] == "failed"
    assert len(restarted.metrics(task["id"])["items"]) == 1
    assert restarted.claim() is None


def test_requested_cuda_fails_but_auto_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device("auto").type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA"):
        select_device("cuda")


@pytest.mark.skipif(os.name != "nt", reason="验证 Windows 父进程句柄监测")
def test_windows_child_exits_when_worker_process_dies(tmp_path):
    import ctypes
    directory = tmp_path / "orphan"
    directory.mkdir()
    values = config(python_source=SOURCE.replace(
        'return Classifier(model_params.get("input_size", 2), model_params.get("classes", 2))',
        'import time\n    time.sleep(60)\n    return Classifier(2, 2)'))
    (directory / "source.py").write_text(values["python_source"], encoding="utf-8")
    (directory / "config.json").write_text(json.dumps(values), encoding="utf-8")
    (directory / "data.json").write_text(json.dumps(RECORDS), encoding="utf-8")
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=os.pathsep.join([
        str(os.getcwd()), str(os.path.join(sys.prefix, "Lib", "site-packages"))]))
    program = '''import os, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-m", "service.training_child", sys.argv[1],
                          "--parent-pid", str(os.getpid())], stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=subprocess.CREATE_NO_WINDOW)
print(child.pid, flush=True)
time.sleep(60)
'''
    parent = subprocess.Popen([sys._base_executable, "-c", program, str(directory)],
                              env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              creationflags=subprocess.CREATE_NO_WINDOW)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = None
    try:
        child_pid = int(parent.stdout.readline())
        handle = kernel.OpenProcess(0x00100000, False, child_pid)
        assert handle
        deadline = time.monotonic() + 15
        while not (directory / "environment.json").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (directory / "environment.json").exists()
        parent.kill()
        parent.wait(timeout=5)
        assert kernel.WaitForSingleObject(handle, 5000) == 0
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        parent.stdout.close()
        parent.stderr.close()
        if handle:
            kernel.CloseHandle(handle)


def test_real_cuda_forward_backward_and_short_training(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("当前测试主机没有可用 CUDA；不能据此宣称 GPU 验证通过")
    assert select_device("cuda").type == "cuda"
    events = train(tmp_path, config(device="cuda", epochs=2))
    assert next(e for e in events if e["type"] == "runtime")["metadata"]["device"] == "cuda"
    history = [e for e in events if e["type"] == "metric" and e["phase"] == "validation"]
    assert len(history) == 2 and math.isfinite(history[-1]["metrics"]["loss"])
    assert (tmp_path / "last.pt").is_file()
