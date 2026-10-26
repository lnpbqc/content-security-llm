"""通用脚本协议、HTTP 隔离和真实子进程监管。"""

import json
import os
import subprocess
import sys
import threading
import time
from datetime import timedelta
from hashlib import sha256

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field

from config import Settings
from main import create_app
from service.script_training_worker import EventReader
from service.training_scripts import SCRIPTS, ScriptDefinition
from service.training_worker import TrainingWorker
from utils.time import utc_now


BASE = "/api/v1/training"
HEADERS = {"Authorization": "Bearer " + "a" * 32}
ADMIN = {**HEADERS, "X-Setup-Secret": "s" * 32}
OTHER = {"Authorization": "Bearer " + "b" * 32}
PRELUDE = '''import argparse, json, time, sys
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
config = json.loads(Path(parser.parse_args().config).read_text(encoding="utf-8"))
output = Path(config["output_dir"])
def emit(event):
    with Path(config["events_path"]).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\\n")
'''


class Inputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = "通用输入"


class Parameters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    scale: int = Field(default=2, ge=1)


@pytest.fixture
def context(tmp_path, monkeypatch):
    script = tmp_path / "registered.py"
    script.write_text(PRELUDE, encoding="utf-8")
    monkeypatch.setitem(SCRIPTS, "text-example", ScriptDefinition("文本统计", "无分类指标", script, Inputs, Parameters))
    app = create_app(Settings(tmp_path / "db", "s" * 32, Fernet.generate_key()))
    with TestClient(app) as client:
        for label in ("a", "b"):
            app.state.auth_service.create_token("s" * 32, label * 32, utc_now() + timedelta(days=1))
        yield app.state.training_service.repository, client, script


def create(context, body="", **overrides):
    repository, client, script = context
    script.write_text(PRELUDE + body, encoding="utf-8")
    response = client.post(BASE + "/script-tasks", headers=ADMIN,
                           json={"name": "通用脚本", "script_id": "text-example", **overrides})
    assert response.status_code == 202, response.text
    return response.json()["data"]


class NoBusinessData:
    def samples(self, scope):
        raise AssertionError("脚本任务不能隐式读取业务 MySQL")


def test_catalog_defaults_validation_and_auth(context):
    repository, client, _ = context
    assert client.get(BASE + "/scripts").status_code == 401
    scripts = client.get(BASE + "/scripts", headers=HEADERS).json()["data"]
    sample = next(item for item in scripts if item["id"] == "text-example")
    assert sample["parameters_schema"]["properties"]["scale"]["default"] == 2
    assert "path" not in sample
    schema = client.get("/openapi.json").json()
    assert schema["paths"][BASE + "/script-tasks"]["post"]["responses"]["202"]["content"]["application/json"]["schema"]["$ref"]
    assert "latest_metrics" in schema["components"]["schemas"]["ScriptTaskPublic"]["properties"]
    task = create(context)
    assert task["parameters"] == {"scale": 2} and task["inputs"] == {"text": "通用输入"}
    assert task["latest_metrics"] == {} and task["epoch"] is task["step"] is task["progress"] is None
    assert "python_source" not in task and "python_source" not in task["config"]
    assert not repository.directory(task["id"]).exists()
    body = {"name": "无权限", "script_id": "text-example"}
    assert client.post(BASE + "/script-tasks", headers=HEADERS, json=body).status_code == 403
    for override in ({"parameters": {"scale": 0}}, {"parameters": {"scale": True}},
                     {"parameters": {"activation": "relu"}}, {"inputs": {"path": "/etc/passwd"}},
                     {"python_source": "print(1)"}, {"name": " "}):
        assert client.post(BASE + "/script-tasks", headers=ADMIN, json={**body, **override}).status_code == 422
    assert client.post(BASE + "/script-tasks", headers=ADMIN,
                       json={**body, "script_id": "../arbitrary.py"}).status_code == 404
    for suffix in ("", "/metrics", "/artifacts", "/logs"):
        assert client.get(BASE + "/tasks/" + task["id"] + suffix, headers=OTHER).status_code == 403
    assert client.post(BASE + "/tasks/" + task["id"] + "/cancel", headers=OTHER).status_code == 403
    assert client.get(BASE + "/tasks/" + "00000000-0000-0000-0000-000000000000", headers=OTHER).status_code == 404


def test_incremental_events_and_final_unterminated_line(tmp_path):
    path, seen = tmp_path / "events.jsonl", []
    reader = EventReader(path, seen.append)
    reader.read()
    raw = json.dumps({"type": "metric", "phase": "自定义阶段", "metrics": {"任意分数": 1.25}}, ensure_ascii=False).encode()
    path.write_bytes(raw[:25])
    reader.read()
    assert seen == []
    with path.open("ab") as stream:
        stream.write(raw[25:] + b"\n" + b'{"type":"progress","progress":9')
    reader.read()
    reader.read()
    assert len(seen) == 1 and seen[0]["metrics"] == {"任意分数": 1.25}
    with path.open("ab") as stream:
        stream.write(b"}")
    reader.read(final=True)
    reader.read(final=True)
    assert len(seen) == 2 and seen[1]["progress"] == 9


def test_generic_real_process_snapshot_metrics_artifacts_and_logs(context):
    repository, client, script = context
    body = '''print("普通日志，不是 JSON", flush=True)
emit({"type":"metric", "phase":"analysis", "metrics":{"characters":len(config["inputs"]["text"]) * config["parameters"]["scale"]}})
(output / "nested").mkdir()
(output / "nested" / "result.txt").write_text("实际产物", encoding="utf-8")
emit({"type":"artifact", "filename":"nested/result.txt", "kind":"report"})
'''
    task = create(context, body, parameters={"scale": 3})
    digest = sha256(script.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    script.write_text("raise RuntimeError('不应执行登记后变化的源码')", encoding="utf-8")
    assert TrainingWorker(repository, NoBusinessData()).run_once()
    path = BASE + "/tasks/" + task["id"]
    result = client.get(path, headers=HEADERS).json()["data"]
    assert result["status"] == "succeeded", result["error_message"]
    assert result["source_hash"] == digest and result["progress"] is None and result["epoch"] is None
    assert result["latest_metrics"] == {"characters": 12} and result["latest_phase"] == "analysis"
    metrics = client.get(path + "/metrics?limit=1", headers=HEADERS).json()["data"]
    assert len(metrics["items"]) == 1 and not metrics["has_more"]
    assert "epoch" not in metrics["items"][0] and "step" not in metrics["items"][0]
    assert client.get(path + "/metrics", params={"after_id": metrics["next_after_id"]}, headers=HEADERS).json()["data"]["items"] == []
    logs = client.get(path + "/logs", headers=HEADERS).json()["data"]
    assert logs["text"].splitlines() == ["普通日志，不是 JSON"]
    assert logs["next_after_offset"] == len(logs["text"].encode())
    assert client.get(path + "/logs", params={"after_offset": logs["next_after_offset"]}, headers=HEADERS).json()["data"]["text"] == ""
    artifacts = client.get(path + "/artifacts", headers=HEADERS).json()["data"]
    assert {item["filename"] for item in artifacts} >= {"source.py", "config.json", "run.log", "events.jsonl", "outputs/nested/result.txt"}
    for artifact in artifacts:
        download = path + "/artifacts/" + artifact["id"] + "/download"
        response = client.get(download, headers=HEADERS)
        assert response.status_code == 200
        assert sha256(response.content).hexdigest() == artifact["sha256"]
        assert client.get(download, headers=OTHER).status_code == 403


@pytest.mark.parametrize("body, expected", [
    ('print("没有指标，也可以成功")\n', "succeeded"),
    ('print("错误详情", flush=True)\nsys.exit(7)\n', "failed"),
    ('Path(config["events_path"]).write_text("not json\\n")\n', "failed"),
    ('emit({"type":"metric","metrics":{"bad":float("nan")}})\n', "failed"),
    ('emit({"type":"metric","metrics":{"bad":True}})\n', "failed"),
    ('emit({"type":"metric","metrics":{" " :1}})\n', "failed"),
    ('emit({"type":"artifact","filename":"../config.json"})\n', "failed"),
    ('emit({"type":"progress","progress":101})\n', "failed"),
])
def test_failure_and_no_fabricated_metrics(context, body, expected):
    repository, client, _ = context
    task = create(context, body)
    assert TrainingWorker(repository, NoBusinessData()).run_once()
    result = client.get(BASE + "/tasks/" + task["id"], headers=HEADERS).json()["data"]
    assert result["status"] == expected and result["latest_metrics"] == {} and result["progress"] is None
    if expected == "failed":
        assert result["error_message"]
    assert (repository.directory(task["id"]) / "run.log").exists()


def test_log_cursor_utf8_partial_lines_and_limits(context):
    repository, client, _ = context
    task = create(context)
    directory = repository.directory(task["id"])
    directory.mkdir(parents=True)
    path = directory / "run.log"
    path.write_bytes("中文\n未完成".encode())
    url = BASE + "/tasks/" + task["id"] + "/logs"
    data = client.get(url, headers=HEADERS).json()["data"]
    assert data["text"] == "中文\n" and data["next_after_offset"] == 7 and data["has_more"]
    assert client.get(url, headers=HEADERS, params={"after_offset": 1}).status_code == 422
    assert client.get(url, headers=HEADERS, params={"after_offset": 999}).status_code == 422
    assert client.get(url, headers=HEADERS, params={"limit": 3}).status_code == 422
    assert client.get(url, headers=HEADERS, params={"after_offset": 7}).json()["data"]["text"] == ""
    repository.claim()
    repository.finish(task["id"], "succeeded")
    assert client.get(url, headers=HEADERS, params={"after_offset": 7}).json()["data"]["text"] == "未完成"


def test_interleaved_metric_pagination_and_final_artifact_hash(context):
    repository, client, _ = context
    task = create(context, '''emit({"type":"metric", "metrics":{"score":1}})
emit({"type":"progress", "progress":25})
(output / "result.txt").write_text("first")
emit({"type":"artifact", "filename":"result.txt"})
time.sleep(0.3)
(output / "result.txt").write_text("final contents")
emit({"type":"metric", "phase":"custom", "step":9, "metrics":{"score":2}})
emit({"type":"progress", "progress":100})
''')
    TrainingWorker(repository, NoBusinessData()).run_once()
    path = BASE + "/tasks/" + task["id"]
    first = client.get(path + "/metrics", headers=HEADERS, params={"limit": 1}).json()["data"]
    assert first["has_more"] and first["items"][0]["metrics"] == {"score": 1}
    second = client.get(path + "/metrics", headers=HEADERS,
                        params={"after_id": first["next_after_id"], "limit": 1}).json()["data"]
    assert not second["has_more"] and second["items"][0]["metrics"] == {"score": 2}
    assert second["next_after_id"] > first["next_after_id"] + 1
    public = client.get(path, headers=HEADERS).json()["data"]
    assert public["status"] == "succeeded" and public["progress"] == 100
    assert public["epoch"] is None and public["step"] == 9
    artifact = next(item for item in repository.artifacts(task["id"]) if item["filename"] == "outputs/result.txt")
    assert artifact["size"] == len(b"final contents")
    assert artifact["sha256"] == sha256(b"final contents").hexdigest()


@pytest.mark.parametrize("cooperative", [True, False])
def test_cancel_script_process(context, cooperative):
    repository, client, _ = context
    body = '''print("ready", flush=True)
while True:
    time.sleep(0.02)
'''
    if cooperative:
        body += '    if Path(config["cancel_path"]).exists():\n        break\n'
    task = create(context, body)
    worker = threading.Thread(target=TrainingWorker(repository, NoBusinessData(), cancel_timeout=0.3).run_once)
    worker.start()
    deadline = time.monotonic() + 15
    log = repository.directory(task["id"]) / "run.log"
    while time.monotonic() < deadline and not (log.exists() and b"ready" in log.read_bytes()):
        time.sleep(0.02)
    assert worker.is_alive()
    client.post(BASE + "/tasks/" + task["id"] + "/cancel", headers=HEADERS)
    worker.join(10)
    assert not worker.is_alive() and repository.get(task["id"])["status"] == "canceled"


def test_pending_cancel_and_recovery(context):
    repository, client, _ = context
    task = create(context)
    response = client.post(BASE + "/tasks/" + task["id"] + "/cancel", headers=HEADERS)
    assert response.json()["data"]["status"] == "canceled"
    assert not TrainingWorker(repository, NoBusinessData()).run_once()
    other = create(context)
    repository.claim()
    repository.recover()
    assert repository.get(other["id"])["status"] == "failed"
    assert not TrainingWorker(repository, NoBusinessData()).run_once()


def test_cifar_parameter_validation(context):
    _, client, _ = context
    request = {"name": "CIFAR", "script_id": "cifar100-cnn"}
    task = client.post(BASE + "/script-tasks", headers=ADMIN, json=request).json()["data"]
    assert task["parameters"]["device"] == "cuda" and task["parameters"]["epochs"] == 3
    for parameters in ({"train_samples": 99}, {"validation_samples": 99},
                       {"train_samples": 40000, "validation_samples": 20000},
                       {"device": "auto"}, {"batch_size": True}):
        assert client.post(BASE + "/script-tasks", headers=ADMIN,
                           json={**request, "parameters": parameters}).status_code == 422


def process_alive(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        from pathlib import Path
        status = Path("/proc") / str(pid) / "stat"
        if status.exists() and status.read_text().split(")", 1)[1].strip().startswith("Z"):
            return False
        return True
    except ProcessLookupError:
        return False


def wait_for_file(path, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size:
            return
        time.sleep(0.02)
    pytest.fail("子进程未在期限内写入 PID")


def test_worker_death_cleans_script_and_descendant(context):
    repository, _, _ = context
    task = create(context, '''import subprocess, os
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
(output / "pids.json").write_text(json.dumps([os.getpid(), child.pid]))
time.sleep(60)
''')
    launcher = repository.root.parent / "worker_launcher.py"
    launcher.write_text('''import sys
from pathlib import Path
from config import Settings
from db.database import Database
from db.training import TrainingRepository
from service.training_worker import TrainingWorker
from dataclasses import replace
TrainingWorker(TrainingRepository(Database(replace(Settings.load(), data_dir=Path(sys.argv[1])))), None).run_once()
''', encoding="utf-8")
    from pathlib import Path
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join([
        os.getcwd(), str(Path(sys.prefix) / "Lib" / "site-packages"), os.environ.get("PYTHONPATH", "")])}
    executable = sys._base_executable if os.name == "nt" else sys.executable
    process = subprocess.Popen([executable, str(launcher), str(repository.database.data_dir)],
                               env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        path = repository.directory(task["id"]) / "outputs" / "pids.json"
        wait_for_file(path)
        pids = json.loads(path.read_text())
        assert all(process_alive(pid) for pid in pids)
        process.kill()
        process.wait(timeout=10)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and any(process_alive(pid) for pid in pids):
            time.sleep(0.05)
        assert not any(process_alive(pid) for pid in pids)
        repository.recover()
        assert repository.get(task["id"])["status"] == "failed"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stderr.close()


def test_cifar_stratified_split_and_fixed_model():
    from training_scripts.cifar100 import SmallCNN, stratified_split
    targets = [label for label in range(100) for _ in range(500)]
    train, validation = stratified_split(targets, 10003, 2007, 42)
    assert len(train) == 10003 and len(validation) == 2007
    assert not set(train) & set(validation)
    assert {targets[index] for index in train} == set(range(100))
    assert {targets[index] for index in validation} == set(range(100))
    assert stratified_split(targets, 10003, 2007, 42) == (train, validation)
    assert len(SmallCNN().state_dict()) > 0
