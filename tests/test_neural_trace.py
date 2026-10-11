"""接口、持久化、流水线和失败恢复验收；不导入 GPU 库，不加载模型。"""

import importlib.util
import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from config import Settings
from db.neural_trace import NeuralTraceRepository
from main import create_app
from service.neural_trace import STAGES
from service import neural_trace_worker as worker_module
from utils.time import utc_now


BASE = "/api/v1/neural-trace"
CASE = "CASE_HIS_028_V01"
AUTH = {"Authorization": "Bearer " + "a" * 64}
OTHER = {"Authorization": "Bearer " + "b" * 64}


@pytest.fixture
def api(tmp_path):
    with TestClient(create_app(Settings(tmp_path / "db", "s" * 32, Fernet.generate_key()))) as client:
        for token in ("a" * 64, "b" * 64):
            response = client.post("/api/auth/token", headers={"X-Setup-Secret": "s" * 32}, json={
                "token": token, "expires_at": (utc_now() + timedelta(days=1)).isoformat()})
            assert response.status_code == 201
        yield client


def submit(api, case=CASE):
    response = api.post(BASE + "/tasks", headers=AUTH, json={"case_id": case})
    assert response.status_code == 202, response.text
    return response.json()["data"]["task_id"]


def stage_outputs():
    common = {"case_id": CASE, "summary": {}, "technical_checks": {"technical_pass": True}}
    generation = {"patched_answer": "正确答案", "patched_answer_matches_gold": True,
                  "patched_answer_matches_poison": False}
    layer = {"layer_index": 1, "crs": 1.2, "patched_margin": 0.5, "margin_delta": 0.7}
    return {
        "document_trace": {**common, "formal_result": True, "summary": {"document_trace_success": True},
                           "conditions": {"original_poison": {"generation": {"model_answer": "污染答案"},
                           "margin": -0.2, "gold_answer_match": False, "poison_answer_match": True}}},
        "prepare_inputs": {**common, "validation": {"alignment_valid": True},
                           "prompt_files": {"clean_prompt_sha256": "clean-hash", "poison_prompt_sha256": "poison-hash"}},
        "layer_scan": {**common, "intervention": {"all_layers_scanned": True, "patch_scope": "candidate_prediction_span"},
                       "environment": {"model_path": "/private/model", "dtype": "bfloat16"},
                       "layer_results": [layer], "summary": {"max_crs": 1.2, "max_crs_layer": 1}},
        "activation_difference": {**common, "layer_results": [
            {"layer_index": 1, "normalized_l2": 0.12, "cosine_distance": 0.02}]},
        "component_scan": {**common, "intervention": {"actual_generation_performed": True,
                           "patch_scope": "candidate_prediction_span", "generation_patch_scope": "generation_prompt_last_token"},
                           "baseline_generation": {"clean": {"model_answer": "正确答案"},
                                                   "poison": {"model_answer": "污染答案"}},
                           "layer_results": [{"layer_index": 1,
                               "full_layer_reference": {**layer, "generation": generation},
                               "component_results": {"self_attn": {**layer, **generation},
                                                     "mlp": {**layer, **generation}}}]},
    }


def fake_gpu(monkeypatch, tmp_path, fail_stage=None, invalid_stage=None):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(worker_module, "check_model", lambda: None)
    monkeypatch.setattr(worker_module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(worker_module.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=""))
    original_collect = worker_module.collect_result
    monkeypatch.setattr(worker_module, "collect_result", lambda task, outputs: original_collect(task, outputs, model))
    seen = []

    def run(command, cwd, log_path):
        stage = log_path.stem
        seen.append(stage)
        if stage == fail_stage:
            raise RuntimeError("测试子进程失败")
        assert command[0] == sys.executable
        assert "--allow-busy-gpu" not in command and "--overwrite" not in command
        directory = Path(command[command.index("--output-dir") + 1])
        if stage == "prepare_inputs":
            path = directory / CASE / "token_alignment.json"
        else:
            suffix = {"document_trace": ".json", "layer_scan": ".layer_scan.candidate_prediction_span.json",
                      "activation_difference": ".activation_difference.json", "component_scan": ".component_scan.json"}
            path = directory / (CASE + suffix[stage])
        value = stage_outputs()[stage]
        if invalid_stage == stage:
            value["technical_checks"] = {"technical_pass": False}
            value["formal_result"] = False
            value["validation"] = {"alignment_valid": False}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        if stage == "document_trace":
            source = Path(command[command.index("--results-dir") + 1]) / CASE / "E1.json"
            assert json.loads(source.read_text(encoding="utf-8"))["case_id"] == CASE
        if stage == "layer_scan":
            assert command[command.index("--patch-scope") + 1] == "candidate_prediction_span"
        if stage == "component_scan":
            assert "--generate-patched-answers" in command
    monkeypatch.setattr(worker_module, "run_command", run)
    return seen


def test_api_validates_and_isolates_tasks_without_loading(api):
    assert api.get(BASE + "/cases").status_code == 401
    cases = api.get(BASE + "/cases", headers=AUTH).json()["data"]
    assert len(cases) == 5 and CASE in [item["case_id"] for item in cases]
    for payload, status in (({"case_id": "../escape"}, 422), ({"case_id": "unknown"}, 404),
                            ({"case_id": CASE, "model_path": "outside"}, 422)):
        assert api.post(BASE + "/tasks", headers=AUTH, json=payload).status_code == status
    task_id = submit(api)
    task = api.get(BASE + "/tasks/" + task_id, headers=AUTH).json()["data"]
    assert task["status"] == "pending" and task["stage"] == "queued"
    assert "token_hash" not in task and "input" not in task
    assert api.get(BASE + "/tasks", headers=OTHER).json()["data"]["total"] == 0
    for suffix in ("", "/result"):
        assert api.get(BASE + "/tasks/" + task_id + suffix, headers=OTHER).status_code == 404
    assert api.get(BASE + "/tasks/" + task_id + "/result", headers=AUTH).status_code == 409


def test_success_pipeline_and_persisted_result(api, monkeypatch, tmp_path):
    seen = fake_gpu(monkeypatch, tmp_path)
    task_id = submit(api)
    repo = api.app.state.neural_trace_service.repository
    worker = worker_module.NeuralTraceWorker(repo)
    assert worker.run_once() and not worker.run_once()
    assert seen == list(STAGES)
    response = api.get(BASE + "/tasks/" + task_id + "/result", headers=AUTH)
    assert response.status_code == 200, response.text
    result = response.json()["data"]
    assert result["layer_scan"]["rows"][0]["crs"] == 1.2  # CRS 不截断成准确率
    assert result["activation_difference"]["rows"][0]["normalized_l2"] == 0.12
    assert len(result["component_scan"]["rows"]) == 3
    assert "model_path" not in result["model"]["environment"]
    assert result["scoring_patch_scope"] != result["generation_patch_scope"]
    token_hash = api.app.state.auth_service.valid_token_hash("a" * 64)
    saved = NeuralTraceRepository(repo.database).get(task_id, token_hash)
    assert saved["status"] == "succeeded" and saved["finished_at"] and saved["result"] == result


@pytest.mark.parametrize("stage", STAGES)
def test_technical_failure_stops_pipeline(api, monkeypatch, tmp_path, stage):
    seen = fake_gpu(monkeypatch, tmp_path, invalid_stage=stage)
    task_id = submit(api)
    assert worker_module.NeuralTraceWorker(api.app.state.neural_trace_service.repository).run_once()
    task = api.get(BASE + "/tasks/" + task_id, headers=AUTH).json()["data"]
    assert task["status"] == "failed" and task["stage"] == stage and task["error_message"]
    assert seen == list(STAGES[:STAGES.index(stage) + 1])
    assert api.get(BASE + "/tasks/" + task_id + "/result", headers=AUTH).status_code == 409


def test_subprocess_failure_and_restart_keep_failed_tasks(api, monkeypatch, tmp_path):
    fake_gpu(monkeypatch, tmp_path, fail_stage="layer_scan")
    first = submit(api)
    repo = api.app.state.neural_trace_service.repository
    worker_module.NeuralTraceWorker(repo).run_once()
    assert api.get(BASE + "/tasks/" + first, headers=AUTH).json()["data"]["error_message"] == "测试子进程失败"
    second = submit(api)
    assert repo.claim()["id"] == second and repo.claim() is None
    repo.recover_running()
    assert api.get(BASE + "/tasks/" + second, headers=AUTH).json()["data"]["status"] == "failed"
    assert repo.claim() is None


def test_missing_model_fails_without_starting_experiment(api, monkeypatch, tmp_path):
    original = worker_module.check_model
    monkeypatch.setattr(worker_module, "check_model", lambda: original(tmp_path))
    monkeypatch.setattr(worker_module, "run_command", lambda *a: pytest.fail("不应启动实验"))
    task_id = submit(api)
    worker_module.NeuralTraceWorker(api.app.state.neural_trace_service.repository).run_once()
    task = api.get(BASE + "/tasks/" + task_id, headers=AUTH).json()["data"]
    assert task["status"] == "failed" and "模型文件不完整" in task["error_message"]


def test_model_shards_checked_without_importing_libraries(tmp_path):
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"one": "model-00001.safetensors", "two": "model-00002.safetensors"}}))
    (tmp_path / "model-00001.safetensors").write_bytes(b"not-real-weights")
    with pytest.raises(RuntimeError, match="模型文件不完整"):
        worker_module.check_model(tmp_path)
    (tmp_path / "model-00002.safetensors").write_bytes(b"not-real-weights")
    worker_module.check_model(tmp_path)


def test_command_logs_and_failure_without_gpu(tmp_path):
    log = tmp_path / "stage.log"
    worker_module.run_command([sys.executable, "-c", "print('no model loaded')"], tmp_path, log)
    assert "no model loaded" in log.read_text(encoding="utf-8")
    with pytest.raises(RuntimeError, match="退出码 3"):
        worker_module.run_command([sys.executable, "-c", "raise SystemExit(3)"], tmp_path, log)


def test_runtime_import_is_lazy():
    root = Path(__file__).resolve().parents[1]
    command = """import importlib.util, sys
def block(event, args):
    if event == 'import' and args[0].split('.')[0] in ('torch', 'transformers', 'nnsight'):
        raise RuntimeError('unexpected GPU import')
sys.addaudithook(block)
spec = importlib.util.spec_from_file_location('runtime', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.MODEL_PATH == sys.argv[2]
"""
    subprocess.run([sys.executable, "-c", command, str(root / "neurotrace-run/core/model_runtime.py"),
                    str(root / "neurotrace-run/models/Qwen3-4B")], check=True)


def test_queued_input_is_immutable(api, tmp_path):
    import shutil
    service = api.app.state.neural_trace_service
    copied = tmp_path / "data"
    shutil.copytree(service.data_root, copied)
    service.data_root = copied
    first = submit(api)
    original = service.snapshot(CASE)["case"]["query"]
    for name in ("E1", "E2"):
        path = copied / "upstream" / CASE / f"{name}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["query"] = "后续修改的问题"
        path.write_text(json.dumps(value), encoding="utf-8")
    token_hash = api.app.state.auth_service.valid_token_hash("a" * 64)
    saved = service.repository.get(first, token_hash)
    assert saved["input"]["case"]["query"] == original
    assert json.loads(saved["input"]["sources"]["E1"])["query"] == original
    second = submit(api)
    assert service.repository.get(second, token_hash)["input"]["case"]["query"] == "后续修改的问题"


def test_busy_gpu_does_not_start_model(api, monkeypatch, tmp_path):
    seen = fake_gpu(monkeypatch, tmp_path)
    monkeypatch.setattr(worker_module.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="12345\n"))
    task_id = submit(api)
    worker_module.NeuralTraceWorker(api.app.state.neural_trace_service.repository).run_once()
    task = api.get(BASE + "/tasks/" + task_id, headers=AUTH).json()["data"]
    assert task["status"] == "failed" and task["stage"] == "preflight" and "GPU 已有" in task["error_message"]
    assert seen == []


def test_interrupted_worker_marks_task_failed(api, monkeypatch, tmp_path):
    fake_gpu(monkeypatch, tmp_path)
    def interrupt(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(worker_module, "run_command", interrupt)
    task_id = submit(api)
    with pytest.raises(KeyboardInterrupt):
        worker_module.NeuralTraceWorker(api.app.state.neural_trace_service.repository).run_once()
    task = api.get(BASE + "/tasks/" + task_id, headers=AUTH).json()["data"]
    assert task["status"] == "failed" and task["finished_at"]
