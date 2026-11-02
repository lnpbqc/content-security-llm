import json
from datetime import timedelta

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from config import Settings
from main import create_app
from utils.time import utc_now


@pytest.fixture
def api(tmp_path):
    sent = []
    def transport(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 1,
            "model": "qwen3-8b", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "已保护测试代号"}}]})
    from openai import OpenAI
    def factory(**kwargs):
        return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(transport)))
    with TestClient(create_app(Settings(tmp_path / "data", "s" * 32, Fernet.generate_key()), client_factory=factory)) as client:
        admin = {"X-Setup-Secret": "s" * 32, "Authorization": "Bearer " + "a" * 64}
        client.post("/api/auth/token", headers=admin, json={"token": "a" * 64,
            "expires_at": (utc_now() + timedelta(days=1)).isoformat()}).raise_for_status()
        response = client.post("/api/models", headers=admin, json={"name": "qwen", "upstream_model_id": "/models/Qwen3-8B",
            "base_url": "https://model.invalid/v1", "api_key": "test-provider-secret"})
        response.raise_for_status()
        model_id = response.json()["model_id"]
        # 业务版本表并非 llm 表；测试数据库中模拟已有业务版本，不操作生产库。
        with client.app.state.database.connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS model_versions (id BIGINT PRIMARY KEY, model_id BIGINT, version VARCHAR(100), task_id VARCHAR(64))")
            if not connection.execute("SHOW COLUMNS FROM model_versions LIKE 'task_id'").fetchone():
                connection.execute("ALTER TABLE model_versions ADD COLUMN task_id VARCHAR(64)")
            connection.execute("DELETE FROM model_versions")
            connection.execute("INSERT INTO model_versions (id,model_id,version) VALUES (1,10,'v1-policy')")
        yield client, admin, model_id, sent


def test_binding_and_real_sdk_system_prompt(api):
    client, admin, model_id, sent = api
    path = "/api/v1/admin/model-evaluation/bindings/10/v1-policy"
    payload = {"model_id": model_id, "mode": "simulated_policy", "policy_version": "policy-v1",
               "system_prompt": "保护测试代号"}
    assert client.put(path, json=payload).status_code == 403
    response = client.put(path, headers=admin, json=payload)
    assert response.status_code == 200, response.text
    binding = response.json()["data"]
    assert "api_key" not in json.dumps(binding)
    result = client.post("/api/v1/model-evaluation/answer", headers=admin,
        json={"platform_model_id": 10, "version": "v1-policy", "binding_hash": binding["binding_hash"], "question": "测试问题"})
    assert result.status_code == 200, result.text
    assert result.json()["data"]["output"] == "已保护测试代号"
    assert result.json()["data"]["call_ids"]
    assert sent[0]["messages"][0]["content"] == "保护测试代号"
    assert sent[0]["messages"][1]["content"] == "测试问题"
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": False}
    client.patch("/api/models/" + model_id, headers=admin, json={"upstream_model_id": "another-model"}).raise_for_status()
    stale = client.post("/api/v1/model-evaluation/answer", headers=admin,
        json={"platform_model_id": 10, "version": "v1-policy", "binding_hash": binding["binding_hash"], "question": "测试问题"})
    assert stale.status_code == 409
    assert len(sent) == 1


def test_missing_and_invalid_binding(api):
    client, admin, model_id, sent = api
    path = "/api/v1/admin/model-evaluation/bindings/10/v1-policy"
    assert client.put(path, headers=admin, json={"model_id": model_id, "mode": "simulated_policy"}).status_code == 422
    assert client.put(path, headers=admin, json={"model_id": "missing"}).status_code == 404
    assert client.get("/api/v1/model-evaluation/bindings/10/v1-policy", headers=admin).status_code == 409
    assert client.get("/api/v1/model-evaluation/bindings/10/v1-policy").status_code == 401
    assert not sent


def test_prepare_edit_inherits_administrator_template(api):
    client, admin, model_id, _ = api
    # 使用专用测试库的业务表，模拟真实编辑产物与其来源任务。
    with client.app.state.database.connect() as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS tasks (task_id VARCHAR(64) PRIMARY KEY, status VARCHAR(30), "
                           "capability_code VARCHAR(100),input_data LONGTEXT,result LONGTEXT)")
        connection.execute("DELETE FROM tasks")
        connection.execute("INSERT INTO tasks VALUES ('edit-round-1','succeeded','knowledge_edit',%s,%s)",
                           (json.dumps({"model_id": "10", "target_knowledge": "保护目标甲"}),
                            json.dumps({"before_version": "v1", "after_version": "v1-policy"})))
        connection.execute("UPDATE model_versions SET task_id='edit-round-1'")
        connection.execute("INSERT INTO model_versions VALUES (2,10,'v1',NULL)")
    path = "/api/v1/model-evaluation/prepare-edit"
    payload = {"platform_model_id": 10, "version": "v1-policy"}
    assert client.post(path, headers=admin, json=payload).status_code == 409
    client.put("/api/v1/admin/model-evaluation/bindings/10/v1", headers=admin,
               json={"model_id": model_id, "mode": "baseline", "simulation_template": {
                   "version": "template-v1", "system_prompt": "请遵守编辑目标：{target_knowledge}"}}).raise_for_status()
    response = client.post(path, headers=admin, json=payload)
    assert response.status_code == 200, response.text
    binding = response.json()["data"]
    assert binding["model_id"] == model_id and binding["mode"] == "simulated_policy"
    assert binding["system_prompt"] == "请遵守编辑目标：保护目标甲"
    assert client.post(path, headers=admin, json=payload).json()["data"]["binding_hash"] == binding["binding_hash"]
    assert client.post(path, headers=admin, json={**payload, "version": "missing"}).status_code == 409
