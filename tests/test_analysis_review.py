"""安全核查的失败案例回归；仅模拟上游，不访问真实服务。"""

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from openai import OpenAI

from test_analysis import api, audit, audit_output, auth, governed, history, output, post, scenario


def test_startup_finishes_interrupted_analysis_and_preserves_history(api):
    from config import Settings
    from main import create_app

    client, model, _, requests = api
    repository = client.app.state.analysis_service.repository
    token_hash = client.app.state.auth_service.valid_token_hash("a" * 64)
    payload = scenario(model)
    state = {"status": "running", "stage": "public_opinion", "result": None, "original_output": "已生成正文",
             "calls": [{"stage": "generation", "call_id": "saved-call", "status": "success", "error_code": None}],
             "error_code": None, "error_message": None, "elapsed_ms": 1234, "finished_at": None}
    interrupted = repository.create(token_hash, "scenario", payload, state)
    finished = []
    for status in ("succeeded", "failed"):
        record_id = repository.create(token_hash, "scenario", payload, state)
        repository.save(record_id, token_hash, {**state, "status": status, "finished_at": "2026-10-10T00:00:00+00:00"})
        finished.append(repository.get(record_id, token_hash))
    # 用同一测试库模拟单 API 实例重启，不能连接真实业务库或重放模型请求。
    settings = Settings(client.app.state.database.path, "s" * 32, Fernet.generate_key())

    def reject_provider(**kwargs):
        raise AssertionError("启动恢复不能重放模型请求")

    with TestClient(create_app(settings, client_factory=reject_provider)) as restarted:
        record = restarted.app.state.analysis_service.repository.get(interrupted, token_hash)
        assert record["status"] == "failed"
        assert record["error_code"] == "analysis_interrupted" and record["error_message"]
        assert record["finished_at"] is not None
        assert record["input"] == payload and record["calls"] == state["calls"]
        assert record["original_output"] == state["original_output"] and record["elapsed_ms"] == 1234
        assert restarted.app.state.analysis_service.repository.list(token_hash, 1, 20, status="running")["total"] == 0
        assert all(restarted.app.state.analysis_service.repository.get(row["id"], token_hash) == row for row in finished)
        assert restarted.app.state.analysis_service.repository.get(interrupted, "other-token") is None
    assert requests == []
    with TestClient(create_app(settings, client_factory=reject_provider)) as restarted:
        assert restarted.app.state.analysis_service.repository.get(interrupted, token_hash) == record


def test_analysis_token_must_not_redirect_shared_model_credentials(api):
    client, model, _, requests = api
    changed = client.patch("/api/models/" + model, headers=auth("b" * 64),
                           json={"base_url": "https://untrusted-review.invalid/v1"})
    response = post(client, "scenarios", scenario(model))
    assert response.status_code == 200
    assert requests[0].url.host == "model.invalid"
    assert requests[0].headers["authorization"] == "Bearer provider-test-key"
    # 地址和上游密钥保持原配置，普通分析令牌不能改写共享模型。
    assert changed.status_code == 403


def test_missing_policy_must_not_accept_unreferenced_requirements(api):
    client, model, upstream, _ = api
    payload, result = scenario(model, "smart_government"), output("smart_government")
    payload["materials"] = []
    result.update(insufficient_materials=True, risk_level="unknown", evidence=[])
    result["details"].update(insufficient_materials=True, policy_claims=[], processing_points=[],
                             answer="申请补贴必须向指定账户缴纳1000元。")
    upstream["queue"] = [result]
    response = post(client, "scenarios", payload)
    assert response.status_code == 502


def test_chain_audit_must_not_claim_completion_after_omitting_every_step(api):
    client, model, upstream, _ = api
    result = audit_output("full_chain")
    result.update(checks=[], risks=[], gaps=[], limitations=[], conclusion="全部步骤检查通过")
    upstream["queue"] = [result]
    assert post(client, "audits", audit(model, "full_chain")).status_code == 502


def test_non_string_provider_content_must_leave_a_failed_generation_record(api, monkeypatch):
    client, model, _, _ = api

    def malformed(request):
        return httpx.Response(200, json={"id": "bad-completion", "object": "chat.completion", "created": 1,
            "model": "test", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "bad format"}]}}]})

    def factory(**kwargs):
        return OpenAI(**kwargs, max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(malformed)))

    monkeypatch.setattr(client.app.state.model_manager, "client_factory", factory)
    # 不重新启动 lifespan；复用当前临时测试应用，只将未捕获异常转为真实 HTTP 500。
    probe = TestClient(client.app, raise_server_exceptions=False)
    try:
        response = post(probe, "model-governance", {"model_id": model, "prompt": "test"})
        saved = history(client)["items"][0]
        assert (response.status_code, saved["status"]) == (502, "failed")
    finally:
        probe.close()


def test_reject_unbounded_resources_before_provider(api):
    client, model, upstream, requests = api
    upstream["queue"] = ["真实模拟生成正文", governed("真实模拟生成正文")]
    response = post(client, "model-governance", {"model_id": model, "prompt": "a" * 1000000,
                                                "max_tokens": 2147483647})
    assert response.status_code == 422 and not requests and history(client)["total"] == 0


@pytest.mark.parametrize("method,path,payload", [
    ("POST", "", {"upstream_model_id": "new", "name": "new", "base_url": "https://new.invalid/v1", "api_key": "dummy"}),
    ("PATCH", "/{id}", {"name": "changed"}),
    ("DELETE", "/{id}", None),
])
@pytest.mark.parametrize("secret", [None, "wrong"])
def test_model_writes_require_existing_admin_secret(api, method, path, payload, secret):
    client, model, _, requests = api
    headers = auth()
    if secret is not None: headers["X-Setup-Secret"] = secret
    response = client.request(method, "/api/models" + path.format(id=model), headers=headers, json=payload)
    assert response.status_code == 403
    assert client.get("/api/models/" + model, headers=auth()).json()["name"] == "测试模型"
    assert not requests


def test_admin_model_writes_still_work_and_do_not_send_setup_secret_upstream(api):
    client, model, _, requests = api
    headers = {**auth(), "X-Setup-Secret": "s" * 32}
    assert client.patch("/api/models/" + model, headers=headers, json={"name": "管理员修改"}).status_code == 200
    response = post(client, "scenarios", scenario(model))
    assert response.status_code == 200 and "x-setup-secret" not in requests[0].headers
    assert client.delete("/api/models/" + model, headers=headers).status_code == 204


def test_analysis_concurrency_limit_does_not_create_or_leak_records(api):
    client, model, _, requests = api
    service = client.app.state.analysis_service
    for _ in range(4): assert service.slots.acquire(blocking=False)
    try:
        assert post(client, "scenarios", scenario(model)).status_code == 429
        assert not requests and history(client)["total"] == 0
    finally:
        for _ in range(4): service.slots.release()
    assert post(client, "scenarios", scenario(model)).status_code == 200


def test_analysis_overrides_sdk_retries_and_timeout(api, monkeypatch):
    client, model, _, _ = api
    requests, sdk_options = [], []
    def handler(request):
        requests.append(request)
        return httpx.Response(500, json={"error": {"message": "test failure"}})
    class ObservedOpenAI(OpenAI):
        def with_options(self, **kwargs):
            sdk_options.append(kwargs)
            return super().with_options(**kwargs)
    def factory(**kwargs):
        return ObservedOpenAI(**kwargs, max_retries=2,
                             http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(client.app.state.model_manager, "client_factory", factory)
    response = post(client, "model-governance", {"model_id": model, "prompt": "test"})
    assert response.status_code == 502 and len(requests) == 1
    assert sdk_options == [{"timeout": 120, "max_retries": 0}]
    assert response.json()["data"]["status"] == "failed"


def test_source_training_requires_admin_before_creating_a_task(api):
    client, _, _, _ = api
    from service.training_examples import example_request
    response = client.post("/api/v1/training/tasks", headers=auth(), json=example_request())
    assert response.status_code == 403
    assert client.get("/api/v1/training/tasks", headers=auth()).json()["data"]["total"] == 0


@pytest.mark.parametrize("bad", ["prompt", "tokens", "materials", "total_content", "steps", "rule_text"])
def test_analysis_input_bounds_are_checked_before_http(api, bad):
    client, model, _, requests = api
    if bad in ("prompt", "tokens"):
        payload = {"model_id": model, "prompt": "a" * 50001 if bad == "prompt" else "test"}
        if bad == "tokens": payload["max_tokens"] = 8193
        path = "model-governance"
    elif bad in ("materials", "total_content"):
        payload = scenario(model)
        count, length = (51, 1) if bad == "materials" else (5, 50000)
        payload["materials"] = [{"id": str(i), "content": "a" * length} for i in range(count)]
        path = "scenarios"
    else:
        payload, path = audit(model), "audits"
        if bad == "steps": payload["steps"] = [{"id": str(i), "label": "s", "detail": "a"} for i in range(101)]
        else: payload["rules"][0]["text"] = "a" * 50001
    assert post(client, path, payload).status_code == 422
    assert not requests and history(client)["total"] == 0
