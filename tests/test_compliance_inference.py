"""业务服务端结构化分析契约；仅模拟上游 HTTP，不替换本项目服务。"""

import json
from copy import deepcopy
from datetime import timedelta

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from openai import OpenAI

from config import Settings
from main import create_app
from utils.time import utc_now


RESULTS = {
    "full_chain_audit": {
        "conclusion": "训练产物登记证据缺失",
        "checks": [{"stage": "训练产物登记", "passed": False, "reason": "未提供登记记录"}],
        "gaps": [{"description": "缺少模型版本登记证据", "reason": "无法核对来源版本"}],
    },
    "reasoning_audit": {
        "steps": [{"label": "输出校验", "detail": "检查已保存输出",
                   "occurredAt": "2026-10-06T10:00:00+08:00", "verificationState": "unknown"}],
        "riskNodes": [{"stepId": "observed-step-1", "ruleRef": "RULE-1",
                       "description": "需要核对来源", "evidenceRefs": ["evidence-1"]}],
        "auditResult": "需要业务方核实引用",
    },
    "semantic_risk": {
        "findings": [{"category": "个人信息暴露", "suggestedLevel": "MEDIUM",
                      "reason": "示例账户标识可关联", "ruleId": "PII-03",
                      "evidenceRefs": ["evidence-1"]}],
        "evidence": [{"quote": "虚构账户标识", "feature": "可关联的账户信息"}],
        "primaryCategory": "个人信息暴露",
        "maximumSuggestedLevel": "MEDIUM",
    },
}


@pytest.fixture
def analysis_api(tmp_path):
    """实际 OpenAI SDK 经 MockTransport 返回，不访问业务库或模型网络。"""
    requests = []
    upstream = {"result": deepcopy(RESULTS["full_chain_audit"])}

    def handle(request):
        requests.append(request)
        if "transport_error" in upstream:
            raise upstream["transport_error"]("private upstream detail", request=request)
        if "status" in upstream:
            return httpx.Response(upstream["status"], json={"error": {
                "message": "private upstream detail", "type": "provider_error",
            }})
        content = upstream.get("content", json.dumps(upstream["result"], ensure_ascii=False))
        choices = [] if upstream.get("empty_choices") else [{
            "index": 0, "finish_reason": upstream.get("finish_reason", "stop"),
            "message": {"role": "assistant", "content": content,
                        "refusal": upstream.get("refusal")},
        }]
        return httpx.Response(200, json={
            "id": "completion-test", "object": "chat.completion", "created": 1,
            "model": "upstream-audit-model", "choices": choices,
        })

    def client_factory(**kwargs):
        return OpenAI(**kwargs, max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(handle)))

    settings = Settings(tmp_path / "db", "s" * 32, Fernet.generate_key())
    with TestClient(create_app(settings, client_factory=client_factory)) as client:
        for token in ("a" * 64, "b" * 64):
            response = client.post("/api/auth/token", headers={"X-Setup-Secret": "s" * 32},
                                   json={"token": token, "expires_at": (utc_now() + timedelta(days=1)).isoformat(),
                                         "label": "业务接入测试"})
            assert response.status_code == 201, response.text
        response = client.post("/api/models", headers={**auth(), "X-Setup-Secret": "s" * 32}, json={
            "upstream_model_id": "upstream-audit-model", "name": "审计测试模型",
            "base_url": "https://model.invalid/v1", "api_key": "provider-key-for-test",
        })
        assert response.status_code == 201, response.text
        yield client, response.json()["model_id"], upstream, requests


def auth(token="a" * 64):
    return {"Authorization": "Bearer " + token}


def invoke(client, model_id, response_type="full_chain_audit", **kwargs):
    return client.post("/api/llm/invoke", headers=kwargs.pop("headers", auth()), json={
        "model_id": model_id, "response_type": response_type,
        "input": "请分析以下已提供证据，不补造记录。证据编号：evidence-1。",
    }, **kwargs)


@pytest.mark.parametrize("response_type", RESULTS)
def test_structured_analysis_http_and_saved_result(analysis_api, response_type):
    """三类结果经真实 SDK、结构校验、接口序列化和记录查询保持一致。"""
    client, model_id, upstream, requests = analysis_api
    upstream["result"] = deepcopy(RESULTS[response_type])
    types = client.get("/api/llm/response-types", headers=auth())
    assert types.status_code == 200 and response_type in types.json()
    response = invoke(client, model_id, response_type)
    assert response.status_code == 200, response.text
    assert response.json() == {"model_id": model_id, "response_type": response_type,
                               "result": RESULTS[response_type]}
    assert len(requests) == 1
    sent = json.loads(requests[0].content)
    assert sent["model"] == "upstream-audit-model"
    assert sent["messages"][-1]["content"] == "请分析以下已提供证据，不补造记录。证据编号：evidence-1。"
    assert "JSON Schema" in sent["messages"][0]["content"]
    assert requests[0].headers["authorization"] == "Bearer provider-key-for-test"
    assert "x-setup-secret" not in requests[0].headers
    records = client.get("/api/calls", headers=auth()).json()
    assert len(records) == 1
    assert records[0]["output"] == RESULTS[response_type]
    assert records[0]["response_type"] == response_type
    assert records[0]["status"] == "success" and records[0]["error_code"] is None
    assert "token_hash" not in records[0]


@pytest.mark.parametrize("response_type", RESULTS)
@pytest.mark.parametrize("invalid", ["not-json", "missing", "extra"])
def test_invalid_results_never_become_success_records(analysis_api, response_type, invalid):
    """非法 JSON、缺字段和未约定字段统一失败，不返回或保存成功结果。"""
    client, model_id, upstream, requests = analysis_api
    output = deepcopy(RESULTS[response_type])
    if invalid == "not-json":
        upstream["content"] = "private invalid model output"
    elif invalid == "missing":
        output.pop(next(iter(output)))
    else:
        output["unrequested_field"] = "private invalid model output"
    upstream["result"] = output
    response = invoke(client, model_id, response_type)
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "invalid_structured_output"
    assert "private invalid model output" not in response.text
    assert len(requests) == 1
    records = client.get("/api/calls", headers=auth()).json()
    assert len(records) == 1
    assert records[0]["status"] == "error" and records[0]["output"] is None
    assert records[0]["error_code"] == "invalid_structured_output"
    assert records[0]["response_type"] == response_type


@pytest.mark.parametrize("change,code", [
    ({"refusal": "private refusal"}, "model_refusal"),
    ({"finish_reason": "content_filter"}, "model_refusal"),
    ({"finish_reason": "length"}, "incomplete_response"),
    ({"content": None}, "empty_response"),
    ({"content": ""}, "empty_response"),
    ({"empty_choices": True}, "empty_response"),
])
def test_refused_incomplete_and_empty_analysis(analysis_api, change, code):
    """即使同时返回合法 JSON，拒答或截断也不能成为成功分析。"""
    client, model_id, upstream, _ = analysis_api
    upstream.update(change)
    response = invoke(client, model_id)
    assert response.status_code == 502 and response.json()["detail"]["code"] == code
    assert "private refusal" not in response.text
    record = client.get("/api/calls", headers=auth()).json()[0]
    assert record["status"] == "error" and record["output"] is None
    assert record["error_code"] == code


@pytest.mark.parametrize("status", [401, 402, 429, 500])
def test_provider_http_failure_is_not_local_token_failure(analysis_api, status):
    """上游错误保留状态码用于诊断，但不透传上游正文。"""
    client, model_id, upstream, _ = analysis_api
    upstream["status"] = status
    response = invoke(client, model_id)
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "provider_error"
    assert response.json()["detail"]["upstream_status"] == status
    assert "private upstream detail" not in response.text
    record = client.get("/api/calls", headers=auth()).json()[0]
    assert record["status"] == "error" and record["output"] is None
    assert record["error_code"] == "provider_error"


@pytest.mark.parametrize("error,status,code", [
    (httpx.ConnectError, 502, "provider_unavailable"),
    (httpx.ReadTimeout, 504, "provider_timeout"),
])
def test_provider_connection_and_timeout(analysis_api, error, status, code):
    client, model_id, upstream, _ = analysis_api
    upstream["transport_error"] = error
    response = invoke(client, model_id)
    assert response.status_code == status and response.json()["detail"]["code"] == code
    assert "private upstream detail" not in response.text
    record = client.get("/api/calls", headers=auth()).json()[0]
    assert record["error_code"] == code and record["output"] is None


@pytest.mark.parametrize("headers", [{}, auth("not-a-model-token"), {"X-Setup-Secret": "s" * 32}])
def test_access_rejected_before_model_request(analysis_api, headers):
    client, model_id, _, requests = analysis_api
    for path, method in [("/api/llm/response-types", "GET"), ("/api/llm/invoke", "POST")]:
        response = (client.get(path, headers=headers) if method == "GET"
                    else invoke(client, model_id, headers=headers))
        assert response.status_code == 401 and response.json()["detail"]["code"] == "invalid_token"
        assert response.headers["www-authenticate"] == "Bearer"
    assert requests == [] and client.get("/api/calls", headers=auth()).json() == []


@pytest.mark.parametrize("state", ["disabled", "expired"])
def test_inactive_token_stops_analysis(analysis_api, monkeypatch, state):
    client, model_id, _, requests = analysis_api
    if state == "disabled":
        changed = client.patch("/api/auth/token/status", headers={"X-Setup-Secret": "s" * 32},
                               json={"token": "a" * 64, "enabled": False})
        assert changed.status_code == 200
    else:
        monkeypatch.setattr("service.auth.utc_now", lambda: utc_now() + timedelta(days=2))
    response = invoke(client, model_id)
    assert response.status_code == 401 and response.json()["detail"]["code"] == "invalid_token"
    assert requests == []


@pytest.mark.parametrize("kind", ["unknown_type", "missing_model"])
def test_invalid_target_does_not_call_provider_or_create_record(analysis_api, kind):
    client, model_id, _, requests = analysis_api
    response = invoke(client, "missing-model" if kind == "missing_model" else model_id,
                      "unknown-type" if kind == "unknown_type" else "full_chain_audit")
    assert response.status_code == (404 if kind == "missing_model" else 422)
    assert response.json()["detail"]["code"] == (
        "model_not_found" if kind == "missing_model" else "unknown_response_type")
    assert requests == [] and client.get("/api/calls", headers=auth()).json() == []


@pytest.mark.parametrize("field", ["model_id", "input", "response_type"])
def test_empty_request_field_is_rejected_before_analysis(analysis_api, field):
    client, model_id, _, requests = analysis_api
    payload = {"model_id": model_id, "input": "证据", "response_type": "full_chain_audit"}
    payload[field] = ""
    response = client.post("/api/llm/invoke", headers=auth(), json=payload)
    assert response.status_code == 422 and isinstance(response.json()["detail"], list)
    assert requests == [] and client.get("/api/calls", headers=auth()).json() == []


def test_call_history_isolation_and_pagination_after_success_and_failure(analysis_api):
    """第二枚令牌不能查询第一枚的结果，分页只作用于本令牌的记录。"""
    client, model_id, upstream, _ = analysis_api
    assert invoke(client, model_id).status_code == 200
    own_first = client.get("/api/calls", headers=auth()).json()
    assert client.get("/api/calls", headers=auth("b" * 64)).json() == []
    assert invoke(client, model_id, headers=auth("b" * 64)).status_code == 200
    upstream["content"] = "invalid output"
    assert invoke(client, model_id).status_code == 502
    own = client.get("/api/calls", headers=auth()).json()
    other = client.get("/api/calls", headers=auth("b" * 64)).json()
    assert len(own) == 2 and len(other) == 1
    assert {row["id"] for row in own}.isdisjoint({row["id"] for row in other})
    assert own[1] == own_first[0] and own[0]["status"] == "error"
    for offset, expected in enumerate(own):
        page = client.get("/api/calls", headers=auth(), params={"limit": 1, "offset": offset})
        assert page.json() == [expected]
    assert client.get("/api/calls", headers=auth(), params={"limit": 1, "offset": 2}).json() == []
