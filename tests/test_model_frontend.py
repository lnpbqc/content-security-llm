"""前端直连包裹接口回归；真实 SDK/SQLite，仅模拟上游 HTTP。"""

import json
from copy import deepcopy
from datetime import timedelta

import httpx
import pytest
from openai import OpenAI

from test_compliance_inference import RESULTS, analysis_api, auth
from test_analysis import output, scenario
from utils.time import utc_now


def invoke(client, model, **kwargs):
    return client.post("/api/v1/llm/invoke", headers=kwargs.pop("headers", auth()),
                       json=kwargs.pop("payload", {"model_id": model, "input": "请检查提交材料",
                                                  "response_type": "full_chain_audit"}), **kwargs)


def envelope(response, status=200):
    assert response.status_code == status
    body = response.json()
    assert {"code", "message", "data", "trace_id", "timestamp"} <= body.keys()
    assert body["code"] == (0 if status == 200 else status)
    assert body["message"] and body["trace_id"] and body["timestamp"]
    return body


def test_model_catalog_and_types_are_wrapped_without_changing_old_routes(analysis_api):
    client, model, _, requests = analysis_api
    old_models = client.get("/api/models", headers=auth()).json()
    models = envelope(client.get("/api/v1/models", headers={**auth(), "X-Trace-Id": "front-test"}))
    assert models["data"] == old_models and models["trace_id"] == "front-test"
    detail = envelope(client.get("/api/v1/models/" + model, headers=auth()))["data"]
    assert detail == client.get("/api/models/" + model, headers=auth()).json()
    assert "api_key" not in detail and "encrypted_api_key" not in detail
    types = envelope(client.get("/api/v1/llm/response-types", headers=auth()))["data"]
    assert types == client.get("/api/llm/response-types", headers=auth()).json()
    assert "semantic_risk" in types and not requests
    for method in ("POST", "PATCH", "DELETE"):
        path = "/api/v1/models" if method == "POST" else "/api/v1/models/" + model
        assert client.request(method, path, headers=auth()).status_code == 405


@pytest.mark.parametrize("response_type", RESULTS)
def test_direct_invoke_calls_and_legacy_response_are_compatible(analysis_api, response_type):
    client, model, upstream, requests = analysis_api
    upstream["result"] = deepcopy(RESULTS[response_type])
    payload = {"model_id": model, "input": "请检查提交材料", "response_type": response_type}
    body = envelope(invoke(client, model, payload=payload))
    assert body["data"] == {"model_id": model, "response_type": response_type, "result": RESULTS[response_type]}
    old = client.post("/api/llm/invoke", headers=auth(), json=payload)
    assert old.status_code == 200 and old.json() == body["data"]
    assert len(requests) == 2 and json.loads(requests[0].content)["messages"][-1]["content"] == payload["input"]
    calls = envelope(client.get("/api/v1/calls", headers=auth()))["data"]
    assert calls == client.get("/api/calls", headers=auth()).json() and len(calls) == 2
    assert all(c["status"] == "success" and c["output"] == RESULTS[response_type] for c in calls)
    assert all("token_hash" not in c for c in calls)
    assert client.get("/api/v1/analysis/records", headers=auth()).json()["data"]["total"] == 0


@pytest.mark.parametrize("method,path", [
    ("GET", "/models"), ("GET", "/models/{id}"), ("GET", "/llm/response-types"),
    ("POST", "/llm/invoke"), ("GET", "/calls"),
])
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer invalid"}])
def test_authentication_precedes_new_frontend_endpoints(analysis_api, method, path, headers):
    client, model, _, requests = analysis_api
    response = client.request(method, "/api/v1" + path.format(id=model), headers=headers,
                              json={"model_id": model, "input": "test", "response_type": "full_chain_audit"}
                              if method == "POST" else None)
    body = envelope(response, 401)
    assert body["data"] is None and body["error_code"] == "invalid_token"
    assert response.headers["WWW-Authenticate"] == "Bearer" and not requests
    assert envelope(client.get("/api/v1/calls", headers=auth()))["data"] == []


@pytest.mark.parametrize("state", ["disabled", "expired"])
def test_inactive_token_cannot_read_or_execute_new_endpoints(analysis_api, monkeypatch, state):
    client, model, _, requests = analysis_api
    if state == "disabled":
        assert client.patch("/api/auth/token/status", headers={"X-Setup-Secret": "s" * 32},
                            json={"token": "a" * 64, "enabled": False}).status_code == 200
    else:
        monkeypatch.setattr("service.auth.utc_now", lambda: utc_now() + timedelta(days=2))
    for path in ("/models", "/models/" + model, "/llm/response-types", "/calls"):
        assert envelope(client.get("/api/v1" + path, headers=auth()), 401)["error_code"] == "invalid_token"
    assert envelope(invoke(client, model), 401)["error_code"] == "invalid_token" and not requests


def test_framework_errors_are_wrapped_only_for_v1(analysis_api):
    client, model, _, requests = analysis_api
    assert envelope(client.post("/api/v1/models", headers=auth()), 405)["data"] is None
    assert envelope(client.get("/api/v1/not-a-route", headers=auth()), 404)["data"] is None
    assert client.get("/api/not-a-route", headers=auth()).json() == {"detail": "Not Found"}
    invalid_json = client.post("/api/v1/llm/invoke", headers={**auth(), "Content-Type": "application/json"},
                               content="{invalid")
    assert envelope(invalid_json, 422)["error_code"] == "validation_error" and not requests


@pytest.mark.parametrize("bad_input", ["", "  \n ", "x" * 50001], ids=["empty", "whitespace", "too-long"])
def test_invalid_input_is_rejected_before_sdk_or_records(analysis_api, bad_input):
    client, model, _, requests = analysis_api
    body = envelope(invoke(client, model, payload={"model_id": model, "input": bad_input,
                                                  "response_type": "full_chain_audit"}), 422)
    assert body["data"] is None and body["error_code"] == "validation_error"
    assert not requests and envelope(client.get("/api/v1/calls", headers=auth()))["data"] == []


def test_input_limit_is_inclusive_and_new_calls_are_token_isolated(analysis_api):
    client, model, _, requests = analysis_api
    assert invoke(client, model, payload={"model_id": model, "input": "x" * 50000,
                                         "response_type": "full_chain_audit"}).status_code == 200
    assert len(requests) == 1
    assert envelope(client.get("/api/v1/calls", headers=auth("b" * 64)))["data"] == []
    assert invoke(client, model, headers=auth("b" * 64)).status_code == 200
    a_calls = envelope(client.get("/api/v1/calls", headers=auth()))["data"]
    b_calls = envelope(client.get("/api/v1/calls", headers=auth("b" * 64)))["data"]
    assert len(a_calls) == len(b_calls) == 1 and a_calls[0]["id"] != b_calls[0]["id"]


def test_unknown_model_or_type_preserves_http_errors_without_calls(analysis_api):
    client, model, _, requests = analysis_api
    assert envelope(client.get("/api/v1/models/missing", headers=auth()), 404)["error_code"] == "model_not_found"
    for bad_model, response_type, status, error in [
        ("missing", "full_chain_audit", 404, "model_not_found"),
        (model, "not-registered", 422, "unknown_response_type"),
    ]:
        body = envelope(invoke(client, model, payload={"model_id": bad_model, "input": "test",
                                                       "response_type": response_type}), status)
        assert body["error_code"] == error and body["data"] is None
    assert not requests and envelope(client.get("/api/v1/calls", headers=auth()))["data"] == []


@pytest.mark.parametrize("failure,status,code", [
    ("connection", 502, "provider_unavailable"), ("timeout", 504, "provider_timeout"),
    ("http", 502, "provider_error"), ("invalid", 502, "invalid_structured_output"),
    ("empty", 502, "empty_response"), ("refusal", 502, "model_refusal"),
    ("truncated", 502, "incomplete_response"),
])
def test_upstream_failures_keep_status_code_and_private_details_hidden(analysis_api, failure, status, code):
    client, model, upstream, requests = analysis_api
    if failure in ("connection", "timeout"):
        upstream["transport_error"] = httpx.ConnectError if failure == "connection" else httpx.ReadTimeout
    elif failure == "http": upstream["status"] = 429
    elif failure == "invalid": upstream["content"] = "private invalid model output"
    elif failure == "empty": upstream["empty_choices"] = True
    elif failure == "refusal": upstream["refusal"] = "private refusal details"
    else: upstream["finish_reason"] = "length"
    body = envelope(invoke(client, model), status)
    assert body["error_code"] == code and body["data"] is None
    if failure == "http": assert body["upstream_status"] == 429
    assert "private" not in json.dumps(body) and len(requests) == 1
    calls = envelope(client.get("/api/v1/calls", headers=auth()))["data"]
    assert len(calls) == 1 and calls[0]["status"] == "error" and calls[0]["error_code"] == code
    assert calls[0]["output"] is None
    # 失败也必须释放共享槽位。
    slots = client.app.state.analysis_service.slots
    acquired = [slots.acquire(blocking=False) for _ in range(4)]
    try: assert all(acquired)
    finally:
        for obtained in acquired:
            if obtained: slots.release()


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 201}, {"offset": -1}])
def test_call_pagination_invalid_parameters(analysis_api, params):
    client, _, _, requests = analysis_api
    assert envelope(client.get("/api/v1/calls", headers=auth(), params=params), 422)["data"] is None
    assert not requests


def test_call_pagination_has_no_fake_total_and_queries_do_not_execute_models(analysis_api):
    client, model, _, requests = analysis_api
    for _ in range(3): assert invoke(client, model).status_code == 200
    pages = [envelope(client.get("/api/v1/calls", headers=auth(), params={"limit": 1, "offset": n}))
             for n in range(4)]
    assert all(isinstance(p["data"], list) and "total" not in p for p in pages)
    assert len({p["data"][0]["id"] for p in pages[:3]}) == 3 and pages[3]["data"] == []
    assert len(requests) == 3


def test_generic_invocation_and_analysis_share_four_slots(analysis_api):
    client, model, upstream, requests = analysis_api
    slots = client.app.state.analysis_service.slots
    for _ in range(4): assert slots.acquire(blocking=False)
    try:
        assert envelope(invoke(client, model), 429)["error_code"] == "analysis_busy"
        assert client.post("/api/v1/analysis/scenarios", headers=auth(), json=scenario(model)).status_code == 429
        assert not requests and envelope(client.get("/api/v1/calls", headers=auth()))["data"] == []
    finally:
        for _ in range(4): slots.release()
    assert invoke(client, model).status_code == 200
    upstream["result"] = output()
    assert client.post("/api/v1/analysis/scenarios", headers=auth(), json=scenario(model)).status_code == 200


@pytest.mark.parametrize("failed", [True, False])
def test_new_generic_call_sets_timeout_no_retry_and_closes_http_client(analysis_api, monkeypatch, failed):
    client, model, _, _ = analysis_api
    requests, options, http_clients = [], [], []

    def handler(request):
        requests.append(request)
        if failed:
            return httpx.Response(500, json={"error": {"message": "private failure"}})
        return httpx.Response(200, json={"id": "completion", "object": "chat.completion", "created": 1,
            "model": "test", "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(RESULTS["full_chain_audit"], ensure_ascii=False)}}]})

    class ObservedOpenAI(OpenAI):
        def with_options(self, **kwargs):
            options.append(kwargs)
            return super().with_options(**kwargs)

    def factory(**kwargs):
        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        http_clients.append(http_client)
        return ObservedOpenAI(**kwargs, max_retries=2, http_client=http_client)

    monkeypatch.setattr(client.app.state.model_manager, "client_factory", factory)
    assert invoke(client, model).status_code == (502 if failed else 200)
    assert options == [{"timeout": 120, "max_retries": 0}] and len(requests) == 1
    assert all(c.is_closed for c in http_clients)
