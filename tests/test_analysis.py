"""真实 SDK/HTTP 调用链的分析验收；只模拟上游，不替换本项目服务。"""

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

TEXT = "请提交身份证明；不要点击陌生链接。"
CITE = {"source_id": "m1", "quote": "请提交身份证明"}
CLAIM = {"text": "请提交身份证明", "evidence": [CITE]}
DETAILS = {
    "public_opinion": {"issues": [CLAIM], "sentiment": "neutral", "sentiment_evidence": [CITE],
                       "controversies": [], "communication_advice": ["说明要求"]},
    "hot_events": {"event_summary": "提交材料", "timeline": [{**CLAIM, "source_id": "m1", "time": "2026-10-06"}],
                   "focal_points": [CLAIM], "contradictions": [], "information_gaps": ["缺少后续记录"], "trend": None},
    "cross_cultural": {"cultural_contexts": [CLAIM], "ambiguities": [],
                       "expression_suggestions": ["解释证件含义"], "translation": None},
    "ethnic_governance": {"issues": [CLAIM], "misunderstandings": [], "conflict_expressions": [],
                          "coordination_advice": ["解释要求"]},
    "smart_government": {"answer": "请提交身份证明", "policy_claims": [CLAIM],
                         "processing_points": [CLAIM], "insufficient_materials": False},
    "cyber_security": {"risk_types": [CLAIM], "suspicious_signs": [CLAIM], "handling_advice": ["核对来源"]},
}
SUBJECT = {"id": "model-version-1", "type": "model", "version": "v1"}
FINDING = {"step_id": "s1", "rule_id": "r1", "description": "需核对来源", "evidence": [CITE]}


def auth(token="a" * 64):
    return {"Authorization": "Bearer " + token}


def scenario(model_id, code="public_opinion"):
    return {"model_id": model_id, "scenario_code": code, "content": TEXT, "language": "zh",
            "materials": [{"id": "m1", "content": TEXT, "title": "办事材料", "time": "2026-10-06"}]}


def output(code="public_opinion"):
    return deepcopy({"scenario_code": code, "summary": "提交材料分析", "risk_level": "medium",
                     "evidence": [CITE], "recommendations": ["核对来源"], "limitations": ["仅分析提交材料"],
                     "insufficient_materials": False, "details": {"type": code, **DETAILS[code]}})


def audit(model_id, audit_type="reasoning"):
    return {"model_id": model_id, "audit_type": audit_type, "subject": SUBJECT,
            "steps": [{"id": "s1", "label": "输出校验", "detail": "已保存输出", "evidence_refs": ["m1"], "rule_refs": ["r1"]}],
            "evidence": [{"id": "m1", "content": TEXT}], "rules": [{"id": "r1", "text": "检查来源"}]}


def audit_output(audit_type="reasoning"):
    result = {"audit_type": audit_type, "subject": deepcopy(SUBJECT), "conclusion": "需要人工核验",
              "limitations": ["模型建议"], "verification_state": "unverified"}
    if audit_type == "reasoning":
        result["risk_nodes"] = [deepcopy(FINDING)]
    else:
        result.update(checks=[{"step_id": "s1", "rule_id": "r1", "assessment": "risk", "reason": "需核对",
                               "evidence": [deepcopy(CITE)]}], risks=[deepcopy(FINDING)], gaps=["缺少登记记录"])
    return result


@pytest.fixture
def api(tmp_path):
    upstream = {"queue": [output()]}
    requests = []

    def handle(request):
        requests.append(request)
        value = upstream["queue"].pop(0)
        if isinstance(value, type) and issubclass(value, httpx.TransportError):
            raise value("private provider error", request=request)
        if isinstance(value, int):
            return httpx.Response(value, json={"error": {"message": "private provider error"}})
        content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        choices = [] if upstream.get("empty_choices") else [{"index": 0,
            "finish_reason": upstream.get("finish_reason", "stop"),
            "message": {"role": "assistant", "content": content, "refusal": upstream.get("refusal")}}]
        return httpx.Response(200, json={"id": "test-completion", "object": "chat.completion", "created": 1,
            "model": "upstream-test", "choices": choices})

    def factory(**kwargs):
        return OpenAI(**kwargs, max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(handle)))

    with TestClient(create_app(Settings(tmp_path / "db", "s" * 32, Fernet.generate_key()),
                               client_factory=factory)) as client:
        for token in ("a" * 64, "b" * 64):
            assert client.post("/api/auth/token", headers={"X-Setup-Secret": "s" * 32}, json={
                "token": token, "expires_at": (utc_now() + timedelta(days=1)).isoformat()}).status_code == 201
        response = client.post("/api/models", headers={**auth(), "X-Setup-Secret": "s" * 32}, json={"upstream_model_id": "upstream-test",
            "name": "测试模型", "base_url": "https://model.invalid/v1", "api_key": "provider-test-key"})
        assert response.status_code == 201
        yield client, response.json()["model_id"], upstream, requests


def post(client, path, payload, headers=None):
    return client.post("/api/v1/analysis/" + path, headers=auth() if headers is None else headers, json=payload)


def history(client, **params):
    return client.get("/api/v1/analysis/records", headers=auth(), params=params).json()["data"]


@pytest.mark.parametrize("code", DETAILS)
def test_six_distinct_results_and_exact_saved_call(api, code):
    client, model, upstream, requests = api
    expected = output(code)
    upstream["queue"] = [expected]
    payload = scenario(model, code)
    response = post(client, "scenarios", payload)
    assert response.status_code == 200, response.text
    record = response.json()["data"]
    assert record["result"] == expected and record["input"]["materials"] == payload["materials"]
    assert record["status"] == "succeeded" and record["elapsed_ms"] >= 0 and record["finished_at"]
    assert len(requests) == 1
    sent = json.loads(requests[0].content)
    schema = sent["messages"][0]["content"]
    assert code in schema
    assert all(other not in schema for other in DETAILS if other != code)
    assert payload["content"] in sent["messages"][-1]["content"]
    calls = client.get("/api/calls", headers=auth()).json()
    assert record["calls"] == [{"stage": code, "call_id": calls[0]["id"], "status": "success", "error_code": None}]
    assert client.get("/api/v1/analysis/records/" + record["id"], headers=auth()).json()["data"] == record
    assert "token_hash" not in record
    payload["materials"][0]["content"] = "改变外部输入"
    assert history(client)["items"][0]["input"]["materials"][0]["content"] == TEXT


@pytest.mark.parametrize("code", DETAILS)
def test_each_scene_rejects_another_scene_shape(api, code):
    client, model, upstream, _ = api
    other = "cyber_security" if code != "cyber_security" else "public_opinion"
    upstream["queue"] = [output(other)]
    response = post(client, "scenarios", scenario(model, code))
    assert response.status_code == 502
    assert response.json()["data"]["error_code"] == "invalid_structured_output"


@pytest.mark.parametrize("change", ["unknown_scene", "empty", "duplicate", "reserved", "bad_language", "wrong_target", "missing_model"])
def test_invalid_scene_requests_do_not_call_provider(api, change):
    client, model, _, requests = api
    payload = scenario(model)
    if change == "unknown_scene": payload["scenario_code"] = "unknown"
    if change == "empty": payload["content"] = " \n "
    if change == "duplicate": payload["materials"] *= 2
    if change == "reserved": payload["materials"][0]["id"] = "input"
    if change == "bad_language": payload.update(scenario_code="cross_cultural", target_language="???")
    if change == "wrong_target": payload["target_language"] = "en"
    if change == "missing_model": payload["model_id"] = "missing"
    assert post(client, "scenarios", payload).status_code == (404 if change == "missing_model" else 422)
    assert not requests and history(client)["total"] == 0


def test_optional_material_fields_accept_blank_values(api):
    client, model, _, _ = api
    payload = scenario(model)
    payload["materials"][0].update(title="", time=" \t\n ")
    response = post(client, "scenarios", payload)
    assert response.status_code == 200, response.text
    material = response.json()["data"]["input"]["materials"][0]
    assert material["title"] is None and material["time"] is None


def test_scenario_total_limit_and_validation_error_details(api):
    client, model, _, requests = api
    payload = scenario(model)
    payload["content"] = "x" * 50000
    payload["materials"][0]["content"] = TEXT + "x" * (50000 - len(TEXT))
    assert post(client, "scenarios", payload).status_code == 200
    payload["materials"].append({"id": "m2", "content": "x"})
    response = post(client, "scenarios", payload)
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == 422 and body["data"] is None and body["error_code"] == "validation_error"
    assert body["validation_errors"][0]["loc"] == ["body"]
    assert "100000" in body["validation_errors"][0]["message"]
    payload = scenario(model)
    payload["language"] = "private input must not be echoed"
    response = post(client, "scenarios", payload)
    issue = response.json()["validation_errors"][0]
    assert issue["loc"] == ["body", "language"] and issue["type"] == "string_pattern_mismatch"
    assert set(issue) == {"loc", "type", "message"}
    assert payload["language"] not in response.text
    assert len(requests) == 1 and history(client)["total"] == 1


@pytest.mark.parametrize("where", ["common", "details"])
@pytest.mark.parametrize("bad", ["source", "quote"])
def test_all_quote_locations_are_validated(api, where, bad):
    client, model, upstream, _ = api
    result = output()
    citation = result["evidence"][0] if where == "common" else result["details"]["issues"][0]["evidence"][0]
    citation["source_id" if bad == "source" else "quote"] = "伪造内容"
    upstream["queue"] = [result]
    response = post(client, "scenarios", scenario(model))
    assert response.status_code == 502 and response.json()["data"]["result"] is None


@pytest.mark.parametrize("code", DETAILS)
def test_insufficient_materials_are_explicit_for_every_scene(api, code):
    client, model, upstream, _ = api
    result = output(code)
    result.update(insufficient_materials=True, risk_level="unknown", evidence=[])
    for key, value in result["details"].items():
        if isinstance(value, list): result["details"][key] = []
    if code == "public_opinion": result["details"]["sentiment"] = "unknown"
    if code == "smart_government":
        result["details"].update(insufficient_materials=True, answer="资料不足，无法给出办理要求")
    upstream["queue"] = [result]
    payload = scenario(model, code)
    payload["materials"] = []
    response = post(client, "scenarios", payload)
    assert response.status_code == 200, response.text
    assert response.json()["data"]["result"]["insufficient_materials"]


@pytest.mark.parametrize("change", ["invented_time", "no_time", "trend_without_time"])
def test_hot_event_time_constraints(api, change):
    client, model, upstream, _ = api
    payload, result = scenario(model, "hot_events"), output("hot_events")
    if change == "invented_time": result["details"]["timeline"][0]["time"] = "2099-01-01"
    else: payload["materials"][0].pop("time")
    if change == "trend_without_time":
        result["details"].update(timeline=[], trend=deepcopy(CLAIM))
    upstream["queue"] = [result]
    assert post(client, "scenarios", payload).status_code == 502


@pytest.mark.parametrize("language,expected", [("en", 200), ("fr", 502), (None, 502)])
def test_cross_cultural_target_translation(api, language, expected):
    client, model, upstream, _ = api
    payload = scenario(model, "cross_cultural")
    payload["target_language"] = "en"
    result = output("cross_cultural")
    result["details"]["translation"] = {"target_language": language, "text": "Please provide ID."} if language else None
    upstream["queue"] = [result]
    assert post(client, "scenarios", payload).status_code == expected


@pytest.mark.parametrize("bad", ["missing_material", "question_citation", "uncited_answer", "extra_answer", "extra", "empty_result"])
def test_government_policy_and_invalid_structures(api, bad):
    client, model, upstream, _ = api
    payload, result = scenario(model, "smart_government"), output("smart_government")
    if bad == "missing_material": payload["materials"] = []
    if bad == "question_citation": result["details"]["policy_claims"][0]["evidence"][0]["source_id"] = "input"
    if bad == "uncited_answer": result["details"]["policy_claims"] = []
    if bad == "extra_answer": result["details"]["answer"] += "\n还需要缴费100元"
    if bad == "extra": result["unrequested"] = True
    if bad == "empty_result": result["summary"] = " "
    upstream["queue"] = [result]
    assert post(client, "scenarios", payload).status_code == 502


def governed(original):
    return {"originalOutput": original, "governedOutput": "安全回答", "riskLevel": "medium", "reason": "移除风险", "reconstruction": None}


@pytest.mark.parametrize("with_parameters", [False, True])
def test_actual_generation_then_governance_and_parameters(api, with_parameters):
    client, model, upstream, requests = api
    original = "真实原始回答\n包含引号\"和反斜线\\"
    upstream["queue"] = [original, governed(original)]
    payload = {"model_id": model, "prompt": "生成一段回答"}
    if with_parameters: payload.update(temperature=0.3, max_tokens=333)
    response = post(client, "model-governance", payload)
    assert response.status_code == 200, response.text
    record = response.json()["data"]
    assert record["original_output"] == original and record["result"] == governed(original)
    sent = [json.loads(req.content) for req in requests]
    assert sent[0]["messages"][-1]["content"] == payload["prompt"]
    assert json.dumps({"originalOutput": original}, ensure_ascii=False) in sent[1]["messages"][-1]["content"]
    if with_parameters:
        assert sent[0]["temperature"] == 0.3 and sent[0]["max_tokens"] == 333
    else: assert "temperature" not in sent[0] and "max_tokens" not in sent[0]
    assert "temperature" not in sent[1] and "max_tokens" not in sent[1]
    calls = client.get("/api/calls", headers=auth()).json()
    assert [c["call_id"] for c in record["calls"]] == [calls[1]["id"], calls[0]["id"]]
    assert [c["stage"] for c in record["calls"]] == ["generation", "governance"]


@pytest.mark.parametrize("failure", ["tampered", "invalid", "provider", "timeout", "first_stage"])
def test_generation_failure_preserves_correct_stage_and_partial_answer(api, failure):
    client, model, upstream, requests = api
    value = governed("被篡改") if failure == "tampered" else (
        "invalid json" if failure == "invalid" else httpx.ReadTimeout if failure == "timeout" else 500)
    upstream["queue"] = [500] if failure == "first_stage" else ["真实原始回答", value]
    response = post(client, "model-governance", {"model_id": model, "prompt": "生成回答"})
    assert response.status_code == (504 if failure == "timeout" else 502)
    record = response.json()["data"]
    assert record["status"] == "failed" and record["result"] is None
    assert record["stage"] == ("generation" if failure == "first_stage" else "governance")
    assert record["original_output"] == (None if failure == "first_stage" else "真实原始回答")
    assert len(record["calls"]) == len(requests) == (1 if failure == "first_stage" else 2)
    assert record["calls"][-1]["status"] == "error"
    assert history(client, status="failed")["items"][0] == record


@pytest.mark.parametrize("field,value", [("temperature", -1), ("temperature", 3), ("max_tokens", 0), ("max_tokens", 1.2), ("prompt", " ")])
def test_generation_invalid_parameters_before_http(api, field, value):
    client, model, _, requests = api
    payload = {"model_id": model, "prompt": "test", field: value}
    assert post(client, "model-governance", payload).status_code == 422
    assert not requests


@pytest.mark.parametrize("audit_type", ["reasoning", "full_chain"])
def test_two_audits_have_distinct_outputs_and_observed_steps(api, audit_type):
    client, model, upstream, _ = api
    upstream["queue"] = [audit_output(audit_type)]
    payload = audit(model, audit_type)
    response = post(client, "audits", payload)
    assert response.status_code == 200, response.text
    record = response.json()["data"]
    assert record["result"] == audit_output(audit_type)
    assert record["input"]["steps"] == payload["steps"]
    assert record["result"]["verification_state"] == "unverified"


@pytest.mark.parametrize("audit_type", ["reasoning", "full_chain"])
@pytest.mark.parametrize("bad", ["step", "rule", "source", "quote", "mapping", "version", "verified"])
def test_audit_rejects_fake_references_and_business_verification(api, audit_type, bad):
    client, model, upstream, _ = api
    result, payload = audit_output(audit_type), audit(model, audit_type)
    finding = result["risk_nodes"][0] if audit_type == "reasoning" else result["risks"][0]
    if bad == "step": finding["step_id"] = "fake"
    if bad == "rule": finding["rule_id"] = "fake"
    if bad == "source": finding["evidence"][0]["source_id"] = "fake"
    if bad == "quote": finding["evidence"][0]["quote"] = "fake quote"
    if bad == "mapping": payload["steps"][0]["evidence_refs"] = []
    if bad == "version": result["subject"]["version"] = "v2"
    if bad == "verified": result["verification_state"] = "verified"
    upstream["queue"] = [result]
    assert post(client, "audits", payload).status_code == 502


@pytest.mark.parametrize("audit_type", ["reasoning", "full_chain"])
def test_missing_audit_materials_cannot_be_verified(api, audit_type):
    client, model, upstream, _ = api
    payload, result = audit(model, audit_type), audit_output(audit_type)
    payload.update(evidence=[], rules=[], steps=[])
    if audit_type == "reasoning": result["risk_nodes"] = []
    else: result.update(checks=[], risks=[], gaps=["缺少步骤、规则、证据"])
    result["conclusion"] = "资料不足，无法核验"
    upstream["queue"] = [result]
    response = post(client, "audits", payload)
    assert response.status_code == 200 and response.json()["data"]["result"]["verification_state"] == "unverified"


@pytest.mark.parametrize("bad", ["duplicate_steps", "duplicate_evidence", "duplicate_rules", "input_evidence_ref", "input_rule_ref", "blank_version"])
def test_invalid_audit_inputs_before_provider(api, bad):
    client, model, _, requests = api
    payload = deepcopy(audit(model))
    if bad.startswith("duplicate_"): payload[bad.split("_", 1)[1]] *= 2
    if bad == "input_evidence_ref": payload["steps"][0]["evidence_refs"] = ["fake"]
    if bad == "input_rule_ref": payload["steps"][0]["rule_refs"] = ["fake"]
    if bad == "blank_version": payload["subject"]["version"] = " "
    assert post(client, "audits", payload).status_code == 422
    assert not requests


@pytest.mark.parametrize("headers", [{}, auth("invalid"), {"X-Setup-Secret": "s" * 32}])
def test_authentication_precedes_every_analysis_endpoint(api, headers):
    client, model, _, requests = api
    for path, payload in [("scenarios", scenario(model)), ("model-governance", {"model_id": model, "prompt": "p"}),
                          ("audits", audit(model))]:
        assert post(client, path, payload, headers).status_code == 401
    for path in ("options", "records", "records/missing"):
        assert client.get("/api/v1/analysis/" + path, headers=headers).status_code == 401
    assert not requests and history(client)["total"] == 0


def test_options_history_pagination_filters_isolation_and_precise_call_association(api):
    client, model, upstream, _ = api
    options = client.get("/api/v1/analysis/options", headers=auth()).json()["data"]
    assert {item["code"] for item in options["scenarios"]} == set(DETAILS)
    assert history(client)["total"] == 0
    first = post(client, "scenarios", scenario(model)).json()["data"]
    upstream["queue"] = [output("cyber_security"), "bad", audit_output()]
    other = post(client, "scenarios", scenario(model, "cyber_security"), auth("b" * 64)).json()["data"]
    failed = post(client, "scenarios", scenario(model)).json()["data"]
    audited = post(client, "audits", audit(model)).json()["data"]
    assert history(client)["total"] == 3
    assert history(client, kind="audit")["items"] == [audited]
    assert history(client, scenario_code="public_opinion", status="failed")["items"] == [failed]
    all_rows = history(client)["items"]
    for page in range(1, 4):
        assert history(client, page=page, page_size=1)["items"] == [all_rows[page - 1]]
    assert history(client, page=4, page_size=1)["items"] == []
    assert client.get("/api/v1/analysis/records/" + first["id"], headers=auth("b" * 64)).status_code == 404
    assert client.get("/api/v1/analysis/records/" + other["id"], headers=auth()).status_code == 404
    own_calls = {c["id"] for c in client.get("/api/calls", headers=auth()).json()}
    assert {c["call_id"] for row in all_rows for c in row["calls"]} == own_calls
    assert {c["call_id"] for c in other["calls"]}.isdisjoint(own_calls)
    for params in ({"page": 0}, {"page_size": 101}, {"kind": "fake"}, {"status": "fake"}):
        assert client.get("/api/v1/analysis/records", headers=auth(), params=params).status_code == 422


def test_hot_events_without_time_and_with_two_timed_materials(api):
    client, model, upstream, _ = api
    payload, result = scenario(model, "hot_events"), output("hot_events")
    payload["materials"][0].pop("time")
    result["details"]["timeline"] = []
    upstream["queue"] = [deepcopy(result)]
    assert post(client, "scenarios", payload).status_code == 200
    payload["materials"][0]["time"] = "2026-10-06"
    payload["materials"].append({"id": "m2", "content": "更新了材料", "time": "2026-10-07"})
    result["details"]["trend"] = {"text": "出现材料更新", "evidence": [CITE, {"source_id": "m2", "quote": "更新了材料"}]}
    upstream["queue"] = [result]
    assert post(client, "scenarios", payload).status_code == 200


@pytest.mark.parametrize("invalid", ["limits", "level", "sentiment"])
def test_insufficiency_and_sentiment_require_valid_basis(api, invalid):
    client, model, upstream, _ = api
    result = output()
    if invalid == "sentiment": result["details"]["sentiment_evidence"] = []
    else:
        result["insufficient_materials"] = True
        if invalid == "limits": result.update(risk_level="unknown", limitations=[])
    upstream["queue"] = [result]
    assert post(client, "scenarios", scenario(model)).status_code == 502


def test_blank_generation_stops_before_governance(api):
    client, model, upstream, requests = api
    upstream["queue"] = [" \n "]
    response = post(client, "model-governance", {"model_id": model, "prompt": "test"})
    assert response.status_code == 502
    assert response.json()["data"]["error_code"] == "empty_response" and len(requests) == 1


@pytest.mark.parametrize("invalid", ["empty_evidence", "rule_mapping", "unexplained_missing"])
def test_chain_check_cannot_claim_verification_without_mapped_evidence(api, invalid):
    client, model, upstream, _ = api
    payload, result = audit(model, "full_chain"), audit_output("full_chain")
    if invalid == "empty_evidence": result["checks"][0]["evidence"] = []
    if invalid == "rule_mapping": payload["steps"][0]["rule_refs"] = []
    if invalid == "unexplained_missing":
        payload.update(evidence=[], steps=[], rules=[])
        result.update(checks=[], risks=[], limitations=[])
    upstream["queue"] = [result]
    assert post(client, "audits", payload).status_code == 502


def test_unverifiable_chain_check_is_supported(api):
    client, model, upstream, _ = api
    payload, result = audit(model, "full_chain"), audit_output("full_chain")
    payload["evidence"] = []
    payload["steps"][0]["evidence_refs"] = []
    result.update(risks=[])
    result["checks"][0].update(assessment="unverifiable", evidence=[], reason="未提供证据")
    upstream["queue"] = [result]
    assert post(client, "audits", payload).status_code == 200


@pytest.mark.parametrize("state", ["disabled", "expired"])
def test_inactive_token_cannot_start_or_query_analysis(api, monkeypatch, state):
    client, model, _, requests = api
    if state == "disabled":
        assert client.patch("/api/auth/token/status", headers={"X-Setup-Secret": "s" * 32},
                            json={"token": "a" * 64, "enabled": False}).status_code == 200
    else:
        monkeypatch.setattr("service.auth.utc_now", lambda: utc_now() + timedelta(days=2))
    assert post(client, "scenarios", scenario(model)).status_code == 401
    assert client.get("/api/v1/analysis/records", headers=auth()).status_code == 401
    assert not requests


def test_saved_success_and_failure_survive_app_restart(api):
    client, model, upstream, requests = api
    original = post(client, "scenarios", scenario(model)).json()["data"]
    upstream["queue"] = ["invalid json"]
    failure = post(client, "scenarios", scenario(model)).json()["data"]
    settings = Settings(client.app.state.database.data_dir, "s" * 32, Fernet.generate_key())
    with TestClient(create_app(settings)) as restarted:
        for saved in (original, failure):
            found = restarted.get("/api/v1/analysis/records/" + saved["id"], headers=auth())
            assert found.status_code == 200 and found.json()["data"] == saved
    assert len(requests) == 2


def test_running_history_is_complete_and_call_link_is_written_during_execution(api, monkeypatch):
    client, model, _, _ = api
    service = client.app.state.analysis_service
    original_invoke = service.manager.invoke
    observed = []

    def inspect(*args, **kwargs):
        running = history(client, status="running")["items"][0]
        observed.append(running)
        assert running["result"] is None and running["original_output"] is None
        assert "finished_at" in running and "calls" in running and running["stage"] == "public_opinion"
        return original_invoke(*args, **kwargs)

    monkeypatch.setattr(service.manager, "invoke", inspect)
    response = post(client, "scenarios", scenario(model))
    assert response.status_code == 200 and len(observed) == 1


def test_hot_events_rejects_single_material_contradiction(api):
    client, model, upstream, _ = api
    result = output("hot_events")
    result["details"]["contradictions"] = [deepcopy(CLAIM)]
    upstream["queue"] = [result]
    assert post(client, "scenarios", scenario(model, "hot_events")).status_code == 502


@pytest.mark.parametrize("change,code", [({"refusal": "blocked"}, "model_refusal"),
    ({"finish_reason": "length"}, "incomplete_response"),
    ({"finish_reason": "content_filter"}, "model_refusal"), ({"empty_choices": True}, "empty_response")])
def test_raw_generation_refusal_or_truncation_does_not_reach_governance(api, change, code):
    client, model, upstream, requests = api
    upstream.update(change)
    upstream["queue"] = ["partial answer", governed("partial answer")]
    response = post(client, "model-governance", {"model_id": model, "prompt": "p"})
    record = response.json()["data"]
    assert response.status_code == 502 and record["error_code"] == code
    assert record["original_output"] is None and record["result"] is None
    assert len(requests) == 1 and record["calls"][0]["error_code"] == code


def test_credential_failure_keeps_precise_failed_call_without_http(api, monkeypatch):
    client, model, _, requests = api
    def fail(_):
        raise ValueError("private secret failure")
    monkeypatch.setattr(client.app.state.model_manager.cipher, "decrypt", fail)
    response = post(client, "scenarios", scenario(model))
    record = response.json()["data"]
    assert response.status_code == 502 and record["error_code"] == "credential_unavailable"
    assert not requests and "private secret" not in response.text
    assert record["calls"][0]["call_id"] == client.get("/api/calls", headers=auth()).json()[0]["id"]


def test_hot_events_reports_body_date_timeline_violation_without_exposing_output(api):
    client, model, upstream, requests = api
    payload = scenario(model, "hot_events")
    payload["content"] = "2026-10-06：发布活动通知。"
    payload["materials"][0].pop("time")
    result = output("hot_events")
    result["details"]["timeline"] = [{"source_id": "input", "time": "2026-10-06",
        "text": "发布通知", "evidence": [{"source_id": "input", "quote": "发布活动通知"}]}]
    result["summary"] = "private provider output"
    upstream["queue"] = [result]
    response = post(client, "scenarios", payload)
    record = response.json()["data"]
    assert response.status_code == 502 and record["error_code"] == "invalid_structured_output"
    assert "时间线必须对应材料时间及原文" in record["error_message"]
    assert "private provider output" not in response.text
    assert record["result"] is None and len(requests) == 1
    assert history(client)["items"][0]["error_message"] == record["error_message"]


@pytest.mark.parametrize("failure,expected", [("json", "JSON"), ("missing", "必填字段"),
                                              ("extra", "未定义字段"), ("type", "类型或取值")])
def test_analysis_schema_failures_are_specific_without_returning_provider_content(api, failure, expected):
    client, model, upstream, requests = api
    result = output("hot_events")
    if failure == "json": result = "private invalid provider content"
    elif failure == "missing": result.pop("summary")
    elif failure == "extra": result["private-provider-field"] = "private provider content"
    else: result["details"]["timeline"] = "private provider content"
    upstream["queue"] = [result]
    response = post(client, "scenarios", scenario(model, "hot_events"))
    record = response.json()["data"]
    assert response.status_code == 502 and record["error_code"] == "invalid_structured_output"
    assert expected in record["error_message"]
    assert "private" not in response.text and len(requests) == 1
    assert record["result"] is None


@pytest.mark.parametrize("timed", [False, True])
def test_hot_event_prompt_explicitly_limits_time_sources_for_each_input(api, timed):
    client, model, upstream, requests = api
    payload = scenario(model, "hot_events")
    result = output("hot_events")
    if not timed:
        payload["materials"][0].pop("time")
        payload["content"] += "\n2026-10-06：这里的日期不是材料 time 字段。"
        result["details"]["timeline"] = []
    upstream["queue"] = [result]
    response = post(client, "scenarios", payload)
    assert response.status_code == 200
    prompt = json.loads(requests[0].content)["messages"][-1]["content"]
    expected = [{"source_id": "m1", "time": "2026-10-06"}] if timed else []
    assert "时间线允许的来源与时间：" + json.dumps(expected, ensure_ascii=False) in prompt
    assert "input 正文中的日期不属于时间线来源" in prompt
    assert "没有允许项时 timeline 必须为 []" in prompt
    assert "contradictions 必须为 []" in prompt


def test_arbitrary_validator_exception_does_not_leak_private_message(api, monkeypatch):
    client, model, _, _ = api

    def fail(*_):
        raise ValueError("private provider output or credentials")

    monkeypatch.setattr("service.analysis.validate_scenario", fail)
    response = post(client, "scenarios", scenario(model))
    assert response.status_code == 502
    assert response.json()["data"]["error_message"] == "Model returned invalid structured output"
    assert "private" not in response.text
