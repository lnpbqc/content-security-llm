"""Provider seam tests without MySQL or paid API calls."""
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from fastapi import HTTPException
from llm.models import InvocationError
from service.agent_runtime import PrivacyRequest, Turn, complete, detect


@pytest.fixture(autouse=True)
def clean_mysql_tables():
    """These provider-only tests deliberately do not initialize or clear a database."""
    yield


@pytest.fixture(scope="session", autouse=True)
def mysql_test_database():
    yield None


def manager(content="回答", tools=None, usage=True, finish="stop"):
    message = Mock()
    message.model_dump.return_value = {"role": "assistant", "content": content, **({"tool_calls": tools} if tools else {})}
    response = NS(choices=[NS(message=message, finish_reason=finish)], usage=Mock() if usage else None)
    if usage:
        response.usage.model_dump.return_value = {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}
    client = Mock()
    client.with_options.return_value.chat.completions.create.return_value = response
    m = Mock()
    m.get_model.return_value = NS(name="DeepSeek", upstream_model_id="deepseek-chat", base_url="https://example.test/v1", encrypted_api_key="cipher")
    m.cipher.decrypt.return_value = b"not-a-real-key"
    m.client_factory.return_value = client
    return m, client


def test_native_turn_preserves_provider_tools_and_usage_without_retries():
    tools = [{"id": "call-1", "type": "function", "function": {"name": "query_dataset_info", "arguments": "{}"}}]
    m, c = manager(content=None, tools=tools, finish="tool_calls")
    response = complete(m, "m", [{"role": "user", "content": "检查"}], "tokenhash", 10)
    assert response["message"]["tool_calls"] == tools
    assert response["usage"]["total_tokens"] == 25
    c.with_options.assert_called_once_with(timeout=10, max_retries=0)
    params = c.with_options.return_value.chat.completions.create.call_args.kwargs
    assert len(params["tools"]) == 4 and params["max_tokens"] == 800
    c.close.assert_called_once()


@pytest.mark.parametrize("bad", ["bad-json", '{"samples":[]}', '{"samples":[{"id":"1","status":"risk","reason":"风险","quotes":["伪造"]}]}'])
def test_privacy_rejects_missing_or_fabricated_evidence_without_repair(bad):
    m, c = manager(content=bad)
    body = PrivacyRequest(model_id="m", samples=[{"id": "1", "text": "真实文本"}])
    response = detect(m, body, "tokenhash")
    assert response["samples"][0]["status"] == "unknown"
    assert response["validation_error"]
    assert c.with_options.return_value.chat.completions.create.call_count == 1


def test_real_privacy_evidence_is_kept_and_usage_unknown_is_not_zero():
    content = json.dumps({"samples": [{"id": "1", "status": "risk", "reason": "个人联系方式", "quotes": ["a@example.com"]}]})
    m, _ = manager(content=content, usage=False)
    response = detect(m, PrivacyRequest(model_id="m", samples=[{"id": "1", "text": "邮箱a@example.com"}]), "hash")
    assert response["samples"][0]["quotes"] == ["a@example.com"]
    assert response["usage"] is None


def test_limits_and_provider_failures_do_not_retry():
    with pytest.raises(ValidationError):
        PrivacyRequest(model_id="m", samples=[{"id": str(i), "text": "x"} for i in range(11)])
    with pytest.raises(ValidationError):
        Turn(model_id="m", messages=[{}], timeout_seconds=99)
    m, c = manager(finish="length")
    with pytest.raises(InvocationError):
        complete(m, "m", [{"role": "user", "content": "问题"}], "hash", 1)
    assert c.with_options.return_value.chat.completions.create.call_count == 1
    with pytest.raises(HTTPException):
        complete(m, "m", [{"role": "user", "content": "x"*45001}], "hash", 1)


def test_policy_mode_only_offers_queries_and_guard_only_entry_points():
    m, c = manager()
    complete(m, "m", [{"role": "user", "content": "权限测试"}], "hash", 5, mode="policy_test")
    tools = c.with_options.return_value.chat.completions.create.call_args.kwargs["tools"]
    names = {t["function"]["name"] for t in tools}
    assert names == {"query_dataset_info", "query_existing_privacy_checks", "export_report", "send_email"}
    email = next(t["function"] for t in tools if t["function"]["name"] == "send_email")
    assert email["parameters"]["properties"]["to"]["enum"] == ["audit-demo@example.com"]
    assert email["parameters"]["additionalProperties"] is False
    assert "不连接邮件服务" in email["description"]
    assert Turn(model_id="m", messages=[{}]).mode == "privacy"
    with pytest.raises(ValidationError):
        Turn(model_id="m", messages=[{}], mode="anything")
    with pytest.raises(HTTPException):
        complete(m, "m", [{}], "hash", 5, mode="anything")
