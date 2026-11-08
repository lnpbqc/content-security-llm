"""经真实 SDK 检查生成约束，不访问 MySQL 或真实模型。"""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cryptography.fernet import Fernet
import httpx
from openai import OpenAI

from llm.analysis_contracts import ScenarioOutput
from llm.governance_outputs import MultilingualSampleOutput
from llm.models import InvocationError, ModelManager
from llm.outputs import TextAnswer
from llm.structured_output import StructuredOutputValidationError
from service.inference import InferenceService


class StructuredOutputTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.content = '{"answer":"checked"}'
        self.status = 200
        self.accepted_format = None
        self.reasoning_when_thinking = False
        self.reasoning_content = None
        cipher = Fernet(Fernet.generate_key())
        model_patch = patch("llm.models.ModelRepository")
        calls_patch = patch("llm.models.CallRepository")
        models = model_patch.start().return_value
        self.addCleanup(model_patch.stop)
        self.calls = calls_patch.start().return_value
        self.addCleanup(calls_patch.stop)
        models.get.return_value = SimpleNamespace(
            upstream_model_id="upstream-test", base_url="https://model.invalid/v1",
            encrypted_api_key=cipher.encrypt(b"provider-test-key").decode("ascii"),
        )

        def handle(request):
            self.requests.append(json.loads(request.content))
            if (self.accepted_format is not None
                    and self.requests[-1].get("response_format", {}).get("type") != self.accepted_format):
                return httpx.Response(400, json={"error": {
                    "message": "Unsupported response_format.type", "type": "invalid_request_error",
                }})
            status = self.status.pop(0) if isinstance(self.status, list) else self.status
            if status != 200:
                return httpx.Response(status, json={"error": {
                    "message": "private upstream detail", "type": "invalid_request_error",
                }})
            content, reasoning = self.content, self.reasoning_content
            if isinstance(content, list):
                content = content.pop(0)
            if (self.reasoning_when_thinking and self.requests[-1].get("response_format")
                    and self.requests[-1].get("chat_template_kwargs", {}).get("enable_thinking") is not False):
                content, reasoning = "", self.content
            return httpx.Response(200, json={
                "id": "completion-test", "object": "chat.completion", "created": 1,
                "model": "upstream-test", "choices": [{
                    "index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content, "reasoning_content": reasoning},
                }],
            })

        def client_factory(**kwargs):
            client = OpenAI(**kwargs, max_retries=0,
                            http_client=httpx.Client(transport=httpx.MockTransport(handle)))
            self.addCleanup(client.close)
            return client

        self.manager = ModelManager(None, b"", client_factory=client_factory, cipher=cipher)

    def invoke(self, response_model=TextAnswer, **kwargs):
        return self.manager.invoke("model-test", kwargs.pop("input", "check input"), response_model,
                                   token_hash="token-test", **kwargs)

    def test_schema_reaches_provider_and_valid_result_is_saved(self):
        result = self.invoke(temperature=0.1, max_tokens=512)
        sent = self.requests[0]
        self.assertEqual(sent["response_format"], {
            "type": "json_schema", "json_schema": {
                "name": "TextAnswer", "schema": TextAnswer.model_json_schema(),
            },
        })
        self.assertEqual(sent["model"], "upstream-test")
        self.assertEqual(sent["messages"][-1]["content"], "check input")
        self.assertIn("JSON Schema", sent["messages"][0]["content"])
        self.assertEqual((sent["temperature"], sent["max_tokens"]), (0.1, 512))
        self.assertEqual(result.answer, "checked")
        saved = self.calls.create.call_args.kwargs
        self.assertEqual(saved["status"], "success")
        self.assertEqual(json.loads(saved["output_json"]), result.model_dump())

    def test_nested_schemas_reach_provider_without_losing_definitions(self):
        self.content = "{}"
        for response_model in (ScenarioOutput, MultilingualSampleOutput):
            result = self.invoke(response_model, input=json.dumps({"scenario_code": "public_opinion", "sampleId": "s1", "sampleRevisionId": "rev1"}))
            self.assertEqual(result.assessment_status, "unavailable")
            schema = self.requests[-1]["response_format"]["json_schema"]["schema"]
            self.assertEqual(schema, response_model.model_json_schema())
            self.assertIn("$defs", schema)

    def test_deepseek_official_chat_uses_supported_json_object_format(self):
        self.accepted_format = "json_object"
        self.manager.models.get.return_value.upstream_model_id = "deepseek-flash"
        for base_url in ("https://api.deepseek.com/", "https://api.deepseek.com/v1"):
            with self.subTest(base_url=base_url):
                self.manager.models.get.return_value.base_url = base_url
                count = len(self.requests)
                result = self.invoke()
                self.assertEqual(result.answer, "checked")
                self.assertEqual(len(self.requests), count + 1)
                sent = self.requests[-1]
                self.assertEqual(sent["response_format"], {"type": "json_object"})
                self.assertEqual(sent["model"], "deepseek-flash")
                schema = sent["messages"][0]["content"].split("JSON Schema:\n", 1)[1]
                self.assertEqual(json.loads(schema), TextAnswer.model_json_schema())

    def test_deepseek_name_or_url_substring_does_not_change_other_providers(self):
        self.manager.models.get.return_value.upstream_model_id = "deepseek-flash"
        for base_url in ("https://model.invalid/v1", "https://api.deepseek.com.proxy.invalid/v1",
                         "https://proxy.invalid/api.deepseek.com"):
            with self.subTest(base_url=base_url):
                self.manager.models.get.return_value.base_url = base_url
                self.invoke()
                self.assertEqual(self.requests[-1]["response_format"]["type"], "json_schema")

    def test_deepseek_json_mode_still_enforces_schema_and_business_rules(self):
        self.manager.models.get.return_value.base_url = "https://api.deepseek.com/"
        self.accepted_format = "json_object"
        for content in ("{}", '{"answer":"checked","extra":true}'):
            self.content = content
            before = len(self.requests)
            result = self.invoke()
            self.assertEqual(result.assessment_status, "unavailable")
            self.assertEqual(len(self.requests) - before, 2)
        self.content = '{"answer":"checked"}'
        def validate(result):
            raise StructuredOutputValidationError("引用来源或原文不匹配")
        result = self.invoke(validator=validate)
        self.assertEqual(result.assessment_status, "unavailable")
        self.assertNotEqual(result.answer, "checked")

    def test_deepseek_text_generation_does_not_request_json(self):
        self.manager.models.get.return_value.base_url = "https://api.deepseek.com/"
        self.content = "plain answer"
        result = self.manager.invoke_text("model-test", "check input", token_hash="token-test")
        self.assertEqual(result, "plain answer")
        self.assertNotIn("response_format", self.requests[-1])

    def test_deepseek_anomaly_repair_matches_frontend_invocation(self):
        self.manager.models.get.return_value.base_url = "https://api.deepseek.com/"
        self.manager.models.get.return_value.upstream_model_id = "deepseek-flash"
        self.accepted_format = "json_object"
        expected = {"findings": [], "primaryType": "无异常", "source": {"filename": "input", "line": 1},
                    "fieldChanges": [], "reason": "未发现异常", "validationResults": []}
        self.content = json.dumps(expected, ensure_ascii=False)
        service = InferenceService(self.manager)
        result = service.invoke("model-test", "1111", "anomaly_repair", "token-test")
        self.assertEqual({k: result.result[k] for k in expected}, expected)
        self.assertEqual(result.response_type, "anomaly_repair")
        self.assertEqual(self.requests[-1]["messages"][-1]["content"], "1111")
        self.assertEqual(self.requests[-1]["response_format"], {"type": "json_object"})
        self.content = "{}"
        fallback = service.invoke("model-test", "1111", "anomaly_repair", "token-test")
        self.assertEqual(fallback.result["assessment_status"], "unavailable")
        self.assertEqual(fallback.result["fieldChanges"], [])

    def test_model_governance_redacts_before_returning_and_saving_success(self):
        service = InferenceService(self.manager)
        original = "顾客张小明，手机号码：13800138000，刚刚下单了一份黄焖鸡米饭。"
        expected = "顾客[姓名已脱敏]，手机号码：[手机号已脱敏]，刚刚下单了一份黄焖鸡米饭。"
        for risk in ("high", "low"):
            with self.subTest(risk=risk):
                self.content = json.dumps({
                    "originalOutput": original, "governedOutput": original,
                    "reason": "订单涉及张小明的个人信息。", "riskLevel": risk,
                    "reconstruction": None,
                }, ensure_ascii=False)
                result = service.invoke("model-test", original, "model_risk_governance", "token-test")
                self.assertEqual(result.result["originalOutput"], original)
                self.assertEqual(result.result["governedOutput"], expected)
                self.assertEqual(result.result["riskLevel"], "high")
                self.assertNotIn("张小明", result.result["reason"])
                self.assertIn("已补充脱敏并复检", result.result["reason"])
                sent = self.requests[-1]["messages"]
                self.assertIn("实际治理", sent[0]["content"])
                self.assertIn("[手机号已脱敏]", sent[0]["content"])
                self.assertEqual(sent[-1]["content"], original)
                saved = self.calls.create.call_args.kwargs
                self.assertEqual(saved["status"], "success")
                self.assertEqual(json.loads(saved["output_json"]), result.result)

    def test_model_governance_unhandled_risk_never_becomes_success_record(self):
        original = "尚未处理的风险正文。"
        self.content = json.dumps({"originalOutput": original, "governedOutput": original,
                                   "reason": "风险尚未处理。", "riskLevel": "high"}, ensure_ascii=False)
        result = InferenceService(self.manager).invoke("model-test", original, "model_risk_governance", "token-test")
        self.assertEqual(result.result["assessment_status"], "unavailable")
        self.assertEqual(result.result["riskLevel"], "unknown")
        self.assertNotIn(original, result.result["governedOutput"])
        self.assertEqual(len(self.requests), 2)
        attempts = self.calls.create.call_args_list
        self.assertEqual([c.kwargs["status"] for c in attempts], ["error", "error", "success"])

    def test_model_governance_non_sensitive_content_is_not_rewritten(self):
        original = "今天餐厅正常营业。"
        expected = {"originalOutput": original, "governedOutput": original,
                    "reason": "未发现个人信息风险。", "riskLevel": "low", "reconstruction": None}
        self.content = json.dumps(expected, ensure_ascii=False)
        result = InferenceService(self.manager).invoke(
            "model-test", original, "model_risk_governance", "token-test",
        )
        self.assertEqual({k: result.result[k] for k in expected}, expected)

    def test_qwen_structured_request_disables_thinking_to_keep_json_in_content(self):
        self.manager.models.get.return_value.upstream_model_id = "/data/models/Qwen3-8B"
        self.manager.models.get.return_value.base_url = "http://127.0.0.1:1222/v1"
        self.reasoning_when_thinking = True
        result = self.invoke()
        self.assertEqual(result.answer, "checked")
        self.assertEqual(self.requests[-1]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(self.requests[-1]["response_format"]["type"], "json_schema")
        self.assertEqual(len(self.requests), 1)

    def test_internlm_structured_request_has_no_thinking_override(self):
        self.manager.models.get.return_value.upstream_model_id = "/data/models/internlm3-8b-instruct"
        self.manager.models.get.return_value.base_url = "http://127.0.0.1:1221/v1"
        self.invoke()
        self.assertNotIn("chat_template_kwargs", self.requests[-1])
        self.assertEqual(self.requests[-1]["response_format"]["type"], "json_schema")

    def test_qwen_plain_text_request_keeps_existing_thinking_settings(self):
        self.manager.models.get.return_value.upstream_model_id = "/data/models/Qwen3-8B"
        self.content = "plain answer"
        result = self.manager.invoke_text("model-test", "check input", token_hash="token-test")
        self.assertEqual(result, "plain answer")
        self.assertNotIn("chat_template_kwargs", self.requests[-1])
        self.assertNotIn("response_format", self.requests[-1])

    def test_qwen_empty_content_does_not_use_reasoning_as_business_result(self):
        self.manager.models.get.return_value.upstream_model_id = "/data/models/Qwen3-8B"
        self.reasoning_content = '{"answer":"private reasoning"}'
        self.content = ""
        result = self.invoke()
        self.assertEqual(result.assessment_status, "unavailable")
        self.assertNotIn("private reasoning", result.model_dump_json())

    def test_text_generation_does_not_request_json(self):
        self.content = "plain answer"
        result = self.manager.invoke_text("model-test", "check input", token_hash="token-test")
        self.assertNotIn("response_format", self.requests[0])
        self.assertEqual(result, "plain answer")
        self.assertEqual(self.calls.create.call_args.kwargs["status"], "success")

    def test_invalid_provider_output_still_fails_local_validation(self):
        for content in ('```json\n{"answer":"checked"}\n```', "{}", '{"answer":"checked","extra":true}'):
            self.content = content
            before = len(self.requests)
            result = self.invoke()
            self.assertEqual(result.assessment_status, "unavailable")
            self.assertNotEqual(result.answer, "checked")
            self.assertEqual(len(self.requests) - before, 2)

    def test_schema_constraint_does_not_bypass_business_validation(self):
        def validate(result):
            raise StructuredOutputValidationError("引用来源或原文不匹配")
        result = self.invoke(validator=validate)
        self.assertEqual(result.assessment_status, "unavailable")
        self.assertNotEqual(result.answer, "checked")

    def test_provider_rejection_is_reported_without_unconstrained_fallback(self):
        self.status = 400
        with self.assertRaises(InvocationError) as raised:
            self.invoke()
        self.assertEqual(raised.exception.code, "provider_error")
        self.assertEqual(raised.exception.upstream_status, 400)
        self.assertNotIn("private upstream detail", str(raised.exception))
        self.assertEqual(len(self.requests), 1)
        self.assertIn("response_format", self.requests[0])
        self.assertEqual(self.calls.create.call_args.kwargs["error_code"], "provider_error")
