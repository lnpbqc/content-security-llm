"""风险判断约束的独立单元测试，不连接或修改数据库。"""

import unittest
import json
from unittest.mock import MagicMock
from types import SimpleNamespace

from pydantic import ValidationError

from llm.governance_outputs import RiskSampleOutput
from service.governance import GovernanceService
from service.governance_data import BusinessGovernanceData
from llm.models import InvocationError


class RiskConstraintsTest(unittest.TestCase):
    def setUp(self):
        data = BusinessGovernanceData(None, connection_factory=lambda: None)
        self.service = GovernanceService(None, data, None, None)
        self.policy = data.policy("governance-risk", "risk-v1")
        self.sample = {
            "id": "1", "revision_id": "v1:1", "language": "zh",
            "text": "社区举办文化交流活动，现场气氛融洽。",
        }

    def finding(self, quote):
        return {
            "category": "个人信息暴露", "suggestedLevel": "MEDIUM",
            "quote": quote, "reason": "需要人工复核", "ruleId": "PII-03",
            "ruleVersion": "1.0", "evidenceRefs": ["1:e1"],
        }

    def output(self, findings):
        return {"sampleId": "1", "sampleRevisionId": "v1:1", "status": "assessed",
                "findings": findings, "reviewRecommended": bool(findings)}

    def test_prompt_explicitly_allows_safe_and_requires_source_quote(self):
        prompt = self.service._model_input("governance-risk", self.sample, self.policy)
        self.assertIn("findings 为空", prompt["instructions"])
        self.assertIn("逐字出现", prompt["instructions"])
        self.assertIn("不得编造", prompt["instructions"])

    def test_safe_result_needs_no_finding(self):
        result = RiskSampleOutput.model_validate(self.output([]))
        self.service._validate("governance-risk", self.sample, self.policy, result.model_dump())

    def test_quote_missing_from_text_is_rejected(self):
        result = RiskSampleOutput.model_validate(self.output([self.finding("账户标识及注册时间")]))
        with self.assertRaisesRegex(ValueError, "风险证据"):
            self.service._validate("governance-risk", self.sample, self.policy, result.model_dump())

    def test_empty_or_missing_quote_is_rejected(self):
        finding = self.finding("")
        with self.assertRaises(ValidationError):
            RiskSampleOutput.model_validate(self.output([finding]))
        del finding["quote"]
        with self.assertRaises(ValidationError):
            RiskSampleOutput.model_validate(self.output([finding]))

    def test_valid_quote_passes_structural_check(self):
        result = RiskSampleOutput.model_validate(self.output([self.finding("文化交流")]))
        self.service._validate("governance-risk", self.sample, self.policy, result.model_dump())
        # 片段存在只代表证据格式有效，不能证明风险类别或判断理由正确。

    def test_unknown_evidence_reference_is_rejected(self):
        finding = self.finding("文化交流")
        finding["evidenceRefs"] = ["other:e1"]
        result = RiskSampleOutput.model_validate(self.output([finding]))
        with self.assertRaisesRegex(ValueError, "风险证据"):
            self.service._validate("governance-risk", self.sample, self.policy, result.model_dump())

    def test_only_text_is_sent_and_source_cannot_be_used_as_quote(self):
        self.sample["metadata"] = {
            "text": "文化交流活动", "_source": "账户标识及注册时间",
            "source": "秘密来源", "topic_label": "风险标签", "reference_risk_level": "safe",
        }
        prompt = self.service._model_input("governance-risk", self.sample, self.policy)
        self.assertEqual(prompt["text"], "文化交流活动")
        self.assertEqual(prompt["evidence"][0]["quote"], "文化交流活动")
        self.assertNotIn("metadata", prompt)
        self.assertNotIn("reference_risk_level", prompt)
        with self.assertRaisesRegex(ValueError, "引用片段不在当前正文中"):
            self.service._validate("governance-risk", self.sample, self.policy,
                                   self.output([self.finding("账户标识及注册时间")]))

    def run_task(self, metadata, manager):
        self.sample.update(dataset_id=3, version_id="v1", metadata=metadata)
        outcomes = {}
        published = []
        def save(task_id, sample_id, output, error_code=None):
            outcomes[sample_id] = {"status": "error" if error_code else "success",
                                   "output": output, "error_code": error_code}
        self.service.data.samples = lambda scope: [self.sample]
        self.service._base_summary = lambda *args: {}
        self.service.manager = manager
        self.service.repository = SimpleNamespace(
            sample_outcomes=lambda task_id: outcomes, save_sample=save,
            publish=lambda task, summary, rows, **kwargs: published.append(summary),
        )
        self.service._run({"id": "task", "kind": "governance-risk", "model_id": "model",
                           "token_hash": "token", "scope": {"scheme_id": "risk-v1"}})
        return outcomes, published[0]

    def test_blank_text_is_unassessable_without_model_call(self):
        def invoke(*args, **kwargs):
            self.fail("空正文不能调用模型")
        _, summary = self.run_task({"text": "", "source": "不应替代正文"}, SimpleNamespace(invoke=invoke))
        self.assertEqual(summary["unassessable_count"], 1)
        self.assertEqual(summary["risk_count"], 0)

    def test_failed_sample_preserves_reason_without_becoming_safe(self):
        def invoke(*args, **kwargs):
            raise InvocationError("invalid_structured_output", "模型输出缺少原文引用（quote）")
        outcomes, summary = self.run_task(
            {"text": "文化交流", "reference_risk_level": "safe"}, SimpleNamespace(invoke=invoke))
        self.assertEqual(outcomes["1"]["status"], "error")
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(summary["valid_count"], 0)
        self.assertEqual(summary["accuracy_correct_count"], 0)
        self.assertEqual(summary["failed_samples"][0]["reason"], "模型输出缺少原文引用（quote）")

    def test_governance_labels_are_kept_for_evaluation_but_not_sent_to_model(self):
        payload = {"text": "社区文化交流", "language": "zh", "topic_label": "文化",
                   "reference_risk_level": "safe", "reference_value_tier": "high", "reference_is_anomaly": False}
        cursor = MagicMock()
        cursor.__enter__.return_value = cursor
        cursor.fetchone.side_effect = [{"name": "测试", "version": "v1"}, None]
        cursor.fetchall.return_value = [{"id": 1, "payload": payload}]
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.cursor.return_value = cursor
        data = BusinessGovernanceData(None, connection_factory=lambda: connection)
        sample = data.samples({"dataset_id": 3, "version_id": "v1", "language": "zh"})[0]
        for kind, scheme in (("governance-value", "general-v1"), ("governance-anomaly", "anomaly-basic-v1")):
            prompt = self.service._model_input(kind, sample, data.policy(kind, scheme))
            serialized = json.dumps(prompt)
            for field in ("reference_value_tier", "reference_is_anomaly", "reference_risk_level"):
                self.assertNotIn(field, serialized)
                self.assertIn(field, sample["metadata"])

    def test_value_and_anomaly_publish_frozen_answers_even_when_model_fails(self):
        for kind, scheme, field, ref in (("governance-value", "general-v1", "reference_value_tier", "high"),
                                         ("governance-anomaly", "anomaly-basic-v1", "reference_is_anomaly", False)):
            with self.subTest(kind=kind):
                outcomes, published = {}, []
                sample = {**self.sample, "metadata": {field: ref}}
                self.service.data.samples = lambda scope: [sample]
                self.service._base_summary = lambda *args: {}
                def save(task_id, sample_id, output, error_code=None):
                    outcomes[sample_id] = {"status": "error", "output": output, "error_code": error_code}
                def invoke(*args, **kwargs):
                    raise InvocationError("timeout", "连接超时")
                self.service.manager = SimpleNamespace(invoke=invoke)
                self.service.repository = SimpleNamespace(sample_outcomes=lambda task_id: outcomes, save_sample=save,
                    publish=lambda task, summary, rows, **kwargs: published.append(summary))
                self.service._run({"id": "task", "kind": kind, "model_id": "model", "token_hash": "token", "scope": {"scheme_id": scheme, "language": "all"}})
                self.assertEqual(published[0]["evaluation_contract"], "governance-reference-v1")
                self.assertEqual(published[0]["reference_annotations"], {"1": ref})
                self.assertEqual(published[0]["failed_count"], 1)


if __name__ == "__main__":
    unittest.main()
