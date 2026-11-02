"""新版主题错标规则的独立回归测试，不连接数据库或调用真实模型。"""
import json
import unittest
from types import SimpleNamespace

from service.governance import GovernanceService
from service.governance_data import BusinessGovernanceData
from llm.governance_outputs import AnomalyLabelJudgment, AnomalyRepairOutput, AnomalySampleOutput
from llm.structured_output import schema_system_prompt, StructuredOutputValidationError


class AnomalyConstraintsTest(unittest.TestCase):
    def setUp(self):
        self.data = BusinessGovernanceData(None, connection_factory=lambda: None)
        self.service = GovernanceService(None, self.data, None, None)
        self.policy = self.data.policy("governance-anomaly", "anomaly-basic-v2")
        self.sample = {"id": "1", "revision_id": "v1:1", "dataset_id": 3, "version_id": "v1", "language": "zh",
                       "text": "_source: private\ntopic_label: 各地美食\ntext: 烤馕刚出炉，香气浓郁。",
                       "metadata": {"text": "烤馕刚出炉，香气浓郁。", "topic_label": "各地美食",
                                    "_source": {"file": "private.csv", "content_hash": "secret"},
                                    "reference_is_anomaly": False, "reference_risk_level": "safe",
                                    "reference_value_tier": "high"}}

    def output(self, quote="烤馕", reason="当前标签是体育赛事，正文实际介绍食物"):
        return {"sampleId": "1", "sampleRevisionId": "v1:1", "status": "assessed", "findings": [{
            "type": "标签异常", "field": "topic_label", "ruleId": "anomaly-rule-1",
            "quote": quote, "reason": reason,
            "fieldChanges": [{"field": "topic_label", "before": self.sample["metadata"]["topic_label"], "after": "美食"}],
            "validationResults": []}]}

    def test_input_contains_only_body_label_and_rules(self):
        prompt = self.service._model_input("governance-anomaly", self.sample, self.policy)
        self.assertEqual(prompt["text"], self.sample["metadata"]["text"])
        self.assertEqual(prompt["metadata"], {"topic_label": "各地美食"})
        for forbidden in ("_source", "private.csv", "secret", "reference_is_anomaly", "reference_risk_level", "reference_value_tier"):
            self.assertNotIn(forbidden, json.dumps(prompt))
        self.assertIsNone(prompt["allowedLabels"])
        self.assertIn("上下级", prompt["instructions"])
        self.assertIn("不得为了统一名称", prompt["instructions"])

    def test_normal_and_unavailable_have_no_findings(self):
        for label in ("各地美食", "地方饮食", "美食"):
            with self.subTest(label=label):
                self.sample["metadata"]["topic_label"] = label
                output = AnomalySampleOutput(sampleId="1", sampleRevisionId="v1:1", status="assessed", findings=[])
                self.service._validate("governance-anomaly", self.sample, self.policy, output.model_dump())
        AnomalySampleOutput(sampleId="1", sampleRevisionId="v1:1", status="unavailable", findings=[], unavailableReason="正文过短")

    def test_label_only_or_metadata_quote_cannot_prove_conflict(self):
        for quote in ("各地美食", "private", "不存在的正文", " "):
            with self.subTest(quote=quote), self.assertRaises(ValueError):
                self.service._validate("governance-anomaly", self.sample, self.policy, self.output(quote))

    def test_unconfigured_enumeration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "未配置标签清单"):
            self.service._validate("governance-anomaly", self.sample, self.policy, self.output(reason="不符合标签枚举值"))

    def test_conflict_with_body_evidence_can_pass_technical_checks(self):
        self.sample["metadata"]["topic_label"] = "篮球赛事"
        self.service._validate("governance-anomaly", self.sample, self.policy, self.output())

    def test_legacy_scheme_is_preserved(self):
        old = self.data.policy("governance-anomaly", "anomaly-basic-v1")
        self.assertEqual(old["name"], "基础异常检测 v1.0")
        self.assertNotIn("allowed_labels", old)

    def test_missing_body_or_label_does_not_call_model(self):
        for metadata in ({"text": "", "topic_label": "美食"}, {"topic_label": "美食"},
                         {"text": "烤馕"}, {"text": 42, "topic_label": "美食"},
                         {"text": "烤馕", "topic_label": " "}):
            with self.subTest(metadata=metadata):
                sample = {**self.sample, "metadata": metadata}
                outcomes, published = {}, []
                self.data.samples = lambda scope: [sample]
                def save(task_id, sample_id, output):
                    outcomes[sample_id] = {"status": "success", "output": output}
                self.service.repository = SimpleNamespace(sample_outcomes=lambda task_id: outcomes, save_sample=save,
                    publish=lambda task, summary, rows, **kwargs: published.append(summary))
                self.service.manager = SimpleNamespace(invoke=lambda *a, **k: self.fail("无效正文或标签不应调用模型"))
                self.service._base_summary = lambda *args: {}
                self.service._run({"id": "task", "kind": "governance-anomaly", "model_id": "model", "token_hash": "token",
                                   "scope": {"scheme_id": "anomaly-basic-v2", "language": "all"}})
                self.assertEqual(published[0]["unavailable_count"], 1)
                self.assertEqual(published[0]["anomaly_count"], 0)

    def judge(self, relation="compatible", repair=None, wrong_id=False, judgment_quote="烤馕"):
        policy = self.data.policy("governance-anomaly", "anomaly-basic-v2.1")
        prompt = self.service._model_input("governance-anomaly", self.sample, policy)
        calls = []
        def invoke(model_id, raw, response_model, **kwargs):
            data = json.loads(raw)
            calls.append((response_model, data, kwargs))
            if response_model is AnomalyLabelJudgment:
                result = response_model(sampleId="other" if wrong_id else "1", sampleRevisionId="v1:1",
                    relation=relation, textTopic="烤馕饮食介绍", labelMeaning="各种地方食物",
                    quote=judgment_quote, reason="正文介绍食物，属于标签覆盖范围" if relation == "compatible" else "正文与标签主题不相容或不足以判断")
            else:
                result = response_model.model_validate(repair)
            kwargs["validator"](result)
            return result
        self.service.manager = SimpleNamespace(invoke=invoke)
        result = self.service._judge_anomaly({"model_id": "model", "token_hash": "token"}, self.sample, policy, prompt)
        return result, calls

    def test_compatible_labels_never_enter_repair_step(self):
        for label in ("各地美食", "地方饮食", "美食"):
            with self.subTest(label=label):
                self.sample["metadata"]["topic_label"] = label
                result, calls = self.judge()
                self.assertEqual(result["status"], "assessed")
                self.assertEqual(result["findings"], [])
                self.assertEqual(len(calls), 1)
                self.assertNotIn("instructions", calls[0][1])
                self.assertNotIn("rules", calls[0][1])
                self.assertEqual(calls[0][2]["temperature"], 0)

    def test_uncertain_does_not_generate_repair(self):
        result, calls = self.judge("uncertain")
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["findings"], [])
        self.assertEqual(len(calls), 1)

    def test_conflict_requires_second_step_and_preserves_judgment(self):
        self.sample["metadata"]["topic_label"] = "篮球赛事"
        result, calls = self.judge("conflict", self.output())
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[1][0], AnomalyRepairOutput)
        self.assertEqual(calls[1][1]["labelJudgment"]["relation"], "conflict")
        self.assertEqual(result["labelJudgment"]["relation"], "conflict")
        self.assertTrue(result["findings"])

    def test_model_false_check_does_not_erase_food_label_conflicts(self):
        cases = (
            ("广州肠粉口感柔软细滑，常见配料有鸡蛋、鲜虾和蔬菜。", "地理自然", "广州肠粉"),
            ("武汉热干面常用芝麻酱拌制，面条筋道，香气浓郁。", "军事科技", "武汉热干面"),
            ("呼和浩特烧麦外皮薄，馅料鲜香，蒸熟后趁热食用口感更好。", "数码科技", "呼和浩特烧麦"),
        )
        for body, label, quote in cases:
            with self.subTest(label=label):
                self.sample["metadata"].update(text=body, topic_label=label)
                self.sample["text"] = body
                repair = self.output(quote=quote, reason="正文介绍饮食，与原标签主题不相容")
                repair["findings"][0]["validationResults"] = [{"name": "主题标签明显冲突", "passed": False}]
                result, _ = self.judge("conflict", repair, judgment_quote=quote)
                self.assertEqual(result["status"], "assessed")
                self.assertEqual(result["findings"], repair["findings"])
                self.service._base_summary = lambda *args: {}
                policy = self.data.policy("governance-anomaly", "anomaly-basic-v2.1")
                summary, rows = self.service._anomaly_snapshot(
                    {}, policy, [self.sample], {"1": {"status": "success", "output": result}}, "result")
                self.assertEqual(summary["anomaly_count"], 1)
                self.assertEqual(summary["unavailable_count"], 0)
                self.assertEqual(rows[0]["status"], "待处理")
                self.assertEqual(rows[0]["candidates"][0]["status"], "DRAFT")

    def test_explicit_unavailable_repair_remains_unavailable(self):
        repair = {"sampleId": "1", "sampleRevisionId": "v1:1", "status": "unavailable",
                  "findings": [], "unavailableReason": "无法确认主题冲突"}
        result, _ = self.judge("conflict", repair)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["findings"], [])

    def test_wrong_sample_judgment_is_rejected(self):
        with self.assertRaisesRegex(StructuredOutputValidationError, "编号不一致"):
            self.judge(wrong_id=True)

    def test_conflict_cannot_succeed_with_empty_repair(self):
        repair = {"sampleId": "1", "sampleRevisionId": "v1:1", "status": "assessed", "findings": []}
        with self.assertRaisesRegex(StructuredOutputValidationError, "缺少具体发现"):
            self.judge("conflict", repair)

    def test_system_instructions_cover_relationship_and_separate_repair(self):
        judgment_prompt = schema_system_prompt(AnomalyLabelJudgment)
        self.assertIn("上位分类", judgment_prompt)
        self.assertIn("担担面制作介绍＋各地美食 => compatible", judgment_prompt)
        self.assertIn("担担面介绍＋篮球赛事 => conflict", judgment_prompt)
        self.assertIn("今天不错＋各地美食 => uncertain", judgment_prompt)
        self.assertIn("不要返回修改建议", judgment_prompt)
        self.assertIn("无法确认冲突", schema_system_prompt(AnomalyRepairOutput))


if __name__ == "__main__":
    unittest.main()
