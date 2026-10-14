"""编排治理任务的建单、逐样本模型调用、校验和结果汇总。"""

import json
from typing import Any, Dict, List, Optional, Type
from uuid import uuid4

from pydantic import BaseModel

from db.governance import GovernanceRepository
from db.governance_config import GovernanceConfigRepository
from llm.governance_outputs import AnomalySampleOutput, MultilingualSampleOutput, RiskSampleOutput, ValueSampleOutput
from llm.models import InvocationError, ModelManager, ModelNotFoundError
from service.governance_data import BusinessGovernanceData
from service.multilingual_semantic import build_snapshot
from utils.time import utc_now


KINDS = {"governance-value", "governance-anomaly", "governance-risk", "ethnic-multilingual"}
REFERENCE_RISK_LEVELS = ("safe", "low", "medium", "high")
OUTPUT_TYPES: Dict[str, Type[BaseModel]] = {
    "governance-value": ValueSampleOutput,
    "governance-anomaly": AnomalySampleOutput,
    "governance-risk": RiskSampleOutput,
    "ethnic-multilingual": MultilingualSampleOutput,
}
RESPONSE_TYPES = {
    "governance-value": "governance_value_sample",
    "governance-anomaly": "governance_anomaly_sample",
    "governance-risk": "governance_risk_sample",
    "ethnic-multilingual": "ethnic_multilingual_sample",
}


class ModelNotConfiguredError(Exception):
    """对应任务类型尚未配置可用模型。"""
    pass


class GovernanceService:
    """连接任务存储、样本适配器和现有模型调用能力。"""
    def __init__(self, repository: GovernanceRepository, data: BusinessGovernanceData,
                 manager: ModelManager, config: GovernanceConfigRepository):
        """接收持久化、数据读取、模型调用及按任务类型选模的依赖。"""
        self.repository = repository
        self.data = data
        self.manager = manager
        self.config = config

    def create_task(self, kind: str, name: str, scope: Dict[str, Any],
                    token_hash: str) -> Dict[str, Any]:
        """校验方案与模型配置后保存 pending 任务；业务库由 worker 读取。"""
        if kind not in KINDS:
            raise ValueError("不支持的任务类型")
        self.data.policy(kind, scope["scheme_id"])
        model_id = self.config.get_model_id(kind)
        if not model_id:
            raise ModelNotConfiguredError(kind)
        try:
            model = self.manager.get_model(model_id)
        except ModelNotFoundError as exc:
            raise ModelNotConfiguredError(kind) from exc
        return self.repository.create_task(kind, token_hash, name, scope, model_id,
                                           model.upstream_model_id)

    def create_multilingual_task(self, scope: Dict[str, Any], model_id: str,
                                 token_hash: str) -> Dict[str, Any]:
        """场景工作台显式选模，结果仍由同一队列持久化。"""
        self.data.policy("ethnic-multilingual", scope["scheme_id"])
        model = self.manager.get_model(model_id)
        return self.repository.create_task("ethnic-multilingual", token_hash,
                                           "多语种语义权重与风险分层", scope, model_id,
                                           model.upstream_model_id)

    def run_once(self) -> bool:
        """领取并处理至多一个任务；无待处理任务时返回 False。"""
        task = self.repository.claim()
        if task is None:
            return False
        try:
            self._run(task)
        except Exception as exc:
            self.repository.fail(task["id"], "{}: {}".format(type(exc).__name__, exc))
        return True

    def _run(self, task: Dict[str, Any]) -> None:
        """逐条调用模型并保存检查点，最后发布汇总结果。"""
        scope, kind = task["scope"], task["kind"]
        policy = self.data.policy(kind, scope["scheme_id"])
        if kind == "ethnic-multilingual":
            policy = {**policy, "analysis_mode": scope["analysis_mode"]}
        samples = self.data.samples(scope)
        if kind == "ethnic-multilingual":
            if any(sample["metadata"].get("is_mock") for sample in samples):
                raise ValueError("多语种分析只接受真实数据集记录")
            samples = [sample for sample in samples if sample["language"] in scope["languages"]]
            if not samples:
                raise ValueError("所选语种没有可分析的样本")
        if kind == "governance-risk":
            for sample in samples:
                if ("reference_risk_level" in sample["metadata"] and
                        sample["metadata"]["reference_risk_level"] not in REFERENCE_RISK_LEVELS):
                    raise ValueError("样本 {} 的标准风险档位无效".format(sample["id"]))
        outcomes = self.repository.sample_outcomes(task["id"])
        for sample in samples:
            if sample["id"] in outcomes:
                continue
            prompt = self._model_input(kind, sample, policy)
            try:
                parsed = self.manager.invoke(
                    task["model_id"], json.dumps(prompt, ensure_ascii=False), OUTPUT_TYPES[kind],
                    response_type=RESPONSE_TYPES[kind], token_hash=task["token_hash"],
                    validator=lambda value: self._validate(
                        kind, sample, policy, value.model_dump(mode="json")
                    ),
                    timeout=120, max_retries=0,
                )
                output = parsed.model_dump(mode="json")
                self.repository.save_sample(task["id"], sample["id"], output)
            except InvocationError as exc:
                self.repository.save_sample(task["id"], sample["id"], None, exc.code)
        outcomes = self.repository.sample_outcomes(task["id"])
        if (samples and all(outcomes[s["id"]]["status"] == "error" for s in samples) and
                not (kind == "governance-risk" and
                     any("reference_risk_level" in s["metadata"] for s in samples))):
            self.repository.fail(task["id"], "所有样本的模型调用或业务校验均失败")
            return
        result_id = str(uuid4())
        summary, public_samples = self._snapshot(task, policy, samples, outcomes, result_id)
        if any(sample["id"].startswith("mock-") and sample["metadata"].get("is_mock") is True
               for sample in samples):
            summary["data_source"] = "mock"
        self.repository.publish(task, summary, public_samples, result_id=result_id)

    def _model_input(self, kind: str, sample: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
        """按任务类型组装单条模型输入；风险文本先遮蔽。"""
        if kind == "ethnic-multilingual":
            return {"sampleId": sample["id"], "sampleRevisionId": sample["revision_id"],
                    "text": self.data.risk_text(sample["text"]), "language": sample["language"],
                    "semanticScheme": policy,
                    "instructions": ("仅根据正文估计安全、中性、风险语义权重及风险分；三权重合计为 1。"
                                     "中高风险必须引用正文连续片段；无法评估时说明原因，不按语种或民族身份定风险。"
                                     + ("重点解释语义权重。" if policy["analysis_mode"] == "semantic_weighting"
                                        else "重点解释风险分层依据。"))}
        if kind == "governance-value":
            return {"sample_id": sample["id"], "text": sample["text"],
                    "language": sample["language"], "rubric": {
                        key: policy[key] for key in ("scheme_id", "dimensions", "score_min", "score_max")
                    }}
        if kind == "governance-anomaly":
            return {"sampleId": sample["id"], "sampleRevisionId": sample["revision_id"],
                    "text": sample["text"], "language": sample["language"],
                    "metadata": sample["metadata"], "schemeId": policy["scheme_id"],
                    "ruleVersion": policy["rule_version"], "rules": policy["rules"],
                    "fieldWhitelist": policy["field_whitelist"], "types": policy["types"],
                    "modelApplicableRuleIds": policy["model_applicable_rule_ids"]}
        masked = self.data.risk_text(sample["text"])
        return {"sampleId": sample["id"], "sampleRevisionId": sample["revision_id"],
                "text": masked, "language": sample["language"], "schemeId": policy["scheme_id"],
                "categories": policy["categories"], "levels": policy["levels"],
                "rules": policy["rules"], "evidence": self._risk_evidence(sample)}

    def _risk_evidence(self, sample: Dict[str, Any]) -> List[Dict[str, str]]:
        """生成风险模型可引用的遮蔽文本证据。"""
        return [{"id": sample["id"] + ":e1", "quote": self.data.risk_text(sample["text"]),
                 "feature": "遮蔽后正文"}]

    def _validate(self, kind: str, sample: Dict[str, Any],
                  policy: Dict[str, Any], output: Dict[str, Any]) -> None:
        """核对模型输出的样本、规则、字段和原文证据。"""
        if kind == "governance-value":
            if output["sample_id"] != sample["id"]:
                raise ValueError("样本 ID 不一致")
            if output["status"] == "scored":
                expected = {dimension["name"] for dimension in policy["dimensions"]}
                actual = [dimension["name"] for dimension in output["dimensions"]]
                if len(actual) != len(expected) or set(actual) != expected:
                    raise ValueError("评分维度不完整或重复")
                for dimension in output["dimensions"]:
                    if not dimension["reason"].strip() or not dimension["evidence"] or any(
                        not quote.strip() or quote not in sample["text"] for quote in dimension["evidence"]
                    ):
                        raise ValueError("评分理由或证据无效")
            return
        if output["sampleId"] != sample["id"] or output["sampleRevisionId"] != sample["revision_id"]:
            raise ValueError("样本或修订 ID 不一致")
        if kind == "ethnic-multilingual":
            masked = self.data.risk_text(sample["text"])
            if any(not item["quote"].strip() or not item["reason"].strip() or
                   item["quote"] not in masked for item in output["evidence"]):
                raise ValueError("语义风险引用或理由无效")
            return
        if kind == "governance-anomaly":
            rules = set(policy["model_applicable_rule_ids"])
            fields = set(policy["field_whitelist"])
            for finding in output["findings"]:
                if (finding["ruleId"] not in rules or finding["type"] != "标签异常" or
                    finding["field"] not in fields or not finding["quote"].strip() or
                    finding["quote"] not in sample["text"] or
                    not finding["reason"].strip()):
                    raise ValueError("异常证据或规则无效")
                for change in finding["fieldChanges"]:
                    if (change["field"] not in fields or
                        change["before"] != sample["metadata"].get(change["field"]) or
                        not change["after"].strip()):
                        raise ValueError("异常字段变更无效")
            return
        rules = {rule["id"]: rule for rule in policy["rules"]}
        evidence_ids = {item["id"] for item in self._risk_evidence(sample)}
        for finding in output["findings"]:
            rule = rules.get(finding["ruleId"])
            if (rule is None or rule["version"] != finding["ruleVersion"] or
                rule["category"] != finding["category"] or
                finding["category"] not in policy["categories"] or
                not set(finding["evidenceRefs"]).issubset(evidence_ids) or
                not finding["reason"].strip()):
                raise ValueError("风险证据、类别或规则版本无效")

    def _snapshot(self, task: Dict[str, Any], policy: Dict[str, Any],
                  samples: List[Dict[str, Any]], outcomes: Dict[str, Dict[str, Any]],
                  result_id: str):
        """选择对应任务的汇总方式，生成前端可读取的快照。"""
        kind = task["kind"]
        if kind == "governance-value":
            return self._value_snapshot(task, policy, samples, outcomes, result_id)
        if kind == "governance-anomaly":
            return self._anomaly_snapshot(task, policy, samples, outcomes, result_id)
        if kind == "ethnic-multilingual":
            earlier = self.repository.list_result_summaries(kind, task["token_hash"], task["scope"])
            previous = next((item["summary"] for item in earlier
                             if item["summary"].get("model_id") == task["model_id"]), None)
            base = {**self._base_summary(task, result_id), "model_id": task["model_id"]}
            return build_snapshot(base, policy, samples, outcomes, self.data.risk_text, previous)
        return self._risk_snapshot(task, policy, samples, outcomes, result_id)

    def _base_summary(self, task: Dict[str, Any], result_id: str) -> Dict[str, Any]:
        """生成三类结果共用的 ID、范围和完成时间字段。"""
        scope = task["scope"]
        return {"id": result_id, "task_id": task["id"], "scope": scope,
                "dataset_name": self.data.dataset_name(scope["dataset_id"], scope["version_id"]),
                "finished_at": utc_now().isoformat()}

    def _value_snapshot(self, task, policy, samples, outcomes, result_id):
        """由逐条评分计算综合分、档位、均分和分布。"""
        rows = []
        failed = 0
        for sample in samples:
            outcome = outcomes[sample["id"]]
            if outcome["status"] == "error":
                failed += 1
                continue
            output = outcome["output"]
            row = {"id": sample["id"], "text": sample["text"], "language": sample["language"],
                   "score": None, "tier": "unavailable", "dimensions": output["dimensions"]}
            if output["status"] == "scored":
                scores = {dimension["name"]: dimension["score"] for dimension in output["dimensions"]}
                row["score"] = round(sum(scores[d["name"]] * d["weight"] for d in policy["dimensions"]), 1)
                row["tier"] = ("high" if row["score"] >= policy["high_threshold"] else
                               "medium" if row["score"] >= policy["medium_threshold"] else "low")
            else:
                row["unavailable_reason"] = output["unavailable_reason"]
            rows.append(row)
        valid = [row for row in rows if row["score"] is not None]
        dimensions = []
        for definition in policy["dimensions"] if valid else []:
            name = definition["name"]
            dimensions.append({"name": name,
                               "score": round(sum(next(d["score"] for d in row["dimensions"] if d["name"] == name)
                                                  for row in valid) / len(valid), 1) if valid else 0,
                               "reason": "有效样本的维度均分", "evidence": []})
        bins = [{"id": str(i), "label": "{}–{}".format(i * 20, (i + 1) * 20),
                 "count": sum(1 for row in valid if min(4, int(row["score"] // 20)) == i)} for i in range(5)]
        summary = self._base_summary(task, result_id)
        summary.update({"version_label": task["scope"]["version_id"],
                        "language_name": task["scope"]["language"], "scheme_name": policy["name"],
                        "mean_score": round(sum(row["score"] for row in valid) / len(valid), 1) if valid else None,
                        "valid_count": len(valid), "high_count": sum(row["tier"] == "high" for row in valid),
                        "unavailable_count": len(rows) - len(valid), "failed_count": failed,
                        "target_count": len(samples), "languages": sorted({row["language"] for row in rows}),
                        "high_threshold": policy["high_threshold"],
                        "medium_threshold": policy["medium_threshold"],
                        "dimensions": dimensions, "bins": bins})
        return summary, rows

    def _anomaly_snapshot(self, task, policy, samples, outcomes, result_id):
        """汇总异常发现，并把可用字段建议保存为草稿候选。"""
        rows = []
        failed = unavailable = 0
        now = utc_now().isoformat()
        for sample in samples:
            outcome = outcomes[sample["id"]]
            if outcome["status"] == "error":
                failed += 1
                continue
            output = outcome["output"]
            if output["status"] == "unavailable":
                unavailable += 1
            findings = output["findings"]
            candidates = [{"candidate_id": str(uuid4()),
                           "input_sample_revision_id": sample["revision_id"],
                           "field_changes": finding["fieldChanges"], "reason": finding["reason"],
                           "validation_results": [{"name": "服务端规则、字段及原文引用校验", "passed": True}],
                           "status": "DRAFT"} for finding in findings if finding["fieldChanges"]]
            rows.append({"id": sample["id"], "dataset_id": sample["dataset_id"],
                         "version_id": sample["version_id"], "revision_id": sample["revision_id"],
                         "text": sample["text"], "language": sample["language"],
                         "primary_type": findings[0]["type"] if findings else "",
                         "status": "待处理" if findings or output["status"] == "unavailable" else "已处理",
                         "findings": [
                             {key: finding[key] for key in ("type", "field", "reason", "quote", "ruleId")}
                             for finding in findings],
                         "source": {}, "metadata": sample["metadata"], "candidates": candidates,
                         "timeline": [{"at": now, "message": "模型检测完成"}], "actions": [],
                         **({"unavailable_reason": output["unavailableReason"]}
                            if output["status"] == "unavailable" else {})})
        counts = [{"type": name, "count": sum(row["primary_type"] == name for row in rows)}
                  for name in policy["types"]]
        valid = len(rows) - unavailable
        anomaly_count = sum(bool(row["findings"]) for row in rows)
        summary = self._base_summary(task, result_id)
        summary.update({"version_label": task["scope"]["version_id"], "valid_count": valid,
                        "failed_count": failed, "unavailable_count": unavailable,
                        "anomaly_count": anomaly_count,
                        "ratio": round(anomaly_count / valid * 100, 1) if valid else 0,
                        "pending_count": anomaly_count, "review_count": 0, "processed_count": 0,
                        "primary_type_counts": counts})
        return summary, rows

    def _risk_snapshot(self, task, policy, samples, outcomes, result_id):
        """汇总风险等级和数量，仅保存遮蔽后的正文。"""
        rows = []
        predicted_levels = {}
        failed = unassessable = 0
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NOTICE": 3}
        for sample in samples:
            outcome = outcomes[sample["id"]]
            if outcome["status"] == "error":
                failed += 1
                continue
            output = outcome["output"]
            if output["status"] == "unassessable":
                unassessable += 1
            findings = output["findings"]
            level = min((finding["suggestedLevel"] for finding in findings),
                        key=order.get, default="NOTICE")
            if output["status"] == "assessed":
                predicted_levels[sample["id"]] = ({"LOW": "low", "MEDIUM": "medium", "HIGH": "high"}.get(level)
                                                  if findings else "safe")
            primary = min(findings, key=lambda finding: order[finding["suggestedLevel"]]) if findings else None
            rows.append({"id": sample["id"], "dataset_id": sample["dataset_id"],
                         "version_id": sample["version_id"], "revision_id": sample["revision_id"],
                         "text": self.data.risk_text(sample["text"]), "language": sample["language"],
                         "primary_category": primary["category"] if primary else "",
                         "maximum_suggested_level": level,
                         "status": ("无法评估" if output["status"] == "unassessable" else
                                    "待复核" if findings and output["reviewRecommended"] else "未发起"),
                         "findings": findings, "evidence": self._risk_evidence(sample),
                         "rules": policy["rules"], "cases": [], "actions": [],
                         **({"unassessable_reason": output["unassessableReason"]}
                            if output["status"] == "unassessable" else {})})
        valid = len(rows) - unassessable
        risk_rows = [row for row in rows if row["findings"]]
        levels = []
        for definition in policy["levels"]:
            count = sum(row["maximum_suggested_level"] == definition["level"] for row in risk_rows)
            levels.append({"level": definition["level"], "label": definition["label"],
                           "color": definition["color"], "count": count,
                           "percent": round(count / len(risk_rows) * 100, 1) if risk_rows else 0})
        summary = self._base_summary(task, result_id)
        labeled = [sample for sample in samples if "reference_risk_level" in sample["metadata"]]
        correct = sum(predicted_levels.get(sample["id"]) == sample["metadata"]["reference_risk_level"]
                      for sample in labeled)
        summary.update({"status": "已完成", "valid_count": valid, "risk_count": len(risk_rows),
                        "ratio": round(len(risk_rows) / valid * 100, 1) if valid else 0,
                        "high_count": sum(row["maximum_suggested_level"] == "HIGH" for row in risk_rows),
                        "pending_count": sum(row["status"] == "待复核" for row in rows),
                        "reviewed_count": 0, "unassessable_count": unassessable,
                        "failed_count": failed, "levels": levels, "actions": [],
                        "accuracy_correct_count": correct, "accuracy_test_count": len(labeled)})
        return summary, rows
