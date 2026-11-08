"""Typed, conservative unavailable results shared by all model judgments.

Metadata describes availability, never success of a downstream business action.
Only server-owned factories may replace a rejected answer.
"""
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

UNAVAILABLE = "资料不足，无法判断"
META_FIELDS = {"assessment_status", "unavailable_fields", "assessment_reason"}


class AssessmentModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    assessment_status: Literal["assessed", "partial", "unavailable"] = "assessed"
    unavailable_fields: list[str] = Field(default_factory=list)
    assessment_reason: str | None = None


def metadata(result):
    data = result.model_dump() if isinstance(result, BaseModel) else result
    return {key: data[key] for key in META_FIELDS if key in data}


def request_data(text):
    """Prompts append a JSON request on the last line; never infer IDs from prose."""
    for candidate in (text, text.rsplit("\n", 1)[-1]):
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except (ValueError, TypeError):
            pass
    return {}


def unavailable_result(cls, text, reason=UNAVAILABLE):
    """Explicit templates; no recursive zero-filling of evidence, IDs or timestamps."""
    from llm import outputs as generic
    from llm import governance_outputs as governance
    from llm import analysis_contracts as analysis

    request = request_data(text)
    result = None
    if issubclass(cls, analysis.ScenarioOutput):
        code = request.get("scenario_code") or cls.model_fields["scenario_code"].annotation.__args__[0]
        details = {
            "public_opinion": dict(issues=[], sentiment="unknown", sentiment_evidence=[], controversies=[], communication_advice=[]),
            "hot_events": dict(event_summary=UNAVAILABLE, timeline=[], focal_points=[], contradictions=[], information_gaps=[reason], trend=None),
            "cross_cultural": dict(cultural_contexts=[], ambiguities=[], expression_suggestions=[], translation=None),
            "ethnic_governance": dict(issues=[], misunderstandings=[], conflict_expressions=[], coordination_advice=[]),
            "smart_government": dict(answer="资料不足，无法给出办理要求", policy_claims=[], processing_points=[], insufficient_materials=True),
            "cyber_security": dict(risk_types=[], suspicious_signs=[], handling_advice=[]),
        }[code]
        if code == "cross_cultural" and request.get("target_language"):
            details["translation"] = dict(target_language=request["target_language"], text=UNAVAILABLE)
        result = dict(scenario_code=code, summary=UNAVAILABLE, risk_level="unknown", evidence=[], recommendations=[],
                      limitations=[reason], insufficient_materials=True, details=dict(type=code, **details))
    elif cls in (analysis.ReasoningAuditOutput, analysis.ChainAuditOutput):
        result = dict(audit_type=request["audit_type"], subject=request["subject"], conclusion=UNAVAILABLE,
                      limitations=[reason], verification_state="unverified")
        if cls is analysis.ReasoningAuditOutput:
            result["risk_nodes"] = []
        else:
            result.update(checks=[dict(step_id=s["id"], rule_id=r, assessment="unverifiable", reason=reason, evidence=[])
                                  for s in request.get("steps", []) for r in s.get("rule_refs", [])], risks=[], gaps=[reason])
    elif cls is governance.ValueSampleOutput:
        result = dict(sample_id=request["sampleId"] if "sampleId" in request else request["sample_id"], status="unavailable", dimensions=[], unavailable_reason=reason)
    elif cls is governance.AnomalyLabelJudgment:
        result = dict(sampleId=request["sampleId"], sampleRevisionId=request["sampleRevisionId"], relation="uncertain",
                      textTopic=UNAVAILABLE, labelMeaning=UNAVAILABLE, quote="", reason=reason)
    elif cls in (governance.AnomalySampleOutput, governance.AnomalyRepairOutput):
        result = dict(sampleId=request["sampleId"], sampleRevisionId=request["sampleRevisionId"], status="unavailable", findings=[], unavailableReason=reason)
    elif cls is governance.RiskSampleOutput:
        result = dict(sampleId=request["sampleId"], sampleRevisionId=request["sampleRevisionId"], status="unassessable", findings=[], reviewRecommended=True, unassessableReason=reason)
    elif cls is governance.MultilingualSampleOutput:
        result = dict(sampleId=request["sampleId"], sampleRevisionId=request["sampleRevisionId"], status="unassessable",
                      semanticWeights=dict(safe=0, neutral=0, risk=0), riskScore=0, evidence=[], unassessableReason=reason)
    elif cls is generic.TextAnswer:
        result = dict(answer=UNAVAILABLE)
    elif cls is generic.JudgementAnswer:
        result = dict(score=0, reason=reason)
    elif cls is generic.ModelRiskGovernanceOutput:
        # Never echo rejected provider output (or a sensitive prompt) into public governance text.
        result = dict(originalOutput=request.get("originalOutput", UNAVAILABLE), governedOutput=UNAVAILABLE,
                      reason=reason, riskLevel="unknown", reconstruction=None)
    elif cls is generic.SemanticRiskOutput:
        result = dict(findings=[], evidence=[], primaryCategory=UNAVAILABLE, maximumSuggestedLevel="UNKNOWN")
    elif cls is generic.ValueScoreOutput:
        result = dict(dimensions=[dict(name=n, score=0, reason=reason, evidence=[]) for n in generic.ValueDimensionName.__args__],
                      tier="unavailable", unavailableReason=reason)
    elif cls is generic.AnomalyRepairOutput:
        result = dict(findings=[], primaryType=UNAVAILABLE, source=dict(filename="", line=0), fieldChanges=[], reason=reason, validationResults=[])
    elif cls is generic.FullChainAuditOutput:
        result = dict(conclusion=UNAVAILABLE, checks=[], gaps=[dict(description=UNAVAILABLE, reason=reason)])
    elif cls is generic.ReasoningAuditOutput:
        result = dict(steps=[], riskNodes=[], auditResult=UNAVAILABLE)
    if result is None:
        raise TypeError(f"No unavailable template registered for {cls.__module__}.{cls.__name__}")
    result.update(assessment_status="unavailable", unavailable_fields=[k for k in result if k not in {"sampleId", "sample_id", "sampleRevisionId", "subject", "scenario_code", "audit_type"}], assessment_reason=reason)
    return cls.model_validate(result)


def normalize_availability(value, text=""):
    """Derive availability from existing business states; keep valid partial evidence."""
    if not isinstance(value, AssessmentModel):
        return value
    data = value.model_dump()
    if value.unavailable_fields and value.assessment_status == "assessed":
        value.assessment_status = "partial"
    unavailable = (data.get("status") in {"unavailable", "unassessable"} or data.get("relation") == "uncertain"
                   or data.get("riskLevel") == "unknown" or data.get("maximumSuggestedLevel") == "UNKNOWN"
                   or data.get("tier") == "unavailable")
    request = request_data(text)
    if data.get("audit_type") and (not request.get("evidence") or not request.get("steps") or not request.get("rules")):
        unavailable = True
    checks = data.get("checks") or []
    if data.get("audit_type") and checks:
        missing = sum(c.get("assessment") == "unverifiable" for c in checks)
        if missing:
            unavailable = True
            if missing < len(checks):
                value.assessment_status = "partial"
    if data.get("insufficient_materials"):
        unavailable = True
        if data.get("evidence"):
            value.assessment_status = "partial"
    if unavailable:
        if value.assessment_status != "partial":
            value.assessment_status = "unavailable"
        value.assessment_reason = (data.get("unavailable_reason") or data.get("unavailableReason") or
                                   data.get("unassessableReason") or value.assessment_reason or UNAVAILABLE)
        if not value.unavailable_fields:
            value.unavailable_fields = [k for k in ("score", "riskScore", "risk_level", "riskLevel", "dimensions", "findings", "relation") if k in data]
    if value.assessment_status != "assessed":
        value.assessment_reason = value.assessment_reason or UNAVAILABLE
    return value
