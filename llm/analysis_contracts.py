"""提交材料的分析契约；事实引用与业务核验状态分开。"""

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
InputText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=50000)]
Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
Language = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z]{2,8}(-[a-zA-Z0-9]{2,8})*$")]
ScenarioCode = Literal["public_opinion", "hot_events", "cross_cultural", "ethnic_governance",
                       "smart_government", "cyber_security"]
Kind = Literal["scenario", "model_governance", "audit"]
Risk = Literal["low", "medium", "high", "unknown"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Material(StrictModel):
    id: Identifier
    content: InputText
    title: Optional[Identifier] = None
    time: Optional[Identifier] = None  # 原材料时间原样保留；不推断时区、日期或趋势。


class ScenarioRequest(StrictModel):
    model_id: Identifier
    scenario_code: ScenarioCode
    content: InputText
    language: Language = "zh"
    materials: list[Material] = Field(default_factory=list, max_length=50)
    target_language: Optional[Language] = None

    @model_validator(mode="after")
    def validate_materials(self):
        ids = [item.id for item in self.materials]
        if len(ids) != len(set(ids)) or "input" in ids:
            raise ValueError("材料 ID 必须唯一，input 保留给输入正文")
        if self.target_language and self.scenario_code != "cross_cultural":
            raise ValueError("只有跨文化场景支持目标语言")
        if len(self.content) + sum(len(item.content) for item in self.materials) > 100000:
            raise ValueError("正文及材料合计不能超过 100000 字符")
        return self


class GovernanceRequest(StrictModel):
    model_id: Identifier
    prompt: InputText
    temperature: Optional[float] = Field(default=None, ge=0, le=2, allow_inf_nan=False)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192, strict=True)


class Citation(StrictModel):
    source_id: Text
    quote: Text


class Claim(StrictModel):
    text: Text
    evidence: list[Citation] = Field(min_length=1)


class PublicOpinionDetails(StrictModel):
    type: Literal["public_opinion"]
    issues: list[Claim]
    sentiment: Literal["positive", "negative", "neutral", "mixed", "unknown"]
    sentiment_evidence: list[Citation]
    controversies: list[Claim]
    communication_advice: list[Text]


class TimelineItem(Claim):
    source_id: Text
    time: Text


class HotEventsDetails(StrictModel):
    type: Literal["hot_events"]
    event_summary: Text
    timeline: list[TimelineItem]
    focal_points: list[Claim]
    contradictions: list[Claim]
    information_gaps: list[Text]
    trend: Optional[Claim] = None


class Translation(StrictModel):
    target_language: Language
    text: Text


class CrossCulturalDetails(StrictModel):
    type: Literal["cross_cultural"]
    cultural_contexts: list[Claim]
    ambiguities: list[Claim]
    expression_suggestions: list[Text]
    translation: Optional[Translation] = None


class EthnicGovernanceDetails(StrictModel):
    type: Literal["ethnic_governance"]
    issues: list[Claim]
    misunderstandings: list[Claim]
    conflict_expressions: list[Claim]
    coordination_advice: list[Text]


class SmartGovernmentDetails(StrictModel):
    type: Literal["smart_government"]
    answer: Text
    policy_claims: list[Claim]
    processing_points: list[Claim]
    insufficient_materials: bool


class CyberSecurityDetails(StrictModel):
    type: Literal["cyber_security"]
    risk_types: list[Claim]
    suspicious_signs: list[Claim]
    handling_advice: list[Text]


Details = Annotated[Union[PublicOpinionDetails, HotEventsDetails, CrossCulturalDetails,
                         EthnicGovernanceDetails, SmartGovernmentDetails, CyberSecurityDetails],
                    Field(discriminator="type")]


class ScenarioOutput(StrictModel):
    scenario_code: ScenarioCode
    summary: Text
    risk_level: Risk
    evidence: list[Citation]
    recommendations: list[Text]
    limitations: list[Text]
    insufficient_materials: bool
    details: Details


class PublicOpinionOutput(ScenarioOutput):
    scenario_code: Literal["public_opinion"]
    details: PublicOpinionDetails


class HotEventsOutput(ScenarioOutput):
    scenario_code: Literal["hot_events"]
    details: HotEventsDetails


class CrossCulturalOutput(ScenarioOutput):
    scenario_code: Literal["cross_cultural"]
    details: CrossCulturalDetails


class EthnicGovernanceOutput(ScenarioOutput):
    scenario_code: Literal["ethnic_governance"]
    details: EthnicGovernanceDetails


class SmartGovernmentOutput(ScenarioOutput):
    scenario_code: Literal["smart_government"]
    details: SmartGovernmentDetails


class CyberSecurityOutput(ScenarioOutput):
    scenario_code: Literal["cyber_security"]
    details: CyberSecurityDetails


class ObservedStep(StrictModel):
    id: Identifier
    label: Identifier
    detail: InputText
    evidence_refs: list[Identifier] = Field(default_factory=list, max_length=50)
    rule_refs: list[Identifier] = Field(default_factory=list, max_length=100)


class AuditRule(StrictModel):
    id: Identifier
    text: InputText


class AuditSubject(StrictModel):
    id: Identifier
    type: Identifier
    version: Identifier


class AuditRequest(StrictModel):
    model_id: Identifier
    audit_type: Literal["reasoning", "full_chain"]
    subject: AuditSubject
    steps: list[ObservedStep] = Field(max_length=100)
    evidence: list[Material] = Field(max_length=50)
    rules: list[AuditRule] = Field(max_length=100)

    @model_validator(mode="after")
    def validate_ids(self):
        for collection in (self.steps, self.evidence, self.rules):
            ids = [item.id for item in collection]
            if len(ids) != len(set(ids)):
                raise ValueError("步骤、证据及规则各自 ID 必须唯一")
        evidence = {item.id for item in self.evidence}
        rules = {item.id for item in self.rules}
        for step in self.steps:
            if not set(step.evidence_refs) <= evidence or not set(step.rule_refs) <= rules:
                raise ValueError("输入步骤引用不存在的证据或规则")
        total = sum(len(s.detail) for s in self.steps) + sum(len(e.content) for e in self.evidence)
        if total + sum(len(r.text) for r in self.rules) > 100000:
            raise ValueError("步骤、证据及规则合计不能超过 100000 字符")
        return self


class AuditFinding(StrictModel):
    step_id: Text
    rule_id: Text
    description: Text
    evidence: list[Citation] = Field(min_length=1)


class AuditCheck(StrictModel):
    step_id: Text
    rule_id: Text
    assessment: Literal["consistent", "risk", "unverifiable"]
    reason: Text
    evidence: list[Citation]


class ReasoningAuditOutput(StrictModel):
    audit_type: Literal["reasoning"]
    subject: AuditSubject
    risk_nodes: list[AuditFinding]
    conclusion: Text
    limitations: list[Text]
    verification_state: Literal["unverified"]


class ChainAuditOutput(StrictModel):
    audit_type: Literal["full_chain"]
    subject: AuditSubject
    checks: list[AuditCheck]
    risks: list[AuditFinding]
    gaps: list[Text]
    conclusion: Text
    limitations: list[Text]
    verification_state: Literal["unverified"]
