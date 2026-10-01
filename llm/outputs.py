"""Server-owned Pydantic result types available to HTTP callers."""

from typing import Dict, List, Literal, Optional, Type

from pydantic import BaseModel, ConfigDict, Field, model_validator


ValueDimensionName = Literal["文化价值", "信息价值", "稀缺性", "可信度", "代表性"]
AnomalyType = Literal["标签异常", "重复记录", "格式异常", "字段缺失"]
RiskCategory = Literal["个人信息暴露", "误导信息", "仇恨歧视", "违法有害"]
RiskLevel = Literal["HIGH", "MEDIUM", "LOW", "NOTICE"]


class TextAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str

class JudgementAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float

    reason: str

class ModelRiskGovernanceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    originalOutput: str
    governedOutput: str
    reason: str
    riskLevel: Literal["high", "medium", "low"]
    reconstruction: Optional[str] = None


class SemanticRiskFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: RiskCategory
    suggestedLevel: RiskLevel
    reason: str
    ruleId: str
    evidenceRefs: List[str]


class SemanticRiskEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quote: str
    feature: str


class SemanticRiskOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: List[SemanticRiskFinding]
    evidence: List[SemanticRiskEvidence]
    primaryCategory: str
    maximumSuggestedLevel: RiskLevel


class ValueScoreDimension(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ValueDimensionName
    score: Optional[float] = Field(default=None, ge=0, le=100)
    reason: str
    evidence: List[str]


class ValueScoreOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimensions: List[ValueScoreDimension] = Field(min_length=5, max_length=5)
    tier: str
    unavailableReason: Optional[str] = None

    @model_validator(mode="after")
    def validate_dimensions(self) -> "ValueScoreOutput":
        expected = {"文化价值", "信息价值", "稀缺性", "可信度", "代表性"}
        actual = {dimension.name for dimension in self.dimensions}
        if actual != expected:
            raise ValueError("dimensions must contain exactly the five required value dimensions")
        return self


class AnomalyFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: AnomalyType
    field: str
    reason: str
    quote: str
    ruleId: str


class AnomalySource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    line: int


class FieldChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    before: str
    after: str


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool


class AnomalyRepairOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: List[AnomalyFinding]
    primaryType: str
    source: AnomalySource
    fieldChanges: List[FieldChange]
    reason: str
    validationResults: List[ValidationResult]


class FullChainCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    passed: bool
    reason: str


class FullChainGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    reason: str


class FullChainAuditOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conclusion: str
    checks: List[FullChainCheck]
    gaps: List[FullChainGap]


class ReasoningStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    detail: str
    occurredAt: str
    verificationState: str


class RiskNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stepId: str
    ruleRef: str
    description: str
    evidenceRefs: List[str]


class ReasoningAuditOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: List[ReasoningStep]
    riskNodes: List[RiskNode]
    auditResult: str


_response_types: Dict[str, Type[BaseModel]] = {
    "text_answer": TextAnswer,
    "judgement": JudgementAnswer,
    "model_risk_governance": ModelRiskGovernanceOutput,
    "semantic_risk": SemanticRiskOutput,
    "value_score": ValueScoreOutput,
    "anomaly_repair": AnomalyRepairOutput,
    "full_chain_audit": FullChainAuditOutput,
    "reasoning_audit": ReasoningAuditOutput,
}


def register_response_type(name: str, response_model: Type[BaseModel]) -> None:
    """Register a business result class under an HTTP-safe name."""
    if not name or not name.replace("_", "").isalnum():
        raise ValueError("Response type name must contain letters, digits, or underscores")
    if not isinstance(response_model, type) or not issubclass(response_model, BaseModel):
        raise TypeError("response_model must be a Pydantic BaseModel class")
    if name in _response_types and _response_types[name] is not response_model:
        raise ValueError("Response type name is already registered")
    _response_types[name] = response_model


def get_response_type(name: str) -> Optional[Type[BaseModel]]:
    return _response_types.get(name)


def list_response_types() -> List[str]:
    return sorted(_response_types)
