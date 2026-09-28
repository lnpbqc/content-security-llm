"""Server-owned Pydantic result types available to HTTP callers."""

from typing import Dict, List, Optional, Type

from pydantic import BaseModel, ConfigDict


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
    riskLevel: str
    reconstruction: str


class SemanticRiskFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    level: str
    reason: str


class SemanticRiskEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quote: str
    feature: str


class SemanticRiskOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: List[SemanticRiskFinding]
    evidence: List[SemanticRiskEvidence]


class ValueScoreDimension(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    score: float
    reason: str
    evidence: str


class ValueScoreOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimensions: List[ValueScoreDimension]
    tier: str
    unavailableReason: Optional[str] = None


class AnomalyFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    field: str
    reason: str
    quote: str


class FieldChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    before: str
    after: str


class AnomalyRepairOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: List[AnomalyFinding]
    fieldChanges: List[FieldChange]


class FullChainAuditOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conclusion: str


class ReasoningStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str


class RiskNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str


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
