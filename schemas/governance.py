"""治理公开响应契约；历史快照中的额外字段继续保留。"""

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class PublicModel(BaseModel):
    model_config = ConfigDict(extra="allow")


T = TypeVar("T")
RiskLevel = Literal["safe", "low", "medium", "high"]


class GovernanceResponse(PublicModel, Generic[T]):
    code: int
    message: str
    data: T
    trace_id: str
    timestamp: str


class RiskWeightAccuracy(PublicModel):
    accuracyPercent: float | None = Field(description="0～100 百分比；没有标准标签时为 null")
    correctCount: int
    totalCount: int
    validCount: int | None = Field(default=None, description="本批次有效检测样本总数；与 totalCount（有标准档位标注数）构成双口径分母")


class MultilingualScopePublic(PublicModel):
    dataset_id: int
    version_id: str
    languages: list[str]
    analysis_mode: Literal["semantic_weighting", "risk_tiering"]
    scheme_id: str


class MultilingualTask(PublicModel):
    task_id: str
    status: Literal["pending", "running", "succeeded", "failed"]
    result_id: str | None
    input: MultilingualScopePublic
    model_id: str
    created_at: str
    error_message: str | None


class SemanticWeights(PublicModel):
    safe: float
    neutral: float
    risk: float


class SemanticEvidence(PublicModel):
    quote: str
    reason: str


class MultilingualSample(PublicModel):
    id: str
    dataset_id: int
    version_id: str
    revision_id: str
    text: str
    language: str
    status: Literal["assessed", "unassessable"]
    semantic_weights: SemanticWeights | None = Field(description="单样本权重为 0～1，三项合计 1")
    risk_score: float | None = Field(description="0～1 风险分；无法评估时为 null")
    risk_level: Literal["safe", "low", "medium", "high", "unknown"]
    evidence: list[SemanticEvidence]
    unassessable_reason: str | None = None


class MultilingualSamplePage(PublicModel):
    items: list[MultilingualSample]
    total: int
    page: int
    page_size: int
    total_pages: int


class RiskThresholds(PublicModel):
    low: float
    medium: float
    high: float


class LanguageDistribution(PublicModel):
    language: str
    sample_count: int
    safe_weight: float = Field(description="0～100 百分比")
    neutral_weight: float = Field(description="0～100 百分比")
    risk_weight: float = Field(description="0～100 百分比")


class RiskDistribution(PublicModel):
    level: RiskLevel
    count: int
    percent: float = Field(description="0～100 百分比")


class MultilingualRiskEvidence(SemanticEvidence):
    sample_id: str
    language: str
    semantic_weight: float = Field(description="0～1 风险语义权重")
    risk_score: float
    risk_level: RiskLevel


class MultilingualSummary(PublicModel):
    id: str
    task_id: str
    scope: MultilingualScopePublic
    dataset_name: str
    model_id: str
    finished_at: str
    data_source: str
    scheme_name: str
    thresholds: RiskThresholds
    target_count: int
    valid_count: int
    unassessable_count: int
    failed_count: int
    overall_risk_score: float | None = Field(description="0～1 风险分")
    overall_risk_level: Literal["safe", "low", "medium", "high", "unknown"]
    semantic_consistency: float | None = Field(description="0～1；有效样本覆盖不足两种语言时为 null")
    semantic_consistency_method: str
    high_risk_semantic_weight: float | None = Field(description="0～100 百分比")
    covered_language_count: int
    language_distribution: list[LanguageDistribution]
    risk_distribution: list[RiskDistribution]
    risk_evidence: list[MultilingualRiskEvidence]
    semantic_consistency_delta: float | None = Field(description="0～1 指标的分值差，可为负")
    high_risk_semantic_weight_delta: float | None = Field(description="百分点差，可为负")
