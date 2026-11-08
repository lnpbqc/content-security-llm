"""定义三类治理任务要求模型逐条返回的结构。"""

from typing import List, Literal, Optional

from llm.assessment import AssessmentModel

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm.outputs import AnomalyType, RiskCategory, RiskLevel, ValueDimensionName


class StrictOutput(BaseModel):
    """拒绝模型返回约定以外的字段。"""
    model_config = ConfigDict(extra="forbid")


class ValueDimensionOutput(StrictOutput):
    """一条样本在一个价值维度上的评分和证据。"""
    name: ValueDimensionName
    score: float = Field(ge=0, le=100, allow_inf_nan=False)
    reason: str = Field(min_length=1)
    evidence: List[str] = Field(min_length=1, description="至少一段正文中逐字出现的连续引用。0分也需要引用评估所依据的文本，不能使用空数组。")


class ValueSampleOutput(AssessmentModel):
    """价值分析的单条模型输出；综合分由服务端计算。"""
    sample_id: str = Field(min_length=1)
    status: Literal["scored", "unavailable"]
    dimensions: List[ValueDimensionOutput] = Field(max_length=5)
    unavailable_reason: Optional[str] = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def check_status(self) -> "ValueSampleOutput":
        """要求不可评估样本只有原因、没有虚构的维度分数。"""
        if self.status == "unavailable" and (self.dimensions or not self.unavailable_reason):
            raise ValueError("unavailable output requires a reason and no scores")
        if self.status == "scored" and self.unavailable_reason:
            raise ValueError("scored output cannot have an unavailable reason")
        return self


class FieldChangeOutput(StrictOutput):
    """异常候选建议修改的一个字段。"""
    field: str = Field(min_length=1)
    before: str
    after: str = Field(min_length=1)


class ValidationOutput(StrictOutput):
    """模型给出的校验项；服务端仍会自行校验。"""
    name: str = Field(min_length=1)
    passed: bool


class AnomalyFindingOutput(StrictOutput):
    """单条异常发现及其候选字段变更。"""
    type: AnomalyType
    field: str = Field(min_length=1)
    quote: str = Field(min_length=1, description="正文中支持异常判断的连续原文；仅引用标签本身不能证明主题冲突。")
    ruleId: str = Field(min_length=1)
    reason: str = Field(min_length=1, description="解释字段与正文的具体冲突；近义、合理上下级标签及可选措辞优化不构成异常。")
    fieldChanges: List[FieldChangeOutput]
    validationResults: List[ValidationOutput]


class AnomalyLabelJudgment(AssessmentModel):
    """先判断主题关系，不要求寻找异常或给出替换标签。"""
    sampleId: str = Field(min_length=1)
    sampleRevisionId: str = Field(min_length=1)
    relation: Literal["compatible", "conflict", "uncertain"]
    textTopic: str = Field(min_length=1, description="正文的主要主题；无法确定时说明不足。")
    labelMeaning: str = Field(min_length=1, description="原标签允许覆盖的含义范围，不要求标签与正文同样具体。")
    quote: str = Field(description="正文中的连续引用；无法引用时为空字符串。")
    reason: str = Field(min_length=1, description="解释为什么主题属于、不属于或无法确定属于标签范围。")


class AnomalySampleOutput(AssessmentModel):
    """异常治理的单条模型输出。"""
    sampleId: str = Field(min_length=1)
    sampleRevisionId: str = Field(min_length=1)
    status: Literal["assessed", "unavailable"]
    findings: List[AnomalyFindingOutput]
    unavailableReason: Optional[str] = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def check_status(self) -> "AnomalySampleOutput":
        """要求不可评估时给出原因且不返回异常发现。"""
        if self.status == "unavailable" and (self.findings or not self.unavailableReason):
            raise ValueError("unavailable output requires a reason and no findings")
        return self


class AnomalyRepairOutput(AnomalySampleOutput):
    """仅用于主题冲突已通过第一步判断的修复建议。"""


class RiskFindingOutput(StrictOutput):
    """单条风险发现、建议等级和证据引用。"""
    category: RiskCategory
    suggestedLevel: RiskLevel
    quote: str = Field(min_length=1, description="逐字引用当前遮蔽正文中支持此风险的连续片段")
    reason: str = Field(min_length=1)
    ruleId: str = Field(min_length=1)
    ruleVersion: str = Field(min_length=1)
    evidenceRefs: List[str] = Field(min_length=1)


class RiskSampleOutput(AssessmentModel):
    """风险分级的单条模型输出；汇总等级由服务端计算。"""
    sampleId: str = Field(min_length=1)
    sampleRevisionId: str = Field(min_length=1)
    status: Literal["assessed", "unassessable"]
    findings: List[RiskFindingOutput]
    reviewRecommended: bool
    unassessableReason: Optional[str] = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def check_status(self) -> "RiskSampleOutput":
        """要求无法评估时给出原因且不返回风险发现。"""
        if self.status == "unassessable" and (self.findings or not self.unassessableReason):
            raise ValueError("unassessable output requires a reason and no findings")
        return self


class SemanticWeights(StrictOutput):
    safe: float = Field(ge=0, le=1, allow_inf_nan=False)
    neutral: float = Field(ge=0, le=1, allow_inf_nan=False)
    risk: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def check_total(self) -> "SemanticWeights":
        if min(abs(self.safe + self.neutral + self.risk - 1), abs(self.safe + self.neutral + self.risk)) > 0.001:
            raise ValueError("semantic weights must sum to 1")
        return self


class SemanticEvidence(StrictOutput):
    quote: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class MultilingualSampleOutput(AssessmentModel):
    sampleId: str = Field(min_length=1)
    sampleRevisionId: str = Field(min_length=1)
    status: Literal["assessed", "unassessable"]
    semanticWeights: Optional[SemanticWeights] = None
    riskScore: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    evidence: List[SemanticEvidence] = Field(default_factory=list, max_length=10)
    unassessableReason: Optional[str] = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def check_status(self) -> "MultilingualSampleOutput":
        if self.status == "assessed":
            if (self.semanticWeights is None or self.riskScore is None or self.unassessableReason or
                    abs(self.semanticWeights.safe + self.semanticWeights.neutral + self.semanticWeights.risk - 1) > 0.001):
                raise ValueError("assessed output requires weights and risk score")
            if self.riskScore >= 0.5 and not self.evidence:
                raise ValueError("medium or high risk requires source evidence")
        elif ((self.semanticWeights is not None and any((self.semanticWeights.safe, self.semanticWeights.neutral, self.semanticWeights.risk)))
              or self.riskScore not in (None, 0) or self.evidence or not self.unassessableReason):
            raise ValueError("unassessable output requires only a reason")
        return self
