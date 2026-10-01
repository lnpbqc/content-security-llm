"""定义三类治理任务要求模型逐条返回的结构。"""

from typing import List, Literal, Optional

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
    evidence: List[str]


class ValueSampleOutput(StrictOutput):
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
    quote: str = Field(min_length=1)
    ruleId: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    fieldChanges: List[FieldChangeOutput]
    validationResults: List[ValidationOutput]


class AnomalySampleOutput(StrictOutput):
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


class RiskFindingOutput(StrictOutput):
    """单条风险发现、建议等级和证据引用。"""
    category: RiskCategory
    suggestedLevel: RiskLevel
    reason: str = Field(min_length=1)
    ruleId: str = Field(min_length=1)
    ruleVersion: str = Field(min_length=1)
    evidenceRefs: List[str] = Field(min_length=1)


class RiskSampleOutput(StrictOutput):
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
