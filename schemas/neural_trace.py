"""前端可调用的溯源任务与结果契约；CRS 和观测性激活差异分别返回。"""

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class TraceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$", max_length=100,
                         description="从 /cases 选取；不接收本地路径或任意执行参数")


class TraceCase(BaseModel):
    case_id: str
    query: str
    gold_answer: str
    poison_target_answer: str


class TraceTask(BaseModel):
    task_id: str
    case_id: str
    status: Literal["pending", "running", "succeeded", "failed"]
    stage: str
    error_message: str | None
    created_at: str
    updated_at: str
    started_at: str | None
    finished_at: str | None


class TracePage(BaseModel):
    items: list[TraceTask]
    total: int


class TraceSection(BaseModel):
    summary: dict[str, Any]
    rows: list[dict[str, Any]]


class TraceResult(BaseModel):
    schema_version: Literal["neural-trace-result-v1"] = "neural-trace-result-v1"
    task_id: str
    case: TraceCase
    model: dict[str, Any]
    input_hashes: dict[str, str]
    document_trace: TraceSection
    layer_scan: TraceSection
    activation_difference: TraceSection
    component_scan: TraceSection
    baseline_generation: dict[str, Any]
    scoring_patch_scope: Literal["candidate_prediction_span"] = "candidate_prediction_span"
    generation_patch_scope: Literal["generation_prompt_last_token"] = "generation_prompt_last_token"


T = TypeVar("T")


class TraceResponse(BaseModel, Generic[T]):
    code: int = 0
    message: str = "success"
    data: T
    trace_id: str
    timestamp: str
