"""通用脚本任务与逐行事件协议，不要求具体算法或指标。"""

from typing import Annotated, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictFloat, StrictInt, TypeAdapter, field_validator


class ScriptTrainingCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str = Field(min_length=1, max_length=100)
    script_id: str = Field(min_length=1, max_length=100)
    inputs: dict[str, JsonValue] = Field(default_factory=dict)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("name", "script_id")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("字段不能为空白")
        return value.strip()


T = TypeVar("T")


class TrainingResponse(BaseModel, Generic[T]):
    code: int = 0
    message: str = "success"
    data: T
    trace_id: str
    timestamp: str


class ScriptCatalogEntry(BaseModel):
    id: str
    name: str
    description: str
    inputs_schema: dict[str, JsonValue]
    parameters_schema: dict[str, JsonValue]


class ScriptTaskPublic(ScriptTrainingCreate):
    runner: Literal["script"]
    id: str
    config: dict[str, JsonValue]
    status: Literal["pending", "running", "canceling", "succeeded", "failed", "canceled"]
    source_hash: str
    runtime: dict[str, JsonValue]
    created_at: str
    updated_at: str
    started_at: str | None
    finished_at: str | None
    elapsed_seconds: int
    error_message: str | None
    latest_metrics: dict[str, StrictInt | StrictFloat]
    latest_phase: str | None
    epoch: int | None
    step: int | None
    progress: int | float | None


class TrainingLogPage(BaseModel):
    text: str
    next_after_offset: int
    has_more: bool


class ScriptEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class MetricEvent(ScriptEvent):
    type: Literal["metric"]
    metrics: dict[str, StrictInt | StrictFloat] = Field(min_length=1)
    phase: str | None = Field(default=None, min_length=1)
    epoch: StrictInt | None = Field(default=None, ge=0)
    step: StrictInt | None = Field(default=None, ge=0)

    @field_validator("metrics")
    @classmethod
    def metric_names(cls, value):
        if any(not key.strip() for key in value):
            raise ValueError("指标名称不能为空白")
        return value


class ProgressEvent(ScriptEvent):
    type: Literal["progress"]
    progress: StrictInt | StrictFloat = Field(ge=0, le=100)


class ArtifactEvent(ScriptEvent):
    type: Literal["artifact"]
    filename: str = Field(min_length=1)
    kind: str = Field(default="result", min_length=1)


SCRIPT_EVENT = TypeAdapter(Annotated[MetricEvent | ProgressEvent | ArtifactEvent, Field(discriminator="type")])
