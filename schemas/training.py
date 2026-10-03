"""可信 Python 训练任务的请求、默认值和张量契约。"""

import ast
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator


TASK_TYPES = ["multiclass", "binary", "multilabel", "regression", "custom"]
OPTIMIZERS = ["AdamW", "Adam", "SGD"]
SCHEDULERS = ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"]
LOSSES = {"multiclass": "CrossEntropyLoss", "binary": "BCEWithLogitsLoss",
          "multilabel": "BCEWithLogitsLoss", "regression": "MSELoss", "custom": "custom"}
METRIC_NAMES = ("precision", "accuracy", "f1", "recall")


class TensorSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dtype: Literal["float16", "bfloat16", "float32", "float64", "int8", "uint8",
                   "int16", "int32", "int64", "bool"]
    shape: list[StrictInt | None]

    @field_validator("shape")
    @classmethod
    def positive_dimensions(cls, value):
        if any(dimension is not None and dimension < 1 for dimension in value):
            raise ValueError("shape 的固定维度必须为正整数")
        return value


class TensorContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inputs: dict[str, TensorSpec] = Field(min_length=1)
    targets: TensorSpec | dict[str, TensorSpec]
    outputs: TensorSpec | dict[str, TensorSpec]

    @model_validator(mode="after")
    def nonempty_mappings(self):
        for value in (self.inputs, self.targets, self.outputs):
            if isinstance(value, dict) and (not value or any(not key for key in value)):
                raise ValueError("张量字典必须非空，且字段名不能为空")
        return self


class TrainingCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str = Field(min_length=1, max_length=100)
    dataset_id: int = Field(ge=1)
    version_id: str = Field(min_length=1)
    python_source: str = Field(min_length=1)
    contract: TensorContract
    model_id: str | None = None
    base_version: str | None = None
    target_version: str | None = None
    method: str = "custom-pytorch"
    epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=0.0002, gt=0)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    optimizer: Literal["AdamW", "Adam", "SGD"] = "AdamW"
    optimizer_params: dict[str, Any] = Field(default_factory=lambda: {"weight_decay": 0.0})
    scheduler: Literal["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"] | None = None
    scheduler_params: dict[str, Any] = Field(default_factory=dict)
    validation_ratio: float = Field(default=0.2, gt=0, lt=1)
    task_type: Literal["multiclass", "binary", "multilabel", "regression", "custom"] = "multiclass"
    loss: Literal["auto", "CrossEntropyLoss", "BCEWithLogitsLoss", "MSELoss", "custom"] = "auto"
    metrics: Literal["auto", "custom", "none"] = "auto"
    average: Literal["auto", "binary", "macro", "micro", "weighted", "samples"] = "auto"
    threshold: float = Field(default=0.5, gt=0, lt=1)
    pos_label: Literal[0, 1] = 1
    model_params: dict[str, Any] = Field(default_factory=dict)
    data_params: dict[str, Any] = Field(default_factory=dict)
    loss_params: dict[str, Any] = Field(default_factory=dict)
    metric_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "version_id")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("字段不能为空白")
        return value.strip()

    @field_validator("python_source")
    @classmethod
    def syntax_only(cls, value):
        try:
            ast.parse(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("Python 源码语法错误: {}".format(exc)) from exc
        return value

    @model_validator(mode="after")
    def resolve_defaults(self):
        if self.loss == "auto":
            self.loss = LOSSES[self.task_type]
        if self.loss != "custom" and self.loss != LOSSES[self.task_type]:
            raise ValueError("内置 loss 与 task_type 不匹配")
        if self.average == "auto":
            self.average = "binary" if self.task_type == "binary" else "macro"
        if self.metrics == "auto" and self.task_type in ("regression", "custom"):
            self.metrics = "none"
        if self.metrics == "auto":
            if self.average == "binary" and self.task_type != "binary":
                raise ValueError("binary 平均方式只适用于二分类")
            if self.average == "samples" and self.task_type != "multilabel":
                raise ValueError("samples 平均方式只适用于多标签")
        if "lr" in self.optimizer_params:
            raise ValueError("学习率请使用 learning_rate，不能在 optimizer_params 中重复传入")
        self.optimizer_params.setdefault("weight_decay", 0.0)
        if self.loss != "custom" and self.loss_params.get("reduction", "mean") != "mean":
            raise ValueError("内置损失只支持 mean reduction，以保证按样本量汇总")
        if self.scheduler == "StepLR" and "step_size" not in self.scheduler_params:
            self.scheduler_params["step_size"] = 1
        if self.scheduler == "CosineAnnealingLR" and "T_max" not in self.scheduler_params:
            self.scheduler_params["T_max"] = self.epochs
        for params in (self.optimizer_params, self.scheduler_params, self.model_params,
                       self.data_params, self.loss_params, self.metric_params):
            try:
                json.dumps(params, allow_nan=False)
            except (ValueError, TypeError) as exc:
                raise ValueError("自定义参数必须为有限的 JSON 值") from exc
        return self
