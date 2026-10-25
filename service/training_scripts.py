"""服务器登记的脚本入口及其输入、参数校验；不在 API 内导入训练脚本。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError, model_validator

from config import PROJECT_ROOT


class CifarInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    download: StrictBool = True


class CifarParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=128, ge=1)
    learning_rate: float = Field(default=0.001, gt=0)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    device: Literal["cuda", "cpu"] = "cuda"
    train_samples: int = Field(default=10000, ge=100, le=49900)
    validation_samples: int = Field(default=2000, ge=100, le=49900)

    @model_validator(mode="after")
    def disjoint_sample_counts(self):
        if self.train_samples + self.validation_samples > 50000:
            raise ValueError("训练与验证样本合计不能超过 CIFAR-100 训练集的 50000 条")
        return self


@dataclass(frozen=True)
class ScriptDefinition:
    name: str
    description: str
    path: Path
    inputs: type[BaseModel]
    parameters: type[BaseModel]

    def public(self, script_id):
        return {"id": script_id, "name": self.name, "description": self.description,
                "inputs_schema": self.inputs.model_json_schema(),
                "parameters_schema": self.parameters.model_json_schema()}


class MiniInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MiniParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    epochs: int = Field(default=3, ge=1, le=20)
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    samples: int = Field(default=512, ge=64, le=8192)
    hidden_size: int = Field(default=32, ge=4, le=256)


SCRIPTS = {
    "mini-mlp": ScriptDefinition(
        "迷你 MLP（演示链路）", "合成二维数据的真实小网络训练；秒级完成，用于验证脚本任务、事件与产物链路。",
        PROJECT_ROOT / "training_scripts" / "mini_mlp.py", MiniInputs, MiniParameters),
    "cifar100-cnn": ScriptDefinition(
        "CIFAR-100 小型 CNN", "真实百分类训练示例；固定 CNN/AdamW，使用独立数据缓存。",
        PROJECT_ROOT / "training_scripts" / "cifar100.py", CifarInputs, CifarParameters),
}


def get_script(script_id):
    script = SCRIPTS.get(script_id)
    if script is None:
        raise HTTPException(404, detail={"code": "training_script_not_found", "message": "脚本未登记"})
    return script


def prepare_script_task(payload):
    script = get_script(payload.script_id)
    try:
        inputs = script.inputs.model_validate(payload.inputs).model_dump(mode="json")
        parameters = script.parameters.model_validate(payload.parameters).model_dump(mode="json")
    except ValidationError as exc:
        errors = [{"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
                  for error in exc.errors()]
        message = "脚本输入或参数无效：" + "; ".join(
            "{}: {}".format(".".join(map(str, error["loc"])), error["msg"]) for error in errors)
        raise HTTPException(422, detail={"code": "invalid_script_configuration", "message": message}) from exc
    return {"runner": "script", "name": payload.name, "script_id": payload.script_id,
            "inputs": inputs, "parameters": parameters,
            "python_source": script.path.read_text(encoding="utf-8")}
