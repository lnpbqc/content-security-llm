"""版本级推理绑定；评估页只提交业务模型与版本，不接收上游地址或密钥。"""
import hashlib
import json
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm.models import InvocationError, ModelNotFoundError
from router.dependencies import require_admin, require_token
from router.governance import success
from utils.time import utc_now

router = APIRouter(prefix="/api/v1", tags=["model-evaluation"])


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


class SimulationTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str = Field(min_length=1, max_length=30)
    system_prompt: str = Field(min_length=1, max_length=6000,
                               description="必须含 {target_knowledge}，创建模拟版本时替换为该任务的编辑目标")

    @model_validator(mode="after")
    def require_target(self):
        if "{target_knowledge}" not in self.system_prompt:
            raise ValueError("策略模板必须包含 {target_knowledge}")
        return self


class Binding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1, max_length=64, description="8001 模型目录中的模型编号")
    mode: Literal["baseline", "simulated_policy", "edited_model"] = "baseline"
    policy_version: str = Field(default="", max_length=100)
    system_prompt: str = Field(default="", max_length=8000,
                               description="仅模拟治理模式填写；作为系统提示发送，不能视为权重编辑")
    simulation_template: SimulationTemplate | None = Field(default=None,
        description="仅基线配置。允许其模拟编辑产物首次评估时自动生成策略绑定")

    @model_validator(mode="after")
    def validate_policy(self):
        if self.mode != "baseline" and self.simulation_template is not None:
            raise ValueError("只有基线绑定可以设置模拟编辑模板")
        if self.mode == "simulated_policy":
            if not self.policy_version.strip() or not self.system_prompt.strip():
                raise ValueError("模拟治理必须填写策略版本和系统提示")
        elif self.system_prompt or self.policy_version:
            raise ValueError("只有模拟治理模式允许填写策略")
        return self


def resolve(request, platform_model_id, version):
    with request.app.state.database.connect() as connection:
        row = connection.execute(
            "SELECT config_json FROM llm_model_evaluation_bindings WHERE platform_model_id=%s AND version=%s",
            (platform_model_id, version)).fetchone()
    if not row:
        raise HTTPException(409, "该模型版本尚未配置评估推理绑定，请在 8001 Swagger 中配置")
    config = json.loads(row["config_json"])
    try:
        model = request.app.state.model_manager.get_model(config["model_id"])
    except ModelNotFoundError as exc:
        raise HTTPException(409, "绑定的推理模型已不存在") from exc
    public = {**config, "platform_model_id": platform_model_id, "version": version,
              "upstream_model_id": model.upstream_model_id, "upstream_name": model.name,
              "base_url": model.base_url, "model_updated_at": model.updated_at.isoformat()}
    return {**public, "binding_hash": digest(public)}


@router.get("/admin/model-evaluation/bindings", dependencies=[Depends(require_admin)])
def list_bindings(request: Request):
    """列出版本绑定；不返回上游密钥。"""
    with request.app.state.database.connect() as connection:
        rows = connection.execute("SELECT platform_model_id, version, config_json, updated_at "
                                  "FROM llm_model_evaluation_bindings ORDER BY platform_model_id, version").fetchall()
    return success(request, [{"platform_model_id": row["platform_model_id"], "version": row["version"], "updated_at": row["updated_at"], "config": json.loads(row["config_json"])} for row in rows])


@router.put("/admin/model-evaluation/bindings/{platform_model_id}/{version}",
            dependencies=[Depends(require_admin)])
def set_binding(request: Request, platform_model_id: int, version: str, payload: Binding):
    """业务模型版本 → 8001 模型。先在模型管理/编辑页登记版本，再在此绑定。"""
    if len(version) > 100:
        raise HTTPException(422, "版本号过长")
    try:
        request.app.state.model_manager.get_model(payload.model_id)
    except ModelNotFoundError as exc:
        raise HTTPException(404, "8001 模型不存在") from exc
    with request.app.state.database.connect() as connection:
        exists = connection.execute("SELECT id FROM model_versions WHERE model_id=%s AND version=%s",
                                    (platform_model_id, version)).fetchone()
        if not exists:
            raise HTTPException(404, "平台模型版本不存在，请先登记版本")
        connection.execute(
            "INSERT INTO llm_model_evaluation_bindings (platform_model_id,version,config_json,updated_at) "
            "VALUES (%s,%s,%s,%s) ON DUPLICATE KEY UPDATE config_json=VALUES(config_json),updated_at=VALUES(updated_at)",
            (str(platform_model_id), version, payload.model_dump_json(), utc_now().isoformat()))
    return success(request, resolve(request, str(platform_model_id), version))


@router.get("/model-evaluation/bindings/{platform_model_id}/{version}", dependencies=[Depends(require_token)])
def get_binding(request: Request, platform_model_id: int, version: str):
    return success(request, resolve(request, str(platform_model_id), version))


class PrepareEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    platform_model_id: int = Field(gt=0)
    version: str = Field(min_length=1, max_length=100)


@router.post("/model-evaluation/prepare-edit", dependencies=[Depends(require_token)])
def prepare_edit(request: Request, payload: PrepareEdit):
    """按管理员预设模板绑定已成功的模拟编辑产物；幂等且不覆盖既有绑定。"""
    platform_id = str(payload.platform_model_id)
    with request.app.state.database.connect() as connection:
        existing = connection.execute("SELECT config_json FROM llm_model_evaluation_bindings "
                                      "WHERE platform_model_id=%s AND version=%s", (platform_id, payload.version)).fetchone()
        if existing:
            return success(request, resolve(request, platform_id, payload.version))
        row = connection.execute(
            "SELECT t.task_id,t.status,t.capability_code,t.input_data,t.result FROM model_versions v "
            "JOIN tasks t ON t.task_id=v.task_id WHERE v.model_id=%s AND v.version=%s",
            (payload.platform_model_id, payload.version)).fetchone()
    if not row or row["status"] != "succeeded" or row["capability_code"] != "knowledge_edit":
        raise HTTPException(409, f"版本 {payload.version} 未配置推理绑定，且不是已完成的模拟编辑产物")
    task_input, result = json.loads(row["input_data"]), json.loads(row["result"])
    baseline = result.get("before_version")
    target = task_input.get("target_knowledge") or task_input.get("targetKnowledge")
    if (str(task_input.get("model_id") or task_input.get("modelId")) != platform_id
            or result.get("after_version") != payload.version or not baseline
            or not isinstance(target, str) or not target.strip()):
        raise HTTPException(409, "编辑任务与模型版本不一致，不能自动绑定")
    base = resolve(request, platform_id, baseline)
    template = base.get("simulation_template")
    if base["mode"] != "baseline" or not template:
        raise HTTPException(409, f"基线 {baseline} 尚未配置模拟编辑模板，请管理员在 8001 配置一次")
    prompt = template["system_prompt"].replace("{target_knowledge}", target)
    if len(prompt) > 8000:
        raise HTTPException(400, "编辑目标与模板合并后过长，请缩短编辑目标")
    binding = Binding(model_id=base["model_id"], mode="simulated_policy",
                      policy_version=template["version"] + ":" + row["task_id"], system_prompt=prompt)
    with request.app.state.database.connect() as connection:
        connection.execute(
            "INSERT IGNORE INTO llm_model_evaluation_bindings (platform_model_id,version,config_json,updated_at) "
            "VALUES (%s,%s,%s,%s)", (platform_id, payload.version, binding.model_dump_json(), utc_now().isoformat()))
    return success(request, resolve(request, platform_id, payload.version))


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    platform_model_id: int
    version: str = Field(min_length=1, max_length=100)
    binding_hash: str = Field(pattern="^[a-f0-9]{64}$")
    question: str = Field(min_length=1, max_length=16000)


@router.post("/model-evaluation/answer")
def answer(request: Request, payload: Answer, token_hash: str = Depends(require_token)):
    """一次真实推理；绑定或上游配置变化时拒绝继续，避免同一评估混用配置。"""
    config = resolve(request, str(payload.platform_model_id), payload.version)
    if config["binding_hash"] != payload.binding_hash:
        raise HTTPException(409, "评估期间推理配置发生变化，请重新发起评估")
    slots = request.app.state.analysis_service.slots
    if not slots.acquire(blocking=False):
        raise HTTPException(429, "模型服务繁忙，请稍后重试")
    calls = []
    try:
        output = request.app.state.model_manager.invoke(
            config["model_id"], payload.question, None, response_type="model-evaluation-answer",
            token_hash=token_hash, temperature=0, max_tokens=1024, timeout=90, max_retries=0,
            system_prompt=config["system_prompt"] or None, on_record=lambda row: calls.append(row.id))
    except InvocationError as exc:
        raise HTTPException(exc.status_code, detail={"code": exc.code, "message": str(exc)}) from exc
    finally:
        slots.release()
    return success(request, {"output": output, "call_ids": calls, "binding_hash": config["binding_hash"]})
