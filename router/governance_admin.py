"""管理员配置治理任务模型、业务库连接和调用归属。"""

from typing import Literal, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from llm.models import ModelNotFoundError
from router.governance import success
from schemas.auth import AdminTokenPublic
from schemas.calls import CallPublic
from service.auth import InvalidSetupSecretError


router = APIRouter(prefix="/api/v1/admin", tags=["governance-admin"])
GovernanceKind = Literal["governance-value", "governance-anomaly", "governance-risk"]


class ModelBinding(BaseModel):
    """管理员为一种治理任务指定已创建的本地模型。"""

    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1)


class BusinessDatabaseInput(BaseModel):
    """业务 MySQL 的五项私有连接参数。"""

    model_config = ConfigDict(extra="forbid")
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)
    database: str = Field(min_length=1)
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


class TokenLabelInput(BaseModel):
    """为尚未标注的令牌登记持有人或用途。"""

    model_config = ConfigDict(extra="forbid")
    label: str = Field(min_length=1, max_length=100)


def require_admin(request: Request, x_setup_secret: Optional[str] = Header(default=None)) -> None:
    """复用现有初始化密钥校验管理员身份。"""
    try:
        request.app.state.auth_service._require_setup_secret(x_setup_secret)
    except InvalidSetupSecretError as exc:
        raise HTTPException(403, detail={"code": "invalid_setup_secret"}) from exc


@router.get("/governance/models", dependencies=[Depends(require_admin)])
def list_model_bindings(request: Request):
    """列出 SQLite 中当前三类治理任务的模型绑定。"""
    return success(request, request.app.state.governance_config.list_model_ids())


@router.put("/governance/models/{kind}", dependencies=[Depends(require_admin)])
def set_model_binding(request: Request, kind: GovernanceKind, payload: ModelBinding):
    """校验本地模型存在后写入绑定，新任务使用新绑定。"""
    try:
        request.app.state.model_manager.get_model(payload.model_id)
    except ModelNotFoundError as exc:
        raise HTTPException(404, detail={"code": "model_not_found"}) from exc
    request.app.state.governance_config.set_model_id(kind, payload.model_id)
    return success(request, {"kind": kind, "model_id": payload.model_id})


@router.get("/governance/business-database", dependencies=[Depends(require_admin)])
def get_business_database(request: Request):
    """只返回可公开给管理员的连接元数据，绝不返回密码。"""
    details = request.app.state.governance_config.get_business_database()
    if details is None:
        return success(request, None)
    return success(request, {"host": details["host"], "port": details["port"],
                             "database": details["database_name"],
                             "username": details["username"], "password_configured": True})


@router.put("/governance/business-database", dependencies=[Depends(require_admin)])
def set_business_database(request: Request, payload: BusinessDatabaseInput):
    """用现有 Fernet 密钥加密密码并保存连接配置；不立即连接 MySQL。"""
    cipher = request.app.state.model_manager.cipher
    request.app.state.governance_config.set_business_database(
        host=payload.host, port=payload.port, database_name=payload.database,
        username=payload.username,
        encrypted_password=cipher.encrypt(payload.password.encode("utf-8")).decode("ascii"),
    )
    return get_business_database(request)


@router.get("/tokens", dependencies=[Depends(require_admin)])
def list_tokens(request: Request):
    """列出令牌 ID 和标签，便于标记迁移前的旧令牌。"""
    records = request.app.state.auth_service.repository.list()
    return success(request, [AdminTokenPublic.from_record(record).model_dump(mode="json")
                             for record in records])


@router.patch("/tokens/{token_id}/label", dependencies=[Depends(require_admin)])
def set_token_label(request: Request, token_id: str, payload: TokenLabelInput):
    """只允许设置一次标签，避免同一令牌的历史调用换名。"""
    repository = request.app.state.auth_service.repository
    record = repository.get(token_id)
    if record is None:
        raise HTTPException(404, detail={"code": "token_not_found"})
    if record.label or not repository.set_label_if_empty(token_id, payload.label):
        raise HTTPException(409, detail={"code": "token_already_labeled"})
    record = repository.get(token_id)
    return success(request, AdminTokenPublic.from_record(record).model_dump(mode="json"))


@router.get("/calls", dependencies=[Depends(require_admin)])
def list_admin_calls(request: Request, limit: int = Query(default=50, ge=1, le=200),
                     offset: int = Query(default=0, ge=0)):
    """跨令牌查询调用记录，附令牌 ID 和持有人标签。"""
    rows = request.app.state.model_manager.calls.list_for_admin(limit=limit, offset=offset)
    return success(request, [{**CallPublic.from_record(row["record"]).model_dump(mode="json"),
                              "token_id": row["token_id"], "token_label": row["token_label"]}
                             for row in rows])
