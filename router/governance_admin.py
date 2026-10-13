"""管理员配置治理任务模型和调用归属。"""

from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from llm.models import ModelNotFoundError
from router.governance import success
from schemas.auth import AdminTokenPublic, TokenStatusUpdate
from schemas.calls import CallPublic
from router.dependencies import require_admin
from service.business_users import BusinessUsersTimeoutError, BusinessUsersUnavailableError


router = APIRouter(prefix="/api/v1/admin", tags=["governance-admin"])
GovernanceKind = Literal["governance-value", "governance-anomaly", "governance-risk"]


class ModelBinding(BaseModel):
    """管理员为一种治理任务指定已创建的本地模型。"""

    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1)


class TokenLabelInput(BaseModel):
    """为尚未标注的令牌登记持有人或用途。"""

    model_config = ConfigDict(extra="forbid")
    label: str = Field(min_length=1, max_length=100)


@router.get("/business-users", dependencies=[Depends(require_admin)])
def list_business_users(request: Request, page: int = Query(default=1, ge=1),
                        page_size: int = Query(default=20, ge=1, le=100),
                        keyword: Optional[str] = Query(default=None)):
    """读取原业务用户目录，供管理端选择独立模型令牌的持有人。"""
    try:
        result = request.app.state.business_user_service.list_users(page, page_size, keyword)
    except BusinessUsersTimeoutError as exc:
        raise HTTPException(504, detail={"message": str(exc)}) from exc
    except BusinessUsersUnavailableError as exc:
        raise HTTPException(502, detail={"message": str(exc)}) from exc
    return success(request, result)


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


@router.get("/tokens", dependencies=[Depends(require_admin)])
def list_tokens(request: Request):
    """列出令牌 ID 和标签，便于标记迁移前的旧令牌。"""
    records = request.app.state.auth_service.repository.list()
    return success(request, [AdminTokenPublic.from_record(record).model_dump(mode="json")
                             for record in records])


@router.patch("/tokens/{token_id}/status", dependencies=[Depends(require_admin)])
def set_token_status_by_id(request: Request, token_id: str, payload: TokenStatusUpdate):
    """按令牌 ID 启用或禁用，并返回管理员可见的最新状态。"""
    record = request.app.state.auth_service.repository.set_enabled_by_id(token_id, payload.enabled)
    if record is None:
        raise HTTPException(404, detail={"code": "token_not_found"})
    return success(request, AdminTokenPublic.from_record(record).model_dump(mode="json"))


@router.delete("/tokens/{token_id}", dependencies=[Depends(require_admin)])
def delete_token(request: Request, token_id: str):
    """软删指定令牌，并保留历史调用可关联的令牌元数据。"""
    if not request.app.state.auth_service.repository.soft_delete(token_id):
        raise HTTPException(404, detail={"code": "token_not_found"})
    return success(request, {"id": token_id, "deleted": True})


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
