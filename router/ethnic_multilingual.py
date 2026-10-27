"""固定数据集版本的多语种语义权重与风险分层接口。"""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from llm.analysis_contracts import Identifier, Language
from llm.models import ModelNotFoundError
from router.dependencies import require_token
from router.governance import get_service, success
from schemas.governance import (GovernanceResponse, MultilingualSamplePage,
                                MultilingualSummary, MultilingualTask)
from service.governance import GovernanceService
from utils.worker_status import stale_hint


KIND = "ethnic-multilingual"
router = APIRouter(prefix="/api/v1/analysis/ethnic-multilingual", tags=["analysis"],
                   dependencies=[Depends(require_token)])


def _error_message(task: dict, service: GovernanceService) -> str | None:
    """失败原因后补上 worker 过期提示（治理 worker 不热更新代码）。"""
    message = task["error_message"]
    if message:
        message = message + stale_hint(service.repository.database.data_dir)
    return message


class Scope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: int = Field(ge=1)
    version_id: Identifier
    languages: list[Language] = Field(min_length=1, max_length=32)
    analysis_mode: Literal["semantic_weighting", "risk_tiering"]
    scheme_id: Literal["multilingual-v1"] = "multilingual-v1"

    @field_validator("languages")
    @classmethod
    def normalize_languages(cls, languages):
        if len(languages) != len(set(languages)):
            raise ValueError("所选语种不能重复")
        return sorted(languages)


class TaskCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: Identifier
    input: Scope


def query_scope(dataset_id: int = Query(..., ge=1), version_id: Identifier = Query(...),
                languages: list[Language] | None = Query(None),
                analysis_mode: Literal["semantic_weighting", "risk_tiering"] = Query(...),
                scheme_id: Literal["multilingual-v1"] = Query("multilingual-v1"),
                bracket_languages: list[Language] | None = Query(None, alias="languages[]")):
    if languages is None and bracket_languages is None:
        raise RequestValidationError([{"loc": ("query", "languages"), "type": "missing",
                                       "msg": "至少提供 languages 或 languages[]"}])
    if languages is not None and bracket_languages is not None and sorted(languages) != sorted(bracket_languages):
        raise RequestValidationError([{"loc": ("query", "languages"), "type": "value_error",
                                       "msg": "languages 与 languages[] 的内容不一致"}])
    try:
        return Scope(dataset_id=dataset_id, version_id=version_id,
                     languages=languages if languages is not None else bracket_languages,
                     analysis_mode=analysis_mode, scheme_id=scheme_id).model_dump()
    except ValidationError as exc:
        raise RequestValidationError([{**error, "loc": ("query", *error["loc"])}
                                      for error in exc.errors()]) from exc


def get_result(service, result_id, token_hash, summary_only=False):
    read = service.repository.get_result_summary if summary_only else service.repository.get_result
    result = read(result_id, token_hash, KIND)
    if result is None:
        raise HTTPException(404, detail={"message": "结果不存在"})
    return result


@router.post("/tasks", response_model=GovernanceResponse[MultilingualTask], response_model_exclude_unset=True)
def create_task(request: Request, payload: TaskCreate,
                service: GovernanceService = Depends(get_service),
                token_hash: str = Depends(require_token)):
    try:
        task = service.create_multilingual_task(payload.input.model_dump(), payload.model_id, token_hash)
    except ModelNotFoundError as exc:
        raise HTTPException(404, detail={"message": "模型不存在或已删除"}) from exc
    return success(request, {"task_id": task["id"], "status": task["status"],
                             "result_id": task["result_id"], "input": task["scope"],
                             "model_id": task["model_id"], "created_at": task["created_at"],
                             "error_message": _error_message(task, service)})


@router.get("/tasks/{task_id}", response_model=GovernanceResponse[MultilingualTask], response_model_exclude_unset=True)
def get_task(request: Request, task_id: str,
             service: GovernanceService = Depends(get_service),
             token_hash: str = Depends(require_token)):
    task = service.repository.get_task(task_id, token_hash)
    if task is None or task["kind"] != KIND:
        raise HTTPException(404, detail={"message": "任务不存在"})
    return success(request, {"task_id": task["id"], "status": task["status"],
                             "result_id": task["result_id"], "input": task["scope"],
                             "model_id": task["model_id"], "created_at": task["created_at"],
                             "error_message": _error_message(task, service)})


@router.get("/results/latest", response_model=GovernanceResponse[MultilingualSummary | None],
            response_model_exclude_unset=True)
def latest_result(request: Request, scope: dict = Depends(query_scope),
                  service: GovernanceService = Depends(get_service),
                  token_hash: str = Depends(require_token)):
    results = service.repository.list_result_summaries(KIND, token_hash, scope, limit=1)
    return success(request, results[0]["summary"] if results else None)


@router.get("/results", response_model=GovernanceResponse[list[MultilingualSummary]], response_model_exclude_unset=True)
def result_history(request: Request, scope: dict = Depends(query_scope),
                   service: GovernanceService = Depends(get_service),
                   token_hash: str = Depends(require_token)):
    return success(request, [item["summary"] for item in
                             service.repository.list_result_summaries(KIND, token_hash, scope)])


@router.get("/results/{result_id}", response_model=GovernanceResponse[MultilingualSummary],
            response_model_exclude_unset=True)
def result(request: Request, result_id: str,
           service: GovernanceService = Depends(get_service),
           token_hash: str = Depends(require_token)):
    return success(request, get_result(service, result_id, token_hash, summary_only=True)["summary"])


@router.get("/results/{result_id}/samples", response_model=GovernanceResponse[MultilingualSamplePage],
            response_model_exclude_unset=True)
def result_samples(request: Request, result_id: str, page: int = Query(1, ge=1),
                   page_size: int = Query(20, ge=1, le=200),
                   service: GovernanceService = Depends(get_service),
                   token_hash: str = Depends(require_token)):
    rows = get_result(service, result_id, token_hash)["samples"]
    total = len(rows)
    return success(request, {"items": rows[(page - 1) * page_size:page * page_size],
                             "total": total, "page": page, "page_size": page_size,
                             "total_pages": (total + page_size - 1) // page_size})
