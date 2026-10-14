"""固定数据集版本的多语种语义权重与风险分层接口。"""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm.analysis_contracts import Identifier, Language
from llm.models import ModelNotFoundError
from router.dependencies import require_token
from router.governance import get_service, success
from service.governance import GovernanceService


KIND = "ethnic-multilingual"
router = APIRouter(prefix="/api/v1/analysis/ethnic-multilingual", tags=["analysis"],
                   dependencies=[Depends(require_token)])


class Scope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: int = Field(ge=1)
    version_id: Identifier
    languages: list[Language] = Field(min_length=1, max_length=32)
    analysis_mode: Literal["semantic_weighting", "risk_tiering"]
    scheme_id: Literal["multilingual-v1"] = "multilingual-v1"

    @model_validator(mode="after")
    def normalize_languages(self):
        if len(self.languages) != len(set(self.languages)):
            raise ValueError("所选语种不能重复")
        self.languages = sorted(self.languages)
        return self


class TaskCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: Identifier
    input: Scope


def query_scope(dataset_id: int = Query(..., ge=1), version_id: Identifier = Query(...),
                languages: list[Language] = Query(...),
                analysis_mode: Literal["semantic_weighting", "risk_tiering"] = Query(...),
                scheme_id: Literal["multilingual-v1"] = Query("multilingual-v1")):
    try:
        return Scope(dataset_id=dataset_id, version_id=version_id, languages=languages,
                     analysis_mode=analysis_mode, scheme_id=scheme_id).model_dump()
    except ValueError as exc:
        raise HTTPException(422, detail={"message": str(exc)}) from exc


def get_result(service, result_id, token_hash):
    result = service.repository.get_result(result_id, token_hash, KIND)
    if result is None:
        raise HTTPException(404, detail={"message": "结果不存在"})
    return result


@router.post("/tasks")
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
                             "error_message": task["error_message"]})


@router.get("/tasks/{task_id}")
def get_task(request: Request, task_id: str,
             service: GovernanceService = Depends(get_service),
             token_hash: str = Depends(require_token)):
    task = service.repository.get_task(task_id, token_hash)
    if task is None or task["kind"] != KIND:
        raise HTTPException(404, detail={"message": "任务不存在"})
    return success(request, {"task_id": task["id"], "status": task["status"],
                             "result_id": task["result_id"], "input": task["scope"],
                             "model_id": task["model_id"], "created_at": task["created_at"],
                             "error_message": task["error_message"]})


@router.get("/results/latest")
def latest_result(request: Request, scope: dict = Depends(query_scope),
                  service: GovernanceService = Depends(get_service),
                  token_hash: str = Depends(require_token)):
    results = service.repository.list_results(KIND, token_hash, scope)
    return success(request, results[0]["summary"] if results else None)


@router.get("/results")
def result_history(request: Request, scope: dict = Depends(query_scope),
                   service: GovernanceService = Depends(get_service),
                   token_hash: str = Depends(require_token)):
    return success(request, [item["summary"] for item in
                             service.repository.list_results(KIND, token_hash, scope)])


@router.get("/results/{result_id}")
def result(request: Request, result_id: str,
           service: GovernanceService = Depends(get_service),
           token_hash: str = Depends(require_token)):
    return success(request, get_result(service, result_id, token_hash)["summary"])


@router.get("/results/{result_id}/samples")
def result_samples(request: Request, result_id: str, page: int = Query(1, ge=1),
                   page_size: int = Query(20, ge=1, le=200),
                   service: GovernanceService = Depends(get_service),
                   token_hash: str = Depends(require_token)):
    rows = get_result(service, result_id, token_hash)["samples"]
    total = len(rows)
    return success(request, {"items": rows[(page - 1) * page_size:page * page_size],
                             "total": total, "page": page, "page_size": page_size,
                             "total_pages": (total + page_size - 1) // page_size})
