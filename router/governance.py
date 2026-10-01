"""提供三类治理任务的创建、轮询和只读结果接口。"""

from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from router.dependencies import require_token
from service.governance import GovernanceService, ModelNotConfiguredError
from utils.time import utc_now


router = APIRouter(prefix="/api/v1", tags=["governance"], dependencies=[Depends(require_token)])


class Scope(BaseModel):
    """指定任务处理范围：哪个数据集版本、哪种语种和哪套方案。"""

    # 不接受前端额外传入样本文本、模型 ID 或自定义规则。
    model_config = ConfigDict(extra="forbid")
    dataset_id: int = Field(ge=1, description="数据集 ID，至少为 1")
    version_id: str = Field(min_length=1, description="须与业务库数据集当前版本一致")
    language: str = Field(min_length=1, description="保留在任务范围中；当前不筛选业务库记录")
    scheme_id: str = Field(min_length=1, description="本次分析使用的方案版本 ID")


class TaskCreate(BaseModel):
    """创建任务时仅接收任务类型、名称和处理范围。"""
    model_config = ConfigDict(extra="forbid")
    kind: str
    name: str = Field(min_length=1)
    input: Scope


def get_service(request: Request) -> GovernanceService:
    """从应用状态中取得治理任务服务。"""
    return request.app.state.governance_service


def query_scope(dataset_id: int = Query(..., ge=1), version_id: str = Query(...),
                language: str = Query(...), scheme_id: str = Query(...)) -> Dict[str, Any]:
    """将四个 URL 查询参数整理成统一的任务范围。"""
    return Scope(dataset_id=dataset_id, version_id=version_id,
                 language=language, scheme_id=scheme_id).model_dump()


def success(request: Request, data: Any) -> Dict[str, Any]:
    """把数据包装成前端约定的成功响应格式。"""
    return {"code": 0, "message": "success", "data": data,
            "trace_id": request.headers.get("X-Trace-Id") or str(uuid4()),
            "timestamp": utc_now().isoformat()}


def _task_public(task: Dict[str, Any], service: GovernanceService) -> Dict[str, Any]:
    """将内部任务记录映射为对应页面的任务字段。"""
    common = {"status": task["status"], "created_at": task["created_at"],
              "result_id": task["result_id"], "error_message": task["error_message"]}
    if task["kind"] == "governance-value":
        return {"task_id": task["id"], "name": task["name"], **common}
    if task["kind"] == "governance-anomaly":
        return {"task_id": task["id"], "input": task["scope"], **common,
                "coverage": len(service.repository.sample_outcomes(task["id"])),
                "rule_version": task["scope"]["scheme_id"],
                "model_version": task["model_version"]}
    return {"id": task["id"], "input": task["scope"], **common}


def _create(request: Request, payload: TaskCreate, kind: str,
            service: GovernanceService, token_hash: str):
    """核对路径与任务类型，创建 pending 任务并包装响应。"""
    if payload.kind != kind:
        raise HTTPException(422, detail={"code": "invalid_kind", "message": "任务类型与路径不一致"})
    try:
        task = service.create_task(kind, payload.name, payload.input.model_dump(), token_hash)
    except ValueError as exc:
        raise HTTPException(422, detail={"code": "invalid_scope", "message": str(exc)}) from exc
    except ModelNotConfiguredError as exc:
        raise HTTPException(503, detail={"code": "model_not_configured", "message": str(exc)}) from exc
    return success(request, _task_public(task, service))


def _get_task(request: Request, task_id: str, kind: str,
              service: GovernanceService, token_hash: str):
    """只返回当前令牌拥有且类型匹配的任务。"""
    task = service.repository.get_task(task_id, token_hash)
    if task is None or task["kind"] != kind:
        raise HTTPException(404, detail={"code": "task_not_found", "message": "任务不存在"})
    return success(request, _task_public(task, service))


def _get_result(service: GovernanceService, result_id: str, kind: str, token_hash: str):
    """只读取当前令牌拥有的结果，缺失时返回 404。"""
    result = service.repository.get_result(result_id, token_hash, kind)
    if result is None:
        raise HTTPException(404, detail={"code": "result_not_found", "message": "结果不存在"})
    return result


def _page(rows: List[Dict[str, Any]], page: int, page_size: int) -> Dict[str, Any]:
    """对已筛选和排序的样本生成分页响应。"""
    total = len(rows)
    return {"items": rows[(page - 1) * page_size:page * page_size], "total": total,
            "page": page, "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size}


@router.post("/tasks")
def create_value_task(request: Request, payload: TaskCreate,
                      service: GovernanceService = Depends(get_service),
                      token_hash: str = Depends(require_token)):
    """创建价值分析任务，不在请求中直接调用模型。"""
    return _create(request, payload, "governance-value", service, token_hash)


@router.get("/tasks")
def list_value_tasks(request: Request, kind: str = Query(...),
                     scope: Dict[str, Any] = Depends(query_scope),
                     page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
                     service: GovernanceService = Depends(get_service),
                     token_hash: str = Depends(require_token)):
    """分页查询当前令牌在指定范围内的价值分析任务。"""
    if kind != "governance-value":
        raise HTTPException(422, detail={"code": "invalid_kind", "message": "不支持的任务类型"})
    result = service.repository.list_tasks(kind, token_hash, scope, page_size, (page - 1) * page_size)
    return success(request, {"items": [_task_public(task, service) for task in result["items"]],
                             "total": result["total"], "page": page, "page_size": page_size,
                             "total_pages": (result["total"] + page_size - 1) // page_size})


@router.get("/tasks/{task_id}")
def get_value_task(request: Request, task_id: str, kind: str = Query(...),
                   service: GovernanceService = Depends(get_service),
                   token_hash: str = Depends(require_token)):
    """查询一项价值分析任务的当前状态和结果 ID。"""
    if kind != "governance-value":
        raise HTTPException(422, detail={"code": "invalid_kind", "message": "不支持的任务类型"})
    return _get_task(request, task_id, kind, service, token_hash)


@router.post("/data-governance/anomaly-tasks")
def create_anomaly_task(request: Request, payload: TaskCreate,
                        service: GovernanceService = Depends(get_service),
                        token_hash: str = Depends(require_token)):
    """创建异常治理任务，不在请求中直接调用模型。"""
    return _create(request, payload, "governance-anomaly", service, token_hash)


@router.get("/data-governance/anomaly-tasks/{task_id}")
def get_anomaly_task(request: Request, task_id: str,
                     kind: str = Query("governance-anomaly"),
                     service: GovernanceService = Depends(get_service),
                     token_hash: str = Depends(require_token)):
    """查询异常治理任务状态、已处理样本数和结果 ID。"""
    if kind != "governance-anomaly":
        raise HTTPException(422, detail={"code": "invalid_kind", "message": "不支持的任务类型"})
    return _get_task(request, task_id, kind, service, token_hash)


@router.post("/data-governance/risk-tasks")
def create_risk_task(request: Request, payload: TaskCreate,
                     service: GovernanceService = Depends(get_service),
                     token_hash: str = Depends(require_token)):
    """创建风险分级任务，不在请求中直接调用模型。"""
    return _create(request, payload, "governance-risk", service, token_hash)


@router.get("/data-governance/risk-tasks/{task_id}")
def get_risk_task(request: Request, task_id: str, kind: str = Query("governance-risk"),
                  service: GovernanceService = Depends(get_service),
                  token_hash: str = Depends(require_token)):
    """查询风险分级任务的当前状态和结果 ID。"""
    if kind != "governance-risk":
        raise HTTPException(422, detail={"code": "invalid_kind", "message": "不支持的任务类型"})
    return _get_task(request, task_id, kind, service, token_hash)


def _latest(request, kind, scope, service, token_hash):
    """读取完整范围内最新的已完成结果；没有则返回 null。"""
    results = service.repository.list_results(kind, token_hash, scope)
    return success(request, results[0]["summary"] if results else None)


def _history(request, kind, scope, service, token_hash):
    """读取完整范围内的已完成结果历史。"""
    results = service.repository.list_results(kind, token_hash, scope)
    return success(request, [result["summary"] for result in results])


@router.get("/data-governance/value-results/latest")
def latest_value(request: Request, scope: Dict[str, Any] = Depends(query_scope),
                 service: GovernanceService = Depends(get_service),
                 token_hash: str = Depends(require_token)):
    """只读查询当前范围最新的价值分析快照。"""
    return _latest(request, "governance-value", scope, service, token_hash)


@router.get("/data-governance/value-results/{result_id}")
def value_result(request: Request, result_id: str,
                 service: GovernanceService = Depends(get_service),
                 token_hash: str = Depends(require_token)):
    """只读查询指定价值分析结果的汇总。"""
    return success(request, _get_result(service, result_id, "governance-value", token_hash)["summary"])


@router.get("/data-governance/value-results/{result_id}/samples")
def value_samples(request: Request, result_id: str, page: int = Query(1, ge=1),
                  page_size: int = Query(10, ge=1, le=200), tier: str = "all",
                  keyword: str = "", bin: str = "", sort_by: Optional[str] = None,
                  sort_order: Optional[str] = None,
                  service: GovernanceService = Depends(get_service),
                  token_hash: str = Depends(require_token)):
    """从已保存的价值样本中先筛选排序，再分页返回。"""
    rows = list(_get_result(service, result_id, "governance-value", token_hash)["samples"])
    if tier not in {"all", "high", "medium", "low", "unavailable"} or (bin and bin not in set("01234")):
        raise HTTPException(422, detail={"code": "invalid_filter", "message": "评分筛选条件无效"})
    rows = [row for row in rows if (tier == "all" or row["tier"] == tier) and
            (not keyword or keyword in row["id"] + row["text"]) and
            (not bin or row["score"] is not None and str(min(4, int(row["score"] // 20))) == bin)]
    if sort_by is not None or sort_order is not None:
        if sort_by not in {"score", "tier"} or sort_order not in {"asc", "desc"}:
            raise HTTPException(422, detail={"code": "invalid_sort", "message": "评分排序条件无效"})
        ranks = {"low": 0, "medium": 1, "high": 2}
        rows.sort(key=lambda row: int(row["id"].rsplit("_", 1)[-1]))
        rows.sort(key=lambda row: (row["score"] if row["score"] is not None else 0)
                  if sort_by == "score" else ranks.get(row["tier"], -1),
                  reverse=sort_order == "desc")
        rows.sort(key=lambda row: row["score"] is None)
    return success(request, _page(rows, page, page_size))


@router.get("/data-governance/anomaly-results/latest")
def latest_anomaly(request: Request, scope: Dict[str, Any] = Depends(query_scope),
                   service: GovernanceService = Depends(get_service),
                   token_hash: str = Depends(require_token)):
    """只读查询当前范围最新的异常治理快照。"""
    return _latest(request, "governance-anomaly", scope, service, token_hash)


@router.get("/data-governance/anomaly-results")
def anomaly_history(request: Request, scope: Dict[str, Any] = Depends(query_scope),
                    service: GovernanceService = Depends(get_service),
                    token_hash: str = Depends(require_token)):
    """只读查询当前范围的异常治理历史快照。"""
    return _history(request, "governance-anomaly", scope, service, token_hash)


@router.get("/data-governance/anomaly-results/{result_id}")
def anomaly_result(request: Request, result_id: str,
                   service: GovernanceService = Depends(get_service),
                   token_hash: str = Depends(require_token)):
    """只读查询指定异常治理结果的汇总。"""
    return success(request, _get_result(service, result_id, "governance-anomaly", token_hash)["summary"])


@router.get("/data-governance/anomaly-results/{result_id}/samples")
def anomaly_samples(request: Request, result_id: str, page: int = Query(1, ge=1),
                    page_size: int = Query(10, ge=1, le=200), keyword: str = "",
                    type: str = "", status: str = "",
                    service: GovernanceService = Depends(get_service),
                    token_hash: str = Depends(require_token)):
    """按关键词、异常类型和状态筛选已保存样本并分页。"""
    rows = _get_result(service, result_id, "governance-anomaly", token_hash)["samples"]
    rows = [row for row in rows if (not keyword or keyword in row["id"] + row["text"]) and
            (not type or any(finding["type"] == type for finding in row["findings"])) and
            (not status or row["status"] == status)]
    return success(request, _page(rows, page, page_size))


@router.get("/data-governance/anomaly-results/{result_id}/samples/{sample_id}")
def anomaly_sample(request: Request, result_id: str, sample_id: str,
                   service: GovernanceService = Depends(get_service),
                   token_hash: str = Depends(require_token)):
    """只读查询一条异常样本及其发现和候选变更。"""
    rows = _get_result(service, result_id, "governance-anomaly", token_hash)["samples"]
    row = next((row for row in rows if row["id"] == sample_id), None)
    if row is None:
        raise HTTPException(404, detail={"code": "sample_not_found", "message": "样本不存在"})
    return success(request, row)


@router.get("/data-governance/risk-results/latest")
def latest_risk(request: Request, scope: Dict[str, Any] = Depends(query_scope),
                service: GovernanceService = Depends(get_service),
                token_hash: str = Depends(require_token)):
    """只读查询当前范围最新的风险分级快照。"""
    return _latest(request, "governance-risk", scope, service, token_hash)


@router.get("/data-governance/risk-results")
def risk_history(request: Request, scope: Dict[str, Any] = Depends(query_scope),
                 service: GovernanceService = Depends(get_service),
                 token_hash: str = Depends(require_token)):
    """只读查询当前范围的风险分级历史快照。"""
    return _history(request, "governance-risk", scope, service, token_hash)


@router.get("/data-governance/risk-results/{result_id}")
def risk_result(request: Request, result_id: str,
                service: GovernanceService = Depends(get_service),
                token_hash: str = Depends(require_token)):
    """只读查询指定风险分级结果的汇总。"""
    return success(request, _get_result(service, result_id, "governance-risk", token_hash)["summary"])


@router.get("/data-governance/risk-results/{result_id}/samples")
def risk_samples(request: Request, result_id: str, page: int = Query(1, ge=1),
                 page_size: int = Query(10, ge=1, le=200), keyword: str = "",
                 level: str = "", status: str = "", sort_by: Optional[str] = None,
                 sort_order: Optional[str] = None,
                 service: GovernanceService = Depends(get_service),
                 token_hash: str = Depends(require_token)):
    """按风险等级等条件筛选已保存样本并分页。"""
    rows = list(_get_result(service, result_id, "governance-risk", token_hash)["samples"])
    rows = [row for row in rows if (not keyword or keyword in row["id"] + row["text"]) and
            (not level or row["findings"] and row["maximum_suggested_level"] == level) and
            (not status or row["status"] == status)]
    if sort_by is not None or sort_order is not None:
        if sort_by not in {"maximum_suggested_level", "status"} or sort_order not in {"asc", "desc"}:
            raise HTTPException(422, detail={"code": "invalid_sort", "message": "风险排序条件无效"})
        ranks = {"NOTICE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
        rows.sort(key=lambda row: row["id"])
        rows.sort(key=lambda row: ranks[row["maximum_suggested_level"]] if sort_by == "maximum_suggested_level"
                  else row["status"], reverse=sort_order == "desc")
    return success(request, _page(rows, page, page_size))


@router.get("/data-governance/risk-results/{result_id}/samples/{sample_id}")
def risk_sample(request: Request, result_id: str, sample_id: str,
                service: GovernanceService = Depends(get_service),
                token_hash: str = Depends(require_token)):
    """只读查询一条遮蔽后的风险样本及其证据。"""
    rows = _get_result(service, result_id, "governance-risk", token_hash)["samples"]
    row = next((row for row in rows if row["id"] == sample_id), None)
    if row is None:
        raise HTTPException(404, detail={"code": "sample_not_found", "message": "样本不存在"})
    return success(request, row)
