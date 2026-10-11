"""GPU 溯源任务只负责入队和查询；HTTP 进程不加载模型。"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from router.dependencies import require_token
from router.governance import success
from schemas.neural_trace import TraceCase, TraceCreate, TracePage, TraceResponse, TraceResult, TraceTask
from service.neural_trace import public_task


router = APIRouter(prefix="/api/v1/neural-trace", tags=["neural-trace"])


def get_task(request, task_id, token_hash):
    task = request.app.state.neural_trace_service.repository.get(task_id, token_hash)
    if task is None:
        raise HTTPException(404, detail={"code": "trace_task_not_found", "message": "溯源任务不存在。"})
    return task


@router.get("/cases", response_model=TraceResponse[list[TraceCase]])
def cases(request: Request, token_hash: str = Depends(require_token)):
    return success(request, request.app.state.neural_trace_service.cases())


@router.post("/tasks", status_code=202, response_model=TraceResponse[TraceTask])
def create(request: Request, payload: TraceCreate, token_hash: str = Depends(require_token)):
    try:
        task = request.app.state.neural_trace_service.create(token_hash, payload.case_id)
    except KeyError:
        raise HTTPException(404, detail={"code": "trace_case_not_found", "message": "案例不存在或未通过正式审核。"})
    return success(request, public_task(task))


@router.get("/tasks", response_model=TraceResponse[TracePage])
def tasks(request: Request, limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
          token_hash: str = Depends(require_token)):
    page = request.app.state.neural_trace_service.repository.list(token_hash, limit, offset)
    return success(request, {"items": [public_task(task) for task in page["items"]], "total": page["total"]})


@router.get("/tasks/{task_id}", response_model=TraceResponse[TraceTask])
def task(request: Request, task_id: str, token_hash: str = Depends(require_token)):
    return success(request, public_task(get_task(request, task_id, token_hash)))


@router.get("/tasks/{task_id}/result", response_model=TraceResponse[TraceResult])
def result(request: Request, task_id: str, token_hash: str = Depends(require_token)):
    saved = get_task(request, task_id, token_hash)
    if saved["status"] != "succeeded":
        raise HTTPException(409, detail={"code": "trace_result_unavailable", "message": "任务尚未成功完成，请查询任务状态。"})
    return success(request, saved["result"])
