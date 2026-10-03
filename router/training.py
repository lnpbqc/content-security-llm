"""与现有训推页面对接的训练任务、增量指标和产物 API。"""

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse

from router.dependencies import require_token
from router.governance import success
from schemas.training import TrainingCreate
from service.training_examples import defaults


router = APIRouter(prefix="/api/v1/training", tags=["training"], dependencies=[Depends(require_token)])


def get_service(request: Request):
    return request.app.state.training_service


@router.get("/defaults")
def get_defaults(request: Request):
    return success(request, defaults())


@router.post("/tasks", status_code=202)
def create_task(request: Request, payload: TrainingCreate,
                token_hash: str = Depends(require_token), service=Depends(get_service)):
    task = service.repository.create(payload.model_dump(), token_hash)
    return success(request, service.repository.public(task))


@router.get("/tasks")
def list_tasks(request: Request, page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
               token_hash: str = Depends(require_token), service=Depends(get_service)):
    return success(request, service.repository.list(token_hash, page, page_size))


@router.get("/tasks/{task_id}")
def get_task(request: Request, task_id: str, token_hash: str = Depends(require_token), service=Depends(get_service)):
    return success(request, service.repository.public(service.get(task_id, token_hash)))


@router.get("/tasks/{task_id}/metrics")
def get_metrics(request: Request, task_id: str, after_id: int = Query(0, ge=0),
                limit: int = Query(200, ge=1, le=1000), token_hash: str = Depends(require_token),
                service=Depends(get_service)):
    service.get(task_id, token_hash)
    return success(request, service.repository.metrics(task_id, after_id, limit))


@router.post("/tasks/{task_id}/cancel")
def cancel_task(request: Request, task_id: str, token_hash: str = Depends(require_token), service=Depends(get_service)):
    service.get(task_id, token_hash)
    return success(request, service.repository.public(service.repository.cancel(task_id, token_hash)))


@router.get("/tasks/{task_id}/artifacts")
def get_artifacts(request: Request, task_id: str, token_hash: str = Depends(require_token), service=Depends(get_service)):
    service.get(task_id, token_hash)
    return success(request, service.repository.artifacts(task_id))


@router.get("/tasks/{task_id}/artifacts/{artifact_id}/download")
def download_artifact(task_id: str, artifact_id: str, token_hash: str = Depends(require_token),
                      service=Depends(get_service)):
    path = service.artifact_path(task_id, artifact_id, token_hash)
    return FileResponse(path, filename=path.name)
