"""前端直连的 v1 包裹入口，保留旧裸响应及模型执行服务。"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from llm.models import InvocationError, ModelManager, ModelNotFoundError
from router.dependencies import get_inference_service, get_model_manager, require_token
from router.errors import model_not_found
from router.governance import success
from schemas.calls import CallPublic
from schemas.inference import FrontendInvokeRequest
from schemas.models import ModelPublic
from service.inference import InferenceService, UnknownResponseTypeError


router = APIRouter(prefix="/api/v1", tags=["model-frontend"], dependencies=[Depends(require_token)])


@router.get("/models")
def list_models(request: Request, manager: ModelManager = Depends(get_model_manager)):
    return success(request, [ModelPublic.from_record(record).model_dump(mode="json")
                             for record in manager.list_models()])


@router.get("/models/{model_id}")
def get_model(request: Request, model_id: str, manager: ModelManager = Depends(get_model_manager)):
    try:
        record = manager.get_model(model_id)
    except ModelNotFoundError as exc:
        raise model_not_found(exc) from exc
    return success(request, ModelPublic.from_record(record).model_dump(mode="json"))


@router.get("/llm/response-types")
def get_response_types(request: Request, service: InferenceService = Depends(get_inference_service)):
    return success(request, service.available_types())


@router.post("/llm/invoke")
def invoke_model(request: Request, payload: FrontendInvokeRequest,
                 service: InferenceService = Depends(get_inference_service),
                 token_hash: str = Depends(require_token)):
    # 与专属分析共用进程级四槽；不改变旧 /api/llm/invoke 的执行配置。
    slots = request.app.state.analysis_service.slots
    if not slots.acquire(blocking=False):
        raise HTTPException(429, detail={"code": "analysis_busy", "message": "模型分析繁忙，请稍后查询记录或重试"})
    try:
        result = service.invoke(payload.model_id, payload.input, payload.response_type, token_hash,
                                timeout=120, max_retries=0)
    except UnknownResponseTypeError as exc:
        raise HTTPException(422, detail={"code": "unknown_response_type"}) from exc
    except ModelNotFoundError as exc:
        raise model_not_found(exc) from exc
    except InvocationError as exc:
        detail = {"code": exc.code, "message": str(exc)}
        if exc.upstream_status is not None:
            detail["upstream_status"] = exc.upstream_status
        raise HTTPException(exc.status_code, detail=detail) from exc
    finally:
        slots.release()
    return success(request, {"model_id": result.model_id, "response_type": result.response_type,
                             "result": result.result})


@router.get("/calls")
def list_calls(request: Request, limit: int = Query(default=50, ge=1, le=200),
               offset: int = Query(default=0, ge=0),
               manager: ModelManager = Depends(get_model_manager),
               token_hash: str = Depends(require_token)):
    return success(request, [CallPublic.from_record(record).model_dump(mode="json")
                             for record in manager.list_calls(token_hash=token_hash, limit=limit, offset=offset)])
