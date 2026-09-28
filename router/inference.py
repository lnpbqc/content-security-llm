"""Structured model invocation routes."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException

from llm.models import InvocationError, ModelNotFoundError
from schemas.inference import InvokeRequest, InvokeResponse
from service.inference import InferenceService, UnknownResponseTypeError

from .dependencies import get_inference_service, require_token
from .errors import model_not_found


router = APIRouter(prefix="/llm", tags=["inference"], dependencies=[Depends(require_token)])


@router.get("/response-types", response_model=List[str])
def get_response_types(service: InferenceService = Depends(get_inference_service)) -> List[str]:
    return service.available_types()


@router.post("/invoke", response_model=InvokeResponse)
def invoke_model(payload: InvokeRequest,
                 service: InferenceService = Depends(get_inference_service)) -> InvokeResponse:
    try:
        result = service.invoke(payload.model_id, payload.input, payload.response_type)
    except UnknownResponseTypeError as exc:
        raise HTTPException(status_code=422, detail={"code": "unknown_response_type"}) from exc
    except ModelNotFoundError as exc:
        raise model_not_found(exc) from exc
    except InvocationError as exc:
        detail = {"code": exc.code, "message": str(exc)}
        if exc.upstream_status is not None:
            detail["upstream_status"] = exc.upstream_status
        raise HTTPException(
            status_code=exc.status_code,
            detail=detail,
        ) from exc
    return InvokeResponse(
        model_id=result.model_id,
        response_type=result.response_type,
        result=result.result,
    )
