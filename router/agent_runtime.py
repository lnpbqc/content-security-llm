"""Authenticated, bounded native tool-call bridge; never executes business tools."""
from fastapi import APIRouter, Depends, Request
from router.dependencies import require_token, get_model_manager
from router.governance import success
from router.errors import model_not_found
from llm.models import ModelNotFoundError, InvocationError
from fastapi import HTTPException
from service.agent_runtime import Turn, PrivacyRequest, complete, detect

router = APIRouter(prefix="/api/v1/agent-runtime", tags=["agent-runtime"])


def invoke(request, payload, manager, token, privacy=False):
    try:
        result = detect(manager, payload, token) if privacy else complete(
            manager, payload.model_id, payload.messages, token, payload.timeout_seconds, mode=payload.mode)
        return success(request, result)
    except ModelNotFoundError as exc:
        raise model_not_found(exc) from exc
    except InvocationError as exc:
        raise HTTPException(exc.status_code, {"code": exc.code, "message": str(exc)}) from exc


@router.post("/turn")
def turn(request: Request, payload: Turn, manager=Depends(get_model_manager), token=Depends(require_token)):
    return invoke(request, payload, manager, token)


@router.post("/privacy")
def privacy(request: Request, payload: PrivacyRequest, manager=Depends(get_model_manager), token=Depends(require_token)):
    return invoke(request, payload, manager, token, True)
