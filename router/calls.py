"""Read-only call record route."""

from typing import List

from fastapi import APIRouter, Depends, Query

from llm.models import ModelManager
from schemas.calls import CallPublic

from .dependencies import get_model_manager, require_token


router = APIRouter(prefix="/calls", tags=["calls"], dependencies=[Depends(require_token)])


@router.get("", response_model=List[CallPublic])
def list_calls(limit: int = Query(default=50, ge=1, le=200),
               offset: int = Query(default=0, ge=0),
               manager: ModelManager = Depends(get_model_manager)) -> List[CallPublic]:
    return [CallPublic.from_record(record) for record in manager.list_calls(limit=limit, offset=offset)]
