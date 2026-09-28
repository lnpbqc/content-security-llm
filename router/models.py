"""Model configuration routes."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException

from llm.models import ModelManager, ModelNotFoundError
from schemas.models import ModelCreate, ModelPublic, ModelUpdate
from service.models import NullModelFieldError, create_model as create_model_record
from service.models import update_model as update_model_record

from .dependencies import get_model_manager, require_token
from .errors import model_not_found


router = APIRouter(prefix="/models", tags=["models"], dependencies=[Depends(require_token)])


@router.post("", response_model=ModelPublic, status_code=201)
def create_model(payload: ModelCreate, manager: ModelManager = Depends(get_model_manager)) -> ModelPublic:
    return ModelPublic.from_record(create_model_record(manager, payload.model_dump()))


@router.get("", response_model=List[ModelPublic])
def list_models(manager: ModelManager = Depends(get_model_manager)) -> List[ModelPublic]:
    return [ModelPublic.from_record(record) for record in manager.list_models()]


@router.get("/{model_id}", response_model=ModelPublic)
def get_model(model_id: str, manager: ModelManager = Depends(get_model_manager)) -> ModelPublic:
    try:
        return ModelPublic.from_record(manager.get_model(model_id))
    except ModelNotFoundError as exc:
        raise model_not_found(exc) from exc


@router.patch("/{model_id}", response_model=ModelPublic)
def update_model(model_id: str, payload: ModelUpdate,
                 manager: ModelManager = Depends(get_model_manager)) -> ModelPublic:
    try:
        record = update_model_record(manager, model_id, payload.model_dump(exclude_unset=True))
    except NullModelFieldError as exc:
        raise HTTPException(status_code=422, detail={"code": "null_model_field", "field": exc.field}) from exc
    except ModelNotFoundError as exc:
        raise model_not_found(exc) from exc
    return ModelPublic.from_record(record)


@router.delete("/{model_id}", status_code=204)
def delete_model(model_id: str, manager: ModelManager = Depends(get_model_manager)) -> None:
    try:
        manager.delete_model(model_id)
    except ModelNotFoundError as exc:
        raise model_not_found(exc) from exc
