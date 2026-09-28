"""Translate domain errors to stable HTTP responses."""

from fastapi import HTTPException

from llm.models import ModelNotFoundError


def model_not_found(exc: ModelNotFoundError) -> HTTPException:
    return HTTPException(status_code=404, detail={"code": "model_not_found", "model_id": str(exc)})
