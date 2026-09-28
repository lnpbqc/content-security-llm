"""Schemas for the structured model invocation API."""

from typing import Any, Dict

from pydantic import BaseModel, Field


class InvokeRequest(BaseModel):
    model_id: str = Field(min_length=1)
    input: str = Field(min_length=1)
    response_type: str = Field(min_length=1)


class InvokeResponse(BaseModel):
    model_id: str
    response_type: str
    result: Dict[str, Any]
