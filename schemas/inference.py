"""Schemas for the structured model invocation API."""

from typing import Any, Dict

from pydantic import BaseModel, Field

from llm.analysis_contracts import InputText


class InvokeRequest(BaseModel):
    model_id: str = Field(min_length=1)
    input: str = Field(min_length=1)
    response_type: str = Field(min_length=1)


class InvokeResponse(BaseModel):
    model_id: str
    response_type: str
    result: Dict[str, Any]


class FrontendInvokeRequest(InvokeRequest):
    """v1 直连正文限长；旧通用请求契约保持不变。"""
    input: InputText
