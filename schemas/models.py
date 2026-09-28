"""Schemas for model configuration APIs."""

from datetime import datetime
from typing import Optional

from pydantic import AnyHttpUrl, BaseModel, Field

from db.entities import ModelRecord


class ModelCreate(BaseModel):
    upstream_model_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: Optional[str] = None
    base_url: AnyHttpUrl
    api_key: str = Field(min_length=1)


class ModelUpdate(BaseModel):
    upstream_model_id: Optional[str] = Field(default=None, min_length=1)
    name: Optional[str] = Field(default=None, min_length=1)
    description: Optional[str] = None
    base_url: Optional[AnyHttpUrl] = None
    api_key: Optional[str] = Field(default=None, min_length=1)


class ModelPublic(BaseModel):
    model_id: str
    upstream_model_id: str
    name: str
    description: Optional[str]
    base_url: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_record(cls, record: ModelRecord) -> "ModelPublic":
        return cls(
            model_id=record.id,
            upstream_model_id=record.upstream_model_id,
            name=record.name,
            description=record.description,
            base_url=record.base_url,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
