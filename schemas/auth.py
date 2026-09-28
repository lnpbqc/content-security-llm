"""Schemas for token creation and administration."""

from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator

from db.entities import AuthRecord
from utils.time import utc_now


class TokenCreate(BaseModel):
    token: str = Field(min_length=32)
    expires_at: datetime

    @field_validator("expires_at")
    @classmethod
    def validate_expiry(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("expires_at must include a timezone")
        if value <= utc_now():
            raise ValueError("expires_at must be in the future")
        return value.astimezone(timezone.utc)


class TokenStatusUpdate(BaseModel):
    enabled: bool


class SpecificTokenStatusUpdate(TokenStatusUpdate):
    token: str = Field(min_length=32)


class TokenPublic(BaseModel):
    created_at: datetime
    expires_at: datetime
    enabled: bool

    @classmethod
    def from_record(cls, record: AuthRecord) -> "TokenPublic":
        return cls(
            created_at=record.created_at,
            expires_at=record.expires_at,
            enabled=record.enabled,
        )


class TokenBulkStatus(BaseModel):
    enabled: bool
    updated_count: int
