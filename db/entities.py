"""Database entities kept separate from public API schemas."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class ModelRecord:
    id: str
    upstream_model_id: str
    name: str
    description: Optional[str]
    base_url: str
    encrypted_api_key: str
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]


@dataclass(frozen=True)
class CallRecord:
    id: str
    model_id: str
    input: str
    output_json: Optional[str]
    response_type: str
    status: str
    error_code: Optional[str]
    created_at: datetime


@dataclass(frozen=True)
class AuthRecord:
    token_hash: str
    created_at: datetime
    expires_at: datetime
