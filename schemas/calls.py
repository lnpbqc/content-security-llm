"""Read-only call record response schema."""

import json
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel

from db.entities import CallRecord


class CallPublic(BaseModel):
    id: str
    model_id: str
    input: str
    output: Optional[Dict[str, Any]]
    response_type: str
    status: str
    error_code: Optional[str]
    created_at: datetime

    @classmethod
    def from_record(cls, record: CallRecord) -> "CallPublic":
        return cls(
            id=record.id,
            model_id=record.model_id,
            input=record.input,
            output=json.loads(record.output_json) if record.output_json is not None else None,
            response_type=record.response_type,
            status=record.status,
            error_code=record.error_code,
            created_at=record.created_at,
        )
