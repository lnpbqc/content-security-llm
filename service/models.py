"""Model configuration validation and update workflow."""

from typing import Any, Dict

from llm.models import ModelManager
from db.entities import ModelRecord


class NullModelFieldError(Exception):
    def __init__(self, field: str):
        self.field = field
        super().__init__(field)


def create_model(manager: ModelManager, values: Dict[str, Any]) -> ModelRecord:
    normalized = dict(values)
    normalized["base_url"] = str(normalized["base_url"])
    return manager.create_model(**normalized)


def update_model(manager: ModelManager, model_id: str, changes: Dict[str, Any]) -> ModelRecord:
    for field in ("upstream_model_id", "name", "base_url", "api_key"):
        if field in changes and changes[field] is None:
            raise NullModelFieldError(field)
    normalized = dict(changes)
    if "base_url" in normalized:
        normalized["base_url"] = str(normalized["base_url"])
    return manager.update_model(model_id, normalized)
