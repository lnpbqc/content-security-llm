"""Select a server-owned output class and run a typed invocation."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from llm.models import ModelManager
from llm.outputs import get_response_type, list_response_types


class UnknownResponseTypeError(Exception):
    pass


@dataclass(frozen=True)
class InvocationResult:
    model_id: str
    response_type: str
    result: Dict[str, Any]


class InferenceService:
    def __init__(self, manager: ModelManager):
        self.manager = manager

    def available_types(self):
        return list_response_types()

    def invoke(self, model_id: str, input: str, response_type: str,
               token_hash: str, *, timeout: Optional[float] = None,
               max_retries: Optional[int] = None) -> InvocationResult:
        response_model = get_response_type(response_type)
        if response_model is None:
            raise UnknownResponseTypeError(response_type)
        options = {}
        if timeout is not None:
            options["timeout"] = timeout
        if max_retries is not None:
            options["max_retries"] = max_retries
        parsed = self.manager.invoke(
            model_id, input, response_model, response_type=response_type,
            token_hash=token_hash, **options,
        )
        return InvocationResult(
            model_id=model_id,
            response_type=response_type,
            result=parsed.model_dump(mode="json"),
        )
