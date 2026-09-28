"""Server-owned Pydantic result types available to HTTP callers."""

from typing import Dict, List, Optional, Type

from pydantic import BaseModel, ConfigDict


class TextAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str

class JudgementAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float

    reason: str


_response_types: Dict[str, Type[BaseModel]] = {
    "text_answer": TextAnswer,
    "judgement": JudgementAnswer
}


def register_response_type(name: str, response_model: Type[BaseModel]) -> None:
    """Register a business result class under an HTTP-safe name."""
    if not name or not name.replace("_", "").isalnum():
        raise ValueError("Response type name must contain letters, digits, or underscores")
    if not isinstance(response_model, type) or not issubclass(response_model, BaseModel):
        raise TypeError("response_model must be a Pydantic BaseModel class")
    if name in _response_types and _response_types[name] is not response_model:
        raise ValueError("Response type name is already registered")
    _response_types[name] = response_model


def get_response_type(name: str) -> Optional[Type[BaseModel]]:
    return _response_types.get(name)


def list_response_types() -> List[str]:
    return sorted(_response_types)
