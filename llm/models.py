"""Model management and typed OpenAI-compatible invocations."""

import json
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    ContentFilterFinishReasonError,
    LengthFinishReasonError,
    OpenAI,
    OpenAIError,
)
from pydantic import BaseModel, ValidationError

from db.database import Database
from db.entities import CallRecord, ModelRecord
from db.repositories import CallRepository, ModelRepository


T = TypeVar("T", bound=BaseModel)


class ModelNotFoundError(Exception):
    pass


class InvocationError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 502):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class ModelManager:
    def __init__(
        self,
        database: Database,
        credential_key: bytes,
        client_factory: Optional[Callable[..., Any]] = None,
        cipher: Optional[Any] = None,
    ):
        self.models = ModelRepository(database)
        self.calls = CallRepository(database)
        if cipher is None:
            from cryptography.fernet import Fernet

            cipher = Fernet(credential_key)
        self.cipher = cipher
        self.client_factory = client_factory or OpenAI

    def create_model(
        self, *, upstream_model_id: str, name: str, description: Optional[str],
        base_url: str, api_key: str
    ) -> ModelRecord:
        return self.models.create(
            upstream_model_id=upstream_model_id,
            name=name,
            description=description,
            base_url=base_url,
            encrypted_api_key=self.cipher.encrypt(api_key.encode("utf-8")).decode("ascii"),
        )

    def get_model(self, model_id: str) -> ModelRecord:
        record = self.models.get(model_id)
        if record is None:
            raise ModelNotFoundError(model_id)
        return record

    def list_models(self) -> List[ModelRecord]:
        return self.models.list()

    def update_model(self, model_id: str, changes: Dict[str, Any]) -> ModelRecord:
        changes = dict(changes)
        if "api_key" in changes:
            api_key = changes.pop("api_key")
            changes["encrypted_api_key"] = self.cipher.encrypt(api_key.encode("utf-8")).decode("ascii")
        record = self.models.update(model_id, changes)
        if record is None:
            raise ModelNotFoundError(model_id)
        return record

    def delete_model(self, model_id: str) -> None:
        if not self.models.soft_delete(model_id):
            raise ModelNotFoundError(model_id)

    def list_calls(self, *, limit: int, offset: int) -> List[CallRecord]:
        return self.calls.list(limit=limit, offset=offset)

    def invoke(
        self, model_id: str, input: str, response_model: Type[T], *,
        response_type: Optional[str] = None
    ) -> T:
        record = self.get_model(model_id)
        type_name = response_type or response_model.__name__
        try:
            api_key = self.cipher.decrypt(record.encrypted_api_key.encode("ascii")).decode("utf-8")
        except Exception as exc:
            self._log_error(model_id, input, type_name, "credential_unavailable")
            raise InvocationError("credential_unavailable", "Stored model credential cannot be read") from exc

        try:
            client = self.client_factory(api_key=api_key, base_url=record.base_url)
            completion = client.chat.completions.parse(
                model=record.upstream_model_id,
                messages=[{"role": "user", "content": input}],
                response_format=response_model,
            )
            if not completion.choices:
                raise InvocationError("empty_response", "Model returned no choices")
            message = completion.choices[0].message
            if getattr(message, "refusal", None):
                raise InvocationError("model_refusal", "Model refused to provide a structured response")
            parsed = getattr(message, "parsed", None)
            if parsed is None:
                raise InvocationError("invalid_structured_output", "Model returned no parsed object")
            if not isinstance(parsed, response_model):
                parsed = response_model.model_validate(parsed)
        except APITimeoutError as exc:
            self._log_error(model_id, input, type_name, "provider_timeout")
            raise InvocationError("provider_timeout", "Model provider timed out", 504) from exc
        except APIConnectionError as exc:
            self._log_error(model_id, input, type_name, "provider_unavailable")
            raise InvocationError("provider_unavailable", "Model provider is unavailable") from exc
        except APIStatusError as exc:
            code = "structured_output_unsupported" if self._is_schema_error(exc) else "provider_error"
            self._log_error(model_id, input, type_name, code)
            raise InvocationError(code, "Model provider rejected the structured request") from exc
        except LengthFinishReasonError as exc:
            self._log_error(model_id, input, type_name, "incomplete_response")
            raise InvocationError("incomplete_response", "Model response was incomplete") from exc
        except ContentFilterFinishReasonError as exc:
            self._log_error(model_id, input, type_name, "model_refusal")
            raise InvocationError("model_refusal", "Model response was blocked by content filtering") from exc
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            self._log_error(model_id, input, type_name, "invalid_structured_output")
            raise InvocationError("invalid_structured_output", "Model returned invalid structured output") from exc
        except OpenAIError as exc:
            self._log_error(model_id, input, type_name, "provider_error")
            raise InvocationError("provider_error", "Model provider request failed") from exc
        except InvocationError as exc:
            self._log_error(model_id, input, type_name, exc.code)
            raise

        self.calls.create(
            model_id=model_id,
            input=input,
            output_json=parsed.model_dump_json(),
            response_type=type_name,
            status="success",
        )
        return parsed

    def _log_error(self, model_id: str, input: str, response_type: str, code: str) -> None:
        self.calls.create(
            model_id=model_id,
            input=input,
            output_json=None,
            response_type=response_type,
            status="error",
            error_code=code,
        )

    @staticmethod
    def _is_schema_error(error: APIStatusError) -> bool:
        if error.status_code != 400:
            return False
        details = str(error).lower()
        return "response_format" in details or "json_schema" in details or "structured output" in details
