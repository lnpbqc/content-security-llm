"""Model management and typed OpenAI-compatible invocations."""

import json
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar
from urllib.parse import urlsplit

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
from llm.structured_output import StructuredOutputValidationError, parse_model_output, schema_system_prompt


T = TypeVar("T", bound=BaseModel)


class ModelNotFoundError(Exception):
    pass


class InvocationError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 502,
                 *, upstream_status: Optional[int] = None):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.upstream_status = upstream_status


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

    def list_calls(self, *, token_hash: str, limit: int, offset: int) -> List[CallRecord]:
        return self.calls.list(token_hash=token_hash, limit=limit, offset=offset)

    def invoke(
        self, model_id: str, input: str, response_model: Optional[Type[T]], *,
        response_type: Optional[str] = None, token_hash: str,
        validator: Optional[Callable[[T], None]] = None,
        on_record: Optional[Callable[[CallRecord], None]] = None,
        temperature: Optional[float] = None, max_tokens: Optional[int] = None,
        timeout: Optional[float] = None, max_retries: Optional[int] = None,
    ) -> T:
        """调用模型并解析结构；可选校验器失败时记为错误调用。"""
        record = self.get_model(model_id)
        type_name = response_type or response_model.__name__
        try:
            api_key = self.cipher.decrypt(record.encrypted_api_key.encode("ascii")).decode("utf-8")
        except Exception as exc:
            self._log_error(model_id, token_hash, input, type_name, "credential_unavailable", on_record)
            raise InvocationError("credential_unavailable", "Stored model credential cannot be read") from exc

        client = None
        try:
            parameters = {}
            if response_model is not None:
                if urlsplit(record.base_url).hostname == "api.deepseek.com":
                    parameters["response_format"] = {"type": "json_object"}
                else:
                    parameters["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {"name": response_model.__name__, "schema": response_model.model_json_schema()},
                    }
                    if record.upstream_model_id.rstrip("/").rsplit("/", 1)[-1].lower() == "qwen3-8b":
                        parameters["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            if temperature is not None:
                parameters["temperature"] = temperature
            if max_tokens is not None:
                parameters["max_tokens"] = max_tokens
            client = self.client_factory(api_key=api_key, base_url=record.base_url)
            options = {}
            if timeout is not None:
                options["timeout"] = timeout
            if max_retries is not None:
                options["max_retries"] = max_retries
            if options:
                client = client.with_options(**options)
            completion = client.chat.completions.create(
                model=record.upstream_model_id,
                messages=[
                    {"role": "system", "content": schema_system_prompt(response_model) if response_model else
                     "请根据用户的提示生成回答，只返回回答正文。"},
                    {"role": "user", "content": input},
                ],
                **parameters,
            )
            choices = getattr(completion, "choices", None)
            if not isinstance(choices, list):
                raise InvocationError("invalid_provider_response", "Model provider returned invalid choices")
            if not choices:
                raise InvocationError("empty_response", "Model returned no choices")
            choice = choices[0]
            if getattr(choice, "finish_reason", None) == "length":
                raise InvocationError("incomplete_response", "Model response was incomplete")
            if getattr(choice, "finish_reason", None) == "content_filter":
                raise InvocationError("model_refusal", "Model response was blocked by content filtering")
            message = getattr(choice, "message", None)
            if message is None:
                raise InvocationError("invalid_provider_response", "Model provider returned an invalid message")
            if getattr(message, "refusal", None):
                raise InvocationError("model_refusal", "Model refused to provide a structured response")
            content = getattr(message, "content", None)
            if content is None or content == "":
                raise InvocationError("empty_response", "Model returned no content")
            if not isinstance(content, str):
                raise InvocationError("invalid_provider_response", "Model provider returned non-text content")
            if response_model is None and not content.strip():
                raise InvocationError("empty_response", "Model returned no content")
            parsed = parse_model_output(content, response_model) if response_model else content
            if validator is not None:
                validator(parsed)
        except APITimeoutError as exc:
            self._log_error(model_id, token_hash, input, type_name, "provider_timeout", on_record)
            raise InvocationError("provider_timeout", "Model provider timed out", 504) from exc
        except APIConnectionError as exc:
            self._log_error(model_id, token_hash, input, type_name, "provider_unavailable", on_record)
            raise InvocationError("provider_unavailable", "Model provider is unavailable") from exc
        except APIStatusError as exc:
            self._log_error(model_id, token_hash, input, type_name, "provider_error", on_record)
            raise InvocationError(
                "provider_error", "Model provider returned an HTTP error",
                upstream_status=exc.status_code,
            ) from exc
        except LengthFinishReasonError as exc:
            self._log_error(model_id, token_hash, input, type_name, "incomplete_response", on_record)
            raise InvocationError("incomplete_response", "Model response was incomplete") from exc
        except ContentFilterFinishReasonError as exc:
            self._log_error(model_id, token_hash, input, type_name, "model_refusal", on_record)
            raise InvocationError("model_refusal", "Model response was blocked by content filtering") from exc
        except StructuredOutputValidationError as exc:
            self._log_error(model_id, token_hash, input, type_name, "invalid_structured_output", on_record)
            raise InvocationError("invalid_structured_output", "模型结果校验失败：" + str(exc)) from exc
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            self._log_error(model_id, token_hash, input, type_name, "invalid_structured_output", on_record)
            message = "Model returned invalid structured output"
            if type_name.startswith("analysis_") and isinstance(exc, ValidationError):
                error_types = {error["type"] for error in exc.errors(include_input=False, include_context=False)}
                message = "模型输出的字段类型或取值不符合接口要求"
                if "json_invalid" in error_types:
                    message = "模型返回的 JSON 格式无效"
                elif "missing" in error_types:
                    message = "模型输出缺少必填字段"
                elif "extra_forbidden" in error_types:
                    message = "模型输出包含未定义字段"
            raise InvocationError("invalid_structured_output", message) from exc
        except OpenAIError as exc:
            self._log_error(model_id, token_hash, input, type_name, "provider_error", on_record)
            raise InvocationError("provider_error", "Model provider request failed") from exc
        except InvocationError as exc:
            self._log_error(model_id, token_hash, input, type_name, exc.code, on_record)
            raise
        finally:
            if client is not None and (timeout is not None or max_retries is not None):
                client.close()

        call = self.calls.create(
            model_id=model_id,
            token_hash=token_hash,
            input=input,
            output_json=parsed.model_dump_json() if response_model else json.dumps({"answer": parsed}),
            response_type=type_name,
            status="success",
        )
        if on_record is not None:
            on_record(call)
        return parsed

    def invoke_text(self, model_id: str, prompt: str, *, token_hash: str,
                    on_record=None, temperature=None, max_tokens=None,
                    timeout=None, max_retries=None) -> str:
        """生成真实正文；不要求用户回答遵循分析 JSON 结构。"""
        return self.invoke(model_id, prompt, None, response_type="analysis_generation",
                           token_hash=token_hash, on_record=on_record,
                           temperature=temperature, max_tokens=max_tokens,
                           timeout=timeout, max_retries=max_retries)

    def _log_error(self, model_id: str, token_hash: str, input: str,
                   response_type: str, code: str, on_record=None) -> None:
        call = self.calls.create(
            model_id=model_id,
            token_hash=token_hash,
            input=input,
            output_json=None,
            response_type=response_type,
            status="error",
            error_code=code,
        )
        if on_record is not None:
            on_record(call)
