"""FastAPI dependencies that adapt application services to HTTP."""

from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from llm.models import ModelManager
from service.auth import AuthService
from service.inference import InferenceService


bearer = HTTPBearer(auto_error=False)


def get_model_manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


def get_auth_service(request: Request) -> AuthService:
    return request.app.state.auth_service


def get_inference_service(request: Request) -> InferenceService:
    return request.app.state.inference_service


def require_token(
    service: AuthService = Depends(get_auth_service),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
) -> None:
    if credentials is None or not service.is_valid_token(credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "invalid_token", "message": "Missing, invalid, or expired token"},
            headers={"WWW-Authenticate": "Bearer"},
        )
