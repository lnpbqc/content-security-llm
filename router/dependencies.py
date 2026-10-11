"""FastAPI dependencies that adapt application services to HTTP."""

from typing import Optional

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from llm.models import ModelManager
from service.auth import AuthService, InvalidSetupSecretError
from service.inference import InferenceService


bearer = HTTPBearer(auto_error=False)


def require_admin(request: Request, x_setup_secret: Optional[str] = Header(default=None)) -> None:
    """模型配置及可信代码操作沿用现有管理员密钥，不引入业务角色。"""
    try:
        request.app.state.auth_service._require_setup_secret(x_setup_secret)
    except InvalidSetupSecretError as exc:
        raise HTTPException(403, detail={"code": "invalid_setup_secret"}) from exc


def get_model_manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


def get_auth_service(request: Request) -> AuthService:
    return request.app.state.auth_service


def get_inference_service(request: Request) -> InferenceService:
    return request.app.state.inference_service


def require_token(
    service: AuthService = Depends(get_auth_service),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
) -> str:
    token_hash = service.valid_token_hash(credentials.credentials) if credentials else None
    if token_hash is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "invalid_token", "message": "Missing, invalid, disabled, or expired token"},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token_hash
