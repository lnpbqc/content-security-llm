"""Token creation and administration routes."""

from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException

from schemas.auth import (
    SpecificTokenStatusUpdate, TokenBulkStatus, TokenCreate, TokenPublic, TokenStatusUpdate,
)
from service.auth import (
    AuthService, InvalidSetupSecretError, TokenAlreadyExistsError, TokenNotFoundError,
)

from .dependencies import get_auth_service


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token", response_model=TokenPublic, status_code=201)
def create_token(
    payload: TokenCreate,
    x_setup_secret: Optional[str] = Header(default=None),
    service: AuthService = Depends(get_auth_service),
) -> TokenPublic:
    try:
        record = service.create_token(x_setup_secret, payload.token, payload.expires_at)
    except InvalidSetupSecretError as exc:
        raise HTTPException(status_code=403, detail={"code": "invalid_setup_secret"}) from exc
    except TokenAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail={"code": "token_already_exists"}) from exc
    return TokenPublic.from_record(record)


@router.patch("/token/status", response_model=TokenPublic)
def set_token_status(
    payload: SpecificTokenStatusUpdate,
    x_setup_secret: Optional[str] = Header(default=None),
    service: AuthService = Depends(get_auth_service),
) -> TokenPublic:
    try:
        record = service.set_token_enabled(x_setup_secret, payload.token, payload.enabled)
    except InvalidSetupSecretError as exc:
        raise HTTPException(status_code=403, detail={"code": "invalid_setup_secret"}) from exc
    except TokenNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"code": "token_not_found"}) from exc
    return TokenPublic.from_record(record)


@router.patch("/tokens/status", response_model=TokenBulkStatus)
def set_all_token_status(
    payload: TokenStatusUpdate,
    x_setup_secret: Optional[str] = Header(default=None),
    service: AuthService = Depends(get_auth_service),
) -> TokenBulkStatus:
    try:
        updated_count = service.set_all_tokens_enabled(x_setup_secret, payload.enabled)
    except InvalidSetupSecretError as exc:
        raise HTTPException(status_code=403, detail={"code": "invalid_setup_secret"}) from exc
    return TokenBulkStatus(enabled=payload.enabled, updated_count=updated_count)
