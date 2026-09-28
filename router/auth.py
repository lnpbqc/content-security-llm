"""Shared-token initialization route."""

from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException

from schemas.auth import TokenCreate, TokenPublic
from service.auth import AuthService, InvalidSetupSecretError

from .dependencies import get_auth_service


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token", response_model=TokenPublic, status_code=201)
def replace_token(
    payload: TokenCreate,
    x_setup_secret: Optional[str] = Header(default=None),
    service: AuthService = Depends(get_auth_service),
) -> TokenPublic:
    if x_setup_secret is None:
        raise HTTPException(status_code=403, detail={"code": "invalid_setup_secret"})
    try:
        record = service.replace_token(x_setup_secret, payload.token, payload.expires_at)
    except InvalidSetupSecretError as exc:
        raise HTTPException(status_code=403, detail={"code": "invalid_setup_secret"}) from exc
    return TokenPublic.from_record(record)
