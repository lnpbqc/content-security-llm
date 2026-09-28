"""Token creation, administration, and verification rules."""

import hashlib
import hmac
import sqlite3
from datetime import datetime
from typing import Optional

from db.entities import AuthRecord
from db.repositories import AuthRepository
from utils.time import utc_now


class InvalidSetupSecretError(Exception):
    pass


class TokenNotFoundError(Exception):
    pass


class TokenAlreadyExistsError(Exception):
    pass


class AuthService:
    def __init__(self, repository: AuthRepository, setup_secret: str):
        self.repository = repository
        self.setup_secret = setup_secret

    def _require_setup_secret(self, setup_secret: Optional[str]) -> None:
        if not hmac.compare_digest(
            (setup_secret or "").encode("utf-8"), self.setup_secret.encode("utf-8")
        ):
            raise InvalidSetupSecretError

    def create_token(self, setup_secret: Optional[str], token: str, expires_at: datetime) -> AuthRecord:
        self._require_setup_secret(setup_secret)
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        try:
            return self.repository.create(digest, expires_at)
        except sqlite3.IntegrityError as exc:
            raise TokenAlreadyExistsError from exc

    def set_token_enabled(self, setup_secret: Optional[str], token: str,
                          enabled: bool) -> AuthRecord:
        self._require_setup_secret(setup_secret)
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        record = self.repository.set_enabled(digest, enabled)
        if record is None:
            raise TokenNotFoundError
        return record

    def set_all_tokens_enabled(self, setup_secret: Optional[str], enabled: bool) -> int:
        self._require_setup_secret(setup_secret)
        return self.repository.set_all_enabled(enabled)

    def is_valid_token(self, token: str) -> bool:
        return self.valid_token_hash(token) is not None

    def valid_token_hash(self, token: str) -> Optional[str]:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        record = self.repository.get_by_hash(digest)
        if record is None or not record.enabled or record.expires_at <= utc_now():
            return None
        return digest if hmac.compare_digest(record.token_hash, digest) else None
