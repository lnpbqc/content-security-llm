"""Shared-token creation and verification rules."""

import hashlib
import hmac
from datetime import datetime

from db.entities import AuthRecord
from db.repositories import AuthRepository
from utils.time import utc_now


class InvalidSetupSecretError(Exception):
    pass


class AuthService:
    def __init__(self, repository: AuthRepository, setup_secret: str):
        self.repository = repository
        self.setup_secret = setup_secret

    def replace_token(self, setup_secret: str, token: str, expires_at: datetime) -> AuthRecord:
        if not hmac.compare_digest(
            setup_secret.encode("utf-8"), self.setup_secret.encode("utf-8")
        ):
            raise InvalidSetupSecretError
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return self.repository.replace(digest, expires_at)

    def is_valid_token(self, token: str) -> bool:
        record = self.repository.get()
        if record is None or record.expires_at <= utc_now():
            return False
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return hmac.compare_digest(record.token_hash, digest)
