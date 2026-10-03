"""读取本地启动配置；治理任务配置由 SQLite 管理。"""

import base64
import binascii
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "settings.local.json"


def _config_path() -> Path:
    configured = os.getenv("APP_CONFIG_FILE")
    if not configured:
        return DEFAULT_CONFIG_PATH
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _read_config(path: Path) -> Dict[str, Any]:
    """读取私有配置，并拒绝未知字段以避免配置拼写错误。"""
    if not path.is_file():
        if os.getenv("APP_CONFIG_FILE"):
            raise RuntimeError("Config file does not exist: {}".format(path))
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Cannot read config file: {}".format(path)) from exc
    if not isinstance(data, dict):
        raise RuntimeError("Config file must contain a JSON object")
    unknown = set(data) - {"setup_secret", "credential_key", "database_path", "governance_model_ids",
                           "business_database_url", "business_api_base_url"}
    if unknown:
        raise RuntimeError("Unknown config fields: {}".format(", ".join(sorted(unknown))))
    return data

@dataclass(frozen=True)
class Settings:
    database_path: Path
    setup_secret: str
    credential_key: bytes
    business_api_base_url: str = "http://127.0.0.1:8000/api/v1"

    @classmethod
    def load(cls) -> "Settings":
        """读取并校验启动配置；旧治理配置字段仅为兼容旧文件而忽略。"""
        path = _config_path()
        data = _read_config(path)
        setup_secret = os.getenv("APP_SETUP_SECRET", data.get("setup_secret", ""))
        if not isinstance(setup_secret, str) or len(setup_secret) < 32:
            raise RuntimeError("setup_secret must contain at least 32 characters")
        key_value = os.getenv("MODEL_CREDENTIAL_KEY", data.get("credential_key", ""))
        if not isinstance(key_value, str) or not key_value:
            raise RuntimeError("credential_key must be configured")
        try:
            credential_key = key_value.encode("ascii")
            decoded_key = base64.b64decode(credential_key, altchars=b"-_", validate=True)
        except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
            raise RuntimeError("credential_key must be a valid Fernet key") from exc
        if len(decoded_key) != 32:
            raise RuntimeError("credential_key must be a valid Fernet key")

        database_value = os.getenv("APP_DB_PATH", data.get("database_path", "db/app.sqlite3"))
        if not isinstance(database_value, str) or not database_value:
            raise RuntimeError("database_path must be a non-empty path")
        database_path = Path(database_value).expanduser()
        if not database_path.is_absolute():
            database_path = path.parent / database_path
        business_api_base_url = data.get("business_api_base_url", "http://127.0.0.1:8000/api/v1")
        try:
            if not isinstance(business_api_base_url, str):
                raise ValueError
            business_api_base_url = business_api_base_url.strip().rstrip("/")
            parsed = urlsplit(business_api_base_url)
            if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                    or parsed.username or parsed.password or parsed.query or parsed.fragment):
                raise ValueError
            parsed.port
        except ValueError as exc:
            raise RuntimeError("business_api_base_url must be an HTTP(S) base URL without credentials, query or fragment") from exc
        return cls(
            database_path=database_path.resolve(),
            setup_secret=setup_secret,
            credential_key=credential_key,
            business_api_base_url=business_api_base_url,
        )
