"""读取本地启动配置；主存储及治理任务配置由 MySQL 管理。"""

import base64
import binascii
import json
import os
from dataclasses import dataclass, field
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
                           "business_database_url", "business_api_base_url", "governance_mock_samples",
                           "data_dir", "mysql_host", "mysql_port", "mysql_user", "mysql_password", "mysql_database"}
    if unknown:
        raise RuntimeError("Unknown config fields: {}".format(", ".join(sorted(unknown))))
    return data

@dataclass(frozen=True)
class Settings:
    data_dir: Path
    setup_secret: str
    credential_key: bytes
    business_api_base_url: str = "http://127.0.0.1:8000/api/v1"
    governance_mock_samples: bool = False

    mysql_host: str = field(default_factory=lambda: os.getenv("MYSQL_HOST", "127.0.0.1"))
    mysql_port: int = field(default_factory=lambda: int(os.getenv("MYSQL_PORT", "3306")))
    mysql_user: str = field(default_factory=lambda: os.getenv("MYSQL_USER", "content_user"))
    mysql_password: str = field(default_factory=lambda: os.getenv("MYSQL_PASSWORD", "content"), repr=False)
    mysql_database: str = field(default_factory=lambda: os.getenv("MYSQL_DATABASE", "content_safety"))

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

        # Legacy database_path only determines the existing artifact directory.
        legacy_path = os.getenv("APP_DB_PATH", data.get("database_path", "db/app.sqlite3"))
        directory = os.getenv("APP_DATA_DIR", data.get("data_dir", str(Path(legacy_path).parent)))
        if not isinstance(directory, str) or not directory:
            raise RuntimeError("data_dir must be a non-empty path")
        data_dir = Path(directory).expanduser()
        if not data_dir.is_absolute():
            data_dir = path.parent / data_dir
        mysql = {}
        for name, default in (("host", "127.0.0.1"), ("user", "content_user"),
                              ("password", "content"), ("database", "content_safety")):
            value = os.getenv("MYSQL_" + name.upper(), data.get("mysql_" + name, default))
            if not isinstance(value, str) or not value:
                raise RuntimeError("mysql_{} must be configured".format(name))
            mysql["mysql_" + name] = value
        try:
            raw_port = os.getenv("MYSQL_PORT", data.get("mysql_port", 3306))
            if isinstance(raw_port, bool) or not isinstance(raw_port, (str, int)):
                raise ValueError
            mysql["mysql_port"] = int(raw_port)
            if not 1 <= mysql["mysql_port"] <= 65535:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise RuntimeError("mysql_port must be an integer between 1 and 65535") from exc
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
        governance_mock_samples = data.get("governance_mock_samples", False)
        if not isinstance(governance_mock_samples, bool):
            raise RuntimeError("governance_mock_samples must be a boolean")
        return cls(
            data_dir=data_dir.resolve(),
            **mysql,
            setup_secret=setup_secret,
            credential_key=credential_key,
            business_api_base_url=business_api_base_url,
            governance_mock_samples=governance_mock_samples,
        )
