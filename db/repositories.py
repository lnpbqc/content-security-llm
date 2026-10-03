"""SQLite queries and row conversion."""

import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .database import Database
from .entities import AuthRecord, CallRecord, ModelRecord
from utils.time import utc_now


def _model_from_row(row: sqlite3.Row) -> ModelRecord:
    return ModelRecord(
        id=row["id"],
        upstream_model_id=row["upstream_model_id"],
        name=row["name"],
        description=row["description"],
        base_url=row["base_url"],
        encrypted_api_key=row["encrypted_api_key"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        deleted_at=datetime.fromisoformat(row["deleted_at"]) if row["deleted_at"] else None,
    )


def _call_from_row(row: sqlite3.Row) -> CallRecord:
    return CallRecord(
        id=row["id"],
        model_id=row["model_id"],
        token_hash=row["token_hash"],
        input=row["input"],
        output_json=row["output_json"],
        response_type=row["response_type"],
        status=row["status"],
        error_code=row["error_code"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


class ModelRepository:
    def __init__(self, database: Database):
        self.database = database

    def create(self, *, upstream_model_id: str, name: str, description: Optional[str],
               base_url: str, encrypted_api_key: str) -> ModelRecord:
        model_id = str(uuid4())
        now = utc_now().isoformat()
        with self.database.connect() as connection:
            connection.execute(
                """INSERT INTO models
                   (id, upstream_model_id, name, description, base_url,
                    encrypted_api_key, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (model_id, upstream_model_id, name, description, base_url,
                 encrypted_api_key, now, now),
            )
        record = self.get(model_id)
        assert record is not None
        return record

    def get(self, model_id: str) -> Optional[ModelRecord]:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM models WHERE id = ? AND deleted_at IS NULL", (model_id,)
            ).fetchone()
        return _model_from_row(row) if row else None

    def list(self) -> List[ModelRecord]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM models WHERE deleted_at IS NULL ORDER BY created_at DESC"
            ).fetchall()
        return [_model_from_row(row) for row in rows]

    def update(self, model_id: str, changes: Dict[str, Any]) -> Optional[ModelRecord]:
        if not changes:
            return self.get(model_id)
        allowed = {"upstream_model_id", "name", "description", "base_url", "encrypted_api_key"}
        if not set(changes).issubset(allowed):
            raise ValueError("Unsupported model field")
        fields = list(changes)
        assignments = ", ".join("{} = ?".format(field) for field in fields)
        values = [changes[field] for field in fields]
        values.extend([utc_now().isoformat(), model_id])
        with self.database.connect() as connection:
            cursor = connection.execute(
                "UPDATE models SET {}, updated_at = ? WHERE id = ? AND deleted_at IS NULL".format(assignments),
                values,
            )
            if cursor.rowcount == 0:
                return None
        return self.get(model_id)

    def soft_delete(self, model_id: str) -> bool:
        now = utc_now().isoformat()
        with self.database.connect() as connection:
            cursor = connection.execute(
                "UPDATE models SET deleted_at = ?, updated_at = ? WHERE id = ? AND deleted_at IS NULL",
                (now, now, model_id),
            )
            return cursor.rowcount > 0


class CallRepository:
    def __init__(self, database: Database):
        self.database = database

    def create(self, *, model_id: str, token_hash: str, input: str, output_json: Optional[str],
               response_type: str, status: str, error_code: Optional[str] = None) -> CallRecord:
        call_id = str(uuid4())
        now = utc_now().isoformat()
        with self.database.connect() as connection:
            connection.execute(
                """INSERT INTO call_records
                   (id, model_id, token_hash, input, output_json, response_type, status, error_code, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (call_id, model_id, token_hash, input, output_json, response_type, status, error_code, now),
            )
            row = connection.execute("SELECT * FROM call_records WHERE id = ?", (call_id,)).fetchone()
        return _call_from_row(row)

    def list(self, *, token_hash: str, limit: int, offset: int) -> List[CallRecord]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM call_records WHERE token_hash = ? "
                "ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                (token_hash, limit, offset),
            ).fetchall()
        return [_call_from_row(row) for row in rows]

    def list_for_admin(self, *, limit: int, offset: int) -> List[Dict[str, Any]]:
        """关联令牌标签供管理员追溯调用，不返回令牌原文或哈希。"""
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT c.*, t.id AS token_id, t.label AS token_label "
                "FROM call_records c LEFT JOIN auth_tokens t ON t.token_hash = c.token_hash "
                "ORDER BY c.created_at DESC, c.id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [{"record": _call_from_row(row), "token_id": row["token_id"],
                 "token_label": row["token_label"]} for row in rows]


class AuthRepository:
    def __init__(self, database: Database):
        self.database = database

    def create(self, token_hash: str, expires_at: datetime, label: str = "") -> AuthRecord:
        token_id = str(uuid4())
        now = utc_now().isoformat()
        with self.database.connect() as connection:
            connection.execute(
                """INSERT INTO auth_tokens (id, token_hash, label, created_at, expires_at, enabled)
                   VALUES (?, ?, ?, ?, ?, 1)""",
                (token_id, token_hash, label, now, expires_at.isoformat()),
            )
        record = self.get(token_id)
        assert record is not None
        return record

    def set_enabled(self, token_hash: str, enabled: bool) -> Optional[AuthRecord]:
        with self.database.connect() as connection:
            cursor = connection.execute(
                "UPDATE auth_tokens SET enabled = ? WHERE token_hash = ? AND deleted_at IS NULL",
                (int(enabled), token_hash),
            )
            if cursor.rowcount == 0:
                return None
        return self.get_by_hash(token_hash)

    def set_enabled_by_id(self, token_id: str, enabled: bool) -> Optional[AuthRecord]:
        """管理员按列表中的令牌 ID 启停，无需令牌原文。"""
        with self.database.connect() as connection:
            cursor = connection.execute(
                "UPDATE auth_tokens SET enabled = ? WHERE id = ? AND deleted_at IS NULL",
                (int(enabled), token_id),
            )
            if cursor.rowcount == 0:
                return None
        return self.get(token_id)

    def set_all_enabled(self, enabled: bool) -> int:
        with self.database.connect() as connection:
            cursor = connection.execute(
                "UPDATE auth_tokens SET enabled = ? WHERE deleted_at IS NULL", (int(enabled),)
            )
            return cursor.rowcount

    def get(self, token_id: str) -> Optional[AuthRecord]:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM auth_tokens WHERE id = ? AND deleted_at IS NULL", (token_id,)
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def get_by_hash(self, token_hash: str) -> Optional[AuthRecord]:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM auth_tokens WHERE token_hash = ? AND deleted_at IS NULL",
                (token_hash,),
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def list(self) -> List[AuthRecord]:
        """列出令牌元数据，供管理员为旧令牌补充标签。"""
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM auth_tokens WHERE deleted_at IS NULL ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def soft_delete(self, token_id: str) -> bool:
        """立即禁用并隐藏令牌，保留令牌行供历史调用关联。"""
        with self.database.connect() as connection:
            cursor = connection.execute(
                "UPDATE auth_tokens SET enabled = 0, deleted_at = ? "
                "WHERE id = ? AND deleted_at IS NULL",
                (utc_now().isoformat(), token_id),
            )
        return cursor.rowcount > 0

    def set_label_if_empty(self, token_id: str, label: str) -> bool:
        """仅为未标注的令牌设置标签，避免历史调用归属被改写。"""
        with self.database.connect() as connection:
            cursor = connection.execute(
                "UPDATE auth_tokens SET label = ? WHERE id = ? AND label = '' AND deleted_at IS NULL",
                (label, token_id),
            )
        return cursor.rowcount > 0

    @staticmethod
    def _from_row(row: sqlite3.Row) -> AuthRecord:
        return AuthRecord(
            id=row["id"],
            token_hash=row["token_hash"],
            label=row["label"],
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
            enabled=bool(row["enabled"]),
        )
