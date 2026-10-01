"""管理 SQLite 连接并建立原有表和治理任务相关表。"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    upstream_model_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    base_url TEXT NOT NULL,
                    encrypted_api_key TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT
                );

                CREATE TABLE IF NOT EXISTS call_records (
                    id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL REFERENCES models(id),
                    token_hash TEXT NOT NULL,
                    input TEXT NOT NULL,
                    output_json TEXT,
                    response_type TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('success', 'error')),
                    error_code TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS ix_call_records_token_created_at
                    ON call_records(token_hash, created_at DESC);

                CREATE TABLE IF NOT EXISTS governance_tasks (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    token_hash TEXT NOT NULL,
                    name TEXT NOT NULL,
                    scope_json TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'succeeded', 'failed')),
                    result_id TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS ix_governance_tasks_scope
                    ON governance_tasks(token_hash, kind, scope_json, created_at DESC);

                CREATE TABLE IF NOT EXISTS governance_task_samples (
                    task_id TEXT NOT NULL REFERENCES governance_tasks(id),
                    sample_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('success', 'error')),
                    output_json TEXT,
                    error_code TEXT,
                    PRIMARY KEY (task_id, sample_id)
                );

                CREATE TABLE IF NOT EXISTS governance_results (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL UNIQUE REFERENCES governance_tasks(id),
                    kind TEXT NOT NULL,
                    token_hash TEXT NOT NULL,
                    scope_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    samples_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS ix_governance_results_scope
                    ON governance_results(token_hash, kind, scope_json, created_at DESC);

                CREATE TABLE IF NOT EXISTS auth_tokens (
                    id TEXT PRIMARY KEY,
                    token_hash TEXT NOT NULL UNIQUE,
                    label TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1))
                );

                CREATE TABLE IF NOT EXISTS governance_model_bindings (
                    kind TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL REFERENCES models(id),
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS business_database_config (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    host TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    database_name TEXT NOT NULL,
                    username TEXT NOT NULL,
                    encrypted_password TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(auth_tokens)")}
            if "label" not in columns:
                connection.execute("ALTER TABLE auth_tokens ADD COLUMN label TEXT NOT NULL DEFAULT ''")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
