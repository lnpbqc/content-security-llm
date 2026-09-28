"""Connection lifecycle and initial SQLite schema."""

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
                    input TEXT NOT NULL,
                    output_json TEXT,
                    response_type TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('success', 'error')),
                    error_code TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS ix_call_records_created_at
                    ON call_records(created_at DESC);

                CREATE TABLE IF NOT EXISTS auth_token (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    token_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                """
            )

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
