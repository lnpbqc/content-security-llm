"""保存治理任务的选模绑定和业务 MySQL 连接配置。"""

from typing import Any, Dict, Optional

from db.database import Database
from utils.time import utc_now


class GovernanceConfigRepository:
    """统一读取管理员写入的治理任务配置。"""

    def __init__(self, database: Database):
        """保存本项目 SQLite 连接入口。"""
        self.database = database

    def get_model_id(self, kind: str) -> Optional[str]:
        """取得某类任务当前绑定的本地模型 ID。"""
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT model_id FROM governance_model_bindings WHERE kind = ?", (kind,)
            ).fetchone()
        return row["model_id"] if row else None

    def list_model_ids(self) -> Dict[str, str]:
        """列出三类任务已配置的模型绑定。"""
        with self.database.connect() as connection:
            rows = connection.execute("SELECT kind, model_id FROM governance_model_bindings").fetchall()
        return {row["kind"]: row["model_id"] for row in rows}

    def set_model_id(self, kind: str, model_id: str) -> None:
        """设置任务类型绑定；已创建任务继续使用原有模型快照。"""
        with self.database.connect() as connection:
            connection.execute(
                "INSERT INTO governance_model_bindings (kind, model_id, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(kind) DO UPDATE SET model_id = excluded.model_id, "
                "updated_at = excluded.updated_at",
                (kind, model_id, utc_now().isoformat()),
            )

    def get_business_database(self) -> Optional[Dict[str, Any]]:
        """读取业务库连接元数据及密文，只有服务端可调用。"""
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM business_database_config WHERE id = 1").fetchone()
        return dict(row) if row else None

    def set_business_database(self, *, host: str, port: int, database_name: str,
                              username: str, encrypted_password: str) -> None:
        """原子更新唯一一条业务库连接配置。"""
        with self.database.connect() as connection:
            connection.execute(
                "INSERT INTO business_database_config "
                "(id, host, port, database_name, username, encrypted_password, updated_at) "
                "VALUES (1, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                "host = excluded.host, port = excluded.port, database_name = excluded.database_name, "
                "username = excluded.username, encrypted_password = excluded.encrypted_password, "
                "updated_at = excluded.updated_at",
                (host, port, database_name, username, encrypted_password, utc_now().isoformat()),
            )
