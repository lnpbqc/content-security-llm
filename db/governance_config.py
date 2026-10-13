"""保存治理任务的选模绑定。"""

from typing import Dict, Optional

from db.database import Database
from utils.time import utc_now


class GovernanceConfigRepository:
    """统一读取管理员写入的治理任务配置。"""

    def __init__(self, database: Database):
        """保存本项目 MySQL 连接入口。"""
        self.database = database

    def get_model_id(self, kind: str) -> Optional[str]:
        """取得某类任务当前绑定的本地模型 ID。"""
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT model_id FROM llm_governance_model_bindings WHERE kind = %s", (kind,)
            ).fetchone()
        return row["model_id"] if row else None

    def list_model_ids(self) -> Dict[str, str]:
        """列出三类任务已配置的模型绑定。"""
        with self.database.connect() as connection:
            rows = connection.execute("SELECT kind, model_id FROM llm_governance_model_bindings").fetchall()
        return {row["kind"]: row["model_id"] for row in rows}

    def set_model_id(self, kind: str, model_id: str) -> None:
        """设置任务类型绑定；已创建任务继续使用原有模型快照。"""
        with self.database.connect() as connection:
            connection.execute(
                "INSERT INTO llm_governance_model_bindings (kind, model_id, updated_at) VALUES (%s, %s, %s) "
                "ON DUPLICATE KEY UPDATE model_id = VALUES(model_id), "
                "updated_at = VALUES(updated_at)",
                (kind, model_id, utc_now().isoformat()),
            )
