"""独立分析记录；输入只在创建时写入，查询始终按令牌过滤。"""

import json
from uuid import uuid4

from db.database import Database
from utils.time import utc_now


class AnalysisRepository:
    def __init__(self, database: Database):
        self.database = database

    def create(self, token_hash, kind, payload, state):
        record_id = str(uuid4())
        with self.database.connect() as connection:
            connection.execute(
                """INSERT INTO analysis_records
                   (id, token_hash, kind, scenario_code, model_id, input_json, state_json, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?)""",
                (record_id, token_hash, kind, payload.get("scenario_code"), payload["model_id"],
                 json.dumps(payload, ensure_ascii=False), json.dumps(state), utc_now().isoformat()),
            )
        return record_id

    def save(self, record_id, token_hash, state):
        with self.database.connect() as connection:
            connection.execute(
                "UPDATE analysis_records SET state_json=?, status=? WHERE id=? AND token_hash=?",
                (json.dumps(state, ensure_ascii=False), state["status"], record_id, token_hash),
            )

    def recover_running(self):
        """单 API 实例启动时结束上次中断的分析，保留输入和已有调用。"""
        finished_at = utc_now().isoformat()
        with self.database.connect() as connection:
            rows = connection.execute("SELECT id, state_json FROM analysis_records WHERE status='running'").fetchall()
            for row in rows:
                state = json.loads(row["state_json"])
                state.update(status="failed", error_code="analysis_interrupted",
                             error_message="模型分析因服务中断而终止，请重新提交", finished_at=finished_at)
                connection.execute(
                    "UPDATE analysis_records SET state_json=?, status='failed' WHERE id=? AND status='running'",
                    (json.dumps(state, ensure_ascii=False), row["id"]),
                )

    @staticmethod
    def public(row):
        return {"id": row["id"], "kind": row["kind"], "scenario_code": row["scenario_code"],
                "model_id": row["model_id"], "input": json.loads(row["input_json"]),
                "created_at": row["created_at"], "status": row["status"],
                **json.loads(row["state_json"])}

    def get(self, record_id, token_hash):
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM analysis_records WHERE id=? AND token_hash=?",
                                     (record_id, token_hash)).fetchone()
        return self.public(row) if row else None

    def list(self, token_hash, page, page_size, kind=None, scenario_code=None, status=None):
        where, values = ["token_hash=?"], [token_hash]
        for column, value in (("kind", kind), ("scenario_code", scenario_code), ("status", status)):
            if value is not None:
                where.append(column + "=?")
                values.append(value)
        clause = " AND ".join(where)
        with self.database.connect() as connection:
            total = connection.execute("SELECT COUNT(*) FROM analysis_records WHERE " + clause,
                                       values).fetchone()[0]
            rows = connection.execute(
                "SELECT * FROM analysis_records WHERE " + clause +
                " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                [*values, page_size, (page - 1) * page_size],
            ).fetchall()
        return {"items": [self.public(row) for row in rows], "total": total,
                "page": page, "page_size": page_size, "total_pages": (total + page_size - 1) // page_size}
