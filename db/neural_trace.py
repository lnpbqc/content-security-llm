"""神经因果溯源任务队列；结果与任务状态在同一事务发布。"""

import json
from uuid import uuid4

from db.database import Database
from utils.time import utc_now


def decode(row):
    if row is None:
        return None
    task = dict(row)
    task["input"] = json.loads(task.pop("input_json"))
    result = task.pop("result_json")
    task["result"] = json.loads(result) if result else None
    return task


class NeuralTraceRepository:
    def __init__(self, database: Database):
        self.database = database

    def create(self, token_hash, case_id, snapshot):
        task_id, now = str(uuid4()), utc_now().isoformat()
        with self.database.connect() as connection:
            connection.execute(
                """INSERT INTO neural_trace_tasks
                   (id, token_hash, case_id, input_json, status, stage, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 'pending', 'queued', ?, ?)""",
                (task_id, token_hash, case_id, json.dumps(snapshot, ensure_ascii=False), now, now),
            )
        return self.get(task_id, token_hash)

    def get(self, task_id, token_hash):
        with self.database.connect() as connection:
            return decode(connection.execute(
                "SELECT * FROM neural_trace_tasks WHERE id = ? AND token_hash = ?",
                (task_id, token_hash),
            ).fetchone())

    def list(self, token_hash, limit, offset):
        with self.database.connect() as connection:
            total = connection.execute(
                "SELECT count(*) FROM neural_trace_tasks WHERE token_hash = ?", (token_hash,),
            ).fetchone()[0]
            rows = connection.execute(
                """SELECT * FROM neural_trace_tasks WHERE token_hash = ?
                   ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?""", (token_hash, limit, offset),
            ).fetchall()
        return {"items": [decode(row) for row in rows], "total": total}

    def claim(self):
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM neural_trace_tasks WHERE status = 'pending' ORDER BY created_at, id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            now = utc_now().isoformat()
            connection.execute(
                """UPDATE neural_trace_tasks SET status = 'running', stage = 'preflight',
                   started_at = ?, updated_at = ? WHERE id = ?""", (now, now, row["id"]),
            )
            return decode(connection.execute(
                "SELECT * FROM neural_trace_tasks WHERE id = ?", (row["id"],),
            ).fetchone())

    def stage(self, task_id, stage):
        with self.database.connect() as connection:
            connection.execute(
                "UPDATE neural_trace_tasks SET stage = ?, updated_at = ? WHERE id = ? AND status = 'running'",
                (stage, utc_now().isoformat(), task_id),
            )

    def finish(self, task_id, result=None, error_message=None):
        now = utc_now().isoformat()
        with self.database.connect() as connection:
            connection.execute(
                """UPDATE neural_trace_tasks SET status = ?, result_json = ?, error_message = ?,
                   stage = CASE WHEN ? IS NULL THEN 'completed' ELSE stage END,
                   finished_at = ?, updated_at = ? WHERE id = ? AND status = 'running'""",
                ("failed" if error_message else "succeeded",
                 json.dumps(result, ensure_ascii=False, allow_nan=False) if result is not None else None,
                 error_message, error_message, now, now, task_id),
            )

    def recover_running(self):
        """中断的任务标记失败，保留旧产物；用户重新建单，不覆盖或隐式重跑。"""
        now = utc_now().isoformat()
        with self.database.connect() as connection:
            connection.execute(
                """UPDATE neural_trace_tasks SET status = 'failed', error_message = ?,
                   finished_at = ?, updated_at = ? WHERE status = 'running'""",
                ("worker 中断，任务未完成；请重新提交任务。", now, now),
            )
