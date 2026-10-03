"""保存治理任务、逐样本检查点和已发布的结果快照。"""

import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from db.database import Database
from utils.time import utc_now


def scope_json(scope: Dict[str, Any]) -> str:
    """把四元组范围转成稳定字符串，供精确匹配查询。"""
    return json.dumps(scope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _task(row: Any) -> Optional[Dict[str, Any]]:
    """把任务数据库行还原为含范围字典的对象。"""
    if row is None:
        return None
    task = dict(row)
    task["scope"] = json.loads(task.pop("scope_json"))
    return task


def _result(row: Any) -> Optional[Dict[str, Any]]:
    """把结果数据库行还原为汇总和样本快照。"""
    if row is None:
        return None
    result = dict(row)
    result["scope"] = json.loads(result.pop("scope_json"))
    result["summary"] = json.loads(result.pop("summary_json"))
    result["samples"] = json.loads(result.pop("samples_json"))
    return result


class GovernanceRepository:
    """集中处理三张治理表的 SQLite 读写。"""
    def __init__(self, database: Database):
        """保存用于访问治理表的数据库连接入口。"""
        self.database = database

    def create_task(self, kind: str, token_hash: str, name: str,
                    scope: Dict[str, Any], model_id: str, model_version: str) -> Dict[str, Any]:
        """将新任务以 pending 状态入队，并返回保存后的记录。"""
        task_id = str(uuid4())
        now = utc_now().isoformat()
        with self.database.connect() as connection:
            connection.execute(
                """INSERT INTO governance_tasks
                   (id, kind, token_hash, name, scope_json, model_id, model_version,
                    status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)""",
                (task_id, kind, token_hash, name, scope_json(scope), model_id, model_version, now, now),
            )
        return self.get_task(task_id, token_hash)

    def get_task(self, task_id: str, token_hash: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """按任务 ID 查询；提供令牌摘要时只允许查该令牌的任务。"""
        query = "SELECT * FROM governance_tasks WHERE id = ?"
        params = [task_id]
        if token_hash is not None:
            query += " AND token_hash = ?"
            params.append(token_hash)
        with self.database.connect() as connection:
            row = connection.execute(query, params).fetchone()
        return _task(row)

    def list_tasks(self, kind: str, token_hash: str, scope: Dict[str, Any],
                   limit: int, offset: int) -> Dict[str, Any]:
        """列出当前令牌、任务类型及范围完全匹配的任务。"""
        params = (token_hash, kind, scope_json(scope))
        with self.database.connect() as connection:
            total = connection.execute(
                "SELECT count(*) FROM governance_tasks WHERE token_hash = ? AND kind = ? AND scope_json = ?",
                params,
            ).fetchone()[0]
            rows = connection.execute(
                """SELECT * FROM governance_tasks WHERE token_hash = ? AND kind = ? AND scope_json = ?
                   ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?""",
                (*params, limit, offset),
            ).fetchall()
        return {"items": [_task(row) for row in rows], "total": total}

    def recover_running(self) -> None:
        """worker 重启时将中断的 running 任务放回待处理队列。"""
        with self.database.connect() as connection:
            connection.execute(
                "UPDATE governance_tasks SET status = 'pending', updated_at = ? WHERE status = 'running'",
                (utc_now().isoformat(),),
            )

    def claim(self) -> Optional[Dict[str, Any]]:
        """原子领取最早的 pending 任务并改为 running。"""
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM governance_tasks WHERE status = 'pending' ORDER BY created_at, id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE governance_tasks SET status = 'running', updated_at = ? WHERE id = ?",
                (utc_now().isoformat(), row["id"]),
            )
            return _task(connection.execute(
                "SELECT * FROM governance_tasks WHERE id = ?", (row["id"],)
            ).fetchone())

    def sample_outcomes(self, task_id: str) -> Dict[str, Dict[str, Any]]:
        """读取任务已保存的逐样本检查点，供续跑和汇总使用。"""
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM governance_task_samples WHERE task_id = ?", (task_id,)
            ).fetchall()
        return {row["sample_id"]: {
            "status": row["status"],
            "output": json.loads(row["output_json"]) if row["output_json"] else None,
            "error_code": row["error_code"],
        } for row in rows}

    def save_sample(self, task_id: str, sample_id: str, output: Optional[Dict[str, Any]],
                    error_code: Optional[str] = None) -> None:
        """首次保存一条样本的成功输出或错误，不覆盖已有检查点。"""
        with self.database.connect() as connection:
            connection.execute(
                """INSERT INTO governance_task_samples
                   (task_id, sample_id, status, output_json, error_code)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(task_id, sample_id) DO NOTHING""",
                (task_id, sample_id, "error" if error_code else "success",
                 json.dumps(output, ensure_ascii=False) if output is not None else None, error_code),
            )

    def publish(self, task: Dict[str, Any], summary: Dict[str, Any],
                samples: List[Dict[str, Any]], result_id: str) -> None:
        """在同一事务中发布只读结果快照并完成任务。"""
        now = utc_now().isoformat()
        with self.database.connect() as connection:
            connection.execute(
                """INSERT INTO governance_results
                   (id, task_id, kind, token_hash, scope_json, summary_json, samples_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (result_id, task["id"], task["kind"], task["token_hash"],
                 scope_json(task["scope"]), json.dumps(summary, ensure_ascii=False),
                 json.dumps(samples, ensure_ascii=False), now),
            )
            connection.execute(
                """UPDATE governance_tasks SET status = 'succeeded', result_id = ?, updated_at = ?
                   WHERE id = ? AND status = 'running'""",
                (result_id, now, task["id"]),
            )

    def fail(self, task_id: str, message: str) -> None:
        """将运行中的任务标记为失败并保存失败原因。"""
        with self.database.connect() as connection:
            connection.execute(
                """UPDATE governance_tasks SET status = 'failed', error_message = ?, updated_at = ?
                   WHERE id = ? AND status = 'running'""",
                (message, utc_now().isoformat(), task_id),
            )

    def get_result(self, result_id: str, token_hash: str,
                   kind: str) -> Optional[Dict[str, Any]]:
        """仅向创建者令牌返回指定类型的结果快照。"""
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM governance_results WHERE id = ? AND token_hash = ? AND kind = ?",
                (result_id, token_hash, kind),
            ).fetchone()
        return _result(row)

    def list_results(self, kind: str, token_hash: str,
                     scope: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """按令牌和类型读取快照；指定范围时做完整匹配，新的排在前面。"""
        query = "SELECT * FROM governance_results WHERE token_hash = ? AND kind = ?"
        params = [token_hash, kind]
        if scope is not None:
            query += " AND scope_json = ?"
            params.append(scope_json(scope))
        query += " ORDER BY created_at DESC, id DESC"
        with self.database.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [_result(row) for row in rows]
