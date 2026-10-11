"""保存独立训练任务、增量指标和产物，所有前端查询按令牌隔离。"""

from __future__ import annotations

import json
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from db.database import Database
from schemas.training import METRIC_NAMES
from utils.time import utc_now


ACTIVE_STATUSES = ("pending", "running", "canceling")


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class TrainingRepository:
    def __init__(self, database: Database):
        self.database = database
        self.root = database.data_dir / "training-runs"

    def directory(self, task_id: str) -> Path:
        return self.root / task_id

    def create(self, config: dict, token_hash: str) -> dict:
        task_id, now = str(uuid4()), utc_now().isoformat()
        source_hash = sha256(config["python_source"].encode("utf-8")).hexdigest()
        with self.database.connect() as connection:
            connection.execute(
                """INSERT INTO llm_training_tasks
                   (id, token_hash, config_json, source_hash, status, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, 'pending', %s, %s)""",
                (task_id, token_hash, encode(config), source_hash, now, now),
            )
        return self.get(task_id, token_hash)

    def get(self, task_id: str, token_hash: str | None = None) -> dict | None:
        with self.database.connect() as connection:
            query, args = "SELECT * FROM llm_training_tasks WHERE id = %s", [task_id]
            if token_hash is not None:
                query += " AND token_hash = %s"
                args.append(token_hash)
            row = connection.execute(query, args).fetchone()
        if row is None:
            return None
        task = dict(row)
        task["config"] = json.loads(task.pop("config_json"))
        task["runtime"] = json.loads(task.pop("runtime_json"))
        task["latest_metrics"] = json.loads(task.pop("latest_metrics_json"))
        return task

    def list(self, token_hash: str, page: int, page_size: int) -> dict:
        with self.database.connect() as connection:
            total = connection.execute("SELECT COUNT(*) AS total FROM llm_training_tasks WHERE token_hash = %s",
                                       (token_hash,)).fetchone()["total"]
            rows = connection.execute(
                "SELECT id FROM llm_training_tasks WHERE token_hash = %s ORDER BY created_at DESC, id LIMIT %s OFFSET %s",
                (token_hash, page_size, (page - 1) * page_size),
            ).fetchall()
        return {"items": [self.public(self.get(row["id"], token_hash)) for row in rows],
                "total": total, "page": page, "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size}

    def claim(self) -> dict | None:
        with self.database.connect() as connection:
            connection.execute("START TRANSACTION")
            row = connection.execute(
                "SELECT id FROM llm_training_tasks WHERE status = 'pending' ORDER BY created_at, id LIMIT 1 FOR UPDATE"
            ).fetchone()
            if row is None:
                return None
            now = utc_now().isoformat()
            connection.execute(
                "UPDATE llm_training_tasks SET status = 'running', started_at = %s, updated_at = %s WHERE id = %s",
                (now, now, row["id"]),
            )
        return self.get(row["id"])

    def cancel(self, task_id: str, token_hash: str) -> dict | None:
        now = utc_now().isoformat()
        with self.database.connect() as connection:
            connection.execute(
                """UPDATE llm_training_tasks SET
                   finished_at = CASE WHEN status = 'pending' THEN %s ELSE finished_at END,
                   updated_at = %s,
                   status = CASE WHEN status = 'pending' THEN 'canceled' ELSE 'canceling' END
                   WHERE id = %s AND token_hash = %s AND status IN ('pending', 'running')""",
                (now, now, task_id, token_hash),
            )
        return self.get(task_id, token_hash)

    def finish(self, task_id: str, status: str, error: str | None = None):
        if status not in ("succeeded", "failed", "canceled"):
            raise ValueError("invalid terminal status")
        now = utc_now().isoformat()
        with self.database.connect() as connection:
            # 已接受的取消请求优先于与其并发发生的训练完成。
            connection.execute(
                """UPDATE llm_training_tasks SET
                   error_message = %s, finished_at = %s, updated_at = %s,
                   progress = CASE WHEN %s = 'succeeded' AND status != 'canceling' THEN 100 ELSE progress END,
                   status = CASE WHEN status = 'canceling' THEN 'canceled' ELSE %s END
                   WHERE id = %s AND status IN ('running', 'canceling')""",
                (error, now, now, status, status, task_id),
            )

    def recover(self):
        now = utc_now().isoformat()
        with self.database.connect() as connection:
            connection.execute(
                """UPDATE llm_training_tasks SET status = 'failed', error_message = '训练 worker 中断；请新建任务重试',
                   updated_at = %s, finished_at = %s WHERE status IN ('running', 'canceling')""", (now, now),
            )

    def runtime(self, task_id: str, metadata: dict):
        with self.database.connect() as connection:
            connection.execute("UPDATE llm_training_tasks SET runtime_json = %s, updated_at = %s WHERE id = %s",
                               (encode(metadata), utc_now().isoformat(), task_id))

    def record_metric(self, task_id: str, event: dict):
        now = utc_now().isoformat()
        with self.database.connect() as connection:
            connection.execute(
                """INSERT INTO llm_training_metrics (task_id, epoch, step, phase, metrics_json, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (task_id, event["epoch"], event["step"], event["phase"], encode(event["metrics"]), now),
            )
            connection.execute(
                """UPDATE llm_training_tasks SET epoch = %s, step = %s, progress = %s,
                   latest_metrics_json = %s, updated_at = %s WHERE id = %s""",
                (event["epoch"], event["step"], event["progress"], encode(event["metrics"]), now, task_id),
            )

    def metrics(self, task_id: str, after_id: int = 0, limit: int = 200) -> dict:
        if self.get(task_id)["config"].get("runner") == "script":
            return self.script_events(task_id, after_id, limit, "metric")
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM llm_training_metrics WHERE task_id = %s AND id > %s ORDER BY id LIMIT %s",
                (task_id, after_id, limit + 1),
            ).fetchall()
        items = []
        for row in rows[:limit]:
            item = dict(row)
            item["metrics"] = json.loads(item.pop("metrics_json"))
            items.append(item)
        return {"items": items, "next_after_id": items[-1]["id"] if items else after_id,
                "has_more": len(rows) > limit}

    def record_script_event(self, task_id: str, event: dict):
        now = utc_now().isoformat()
        with self.database.connect() as connection:
            connection.execute(
                "INSERT INTO llm_training_script_events (task_id, type, event_json, created_at) VALUES (%s, %s, %s, %s)",
                (task_id, event["type"], encode(event), now),
            )
            connection.execute("UPDATE llm_training_tasks SET updated_at = %s WHERE id = %s", (now, task_id))
            if event["type"] == "metric":
                connection.execute("UPDATE llm_training_tasks SET latest_metrics_json = %s WHERE id = %s",
                                   (encode(event["metrics"]), task_id))

    def script_events(self, task_id: str, after_id: int = 0, limit: int = 200, event_type=None):
        with self.database.connect() as connection:
            query = "SELECT * FROM llm_training_script_events WHERE task_id = %s AND id > %s"
            args = [task_id, after_id]
            if event_type is not None:
                query += " AND type = %s"
                args.append(event_type)
            rows = connection.execute(query + " ORDER BY id LIMIT %s", (*args, limit + 1)).fetchall()
        items = [{**json.loads(row["event_json"]), "id": row["id"], "task_id": task_id,
                  "created_at": row["created_at"]} for row in rows[:limit]]
        return {"items": items, "next_after_id": items[-1]["id"] if items else after_id,
                "has_more": len(rows) > limit}

    def public_script(self, task):
        config = {key: value for key, value in task["config"].items() if key != "python_source"}
        with self.database.connect() as connection:
            latest = {}
            for event_type in ("metric", "progress"):
                row = connection.execute(
                    "SELECT event_json FROM llm_training_script_events WHERE task_id = %s AND type = %s ORDER BY id DESC LIMIT 1",
                    (task["id"], event_type),
                ).fetchone()
                latest[event_type] = json.loads(row["event_json"]) if row else {}
        started = datetime.fromisoformat(task["started_at"]) if task["started_at"] else None
        ended = datetime.fromisoformat(task["finished_at"]) if task["finished_at"] else utc_now()
        return {**config, "id": task["id"], "config": config, "status": task["status"],
                **{key: task[key] for key in ("created_at", "updated_at", "started_at", "finished_at",
                                             "source_hash", "runtime", "error_message")},
                "elapsed_seconds": max(0, int((ended - started).total_seconds())) if started else 0,
                "epoch": latest["metric"].get("epoch"), "step": latest["metric"].get("step"),
                "latest_phase": latest["metric"].get("phase"),
                "latest_metrics": latest["metric"].get("metrics", {}),
                "progress": latest["progress"].get("progress")}

    def artifact(self, task_id: str, filename: str, kind: str, epoch: int | None = None,
                 loss: float | None = None):
        root = self.directory(task_id).resolve()
        path = (root / filename).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("产物必须为任务目录内的文件")
        digest = sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        with self.database.connect() as connection:
            connection.execute(
                """INSERT INTO llm_training_artifacts (id, task_id, filename, kind, epoch, loss, size, sha256)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE
                   epoch=VALUES(epoch), loss=VALUES(loss), size=VALUES(size), sha256=VALUES(sha256)""",
                (str(uuid4()), task_id, filename, kind, epoch, loss, path.stat().st_size, digest.hexdigest()),
            )

    def artifacts(self, task_id: str) -> list[dict]:
        with self.database.connect() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT * FROM llm_training_artifacts WHERE task_id = %s ORDER BY kind, filename", (task_id,)
            )]

    def public(self, task: dict) -> dict:
        if task["config"].get("runner") == "script":
            return self.public_script(task)
        config = {key: value for key, value in task["config"].items() if key != "python_source"}
        with self.database.connect() as connection:
            history = connection.execute(
                "SELECT epoch, metrics_json FROM llm_training_metrics WHERE task_id = %s AND phase = 'validation' ORDER BY id",
                (task["id"],),
            ).fetchall()
        history = [{"epoch": row["epoch"], **json.loads(row["metrics_json"])} for row in history]
        with self.database.connect() as connection:
            latest = connection.execute("SELECT phase FROM llm_training_metrics WHERE task_id = %s ORDER BY id DESC LIMIT 1",
                                        (task["id"],)).fetchone()
        checkpoints = [{"name": item["filename"], "artifact_id": item["id"], "epoch": item["epoch"],
                        "loss": item["loss"]} for item in self.artifacts(task["id"]) if item["kind"] == "weights"]
        started = datetime.fromisoformat(task["started_at"]) if task["started_at"] else None
        ended = datetime.fromisoformat(task["finished_at"]) if task["finished_at"] else utc_now()
        elapsed = max(0, int((ended - started).total_seconds())) if started else 0
        return {**config, "id": task["id"], "dataset_version": config["version_id"],
                "description": "自定义 PyTorch 训练", "status": task["status"], "epoch": task["epoch"],
                "step": task["step"], "progress": task["progress"], "elapsed": "{} 秒".format(elapsed),
                "elapsed_seconds": elapsed, "created_at": task["created_at"], "updated_at": task["updated_at"],
                "started_at": task["started_at"], "finished_at": task["finished_at"],
                "config": config, "source_hash": task["source_hash"], "runtime": task["runtime"],
                "error_message": task["error_message"],
                "latest_phase": latest["phase"] if latest else None,
                "latest_metrics": {**dict.fromkeys(METRIC_NAMES), **task["latest_metrics"]},
                "loss": [row["train_loss"] for row in history],
                "validation_loss": [row["loss"] for row in history],
                "metric_history": history, "checkpoints": checkpoints}
