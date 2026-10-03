"""单实例训练 worker：读取业务数据、监管子进程、持久化事件及处理停止。"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from config import PROJECT_ROOT, Settings
from db.database import Database
from db.governance_config import GovernanceConfigRepository
from db.training import TrainingRepository, encode
from llm.models import ModelManager
from service.governance_data import BusinessGovernanceData


@contextmanager
def worker_lock(directory):
    """OS 文件锁确保第二个 worker 不会恢复仍在运行的任务。"""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "worker.lock").open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("训练 worker 已运行，只允许单实例") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


class TrainingWorker:
    def __init__(self, repository, data, cancel_timeout=30):
        self.repository = repository
        self.data = data
        self.cancel_timeout = cancel_timeout

    def _event(self, task_id, event):
        if event["type"] == "metric":
            self.repository.record_metric(task_id, event)
        elif event["type"] == "runtime":
            self.repository.runtime(task_id, event["metadata"])
        elif event["type"] == "artifact":
            filename = event["filename"]
            if filename not in {"environment.json", "split.json", "best.pt", "last.pt"}:
                raise ValueError("未知训练产物")
            self.repository.artifact(task_id, filename, event["kind"], event.get("epoch"), event.get("loss"))
        elif event["type"] != "finished":
            raise ValueError("未知训练事件")

    def run_once(self):
        task = self.repository.claim()
        if task is None:
            return False
        process = None
        directory = self.repository.directory(task["id"])
        try:
            directory.mkdir(parents=True, exist_ok=False)
            config = task["config"]
            (directory / "source.py").write_text(config["python_source"], encoding="utf-8")
            (directory / "config.json").write_text(encode(config), encoding="utf-8")
            for filename in ("source.py", "config.json"):
                self.repository.artifact(task["id"], filename, "source" if filename.endswith("py") else "metadata")
            samples = self.data.samples({"dataset_id": config["dataset_id"], "version_id": config["version_id"]})
            records = [{"id": sample["id"], "payload": sample["metadata"]} for sample in samples]
            (directory / "data.json").write_text(encode(records), encoding="utf-8")
            self.repository.artifact(task["id"], "data.json", "data")
            if self.repository.get(task["id"])["status"] == "canceling":
                self.repository.finish(task["id"], "canceled")
                return True
            events = queue.Queue()
            with (directory / "stderr.log").open("w", encoding="utf-8") as log:
                environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
                executable = sys.executable
                if os.name == "nt":
                    # uv 的虚拟环境 python.exe 是启动器；kill 必须作用于实际训练进程。
                    executable = sys._base_executable
                    environment["PYTHONPATH"] = os.pathsep.join([
                        str(PROJECT_ROOT), str(Path(sys.prefix) / "Lib" / "site-packages"),
                        environment.get("PYTHONPATH", ""),
                    ])
                process = subprocess.Popen(
                    [executable, "-m", "service.training_child", str(directory.resolve()),
                     "--parent-pid", str(os.getpid())],
                    cwd=str(PROJECT_ROOT), env=environment,
                    stdin=subprocess.DEVNULL if os.name == "nt" else subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8",
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )

                def read_events():
                    try:
                        for line in process.stdout:
                            events.put(line)
                    finally:
                        events.put(None)

                reader = threading.Thread(target=read_events, daemon=True)
                reader.start()
                deadline, finished, eof = None, None, False
                while not eof:
                    current = self.repository.get(task["id"])
                    if current["status"] == "canceling" and deadline is None:
                        (directory / "cancel.request").touch()
                        deadline = time.monotonic() + self.cancel_timeout
                    if deadline is not None and time.monotonic() >= deadline and process.poll() is None:
                        process.kill()
                    try:
                        line = events.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if line is None:
                        eof = True
                    else:
                        event = json.loads(line)
                        self._event(task["id"], event)
                        if event["type"] == "finished":
                            finished = event
                exit_code = process.wait()
                reader.join(timeout=1)
            if self.repository.get(task["id"])["status"] == "canceling":
                self.repository.finish(task["id"], "canceled")
            elif not finished or exit_code != 0 or finished["status"] == "failed":
                error = finished.get("error") if finished else None
                self.repository.finish(task["id"], "failed", error or "训练子进程异常退出，exit_code={}".format(exit_code))
            else:
                self.repository.finish(task["id"], finished["status"])
        except Exception as exc:
            self.repository.finish(task["id"], "failed", "{}: {}".format(type(exc).__name__, exc))
        except BaseException:
            self.repository.finish(task["id"], "failed", "训练 worker 中断；请新建任务重试")
            raise
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                if process.stdin is not None:
                    process.stdin.close()
                process.stdout.close()
            if (directory / "stderr.log").exists():
                self.repository.artifact(task["id"], "stderr.log", "log")
        return True


def main():
    parser = argparse.ArgumentParser(description="Process trusted Python training tasks")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    settings = Settings.load()
    database = Database(settings.database_path)
    database.initialize()
    repository = TrainingRepository(database)
    config = GovernanceConfigRepository(database)
    manager = ModelManager(database, settings.credential_key)
    worker = TrainingWorker(repository, BusinessGovernanceData(config, manager.cipher))
    with worker_lock(repository.root):
        repository.recover()
        while True:
            worked = worker.run_once()
            if args.once:
                return
            if not worked:
                time.sleep(1)


if __name__ == "__main__":
    main()
