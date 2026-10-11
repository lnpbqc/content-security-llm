"""统一脚本进程监管；控制台输出与结构化事件分离。"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from config import PROJECT_ROOT
from db.training import encode
from schemas.script_training import SCRIPT_EVENT


class EventReader:
    def __init__(self, path, consume):
        self.path = path
        self.consume = consume
        self.offset = 0
        self.pending = b""

    def read(self, final=False):
        if not self.path.exists():
            return
        with self.path.open("rb") as stream:
            if self.path.stat().st_size < self.offset:
                raise ValueError("events.jsonl 必须追加写入，不能截断")
            stream.seek(self.offset)
            while chunk := stream.read(65536):
                self.offset += len(chunk)
                self.pending += chunk
                while b"\n" in self.pending:
                    line, self.pending = self.pending.split(b"\n", 1)
                    self._line(line)
        if final and self.pending:
            line, self.pending = self.pending, b""
            self._line(line)

    def _line(self, line):
        if not line.strip():
            return
        event = SCRIPT_EVENT.validate_json(line).model_dump(exclude_none=True)
        self.consume(event)


class WindowsJob:
    """关闭 job（包括父 worker 崩溃）时终止关联的全部子进程。"""
    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self.ctypes = ctypes
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        # JOBOBJECT_EXTENDED_LIMIT_INFORMATION，64 位布局：LimitFlags 位于偏移 16。
        information = ctypes.create_string_buffer(144 if ctypes.sizeof(ctypes.c_void_p) == 8 else 112)
        ctypes.c_uint32.from_buffer(information, 16).value = 0x2000  # KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(self.handle, 9, information, len(information)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, process):
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise self.ctypes.WinError(self.ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def run_script_task(repository, task, cancel_timeout=30):
    directory = repository.directory(task["id"]).resolve()
    output = directory / "outputs"
    process, job, completed, failure = None, None, False, None
    try:
        directory.mkdir(parents=True, exist_ok=False)
        output.mkdir()
        config = task["config"]
        (directory / "source.py").write_text(config["python_source"], encoding="utf-8", newline="\n")
        launch = {"task_id": task["id"], "script_id": config["script_id"], "inputs": config["inputs"],
                  "parameters": config["parameters"], "output_dir": str(output),
                  "events_path": str(directory / "events.jsonl"),
                  "cancel_path": str(directory / "cancel.request"),
                  "data_dir": str(repository.root.parent / "training-data" / config["script_id"])}
        (directory / "config.json").write_text(encode(launch), encoding="utf-8")
        for filename, kind in (("source.py", "source"), ("config.json", "metadata")):
            repository.artifact(task["id"], filename, kind)

        def consume(event):
            if event["type"] == "artifact":
                relative = Path(event["filename"])
                path = (output / relative).resolve()
                if relative.is_absolute() or not path.is_relative_to(output.resolve()) or not path.is_file():
                    raise ValueError("脚本产物必须为 output_dir 内的文件")
                repository.artifact(task["id"], path.relative_to(directory).as_posix(), event["kind"])
            repository.record_script_event(task["id"], event)

        reader = EventReader(directory / "events.jsonl", consume)
        if repository.get(task["id"])["status"] == "canceling":
            completed = True
            return
        environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        executable = sys.executable
        if os.name == "nt":
            executable = sys._base_executable
            environment["PYTHONPATH"] = os.pathsep.join([
                str(PROJECT_ROOT), str(Path(sys.prefix) / "Lib" / "site-packages"), environment.get("PYTHONPATH", "")])
            job = WindowsJob()
        with (directory / "run.log").open("wb") as log:
            process = subprocess.Popen(
                [executable, "-m", "service.script_training_child", str(directory)],
                cwd=PROJECT_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL if os.name == "nt" else subprocess.PIPE,
                start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if job is not None:
                job.assign(process)
            deadline = None
            while process.poll() is None:
                reader.read()
                if repository.get(task["id"])["status"] == "canceling":
                    if deadline is None:
                        (directory / "cancel.request").touch()
                        deadline = time.monotonic() + cancel_timeout
                        if os.name != "nt":
                            try:
                                os.killpg(process.pid, signal.SIGTERM)
                            except ProcessLookupError:
                                pass
                    if time.monotonic() >= deadline:
                        if job is not None:
                            job.close()
                        else:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                time.sleep(0.1)
            reader.read(final=True)
            if process.returncode and repository.get(task["id"])["status"] != "canceling":
                raise RuntimeError("脚本异常退出，exit_code={}；详情见 run.log".format(process.returncode))
        completed = True
    except Exception as exc:
        failure = "{}: {}".format(type(exc).__name__, exc)
    except BaseException:
        failure = "脚本 worker 中断；请新建任务重试"
        raise
    finally:
        if job is not None:
            job.close()
        elif process is not None and os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait()
            if process.stdin is not None:
                process.stdin.close()
        try:
            if failure and directory.is_dir():
                with (directory / "run.log").open("ab") as log:
                    log.write(("\n[worker] " + failure + "\n").encode("utf-8"))
            for filename in ("run.log", "events.jsonl"):
                if (directory / filename).is_file():
                    repository.artifact(task["id"], filename, "log")
            # 更新被脚本反复覆盖的权重校验和，发布终态后日志和产物保持稳定。
            for artifact in repository.artifacts(task["id"]):
                if artifact["filename"].startswith("outputs/"):
                    repository.artifact(task["id"], artifact["filename"], artifact["kind"])
        except Exception as exc:
            failure = failure or "产物收集失败：{}: {}".format(type(exc).__name__, exc)
        if failure:
            repository.finish(task["id"], "failed", failure)
        elif completed:
            repository.finish(task["id"], "succeeded")
