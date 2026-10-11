"""在 Linux 上用假服务验证启动脚本，不加载项目配置、数据库或 GPU。"""

import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest


@unittest.skipUnless(sys.platform == "linux", "需要 Linux /proc、setsid 和 flock")
class StartScriptTests(unittest.TestCase):
    def setUp(self):
        self.workspace = Path(__file__).resolve().parents[1]
        self.temp = tempfile.TemporaryDirectory(prefix="launcher-test-", dir=self.workspace)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.script = self.root / "start-content-security.sh"
        shutil.copyfile(self.workspace / self.script.name, self.script)
        (self.root / ".venv/bin").mkdir(parents=True)
        python = self.root / ".venv/bin/python"
        python.write_text("""#!/usr/bin/env bash
if [[ "$1" == -c ]]; then
  if [[ "$2" == *socket* ]]; then exit "${FAKE_PORT_BUSY:-0}"; fi
  exit "${FAKE_CONFIG_BAD:-0}"
fi
module="$2"
[[ "${FAIL_MODULE:-}" != "$module" ]] || exit 1
echo "$module started"
/bin/sleep 300 &
child=$!
echo "$child" > "logs/$module.child.pid"
trap 'kill "$child" 2>/dev/null; wait "$child" 2>/dev/null; exit 0' TERM INT
wait "$child"
""", encoding="utf-8")
        python.chmod(0o700)
        self.addCleanup(self.cleanup_processes)

    def run_script(self, *args, **environment):
        # 从其他工作目录调用，确保脚本按自身位置定位项目。
        return subprocess.run(
            ["bash", str(self.script), *args], cwd=self.workspace,
            env=dict(os.environ, **environment), text=True, capture_output=True, timeout=15,
        )

    def pid(self, name):
        return int((self.root / "logs" / (name + ".pid")).read_text())

    def alive(self, pid):
        try:
            return Path("/proc/{}/stat".format(pid)).read_text().rsplit(")", 1)[1].split()[0] != "Z"
        except FileNotFoundError:
            return False

    def cleanup_processes(self):
        for name in ("api", "governance", "training", "neural-trace"):
            file = self.root / "logs" / (name + ".pid")
            if file.exists():
                try:
                    os.killpg(int(file.read_text()), 15)
                except ProcessLookupError:
                    pass

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_start_repeat_restart_status_and_stop(self):
        self.assert_success(self.run_script())
        original = {name: self.pid(name) for name in ("api", "governance")}
        self.assertFalse((self.root / "logs/training.pid").exists())
        self.assert_success(self.run_script("start"))
        self.assertEqual(original, {name: self.pid(name) for name in original})
        status = self.run_script("status")
        self.assert_success(status)
        self.assertIn("api 运行中", status.stdout)
        self.assertIn("training 未运行", status.stdout)
        self.assert_success(self.run_script("start", "training"))
        original["training"] = self.pid("training")
        self.assert_success(self.run_script("restart"))
        for name, pid in original.items():
            self.assertNotEqual(self.pid(name), pid)
            self.assertFalse(self.alive(pid))
        self.assertFalse((self.root / "logs/neural-trace.pid").exists())
        self.assert_success(self.run_script("stop"))
        self.assertFalse(any((self.root / "logs").glob("api.pid")))

    def test_stop_includes_worker_child(self):
        self.assert_success(self.run_script("start", "training"))
        parent = self.pid("training")
        child = self.pid("service.training_worker.child")
        self.assertTrue(self.alive(child))
        self.assert_success(self.run_script("stop", "training"))
        self.assertFalse(self.alive(parent))
        self.assertFalse(self.alive(child))

    def test_stale_pid_does_not_stop_unrelated_process(self):
        (self.root / "logs").mkdir()
        unrelated = subprocess.Popen(["sleep", "300"], cwd=self.root)
        try:
            (self.root / "logs/api.pid").write_text(str(unrelated.pid))
            self.assert_success(self.run_script("stop", "api"))
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)

    def test_failed_start_removes_pid_and_reports_failure(self):
        result = self.run_script("start", "governance", FAIL_MODULE="service.governance_worker")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("启动失败", result.stderr)
        self.assertFalse((self.root / "logs/governance.pid").exists())

    def test_preflight_rejects_configuration_and_occupied_port(self):
        for environment in ({"FAKE_CONFIG_BAD": "1"}, {"FAKE_PORT_BUSY": "1"}):
            with self.subTest(environment=environment):
                result = self.run_script("start", **environment)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.root / "logs/api.pid").exists())

    def test_concurrent_operation_is_rejected(self):
        import fcntl
        (self.root / "logs").mkdir()
        with (self.root / "logs/launcher.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = self.run_script("start")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("另一条启动/停止命令", result.stderr)

    def test_unknown_service_is_rejected(self):
        result = self.run_script("start", "unknown")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("用法", result.stdout)

    def test_log_follows_service_output_without_stopping_service(self):
        self.assert_success(self.run_script("start", "governance"))
        process = subprocess.Popen(
            ["bash", str(self.script), "log", "governance"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
        )
        try:
            time.sleep(0.2)
            process.terminate()
            output, _ = process.communicate(timeout=5)
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            process.communicate(timeout=5)
        self.assertIn("service.governance_worker started", output)
        self.assertTrue(self.alive(self.pid("governance")))

    def test_foreground_service_exits_with_its_child(self):
        process = subprocess.Popen(
            ["bash", str(self.script), "fg", "training"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
        )
        child_file = self.root / "logs/service.training_worker.child.pid"
        try:
            deadline = time.monotonic() + 5
            while not child_file.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(child_file.exists())
            child = int(child_file.read_text())
            os.killpg(process.pid, signal.SIGINT)
            output, _ = process.communicate(timeout=5)
            self.assertIn("前台运行", output)
            self.assertFalse(self.alive(child))
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
