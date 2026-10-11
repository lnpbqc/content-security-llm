"""隔离 SQLite + 本地真实 HTTP + GPU 训练验收，不读取真实服务配置。"""

import argparse
import json
import secrets
import socket
import sys
import threading
import time
from datetime import timedelta
from hashlib import sha256
from pathlib import Path

import httpx
import uvicorn
from cryptography.fernet import Fernet

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PROJECT_ROOT, Settings
from main import create_app
from service.training_worker import TrainingWorker
from utils.time import utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--validation-samples", type=int, default=2000)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    root = PROJECT_ROOT / "db" / "script-training-validation"
    root.mkdir(parents=True, exist_ok=True)
    setup_secret, token = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    settings = Settings(root / "validation.sqlite3", setup_secret, Fernet.generate_key())
    app = create_app(settings)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", log_level="warning", access_log=False))
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    api = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    api.start()
    worker = None
    try:
        deadline = time.monotonic() + 30
        while not server.started and api.is_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not server.started:
            raise RuntimeError("隔离 HTTP 服务启动失败")
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30) as client:
            response = client.post("/api/auth/token", headers={"X-Setup-Secret": setup_secret}, json={
                "token": token, "expires_at": (utc_now() + timedelta(hours=2)).isoformat(), "label": "隔离 GPU 验收"})
            response.raise_for_status()
            headers = {"Authorization": "Bearer " + token}
            response = client.post("/api/v1/training/script-tasks", headers={**headers, "X-Setup-Secret": setup_secret},
                                   json={"name": "CIFAR-100 本地 GPU 验收", "script_id": "cifar100-cnn",
                                         "parameters": {"epochs": args.epochs, "device": args.device,
                                                        "train_samples": args.train_samples,
                                                        "validation_samples": args.validation_samples}})
            response.raise_for_status()
            task = response.json()["data"]
            path = "/api/v1/training/tasks/" + task["id"]
            repository = app.state.training_service.repository
            worker = threading.Thread(target=TrainingWorker(repository, None).run_once, daemon=True)
            worker.start()
            after_id, offset, observed_metrics, validation = 0, 0, [], []
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                response = client.get(path, headers=headers)
                response.raise_for_status()
                task = response.json()["data"]
                response = client.get(path + "/metrics", headers=headers, params={"after_id": after_id, "limit": 1000})
                response.raise_for_status()
                page = response.json()["data"]
                after_id = page["next_after_id"]
                observed_metrics.extend(page["items"])
                for event in page["items"]:
                    if event.get("phase") == "validation":
                        validation.append(event)
                        print(json.dumps(event, ensure_ascii=False), flush=True)
                response = client.get(path + "/logs", headers=headers, params={"after_offset": offset, "limit": 1048576})
                response.raise_for_status()
                log = response.json()["data"]
                offset = log["next_after_offset"]
                if log["text"]:
                    print(log["text"], end="", flush=True)
                if task["status"] in ("succeeded", "failed", "canceled") and not page["has_more"]:
                    break
                time.sleep(1)
            else:
                client.post(path + "/cancel", headers=headers).raise_for_status()
                raise TimeoutError("验收超过 30 分钟，已请求取消")
            worker.join(35)
            if task["status"] != "succeeded":
                raise RuntimeError(task["error_message"] or task["status"])
            assert len(validation) == args.epochs, "缺少完整验证轮次"
            assert task["progress"] == 100 and observed_metrics, "缺少实际指标或完成事件"
            response = client.get(path + "/artifacts", headers=headers)
            response.raise_for_status()
            artifacts = response.json()["data"]
            assert {item["filename"] for item in artifacts} >= {"outputs/last.pt", "outputs/best.pt", "outputs/split.json"}
            for artifact in artifacts:
                response = client.get(path + "/artifacts/" + artifact["id"] + "/download", headers=headers)
                response.raise_for_status()
                assert len(response.content) == artifact["size"]
                assert sha256(response.content).hexdigest() == artifact["sha256"]
            environment = json.loads((repository.directory(task["id"]) / "outputs" / "environment.json").read_text(encoding="utf-8"))
            assert environment["device"] == args.device
            report = {"task": task, "environment": environment, "metric_count": len(observed_metrics),
                      "validation": validation, "artifacts": artifacts, "transport": "real loopback HTTP",
                      "verified_at": utc_now().isoformat(), "log_bytes": offset}
            report_path = root / "validation-report.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print("验收通过：" + str(report_path), flush=True)
    finally:
        if worker is not None and worker.is_alive():
            # 不让验收进程退出遗留 GPU 任务；失败也请求正常取消。
            repository.cancel(task["id"], repository.get(task["id"])["token_hash"])
            worker.join(35)
        server.should_exit = True
        api.join(10)
        listener.close()


if __name__ == "__main__":
    main()
