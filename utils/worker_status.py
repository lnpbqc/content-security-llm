"""登记治理 worker 的运行状态与已加载的源码指纹。

治理 worker 是独立进程，改代码不会热更新；若它仍在跑旧代码，任务会以
难以理解的错误失败（例如"数据集版本不匹配"）。worker 启动时登记它加载的
源码指纹，接口据此判断 worker 是否过期，并在失败原因后补一句重启提示。
"""

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKER_DIR = "governance-worker"
STATUS_FILE = "worker.status.json"
_CACHE_TTL = 5.0
_cache: Dict[str, Any] = {"key": None, "value": [], "at": 0.0}


def worker_dir(data_dir: Any) -> Path:
    """返回 worker 状态目录；相对路径按项目根目录解析，两端进程结果一致。"""
    path = Path(data_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path / WORKER_DIR


def file_digest(path: Path) -> str:
    """返回文件内容的短摘要。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def imported_sources() -> Dict[str, str]:
    """列出当前进程已加载的项目源码文件及其摘要（不含第三方依赖）。"""
    sources: Dict[str, str] = {}
    for module in list(sys.modules.values()):
        origin = getattr(module, "__file__", None)
        if not origin:
            continue
        path = Path(origin)
        text = str(path)
        if path.suffix != ".py" or "site-packages" in text or ".venv" in text:
            continue
        try:
            relative = path.resolve().relative_to(PROJECT_ROOT)
        except ValueError:
            continue
        if not path.exists():
            continue  # 临时模块已删除：无法登记指纹，跳过
        sources[relative.as_posix()] = file_digest(path)
    return sources


def write_status(data_dir: Any, pid: int, started_at: str) -> Path:
    """写出 worker 状态文件，返回其路径。"""
    folder = worker_dir(data_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / STATUS_FILE
    path.write_text(json.dumps({"pid": pid, "started_at": started_at,
                                "sources": imported_sources()},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def changed_sources(recorded: Dict[str, str], root: Path = PROJECT_ROOT) -> List[str]:
    """返回登记之后被改动或删除的源码文件。"""
    changed: List[str] = []
    for relative, digest in (recorded or {}).items():
        path = root / relative
        if not path.exists() or file_digest(path) != digest:
            changed.append(relative)
    return sorted(changed)


def stale_worker_sources(data_dir: Any) -> List[str]:
    """返回 worker 启动后已变更的源码文件；没有状态文件时为空。"""
    path = worker_dir(data_dir) / STATUS_FILE
    try:
        raw = path.read_bytes()
    except OSError:
        return []
    key = hashlib.sha256(raw).hexdigest()
    now = time.monotonic()
    if _cache["key"] == key and now - _cache["at"] < _CACHE_TTL:
        return list(_cache["value"])
    try:
        status = json.loads(raw.decode("utf-8"))
    except ValueError:
        return []
    value = changed_sources(status.get("sources") or {})
    _cache.update({"key": key, "value": value, "at": now})
    return list(value)


def stale_hint(data_dir: Any) -> str:
    """worker 过期时给出的重启提示；未过期时返回空串。"""
    stale = stale_worker_sources(data_dir)
    if not stale:
        return ""
    return ("（提示：治理 worker 启动后 {} 已被改动，worker 仍在运行旧代码，"
            "请重启 gov-worker 窗口后重试）".format("、".join(stale[:3])))
