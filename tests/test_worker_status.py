"""治理 worker 运行状态与源码指纹的单元测试。"""

import json

from utils import worker_status


def test_imported_sources_only_contains_project_files():
    """登记的源码只包含项目内文件，不含第三方依赖。"""
    sources = worker_status.imported_sources()
    assert any(name.endswith("utils/worker_status.py") for name in sources)
    assert all("site-packages" not in name and ".venv" not in name for name in sources)


def test_changed_sources_reports_modified_and_removed_files(tmp_path):
    """登记之后被改动或删除的文件都会被列出来。"""
    target = tmp_path / "mod.py"
    target.write_text("x = 1", encoding="utf-8")
    recorded = {"mod.py": worker_status.file_digest(target)}
    assert worker_status.changed_sources(recorded, root=tmp_path) == []
    target.write_text("x = 2", encoding="utf-8")
    assert worker_status.changed_sources(recorded, root=tmp_path) == ["mod.py"]
    target.unlink()
    assert worker_status.changed_sources(recorded, root=tmp_path) == ["mod.py"]


def test_write_status_records_sources_and_detects_stale_worker(tmp_path):
    """状态文件登记的指纹过期时，能指出需要重启 worker。"""
    path = worker_status.write_status(tmp_path, 4321, "2026-10-27T00:00:00+00:00")
    assert path.exists()
    status = json.loads(path.read_text(encoding="utf-8"))
    assert status["pid"] == 4321
    assert status["sources"]
    assert worker_status.stale_worker_sources(tmp_path) == []
    assert worker_status.stale_hint(tmp_path) == ""

    status["sources"]["service/governance_data.py"] = "0" * 16
    path.write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")
    assert worker_status.stale_worker_sources(tmp_path) == ["service/governance_data.py"]
    assert "请重启 gov-worker" in worker_status.stale_hint(tmp_path)


def test_missing_status_file_means_no_hint(tmp_path):
    """没有状态文件（worker 从未启动）时不给过期提示。"""
    assert worker_status.stale_worker_sources(tmp_path) == []
    assert worker_status.stale_hint(tmp_path) == ""
