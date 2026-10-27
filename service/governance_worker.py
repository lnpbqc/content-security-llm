"""在独立进程中持续领取并执行 SQLite 中的治理任务。"""

import argparse
import os
import time

from config import Settings
from db.database import Database
from db.governance import GovernanceRepository
from db.governance_config import GovernanceConfigRepository
from llm.models import ModelManager
from service.governance import GovernanceService
from service.governance_data import BusinessGovernanceData
from utils.locking import exclusive_lock
from utils.time import utc_now
from utils.worker_status import write_status


def main() -> None:
    """初始化数据库和模型服务，持续执行队列中的治理任务。"""
    parser = argparse.ArgumentParser(description="Process queued governance model tasks")
    parser.add_argument("--once", action="store_true", help="process at most one queued task")
    args = parser.parse_args()

    settings = Settings.load()
    database = Database(settings)
    database.initialize()
    repository = GovernanceRepository(database)
    config = GovernanceConfigRepository(database)
    manager = ModelManager(database, settings.credential_key)
    service = GovernanceService(
        repository, BusinessGovernanceData(database,
                                           mock_samples=settings.governance_mock_samples), manager, config,
    )
    with exclusive_lock(database.data_dir / "governance-worker" / "worker.lock",
                        "治理 worker 已运行，只允许单实例"):
        # 登记本次加载的源码指纹：改代码后接口据此提示 worker 仍在跑旧版本。
        write_status(database.data_dir, os.getpid(), utc_now().isoformat())
        repository.recover_running()
        while True:
            worked = service.run_once()
            if args.once:
                return
            if not worked:
                time.sleep(1)


if __name__ == "__main__":
    main()
