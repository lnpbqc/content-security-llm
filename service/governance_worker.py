"""在独立进程中持续领取并执行 SQLite 中的治理任务。"""

import argparse
import time

from config import Settings
from db.database import Database
from db.governance import GovernanceRepository
from db.governance_config import GovernanceConfigRepository
from llm.models import ModelManager
from service.governance import GovernanceService
from service.governance_data import BusinessGovernanceData


def main() -> None:
    """初始化数据库和模型服务，持续执行队列中的治理任务。"""
    parser = argparse.ArgumentParser(description="Process queued governance model tasks")
    parser.add_argument("--once", action="store_true", help="process at most one queued task")
    args = parser.parse_args()

    settings = Settings.load()
    database = Database(settings.database_path)
    database.initialize()
    repository = GovernanceRepository(database)
    repository.recover_running()
    config = GovernanceConfigRepository(database)
    manager = ModelManager(database, settings.credential_key)
    service = GovernanceService(
        repository, BusinessGovernanceData(config, manager.cipher,
                                           mock_samples=settings.governance_mock_samples), manager, config,
    )
    while True:
        worked = service.run_once()
        if args.once:
            return
        if not worked:
            time.sleep(1)


if __name__ == "__main__":
    main()
