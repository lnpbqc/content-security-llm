"""All persistence tests run against a dedicated real MySQL database."""

import os
from dataclasses import replace

import pytest

from config import Settings
from db.database import Database, TABLES


@pytest.fixture(scope="session", autouse=True)
def mysql_test_database():
    settings = Settings.load()
    name = os.getenv("TEST_MYSQL_DATABASE", "content_security_llm_test")
    if not name.endswith("_test") or name == settings.mysql_database:
        raise RuntimeError("TEST_MYSQL_DATABASE must be a separate database ending in _test")
    overrides = {
        "MYSQL_HOST": os.getenv("TEST_MYSQL_HOST", settings.mysql_host),
        "MYSQL_PORT": os.getenv("TEST_MYSQL_PORT", str(settings.mysql_port)),
        "MYSQL_USER": os.getenv("TEST_MYSQL_USER", settings.mysql_user),
        "MYSQL_PASSWORD": os.getenv("TEST_MYSQL_PASSWORD", settings.mysql_password),
        "MYSQL_DATABASE": name,
    }
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    settings = replace(settings, mysql_host=overrides["MYSQL_HOST"], mysql_port=int(overrides["MYSQL_PORT"]),
                       mysql_user=overrides["MYSQL_USER"], mysql_password=overrides["MYSQL_PASSWORD"],
                       mysql_database=name)
    database = Database(settings)
    try:
        database.create_schema()
        yield database
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture(autouse=True)
def clean_mysql_tables(mysql_test_database):
    # Only the explicitly guarded test database is cleared; business tables are untouched.
    with mysql_test_database.connect() as connection:
        for table in reversed(TABLES):
            connection.execute(f"DELETE FROM `{table}`")
