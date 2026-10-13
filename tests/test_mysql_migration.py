"""Real MySQL regressions for import, transactions and worker SQL semantics."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from hashlib import sha256
from threading import Barrier

import pytest
from cryptography.fernet import Fernet
from pymysql import IntegrityError

from config import Settings
from db.governance import GovernanceRepository
from db.neural_trace import NeuralTraceRepository
from db.repositories import AuthRepository, ModelRepository
from db.training import TrainingRepository
from scripts.migrate_sqlite_to_mysql import migrate
from service.auth import AuthService
from utils.time import utc_now


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE models (
                id TEXT PRIMARY KEY, upstream_model_id TEXT NOT NULL, name TEXT NOT NULL,
                description TEXT, base_url TEXT NOT NULL, encrypted_api_key TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL, deleted_at TEXT);
            CREATE TABLE auth_tokens (
                id TEXT PRIMARY KEY, token_hash TEXT NOT NULL UNIQUE, label TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL, expires_at TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1,
                deleted_at TEXT);
            CREATE TABLE analysis_records (
                id TEXT PRIMARY KEY, token_hash TEXT NOT NULL, kind TEXT NOT NULL, scenario_code TEXT,
                model_id TEXT NOT NULL REFERENCES models(id), input_json TEXT NOT NULL,
                state_json TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE training_tasks (
                id TEXT PRIMARY KEY, token_hash TEXT NOT NULL, config_json TEXT NOT NULL,
                source_hash TEXT NOT NULL, status TEXT NOT NULL, epoch INTEGER NOT NULL DEFAULT 0,
                step INTEGER NOT NULL DEFAULT 0, progress REAL NOT NULL DEFAULT 0,
                latest_metrics_json TEXT NOT NULL DEFAULT '{}', runtime_json TEXT NOT NULL DEFAULT '{}',
                error_message TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                started_at TEXT, finished_at TEXT);
            CREATE TABLE training_script_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL REFERENCES training_tasks(id),
                type TEXT NOT NULL, event_json TEXT NOT NULL, created_at TEXT NOT NULL);
        """)
        now = utc_now().isoformat()
        connection.execute("INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?)",
                           ("model-1", "upstream", "中文模型🙂", None, "https://example.invalid/v1",
                            "encrypted-original", now, now, None))
        connection.execute("INSERT INTO auth_tokens VALUES (?,?,?,?,?,?,?)",
                           ("token-1", "a" * 64, "用户🙂", now, now, 1, None))
        text = json.dumps({"text": "中文🙂" * 20000}, ensure_ascii=False)
        connection.execute("INSERT INTO analysis_records VALUES (?,?,?,?,?,?,?,?,?)",
                           ("analysis-1", "a" * 64, "scenario", None, "model-1", text,
                            '{"status":"succeeded"}', "succeeded", now))
        connection.execute("INSERT INTO training_tasks "
                           "(id,token_hash,config_json,source_hash,status,created_at,updated_at) "
                           "VALUES (?,?,?,?,?,?,?)",
                           ("training-1", "a" * 64, "{}", "b" * 64, "succeeded", now, now))
        connection.execute("INSERT INTO training_script_events VALUES (?,?,?,?,?)",
                           (487, "training-1", "progress", '{"progress":100}', now))
    return path


def test_import_roundtrip_is_read_only_and_repeatable(source, mysql_test_database):
    before = sha256(source.read_bytes()).hexdigest()
    first = migrate(source, mysql_test_database)
    assert sum(table["inserted"] for table in first.values()) == 5
    second = migrate(source, mysql_test_database)
    assert sum(table["inserted"] for table in second.values()) == 0
    assert sum(table["skipped"] for table in second.values()) == 5
    assert before == sha256(source.read_bytes()).hexdigest()
    with mysql_test_database.connect() as connection:
        assert connection.execute("SELECT name FROM llm_models").fetchone()["name"] == "中文模型🙂"
        text = connection.execute("SELECT input_json FROM llm_analysis_records").fetchone()["input_json"]
        assert json.loads(text)["text"] == "中文🙂" * 20000
        connection.execute("INSERT INTO llm_training_script_events (task_id,type,event_json,created_at) "
                           "VALUES (%s,%s,%s,%s)", ("training-1", "progress", "{}", utc_now().isoformat()))
        assert connection.execute("SELECT MAX(id) AS last FROM llm_training_script_events").fetchone()["last"] > 487


def test_import_ignores_obsolete_business_connection_config(source, mysql_test_database):
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE business_database_config (id INTEGER PRIMARY KEY, host TEXT)")
        connection.execute("INSERT INTO business_database_config VALUES (1, 'obsolete-host')")
    report = migrate(source, mysql_test_database)
    assert "llm_business_database_config" not in report
    assert sum(table["inserted"] for table in report.values()) == 5


def test_import_conflict_rolls_back_earlier_tables(source, mysql_test_database):
    now = utc_now().isoformat()
    with mysql_test_database.connect() as connection:
        connection.execute("INSERT INTO llm_auth_tokens (id,token_hash,label,created_at,expires_at) "
                           "VALUES (%s,%s,%s,%s,%s)", ("token-1", "a" * 64, "different", now, now))
    with pytest.raises(ValueError, match="different content"):
        migrate(source, mysql_test_database)
    with mysql_test_database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) AS total FROM llm_models").fetchone()["total"] == 0
        assert connection.execute("SELECT label FROM llm_auth_tokens").fetchone()["label"] == "different"


def test_migrated_token_authenticates_and_model_secret_decrypts(source, mysql_test_database):
    cipher = Fernet(Fernet.generate_key())
    token = "known-migration-test-token"
    with sqlite3.connect(source) as connection:
        connection.execute("UPDATE auth_tokens SET token_hash=?,expires_at=?",
                           (sha256(token.encode()).hexdigest(), (utc_now() + timedelta(days=1)).isoformat()))
        connection.execute("UPDATE models SET encrypted_api_key=?",
                           (cipher.encrypt(b"original-model-key").decode(),))
    migrate(source, mysql_test_database)
    assert AuthService(AuthRepository(mysql_test_database), "s" * 32).is_valid_token(token)
    model = ModelRepository(mysql_test_database).get("model-1")
    assert cipher.decrypt(model.encrypted_api_key.encode()) == b"original-model-key"


def test_import_rejects_overlong_values_before_inserting(source, mysql_test_database):
    with sqlite3.connect(source) as connection:
        connection.execute("UPDATE auth_tokens SET token_hash=?", ("x" * 65,))
    with pytest.raises(ValueError, match="exceeds MySQL length"):
        migrate(source, mysql_test_database)
    with mysql_test_database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) AS total FROM llm_models").fetchone()["total"] == 0


def test_import_foreign_key_failure_rolls_back(source, mysql_test_database):
    with sqlite3.connect(source) as connection:
        connection.execute("UPDATE analysis_records SET model_id='missing'")
    with pytest.raises(IntegrityError):
        migrate(source, mysql_test_database)
    with mysql_test_database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) AS total FROM llm_models").fetchone()["total"] == 0
        assert connection.execute("SELECT COUNT(*) AS total FROM llm_auth_tokens").fetchone()["total"] == 0


def test_duplicate_token_and_case_sensitive_hash(mysql_test_database):
    repository = AuthRepository(mysql_test_database)
    repository.create("a" * 64, utc_now())
    with pytest.raises(IntegrityError):
        repository.create("a" * 64, utc_now())
    assert repository.create("A" * 64, utc_now()).token_hash == "A" * 64
    assert repository.set_enabled("a" * 64, True) is not None  # matched, unchanged rows


@pytest.mark.parametrize("kind", ["governance", "training", "neural"])
def test_two_connections_cannot_claim_the_same_task(mysql_test_database, kind):
    if kind == "governance":
        repository = GovernanceRepository(mysql_test_database)
        task = repository.create_task(kind="governance-risk", token_hash="a" * 64, name="test",
                                      scope={}, model_id="model", model_version="v1")
    elif kind == "training":
        repository = TrainingRepository(mysql_test_database)
        task = repository.create({"python_source": "pass"}, "a" * 64)
    else:
        repository = NeuralTraceRepository(mysql_test_database)
        task = repository.create("a" * 64, "case-1", {})
    barrier = Barrier(2)

    def claim():
        barrier.wait(timeout=10)
        return repository.claim()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: claim(), range(2)))
    assert [result["id"] for result in results if result] == [task["id"]]


def test_training_cancellation_uses_original_status(mysql_test_database):
    repository = TrainingRepository(mysql_test_database)
    pending = repository.create({"python_source": "pass"}, "a" * 64)
    canceled = repository.cancel(pending["id"], "a" * 64)
    assert canceled["status"] == "canceled" and canceled["finished_at"]
    running = repository.create({"python_source": "pass"}, "a" * 64)
    repository.claim()
    repository.cancel(running["id"], "a" * 64)
    repository.finish(running["id"], "succeeded")
    canceled = repository.get(running["id"])
    assert canceled["status"] == "canceled" and canceled["progress"] == 0
    completed = repository.create({"python_source": "pass"}, "a" * 64)
    repository.claim()
    repository.finish(completed["id"], "succeeded")
    assert repository.get(completed["id"])["progress"] == 100


def test_startup_rejects_missing_table_without_recreating_it(mysql_test_database):
    with mysql_test_database.connect() as connection:
        connection.execute("RENAME TABLE llm_auth_tokens TO hidden_auth_tokens")
    try:
        with pytest.raises(RuntimeError, match="missing table"):
            mysql_test_database.initialize()
        with mysql_test_database.connect() as connection:
            assert connection.execute("SHOW TABLES LIKE 'llm_auth_tokens'").fetchone() is None
    finally:
        with mysql_test_database.connect() as connection:
            connection.execute("RENAME TABLE hidden_auth_tokens TO llm_auth_tokens")


def test_startup_rejects_incompatible_column(mysql_test_database):
    with mysql_test_database.connect() as connection:
        connection.execute("ALTER TABLE llm_auth_tokens MODIFY enabled DOUBLE NOT NULL DEFAULT 1")
    try:
        with pytest.raises(RuntimeError, match="incompatible column"):
            mysql_test_database.initialize()
    finally:
        with mysql_test_database.connect() as connection:
            connection.execute("ALTER TABLE llm_auth_tokens MODIFY enabled BIGINT NOT NULL DEFAULT 1")


def test_startup_rejects_changed_default(mysql_test_database):
    with mysql_test_database.connect() as connection:
        connection.execute("ALTER TABLE llm_auth_tokens ALTER enabled SET DEFAULT 0")
    try:
        with pytest.raises(RuntimeError, match="incompatible default"):
            mysql_test_database.initialize()
    finally:
        with mysql_test_database.connect() as connection:
            connection.execute("ALTER TABLE llm_auth_tokens ALTER enabled SET DEFAULT 1")


def test_mysql_config_precedence_and_legacy_artifact_directory(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"setup_secret": "s" * 32, "credential_key": Fernet.generate_key().decode(),
                               "database_path": "legacy/app.sqlite3", "mysql_host": "json-host",
                               "mysql_port": 3307, "mysql_user": "json-user", "mysql_password": "private",
                               "mysql_database": "json-db"}), encoding="utf-8")
    monkeypatch.setenv("APP_CONFIG_FILE", str(path))
    for name in ("HOST", "PORT", "USER", "PASSWORD", "DATABASE"):
        monkeypatch.delenv("MYSQL_" + name, raising=False)
    monkeypatch.delenv("APP_DB_PATH", raising=False)
    monkeypatch.delenv("APP_DATA_DIR", raising=False)
    settings = Settings.load()
    assert settings.data_dir == tmp_path / "legacy"
    assert settings.mysql_host == "json-host" and settings.mysql_port == 3307
    monkeypatch.setenv("MYSQL_PORT", "13306")
    monkeypatch.setenv("MYSQL_USER", "env-user")
    settings = Settings.load()
    assert settings.mysql_user == "env-user" and settings.mysql_port == 13306
    assert "private" not in repr(settings)
    monkeypatch.setenv("MYSQL_PORT", "0")
    with pytest.raises(RuntimeError, match="mysql_port"):
        Settings.load()
