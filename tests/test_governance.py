"""用模拟模型验证治理任务接口、worker 和结果查询。"""

import json
import sqlite3
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from pydantic import ValidationError

from config import Settings
from db.database import Database
from llm.governance_outputs import AnomalyFindingOutput, RiskFindingOutput, ValueDimensionOutput
from llm.outputs import AnomalyFinding, SemanticRiskFinding, ValueScoreDimension
from main import create_app
from utils.time import utc_now


class FakeCompletionClient:
    """按输入类型返回固定结构，避免测试访问真实模型。"""
    calls = []
    invalid_value = False
    invalid_dimension_name = False

    def __init__(self, **kwargs):
        """提供与 OpenAI 客户端一致的最小调用入口。"""
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        """根据任务输入生成确定的逐条模型响应。"""
        prompt = kwargs["messages"][-1]["content"]
        self.calls.append(prompt)
        if prompt == "hello":
            output = {"answer": "ok"}
        else:
            data = json.loads(prompt)
            if "rubric" in data:
                output = {"sample_id": "wrong" if self.invalid_value else data["sample_id"],
                          "status": "scored", "dimensions": [
                    {"name": "其他" if self.invalid_dimension_name and index == 0 else d["name"],
                     "score": 80, "reason": "有文本证据", "evidence": [data["text"]]}
                    for index, d in enumerate(data["rubric"]["dimensions"])]}
            elif "fieldWhitelist" in data:
                findings = []
                if "文化传承" in data["text"]:
                    findings = [{"type": "标签异常", "field": "topic_label",
                                 "quote": "文化传承", "ruleId": "anomaly-rule-1", "reason": "标签与正文不一致",
                                 "fieldChanges": [{"field": "topic_label", "before": "体育", "after": "文化"}],
                                 "validationResults": []}]
                output = {"sampleId": data["sampleId"], "sampleRevisionId": data["sampleRevisionId"],
                          "status": "assessed", "findings": findings}
            else:
                findings = []
                if "用户ID" in data["text"]:
                    findings = [{"category": "个人信息暴露", "suggestedLevel": "MEDIUM",
                                 "reason": "账户字段与注册时间可关联", "ruleId": "PII-03",
                                 "ruleVersion": "1.0", "evidenceRefs": [data["evidence"][0]["id"]]}]
                output = {"sampleId": data["sampleId"], "sampleRevisionId": data["sampleRevisionId"],
                          "status": "assessed", "findings": findings,
                          "reviewRecommended": bool(findings)}
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=json.dumps(output, ensure_ascii=False), refusal=None)
        )])


class FakeBusinessDatabase:
    """模拟业务库的当前版本与原始记录，记录连接和查询次数。"""

    def __init__(self):
        """准备三个数据集，每个数据集各有三条原始记录。"""
        self.datasets = {i: {"name": "业务数据集 {}".format(i), "version": "v1.0.0"}
                         for i in (1, 2, 3)}
        self.records = {i: [
            {"id": 1, "payload": {"text": "【新闻】北京科技交流会召开。", "topic_label": "综合", "language": "zh"}},
            {"id": 2, "payload": {"text": "用户ID：10086；注册时间：2026/9/20", "topic_label": "综合", "language": "zh"}},
            {"id": 3, "payload": {"text": "社区节庆活动记录：文化传承。", "topic_label": "体育", "language": "zh"}},
        ] for i in (1, 2, 3)}
        self.connections = 0
        self.queries = []
        self.unavailable = False

    def connect(self, **kwargs):
        """提供与 PyMySQL 连接一致的最小测试入口。"""
        if self.unavailable:
            raise OSError("database unavailable")
        self.connections += 1
        return FakeBusinessConnection(self)


class FakeBusinessConnection:
    """提供可关闭、可创建游标的模拟业务库连接。"""

    def __init__(self, database):
        """保存模拟业务库引用。"""
        self.database = database

    def cursor(self):
        """返回可用于 with 的模拟游标。"""
        return FakeBusinessCursor(self.database)

    def close(self):
        """模拟关闭连接。"""
        pass


class FakeBusinessCursor:
    """按业务库两条只读 SQL 返回数据集和记录。"""

    def __init__(self, database):
        """保存模拟数据及本次查询结果。"""
        self.database = database
        self.result = None

    def __enter__(self):
        """进入游标上下文。"""
        return self

    def __exit__(self, *_):
        """退出游标上下文。"""
        return False

    def execute(self, sql, params):
        """核对查询使用参数绑定，并返回对应模拟结果。"""
        self.database.queries.append((sql, params))
        if sql.startswith("SELECT name, version FROM datasets"):
            self.result = self.database.datasets.get(params[0])
        elif sql.startswith("SELECT id, payload FROM dataset_records"):
            self.result = sorted(self.database.records.get(params[0], []), key=lambda row: row["id"])
        else:
            raise AssertionError(sql)

    def fetchone(self):
        """读取单条数据集信息。"""
        return self.result

    def fetchall(self):
        """读取该数据集全部记录。"""
        return self.result


@pytest.fixture
def app_client(tmp_path: Path):
    """建立临时治理库、模拟业务库、三个模型和两枚令牌。"""
    FakeCompletionClient.calls = []
    FakeCompletionClient.invalid_value = False
    FakeCompletionClient.invalid_dimension_name = False
    business = FakeBusinessDatabase()
    settings = Settings(tmp_path / "app.sqlite3", "s" * 32, Fernet.generate_key())
    app = create_app(settings, client_factory=FakeCompletionClient)
    with TestClient(app) as client:
        app.state.test_settings = settings
        app.state.governance_service.data.connection_factory = business.connect
        models = {}
        for kind in ("governance-value", "governance-anomaly", "governance-risk"):
            model = app.state.model_manager.create_model(
                upstream_model_id="fake-" + kind, name=kind, description=None,
                base_url="https://example.invalid/v1", api_key="fake-key",
            )
            models[kind] = model.id
        for kind, model_id in models.items():
            response = client.put("/api/v1/admin/governance/models/" + kind,
                                  headers={"X-Setup-Secret": "s" * 32}, json={"model_id": model_id})
            assert response.status_code == 200, response.text
        response = client.put("/api/v1/admin/governance/business-database",
                              headers={"X-Setup-Secret": "s" * 32},
                              json={"host": "localhost", "port": 3306, "database": "content_safety",
                                    "username": "reader", "password": "secret"})
        assert response.status_code == 200, response.text
        app.state.auth_service.create_token("s" * 32, "a" * 32, utc_now() + timedelta(days=1), "前端")
        app.state.auth_service.create_token("s" * 32, "b" * 32, utc_now() + timedelta(days=1), "业务后端")
        yield app, client, models["governance-value"], business


@pytest.mark.parametrize("kind,scheme,name", [
    ("governance-value", "general-v1", "value"),
    ("governance-anomaly", "anomaly-basic-v1", "anomaly"),
    ("governance-risk", "risk-v1", "risk"),
])
def test_empty_dataset_mock_samples_publish_marked_snapshots(app_client, kind, scheme, name):
    """临时空记录样本经过三类模型校验，并保留来源和范围。"""
    app, client, _, business = app_client
    service = app.state.governance_service
    service.data.mock_samples = True
    business.records[3] = []
    samples = service.data.samples(scope(scheme))
    assert len(samples) == 10
    assert len({row["id"] for row in samples}) == 10
    assert all(row["id"].startswith("mock-3-v1.0.0-") for row in samples)
    assert all(row["dataset_id"] == 3 and row["version_id"] == "v1.0.0" for row in samples)
    assert all(row["metadata"]["is_mock"] and "临时模拟数据" in row["text"] for row in samples)
    assert service.data.dataset_name(3, "v1.0.0") == "业务数据集 3"
    task = create(client, kind, scheme)
    task_id = task.get("task_id") or task["id"]
    assert service.run_once()
    saved = service.repository.get_task(task_id)
    assert saved["status"] == "succeeded", saved
    assert len(FakeCompletionClient.calls) == 10
    base = "/api/v1/data-governance/{}-results/{}".format(name, saved["result_id"])
    summary = client.get(base, headers=headers()).json()["data"]
    assert summary["data_source"] == "mock"
    page = client.get(base + "/samples", headers=headers()).json()["data"]
    assert page["total"] > 0
    assert all("临时模拟数据" in row["text"] for row in page["items"])
    paged_items = []
    for number in range(1, (page["total"] + 1) // 2 + 1):
        response = client.get(base + "/samples", headers=headers(),
                              params={"page": number, "page_size": 2})
        assert response.status_code == 200
        paged_items.extend(response.json()["data"]["items"])
    assert len(paged_items) == page["total"]
    assert {row["id"] for row in paged_items} == {row["id"] for row in page["items"]}
    if name == "risk":
        assert "10086" not in json.dumps(page, ensure_ascii=False)
    # 关闭模拟后发布新的真实结果，旧快照仍明确标记模拟来源。
    service.data.mock_samples = False
    business.records[3] = [{"id": 88, "payload": {"text": "真实记录", "language": "zh"}}]
    task = create(client, kind, scheme)
    assert service.run_once()
    real = service.repository.get_task(task.get("task_id") or task["id"])
    assert real["status"] == "succeeded"
    assert client.get(base, headers=headers()).json()["data"]["data_source"] == "mock"
    assert client.get(base + "/samples", headers=headers()).json()["data"] == page


def test_mock_samples_never_replace_existing_records(app_client):
    """开关开启时，仍优先读取真实记录，不修改业务库数据。"""
    app, _, _, business = app_client
    data = app.state.governance_service.data
    data.mock_samples = True
    rows = data.samples(scope("general-v1"))
    assert [row["id"] for row in rows] == ["1", "2", "3"]
    assert all(not row["metadata"].get("is_mock") for row in rows)
    assert len(business.records[3]) == 3


@pytest.mark.parametrize("failure,message", [
    ("empty", "数据集没有可治理的记录"),
    ("missing_dataset", "数据集不存在"),
    ("wrong_version", "数据集版本不匹配"),
    ("unavailable", "业务数据库连接失败"),
    ("missing_config", "业务数据库连接未配置"),
])
def test_mock_samples_preserve_source_errors(app_client, failure, message):
    """模拟只对明确开启后的空记录生效，不掩盖其他错误。"""
    app, _, _, business = app_client
    data = app.state.governance_service.data
    data.mock_samples = failure != "empty"
    business.records[3] = []
    if failure == "missing_dataset":
        business.datasets.pop(3)
    elif failure == "wrong_version":
        business.datasets[3]["version"] = "v2.0.0"
    elif failure == "unavailable":
        business.unavailable = True
    elif failure == "missing_config":
        with app.state.database.connect() as connection:
            connection.execute("DELETE FROM business_database_config")
    with pytest.raises((ValueError, RuntimeError), match=message):
        data.samples(scope("general-v1"))
    assert not FakeCompletionClient.calls


def test_mock_settings_default_and_api_wiring(tmp_path, monkeypatch):
    """配置默认关闭；开启后注入 API 的治理样本读取器。"""
    path = tmp_path / "settings.json"
    config = {"setup_secret": "s" * 32, "credential_key": Fernet.generate_key().decode(),
              "database_path": "mock.sqlite3"}
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("APP_CONFIG_FILE", str(path))
    assert Settings.load().governance_mock_samples is False
    config["governance_mock_samples"] = True
    path.write_text(json.dumps(config), encoding="utf-8")
    settings = Settings.load()
    assert settings.governance_mock_samples is True
    with TestClient(create_app(settings)) as client:
        assert client.app.state.governance_service.data.mock_samples is True


@pytest.mark.parametrize("value", ["false", "true", 1, None])
def test_mock_settings_require_boolean(tmp_path, monkeypatch, value):
    """拒绝看似布尔值的字符串和数字，防止意外开启。"""
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"setup_secret": "s" * 32,
                               "credential_key": Fernet.generate_key().decode(),
                               "governance_mock_samples": value}), encoding="utf-8")
    monkeypatch.setenv("APP_CONFIG_FILE", str(path))
    with pytest.raises(RuntimeError, match="governance_mock_samples must be a boolean"):
        Settings.load()


def headers(token="a"):
    """返回测试请求使用的 Bearer 和追踪请求头。"""
    return {"Authorization": "Bearer " + token * 32, "X-Trace-Id": "trace-test"}


def scope(scheme, dataset_id=3):
    """构造与模拟业务库当前版本对应的四元组任务范围。"""
    return {"dataset_id": dataset_id, "version_id": "v1.0.0",
            "language": "all", "scheme_id": scheme}


def create(client, kind, scheme, dataset_id=3):
    """通过对应创建接口提交任务，并断言初始状态为 pending。"""
    path = {"governance-value": "/tasks", "governance-anomaly": "/data-governance/anomaly-tasks",
            "governance-risk": "/data-governance/risk-tasks"}[kind]
    response = client.post("/api/v1" + path, headers=headers(),
                           json={"kind": kind, "name": "测试任务", "input": scope(scheme, dataset_id)})
    assert response.status_code == 200, response.text
    assert response.json()["data"]["status"] == "pending"
    return response.json()["data"]


def test_three_tasks_publish_read_only_results_and_preserve_invoke(app_client):
    """验证三类任务结果、只读查询、令牌隔离及旧统一入口。"""
    app, client, model_id, business = app_client
    cases = [
        ("governance-value", "general-v1", "value", 3, "/tasks/{id}?kind=governance-value"),
        ("governance-anomaly", "anomaly-basic-v1", "anomaly", 2,
         "/data-governance/anomaly-tasks/{id}"),
        ("governance-risk", "risk-v1", "risk", 1, "/data-governance/risk-tasks/{id}"),
    ]
    service = app.state.governance_service
    for kind, scheme, name, dataset_id, task_path in cases:
        source_reads = business.connections
        created = create(client, kind, scheme, dataset_id)
        assert business.connections == source_reads
        task_id = created.get("task_id") or created.get("id")
        assert service.repository.get_task(task_id)["model_id"] == service.config.get_model_id(kind)
        assert service.run_once()
        task = client.get("/api/v1" + task_path.format(id=task_id), headers=headers()).json()["data"]
        assert task["status"] == "succeeded"
        result_id = task["result_id"]
        base = "/api/v1/data-governance/{}-results".format(name)
        detail = client.get(base + "/" + result_id, headers=headers()).json()["data"]
        assert detail["id"] == result_id
        assert detail["scope"] == scope(scheme, dataset_id)
        latest = client.get(base + "/latest", headers=headers(),
                            params=scope(scheme, dataset_id)).json()["data"]
        assert latest == detail
        before = len(FakeCompletionClient.calls)
        source_reads = business.connections
        samples = client.get(base + "/" + result_id + "/samples", headers=headers(),
                             params={"page": 1, "page_size": 2}).json()["data"]
        assert samples["total"] == 3
        assert len(samples["items"]) == 2
        if name == "value":
            assert detail["valid_count"] == 3
            assert detail["mean_score"] == 80
            listed = client.get("/api/v1/tasks", headers=headers(),
                                params={"kind": kind, **scope(scheme, dataset_id)}).json()["data"]
            assert listed["total"] == 1
            assert client.get(base + "/" + result_id + "/samples", headers=headers(),
                              params={"tier": "high"}).json()["data"]["total"] == 0
        else:
            item_id = samples["items"][0]["id"]
            sample = client.get(base + "/" + result_id + "/samples/" + item_id,
                                headers=headers()).json()["data"]
            assert sample["id"] == item_id
            history = client.get(base, headers=headers(),
                                 params=scope(scheme, dataset_id)).json()["data"]
            assert history == [detail]
        if name == "anomaly":
            assert detail["anomaly_count"] == 1
            found = client.get(base + "/" + result_id + "/samples", headers=headers(),
                               params={"type": "标签异常"}).json()["data"]
            assert found["total"] == 1
            assert found["items"][0]["candidates"][0]["status"] == "DRAFT"
        if name == "risk":
            assert detail["risk_count"] == 1
            all_samples = client.get(base + "/" + result_id + "/samples", headers=headers(),
                                     params={"page_size": 10}).json()["data"]
            assert "10086" not in json.dumps(all_samples, ensure_ascii=False)
            assert "10086" not in FakeCompletionClient.calls[-2]
            assert client.get(base + "/" + result_id + "/samples", headers=headers(),
                              params={"level": "MEDIUM"}).json()["data"]["total"] == 1
            call_rows = client.get("/api/calls", headers=headers()).json()
            risk_inputs = [row["input"] for row in call_rows
                           if row["response_type"] == "governance_risk_sample"]
            assert all("10086" not in item for item in risk_inputs)
        assert len(FakeCompletionClient.calls) == before
        assert business.connections == source_reads
        assert client.get(base + "/" + result_id, headers=headers("b")).status_code == 404
        assert client.get(base + "/latest", headers=headers("b"),
                          params=scope(scheme, dataset_id)).json()["data"] is None
    assert len(FakeCompletionClient.calls) == 9
    response = client.post("/api/llm/invoke", headers=headers(),
                           json={"model_id": model_id, "input": "hello", "response_type": "text_answer"})
    assert response.status_code == 200
    assert response.json()["result"] == {"answer": "ok"}
    calls = client.get("/api/calls", headers=headers()).json()
    assert {row["response_type"] for row in calls} == {
        "governance_value_sample", "governance_anomaly_sample", "governance_risk_sample", "text_answer"}


def test_restart_resumes_saved_sample_and_rejects_invalid_scope(app_client):
    """验证重启续跑会跳过检查点，且无效方案被拒绝。"""
    app, client, _, business = app_client
    task = create(client, "governance-value", "general-v1")
    service = app.state.governance_service
    claimed = service.repository.claim()
    assert claimed["status"] == "running"
    running = client.get("/api/v1/tasks/" + task["task_id"], headers=headers(),
                         params={"kind": "governance-value"}).json()["data"]
    assert running["status"] == "running"
    sample = service.data.samples(scope("general-v1"))[0]
    saved = {"sample_id": sample["id"], "status": "unavailable", "dimensions": [],
             "unavailable_reason": "证据不足"}
    service.repository.save_sample(task["task_id"], sample["id"], saved)
    service.repository.recover_running()
    assert service.run_once()
    assert len(FakeCompletionClient.calls) == 2
    result_id = service.repository.get_task(task["task_id"])["result_id"]
    result = client.get("/api/v1/data-governance/value-results/" + result_id,
                        headers=headers()).json()["data"]
    assert result["unavailable_count"] == 1
    assert result["valid_count"] == 2
    sorted_rows = client.get("/api/v1/data-governance/value-results/" + result_id + "/samples",
                             headers=headers(), params={"sort_by": "score", "sort_order": "desc"})
    assert sorted_rows.status_code == 200
    assert sorted_rows.json()["data"]["items"][-1]["tier"] == "unavailable"
    invalid = client.post("/api/v1/tasks", headers=headers(),
                          json={"kind": "governance-value", "name": "bad",
                                "input": scope("missing")})
    assert invalid.status_code == 422
    assert invalid.json()["code"] == 422


def test_missing_model_configuration_rejects_creation(app_client):
    """验证缺少模型配置与缺少令牌时不会创建任务。"""
    app, client, _, business = app_client
    with app.state.database.connect() as connection:
        connection.execute("DELETE FROM governance_model_bindings WHERE kind = ?", ("governance-risk",))
    response = client.post("/api/v1/data-governance/risk-tasks", headers=headers(),
                           json={"kind": "governance-risk", "name": "检测",
                                 "input": scope("risk-v1", 1)})
    assert response.status_code == 503
    assert response.json()["code"] == 503
    unauthenticated = client.get("/api/v1/tasks/no-task", params={"kind": "governance-value"})
    assert unauthenticated.status_code == 401
    assert unauthenticated.json()["code"] == 401


def test_all_invalid_business_outputs_fail_without_publishing(app_client):
    """验证全部样本输出无效时任务失败且不发布结果。"""
    app, client, _, business = app_client
    task = create(client, "governance-value", "general-v1")
    FakeCompletionClient.invalid_value = True
    assert app.state.governance_service.run_once()
    saved = client.get("/api/v1/tasks/" + task["task_id"], headers=headers(),
                       params={"kind": "governance-value"}).json()["data"]
    assert saved["status"] == "failed"
    assert saved["result_id"] is None
    assert {row["status"] for row in client.get("/api/calls", headers=headers()).json()} == {"error"}
    assert client.get("/api/v1/data-governance/value-results/latest", headers=headers(),
                      params=scope("general-v1")).json()["data"] is None


def test_governance_enums_match_general_outputs_and_reject_unknown_values(app_client):
    """验证三种任务输出与通用输出共享枚举，未知名称不能进入结果。"""
    pairs = (
        (ValueDimensionOutput, ValueScoreDimension, "name"),
        (AnomalyFindingOutput, AnomalyFinding, "type"),
        (RiskFindingOutput, SemanticRiskFinding, "category"),
    )
    for task_type, general_type, field in pairs:
        assert task_type.model_json_schema()["properties"][field]["enum"] == (
            general_type.model_json_schema()["properties"][field]["enum"]
        )
    with pytest.raises(ValidationError):
        ValueDimensionOutput(name="其他", score=80, reason="有依据", evidence=[])
    with pytest.raises(ValidationError):
        AnomalyFindingOutput(type="其他", field="topic_label", quote="原文", ruleId="anomaly-rule-1",
                             reason="有依据", fieldChanges=[], validationResults=[])
    with pytest.raises(ValidationError):
        RiskFindingOutput(category="其他", suggestedLevel="LOW", reason="有依据", ruleId="PII-03",
                          ruleVersion="1.0", evidenceRefs=["e1"])

    app, client, _, business = app_client
    created = create(client, "governance-value", "general-v1")
    FakeCompletionClient.invalid_dimension_name = True
    assert app.state.governance_service.run_once()
    task = app.state.governance_service.repository.get_task(created["task_id"])
    assert task["status"] == "failed"
    assert task["result_id"] is None
    calls = client.get("/api/calls", headers=headers()).json()
    assert len(calls) == 3
    assert {call["error_code"] for call in calls} == {"invalid_structured_output"}


def test_business_rows_are_ordered_and_all_fields_become_model_text(app_client):
    """验证业务库记录与字段排序、语种不筛选及本地源表已移除。"""
    app, client, _, business = app_client
    business.datasets[99] = {"name": "自定义数据集", "version": "v2"}
    business.records[99] = [
        {"id": 10, "payload": '{"text":"第二条","language":"zh"}'},
        {"id": 4, "payload": {"zeta": {"b": 2, "a": 1}, "alpha": "第一条", "language": "zh"}},
    ]
    custom_scope = {"dataset_id": 99, "version_id": "v2", "language": "en", "scheme_id": "general-v1"}
    response = client.post("/api/v1/tasks", headers=headers(),
                           json={"kind": "governance-value", "name": "查库测试", "input": custom_scope})
    assert response.status_code == 200, response.text
    assert business.connections == 0
    assert app.state.governance_service.run_once()
    first_input = json.loads(FakeCompletionClient.calls[0])
    assert first_input["sample_id"] == "4"
    assert first_input["text"] == 'alpha: 第一条\nlanguage: zh\nzeta: {"a":1,"b":2}'
    assert len(FakeCompletionClient.calls) == 2
    assert business.queries[0][1] == (99,)
    assert "ORDER BY id" in business.queries[1][0]
    task_id = response.json()["data"]["task_id"]
    result_id = app.state.governance_service.repository.get_task(task_id)["result_id"]
    detail = client.get("/api/v1/data-governance/value-results/" + result_id,
                        headers=headers()).json()["data"]
    assert detail["dataset_name"] == "自定义数据集"
    assert detail["target_count"] == 2
    with app.state.database.connect() as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "governance_dataset_versions" not in tables
    assert "governance_source_samples" not in tables


@pytest.mark.parametrize("failure", ["missing_config", "unavailable", "missing_dataset", "wrong_version", "no_rows"])
def test_source_failures_mark_pending_task_failed_without_model_call(app_client, failure):
    """验证业务库无连接、无版本或无记录时仅保存失败任务。"""
    app, client, _, business = app_client
    service = app.state.governance_service
    if failure == "missing_config":
        with app.state.database.connect() as connection:
            connection.execute("DELETE FROM business_database_config")
    elif failure == "unavailable":
        business.unavailable = True
    elif failure == "missing_dataset":
        business.datasets.pop(3)
    elif failure == "wrong_version":
        business.datasets[3]["version"] = "v2"
    else:
        business.records[3] = []
    created = create(client, "governance-value", "general-v1")
    assert service.run_once()
    task = service.repository.get_task(created["task_id"])
    assert task["status"] == "failed"
    assert task["result_id"] is None
    assert task["error_message"]
    assert FakeCompletionClient.calls == []


def test_admin_config_is_persistent_encrypted_and_requires_setup_secret(app_client):
    """验证配置入库、密码密文、管理员权限，以及重启后可读取。"""
    app, client, model_id, business = app_client
    path = "/api/v1/admin/governance/business-database"
    assert client.get(path).status_code == 403
    assert client.get(path, headers=headers()).status_code == 403
    admin = {"X-Setup-Secret": "s" * 32}
    public = client.get(path, headers=admin).json()["data"]
    assert public == {"host": "localhost", "port": 3306, "database": "content_safety",
                      "username": "reader", "password_configured": True}
    assert "secret" not in json.dumps(public)
    with app.state.database.connect() as connection:
        stored = connection.execute("SELECT * FROM business_database_config").fetchone()
    assert stored["encrypted_password"] != "secret"
    assert "secret" not in stored["encrypted_password"]
    assert client.put("/api/v1/admin/governance/models/governance-value", headers=admin,
                      json={"model_id": "missing"}).status_code == 404

    restarted = create_app(app.state.test_settings, client_factory=FakeCompletionClient)
    with TestClient(restarted) as other:
        assert other.get(path, headers=admin).json()["data"] == public
        bindings = other.get("/api/v1/admin/governance/models", headers=admin).json()["data"]
        assert bindings["governance-value"] == model_id
        restarted.state.governance_service.data.connection_factory = business.connect
        created = create(other, "governance-value", "general-v1")
        assert restarted.state.governance_service.run_once()
        assert restarted.state.governance_service.repository.get_task(created["task_id"])["status"] == "succeeded"


def test_admin_calls_identify_token_without_disclosing_it(app_client):
    """验证调用记录能追溯令牌标签，并允许旧令牌补一次标签。"""
    app, client, model_id, business = app_client
    admin = {"X-Setup-Secret": "s" * 32}
    old = app.state.auth_service.create_token("s" * 32, "c" * 32,
                                               utc_now() + timedelta(days=1))
    tokens = client.get("/api/v1/admin/tokens", headers=admin).json()["data"]
    assert next(row for row in tokens if row["id"] == old.id)["label"] == ""
    labeled = client.patch("/api/v1/admin/tokens/" + old.id + "/label", headers=admin,
                           json={"label": "旧系统"})
    assert labeled.status_code == 200
    assert labeled.json()["data"]["label"] == "旧系统"
    assert client.patch("/api/v1/admin/tokens/" + old.id + "/label", headers=admin,
                        json={"label": "其他系统"}).status_code == 409
    response = client.post("/api/llm/invoke", headers=headers("c"),
                           json={"model_id": model_id, "input": "hello", "response_type": "text_answer"})
    assert response.status_code == 200
    calls = client.get("/api/v1/admin/calls", headers=admin).json()["data"]
    assert calls[0]["token_id"] == old.id
    assert calls[0]["token_label"] == "旧系统"
    assert "token_hash" not in calls[0]
    assert "c" * 32 not in json.dumps(calls)
    assert client.get("/api/v1/admin/calls", headers=headers()).status_code == 403


def test_admin_changes_token_status_by_id_without_token_text(app_client):
    """验证管理员凭列表 ID 停用和恢复令牌，无需知道原文。"""
    app, client, _, _ = app_client
    admin = {"X-Setup-Secret": "s" * 32}
    tokens = client.get("/api/v1/admin/tokens", headers=admin).json()["data"]
    token_id = next(row["id"] for row in tokens if row["label"] == "前端")
    path = "/api/v1/admin/tokens/" + token_id + "/status"

    assert client.patch(path, json={"enabled": False}).status_code == 403
    assert client.patch(path, headers=headers(), json={"enabled": False}).status_code == 403
    disabled = client.patch(path, headers=admin, json={"enabled": False})
    assert disabled.status_code == 200
    assert disabled.json()["data"]["id"] == token_id
    assert disabled.json()["data"]["enabled"] is False
    assert "token_hash" not in disabled.json()["data"]
    assert client.get("/api/models", headers=headers()).status_code == 401
    assert client.get("/api/models", headers=headers("b")).status_code == 200

    enabled = client.patch(path, headers=admin, json={"enabled": True})
    assert enabled.status_code == 200
    assert enabled.json()["data"]["enabled"] is True
    assert client.get("/api/models", headers=headers()).status_code == 200
    assert client.patch("/api/v1/admin/tokens/missing/status", headers=admin,
                        json={"enabled": False}).status_code == 404
    assert client.patch(path, headers=admin, json={"enabled": "invalid"}).status_code == 422

    legacy = client.patch("/api/auth/token/status", headers=admin,
                          json={"token": "b" * 32, "enabled": False})
    assert legacy.status_code == 200
    assert legacy.json()["enabled"] is False


def test_admin_soft_deletes_token_and_keeps_call_attribution(app_client):
    """验证按 ID 软删令牌后无法恢复，历史调用仍保留归属。"""
    app, client, model_id, _ = app_client
    admin = {"X-Setup-Secret": "s" * 32}
    token_id = next(row["id"] for row in client.get("/api/v1/admin/tokens", headers=admin)
                    .json()["data"] if row["label"] == "前端")
    response = client.post("/api/llm/invoke", headers=headers(),
                           json={"model_id": model_id, "input": "hello", "response_type": "text_answer"})
    assert response.status_code == 200
    path = "/api/v1/admin/tokens/" + token_id

    assert client.delete(path).status_code == 403
    assert client.delete(path, headers=headers()).status_code == 403
    deleted = client.delete(path, headers=admin)
    assert deleted.status_code == 200
    assert deleted.json()["data"] == {"id": token_id, "deleted": True}
    assert client.get("/api/models", headers=headers()).status_code == 401
    assert token_id not in {row["id"] for row in client.get("/api/v1/admin/tokens", headers=admin)
                            .json()["data"]}
    assert client.patch(path + "/status", headers=admin, json={"enabled": True}).status_code == 404
    assert client.patch("/api/auth/token/status", headers=admin,
                        json={"token": "a" * 32, "enabled": True}).status_code == 404
    assert client.patch("/api/auth/tokens/status", headers=admin,
                        json={"enabled": True}).status_code == 200
    assert client.get("/api/models", headers=headers()).status_code == 401
    assert client.delete(path, headers=admin).status_code == 404

    calls = client.get("/api/v1/admin/calls", headers=admin).json()["data"]
    assert calls[0]["token_id"] == token_id
    assert calls[0]["token_label"] == "前端"


def test_existing_auth_table_gains_label_without_losing_tokens(tmp_path):
    """验证已有 SQLite 令牌表在初始化时仅补充标签列。"""
    path = tmp_path / "existing.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE auth_tokens (id TEXT PRIMARY KEY, token_hash TEXT NOT NULL UNIQUE, "
                           "created_at TEXT NOT NULL, expires_at TEXT NOT NULL, enabled INTEGER NOT NULL)")
        connection.execute("INSERT INTO auth_tokens VALUES (?, ?, ?, ?, ?)",
                           ("old", "hash", utc_now().isoformat(),
                            (utc_now() + timedelta(days=1)).isoformat(), 1))
    database = Database(path)
    database.initialize()
    with database.connect() as connection:
        row = connection.execute("SELECT id, label, deleted_at FROM auth_tokens WHERE id = 'old'").fetchone()
    assert dict(row) == {"id": "old", "label": "", "deleted_at": None}


def test_app_starts_without_mysql_and_worker_marks_task_failed(tmp_path):
    """验证未配置业务库时仍可启动建单，worker 失败且不调用模型。"""
    FakeCompletionClient.calls = []
    settings = Settings(tmp_path / "app.sqlite3", "s" * 32, Fernet.generate_key())
    app = create_app(settings, client_factory=FakeCompletionClient)
    with TestClient(app) as client:
        manager = app.state.model_manager
        model = manager.create_model(upstream_model_id="fake", name="fake", description=None,
                                     base_url="https://example.invalid/v1", api_key="fake-key")
        admin = {"X-Setup-Secret": "s" * 32}
        response = client.put("/api/v1/admin/governance/models/governance-value", headers=admin,
                              json={"model_id": model.id})
        assert response.status_code == 200
        app.state.auth_service.create_token("s" * 32, "a" * 32,
                                            utc_now() + timedelta(days=1), "前端")
        created = create(client, "governance-value", "general-v1")
        assert app.state.governance_service.run_once()
        task = app.state.governance_service.repository.get_task(created["task_id"])
        assert task["status"] == "failed"
        assert "业务数据库连接未配置" in task["error_message"]
        assert FakeCompletionClient.calls == []


def test_frontend_options_empty_overviews_and_business_boundary(app_client):
    """无历史时可初始化；方案只显示模型能力，业务目录不由模型服务接管。"""
    app, client, _, business = app_client
    options = client.get("/api/v1/data-governance/options", headers=headers(),
                         params={"kind": "governance-value"}).json()["data"]
    assert [row["id"] for row in options["schemes"]] == ["general-v1", "general-v2"]
    options = client.get("/api/v1/data-governance/options", headers=headers(),
                         params={"kind": "governance-anomaly"}).json()["data"]
    assert options["schemes"][0]["id"] == "anomaly-basic-v1"
    assert options["types"] == ["标签异常"]
    assert [rule["id"] for rule in options["rules"]] == ["anomaly-rule-1"]
    risk = client.get("/api/v1/data-governance/risk-options", headers=headers()).json()["data"]
    assert risk["schemes"][0]["id"] == "risk-v1"
    assert risk["read_only"] and risk["review_decisions"] == []
    for path in ("/overview", "/data-governance/anomaly-results/overview",
                 "/data-governance/risk-overview"):
        response = client.get("/api/v1" + path, headers=headers())
        assert response.status_code == 200, response.text
        assert all(row["value"] == 0 for row in response.json()["data"]["cards"])
    for path in ("/datasets", "/datasets/3/versions", "/datasets/3/versions/v1.0.0/samples"):
        assert client.get("/api/v1" + path, headers=headers(), params={"ids[]": "1"}).status_code == 404
    assert client.get("/api/v1/data-governance/options",
                      params={"kind": "governance-value"}).status_code == 401
    assert business.connections == 0 and FakeCompletionClient.calls == []


def test_value_samples_add_scope_without_changing_saved_history(app_client):
    """范围字段仅补响应，保存快照不变，历史正文与业务记录后续变更隔离。"""
    app, client, _, business = app_client
    service = app.state.governance_service
    first = create(client, "governance-value", "general-v1")
    assert service.run_once()
    task = service.repository.get_task(first["task_id"])
    result_id = task["result_id"]
    stored = service.repository.get_result(result_id, task["token_hash"], "governance-value")
    assert all("dataset_id" not in row and "version_id" not in row for row in stored["samples"])
    business.records[3][0]["payload"]["text"] = "变更后的正文"
    second = create(client, "governance-value", "general-v1")
    assert service.run_once()
    latest_id = service.repository.get_task(second["task_id"])["result_id"]
    business.unavailable = True
    calls, connections = len(FakeCompletionClient.calls), business.connections
    path = "/api/v1/data-governance/value-results/" + result_id + "/samples"
    response = client.get(path, headers=headers(), params={"page_size": 2})
    assert response.status_code == 200
    rows = response.json()["data"]["items"]
    assert {row["id"] for row in rows} == {"1", "2"}
    assert all(row["dataset_id"] == 3 and row["version_id"] == "v1.0.0" for row in rows)
    assert "变更后的正文" not in rows[0]["text"]
    assert client.get(path, headers=headers("b")).status_code == 404
    assert client.get(path.replace("value-results", "anomaly-results"), headers=headers()).status_code == 404
    latest_path = "/api/v1/data-governance/value-results/" + latest_id + "/samples"
    latest = client.get(latest_path, headers=headers(), params={"page_size": 1}).json()["data"]["items"]
    assert len(latest) == 1 and "变更后的正文" in latest[0]["text"]
    assert service.repository.get_result(result_id, task["token_hash"], "governance-value") == stored
    assert len(FakeCompletionClient.calls) == calls and business.connections == connections


def test_model_overviews_count_latest_snapshot_once_and_isolate_tokens(app_client):
    """重复执行不累加同一版本；统计来自保存结果且不含其他令牌。"""
    app, client, _, business = app_client
    service = app.state.governance_service
    for kind, scheme in (("governance-value", "general-v1"),
                         ("governance-anomaly", "anomaly-basic-v1"),
                         ("governance-risk", "risk-v1")):
        for _ in range(2):
            create(client, kind, scheme)
            assert service.run_once()
    calls, connections = len(FakeCompletionClient.calls), business.connections
    kpis = client.get("/api/v1/kpis", headers=headers(),
                      params={"kind": "governance-value"}).json()["data"]
    assert [row["value"] for row in kpis] == [80, 0, 3, 1]
    anomaly = client.get("/api/v1/data-governance/anomaly-results/overview",
                         headers=headers()).json()["data"]
    assert [row["value"] for row in anomaly["cards"]] == [3, 1, 0, 0]
    risk = client.get("/api/v1/data-governance/risk-overview", headers=headers()).json()["data"]
    assert [row["value"] for row in risk["cards"]] == [3, 1, 0, 1]
    assert len(risk["records"]) == 1
    other = client.get("/api/v1/data-governance/risk-overview", headers=headers("b")).json()["data"]
    assert other["records"] == [] and all(row["value"] == 0 for row in other["cards"])
    assert len(FakeCompletionClient.calls) == calls and business.connections == connections


def test_risk_export_and_knowledge_use_saved_masked_samples(app_client):
    """导出包含所有匹配行、保持遮蔽，规则检索不触发模型或业务读写。"""
    app, client, _, business = app_client
    service = app.state.governance_service
    created = create(client, "governance-risk", "risk-v1")
    assert service.run_once()
    result_id = service.repository.get_task(created["id"])["result_id"]
    base = "/api/v1/data-governance/risk-results/" + result_id
    detail = client.get(base, headers=headers()).json()["data"]
    assert detail["actions"] == ["viewTask", "export"]
    token_hash = service.repository.get_task(created["id"])["token_hash"]
    assert service.repository.get_result(result_id, token_hash, "governance-risk")["summary"]["actions"] == []
    calls, connections = len(FakeCompletionClient.calls), business.connections
    page = client.get(base + "/samples", headers=headers(), params={"page_size": 1}).json()["data"]
    assert len(page["items"]) == 1 and page["total"] == 3
    rows = client.post(base + "/export", headers=headers(), json={}).json()["data"]
    assert len(rows) == 3 and "10086" not in json.dumps(rows, ensure_ascii=False)
    filtered = client.post(base + "/export", headers=headers(),
                           json={"keyword": "用户ID", "level": "MEDIUM", "status": "待复核"}).json()["data"]
    assert len(filtered) == 1
    params = {"result_id": result_id, "sample_id": filtered[0]["id"], "keyword": "pii-03"}
    rules = client.get("/api/v1/risk-knowledge", headers=headers(), params=params).json()["data"]
    assert [rule["id"] for rule in rules] == ["PII-03"]
    assert client.post(base + "/export", headers=headers("b"), json={}).status_code == 404
    assert client.get("/api/v1/risk-knowledge", headers=headers("b"), params=params).status_code == 404
    params["sample_id"] = "missing"
    assert client.get("/api/v1/risk-knowledge", headers=headers(), params=params).status_code == 404
    assert len(FakeCompletionClient.calls) == calls and business.connections == connections


def test_model_read_only_contract_rejects_business_writes(app_client):
    """只读页初始化无修改集，审批、发布和复核明确拒绝而不伪造业务成功。"""
    app, client, _, business = app_client
    response = client.get("/api/v1/data-governance/change-sets/current", headers=headers(),
                          params=scope("anomaly-basic-v1"))
    assert response.status_code == 200 and response.json()["data"] is None
    for path in ("/data-governance/change-sets/publish",
                 "/data-governance/anomaly-results/r/samples/s/candidates/approve",
                 "/data-governance/risk-results/r/samples/s/reviews"):
        response = client.post("/api/v1" + path, headers=headers(), json={})
        assert response.status_code == 405
        assert response.json()["code"] == 405
        assert "仅支持分析查看" in response.json()["message"]
    assert business.connections == 0 and FakeCompletionClient.calls == []


@pytest.mark.parametrize("path,scheme", [
    ("/data-governance/options", "general-v1"),
    ("/kpis", "general-v1"),
    ("/data-governance/value-results/latest", "general-v1"),
    ("/data-governance/value-results/missing", "general-v1"),
    ("/data-governance/value-results/missing/samples", "general-v1"),
    ("/data-governance/anomaly-results/latest", "anomaly-basic-v1"),
    ("/data-governance/anomaly-results", "anomaly-basic-v1"),
    ("/data-governance/anomaly-results/missing", "anomaly-basic-v1"),
    ("/data-governance/anomaly-results/missing/samples", "anomaly-basic-v1"),
    ("/data-governance/anomaly-results/missing/samples/1", "anomaly-basic-v1"),
    ("/data-governance/risk-results/latest", "risk-v1"),
    ("/data-governance/risk-results", "risk-v1"),
    ("/data-governance/risk-results/missing", "risk-v1"),
    ("/data-governance/risk-results/missing/samples", "risk-v1"),
    ("/data-governance/risk-results/missing/samples/1", "risk-v1"),
    ("/data-governance/risk-options", "risk-v1"),
    ("/data-governance/risk-overview", "risk-v1"),
    ("/overview", "anomaly-basic-v1"),
    ("/data-governance/anomaly-results/overview", "anomaly-basic-v1"),
    ("/data-governance/change-sets/current", "anomaly-basic-v1"),
    ("/risk-knowledge", "risk-v1"),
])
def test_frontend_read_routes_reject_wrong_kind(app_client, path, scheme):
    """清单要求 kind 生效，不能在只读兼容接口上悄悄忽略错误类型。"""
    app, client, _, business = app_client
    response = client.get("/api/v1" + path, headers=headers(),
                          params={**scope(scheme), "kind": "wrong", "result_id": "r", "sample_id": "1"})
    assert response.status_code == 400, response.text
    assert response.json()["code"] == 400 and response.json()["data"] is None
    assert business.connections == 0 and FakeCompletionClient.calls == []


def test_frontend_export_rejects_wrong_kind_before_returning_samples(app_client):
    """导出与查询执行相同的任务类型约束。"""
    app, client, _, business = app_client
    response = client.post("/api/v1/data-governance/risk-results/missing/export",
                           headers=headers(), params={"kind": "governance-anomaly"}, json={})
    assert response.status_code == 400, response.text
    assert business.connections == 0 and FakeCompletionClient.calls == []


def test_frontend_task_routes_reject_wrong_kind(app_client):
    """任务入口不能混用治理类型；清单中的错误 kind 返回 400。"""
    app, client, _, business = app_client
    for path, scheme in (("/tasks", "general-v1"),
                         ("/data-governance/anomaly-tasks", "anomaly-basic-v1"),
                         ("/data-governance/risk-tasks", "risk-v1")):
        created = client.post("/api/v1" + path, headers=headers(),
                              json={"kind": "wrong", "name": "错误类型", "input": scope(scheme)})
        assert created.status_code == 400 and created.json()["data"] is None
        queried = client.get("/api/v1" + path + "/missing", headers=headers(), params={"kind": "wrong"})
        assert queried.status_code == 400 and queried.json()["data"] is None
    assert business.connections == 0 and FakeCompletionClient.calls == []


@pytest.mark.parametrize("name,scheme", [
    ("value", "general-v1"), ("anomaly", "anomaly-basic-v1"), ("risk", "risk-v1"),
])
def test_frontend_checklist_read_only_chain(app_client, name, scheme):
    """按清单正确 kind 和分页参数执行从建单到保存结果、校验和全量导出的链路。"""
    app, client, _, business = app_client
    service = app.state.governance_service
    kind = "governance-" + name
    base = "/api/v1/data-governance/" + name + "-results"
    params = {"kind": kind, **scope(scheme)}
    assert client.get(base + "/latest", headers=headers(), params=params).json()["data"] is None
    created = create(client, kind, scheme)
    task_id = created.get("task_id") or created["id"]
    task_path = "/api/v1/tasks/" if name == "value" else "/api/v1/data-governance/" + name + "-tasks/"
    pending = client.get(task_path + task_id, headers=headers(), params={"kind": kind}).json()["data"]
    assert pending["status"] == "pending" and pending["result_id"] is None
    assert FakeCompletionClient.calls == []
    assert service.repository.claim()["id"] == task_id
    running = client.get(task_path + task_id, headers=headers(), params={"kind": kind}).json()["data"]
    assert running["status"] == "running" and running["result_id"] is None
    service.repository.recover_running()
    assert service.run_once()
    completed = client.get(task_path + task_id, headers=headers(), params={"kind": kind}).json()["data"]
    assert completed["status"] == "succeeded" and completed["result_id"]
    calls, connections = len(FakeCompletionClient.calls), business.connections
    result_path = base + "/" + completed["result_id"]
    response = client.get(result_path, headers=headers(), params={"kind": kind})
    assert response.status_code == 200
    envelope = response.json()
    assert envelope["code"] == 0 and envelope["trace_id"] == "trace-test" and envelope["timestamp"]
    result = envelope["data"]
    assert result["scope"] == scope(scheme) and result["task_id"] == task_id
    assert result == client.get(base + "/latest", headers=headers(), params=params).json()["data"]
    if name != "value":
        assert client.get(base, headers=headers(), params=params).json()["data"] == [result]
    rows = []
    for page in (1, 2, 3):
        saved = client.get(result_path + "/samples", headers=headers(),
                           params={"kind": kind, "page": page, "page_size": 1}).json()["data"]
        assert saved["page"] == page and saved["page_size"] == 1
        assert saved["total"] == 3 and saved["total_pages"] == 3
        rows.extend(saved["items"])
    assert {row["id"] for row in rows} == {"1", "2", "3"} and len(rows) == 3
    if name == "risk":
        exported = client.post(result_path + "/export", headers=headers(),
                               params={"kind": kind}, json={}).json()["data"]
        assert exported == rows
        assert "10086" not in json.dumps(exported, ensure_ascii=False)
    assert all(row["dataset_id"] == result["scope"]["dataset_id"]
               and row["version_id"] == result["scope"]["version_id"] for row in rows)
    assert client.get(result_path, headers=headers(), params={"kind": "wrong"}).status_code == 400
    assert len(FakeCompletionClient.calls) == calls and business.connections == connections
