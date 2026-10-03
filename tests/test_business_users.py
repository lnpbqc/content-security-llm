"""管理员业务用户查询与独立模型令牌流程。"""

import json
from datetime import timedelta

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from config import Settings
from main import create_app
from utils.time import utc_now


def user_page():
    return {"code": 0, "data": {
        "items": [{"id": 12, "username": "zhangsan", "display_name": "张三",
                   "role": "普通用户", "status": "normal", "email": "private@example.test",
                   "token": "must-not-return", "password": "must-not-return"}],
        "total": 21, "page": 2, "page_size": 20, "total_pages": 2,
    }}


@pytest.fixture
def client(tmp_path, monkeypatch):
    calls = []
    upstream = {"body": user_page(), "status": 200}

    def handle_request(transport, request):
        calls.append(request)
        if "error" in upstream:
            raise upstream["error"]
        if "raw" in upstream:
            return httpx.Response(upstream["status"], content=upstream["raw"])
        return httpx.Response(upstream["status"], json=upstream["body"])

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)
    settings = Settings(tmp_path / "app.sqlite3", "s" * 32, Fernet.generate_key(),
                        business_api_base_url="https://business.example/api/v1/")
    with TestClient(create_app(settings)) as api:
        yield api, upstream, calls


def admin_headers():
    return {"X-Setup-Secret": "s" * 32, "X-Trace-Id": "business-users-test",
            "Authorization": "Bearer must-not-forward"}


@pytest.mark.parametrize("headers", [{}, {"X-Setup-Secret": "wrong"},
                                      {"Authorization": "Bearer model-token"}])
def test_admin_rejected_before_business_request(client, headers):
    api, _, calls = client
    response = api.get("/api/v1/admin/business-users", headers=headers)
    assert response.status_code == 403
    assert response.json()["data"] is None
    assert calls == []


def test_users_forward_pagination_and_strip_private_fields(client):
    api, _, calls = client
    response = api.get("/api/v1/admin/business-users", headers=admin_headers(),
                       params={"page": 2, "page_size": 20, "keyword": "张"})
    assert response.status_code == 200, response.text
    envelope = response.json()
    assert envelope["code"] == 0 and envelope["trace_id"] == "business-users-test"
    assert envelope["timestamp"]
    assert envelope["data"] == {
        "items": [{"id": 12, "username": "zhangsan", "display_name": "张三",
                   "role": "普通用户", "enabled": True}],
        "total": 21, "page": 2, "page_size": 20, "total_pages": 2,
    }
    assert len(calls) == 1
    request = calls[0]
    assert str(request.url).startswith("https://business.example/api/v1/users?")
    assert dict(request.url.params) == {"page": "2", "page_size": "20", "keyword": "张"}
    assert "authorization" not in request.headers and "x-setup-secret" not in request.headers
    assert set(request.extensions["timeout"].values()) == {5.0}


@pytest.mark.parametrize("status,enabled", [("normal", True), ("pending", False),
                                           ("disabled", False), ("unknown", False)])
def test_business_user_status_conversion(client, status, enabled):
    api, upstream, _ = client
    upstream["body"]["data"]["items"][0]["status"] = status
    result = api.get("/api/v1/admin/business-users", headers=admin_headers())
    assert result.status_code == 200
    assert result.json()["data"]["items"][0]["enabled"] is enabled


def test_empty_users_and_default_query(client):
    api, upstream, calls = client
    page = {"items": [], "total": 0, "page": 1, "page_size": 20, "total_pages": 0}
    upstream["body"] = {"code": 0, "data": page}
    result = api.get("/api/v1/admin/business-users", headers=admin_headers())
    assert result.status_code == 200 and result.json()["data"] == page
    assert dict(calls[0].url.params) == {"page": "1", "page_size": "20"}


@pytest.mark.parametrize("params", [{"page": 0}, {"page": "wrong"},
                                     {"page_size": 0}, {"page_size": 101}])
def test_invalid_pagination_does_not_request_business(client, params):
    api, _, calls = client
    result = api.get("/api/v1/admin/business-users", headers=admin_headers(), params=params)
    assert result.status_code == 422 and result.json()["data"] is None
    assert calls == []


@pytest.mark.parametrize("error,code", [
    (httpx.ConnectError("private upstream details"), 502),
    (httpx.ReadTimeout("private upstream details"), 504),
    (httpx.ConnectTimeout("private upstream details"), 504),
])
def test_transport_errors_are_explicit(client, error, code):
    api, upstream, _ = client
    upstream["error"] = error
    result = api.get("/api/v1/admin/business-users", headers=admin_headers())
    assert result.status_code == code and result.json()["code"] == code
    assert result.json()["data"] is None
    assert "private upstream details" not in result.text


@pytest.mark.parametrize("status", [302, 401, 404, 500])
def test_non_success_business_response(client, status):
    api, upstream, _ = client
    upstream["status"] = status
    result = api.get("/api/v1/admin/business-users", headers=admin_headers())
    assert result.status_code == 502 and result.json()["data"] is None


@pytest.mark.parametrize("body", [
    [], {"code": 1, "data": user_page()["data"]}, {"code": 0, "data": None},
    {"code": 0, "data": {**user_page()["data"], "items": [{}]}},
    {"code": 0, "data": {**user_page()["data"], "total": -1}},
    {"code": 0, "data": {**user_page()["data"], "total": "21"}},
])
def test_malformed_business_response_is_not_empty_success(client, body):
    api, upstream, _ = client
    upstream["body"] = body
    result = api.get("/api/v1/admin/business-users", headers=admin_headers())
    assert result.status_code == 502 and result.json()["data"] is None


def test_invalid_json_business_response(client):
    api, upstream, _ = client
    upstream["raw"] = "<html>not JSON</html>"
    result = api.get("/api/v1/admin/business-users", headers=admin_headers())
    assert result.status_code == 502 and result.json()["data"] is None


def test_existing_token_flow_survives_business_failure(client):
    api, upstream, _ = client
    upstream["error"] = httpx.ConnectError("offline")
    assert api.get("/api/v1/admin/business-users", headers=admin_headers()).status_code == 502
    token = "a" * 64
    created = api.post("/api/auth/token", headers=admin_headers(), json={
        "token": token, "expires_at": (utc_now() + timedelta(days=1)).isoformat(),
        "label": "business-user:12",
    })
    assert created.status_code == 201 and created.json()["enabled"] is True
    listed = api.get("/api/v1/admin/tokens", headers=admin_headers()).json()["data"]
    assert listed[0]["label"] == "business-user:12" and token not in json.dumps(listed)
    bearer = {"Authorization": "Bearer " + token}
    assert api.get("/api/v1/kpis?kind=governance-value", headers=bearer).status_code == 200
    for enabled, expected_status in ((False, 401), (True, 200)):
        changed = api.patch("/api/auth/token/status", headers=admin_headers(),
                            json={"token": token, "enabled": enabled})
        assert changed.status_code == 200 and changed.json()["enabled"] is enabled
        assert api.get("/api/v1/kpis?kind=governance-value", headers=bearer).status_code == expected_status


def test_business_api_settings_default_and_local_file(tmp_path, monkeypatch):
    for name in ("APP_SETUP_SECRET", "MODEL_CREDENTIAL_KEY", "APP_DB_PATH"):
        monkeypatch.delenv(name, raising=False)
    config_file = tmp_path / "settings.json"
    data = {"setup_secret": "s" * 32, "credential_key": Fernet.generate_key().decode()}
    config_file.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("APP_CONFIG_FILE", str(config_file))
    assert Settings.load().business_api_base_url == "http://127.0.0.1:8000/api/v1"
    data["business_api_base_url"] = "https://business.example/api/v1/"
    config_file.write_text(json.dumps(data), encoding="utf-8")
    assert Settings.load().business_api_base_url == "https://business.example/api/v1"


@pytest.mark.parametrize("url", [None, "", "file:///tmp/users", "not-a-url",
                                "https://business.example/api/v1?key=secret"])
def test_invalid_business_api_settings(tmp_path, monkeypatch, url):
    config_file = tmp_path / "settings.json"
    config_file.write_text(json.dumps({"setup_secret": "s" * 32,
                                      "credential_key": Fernet.generate_key().decode(),
                                      "business_api_base_url": url}), encoding="utf-8")
    monkeypatch.setenv("APP_CONFIG_FILE", str(config_file))
    with pytest.raises(RuntimeError, match="business_api_base_url"):
        Settings.load()
