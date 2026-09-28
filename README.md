# Content Security LLM

这是一个 FastAPI 服务：在 SQLite 中管理模型配置、访问令牌和模型调用记录，通过 OpenAI 兼容接口调用模型，并把输出解析为服务端定义的 Pydantic 对象。

## 配置文件说明

仓库中的 `settings.example.json` 只是模板。复制成项目根目录的 `settings.local.json` 并替换占位值后，程序才会读取该文件：

| 字段 | 作用 | 要填什么 |
| --- | --- | --- |
| `setup_secret` | 管理员初始化密钥。创建令牌及启用、禁用令牌时放在 `X-Setup-Secret` 请求头。 | 至少 32 个字符的随机字符串；只保存在服务端。 |
| `credential_key` | 本服务加密、解密 SQLite 中模型 `api_key` 的密钥。它**不参与前端鉴权**，也不是模型服务的 API Key。 | 一次生成的 Fernet 密钥；重启时必须保持不变。 |
| `database_path` | SQLite 数据库文件的位置；首次启动时自动创建。 | 默认 `db/app.sqlite3` 即可；相对路径按配置文件所在目录解析。 |

这几个值不要混用：`setup_secret` 负责签发与管理访问令牌；访问令牌由 `POST /api/auth/token` 的请求体提供，供调用者访问本服务；添加模型时提交的 `api_key` 则用于调用上游模型服务，由 `credential_key` 加密保存。

## 本地启动

需要 Python 3.9+ 和 `uv`。在项目根目录依次执行：

```powershell
uv sync
Copy-Item settings.example.json settings.local.json
```

生成两个随机值：

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
python -c "import base64,secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"
```

编辑 `settings.local.json`，把第一条命令的结果填入 `setup_secret`，第二条命令的结果填入 `credential_key`。保留默认的 `database_path` 即可。不要保留模板中的 `REPLACE_WITH_...` 占位值。

```powershell
uv run fastapi dev main.py
```

打开 `http://127.0.0.1:8000/docs` 查看和调用接口。启动时如果配置缺失或格式有误，服务会直接报错并指出相应字段。

默认读取项目根目录的 `settings.local.json`，**不要求设置环境变量**。如需指定其他文件，设置 `APP_CONFIG_FILE`；相对路径从项目根目录计算。环境变量 `APP_SETUP_SECRET`、`MODEL_CREDENTIAL_KEY`、`APP_DB_PATH` 可以逐项覆盖文件中的 `setup_secret`、`credential_key`、`database_path`。数据库相对路径从配置文件所在目录计算。

## 接口说明

本地服务地址为 `http://127.0.0.1:8000`，请求体和响应体均为 JSON（`204` 响应没有响应体）。`GET /` 无需鉴权，返回 `{"message":"Content Security LLM"}`。`/api/auth/token` 下的令牌创建与状态接口使用 `X-Setup-Secret`；其余 `/api` 接口都需要 `Authorization: Bearer <访问令牌>`。可在 `/docs` 中交互式调用。

推荐调用顺序：创建访问令牌 → 添加模型 → 查询可用结果类型 → 调用模型 → 查询调用记录。时间字段为带时区的 ISO 8601 字符串，下面示例均使用 UTC。

| 方法与路径 | 用途 | 成功状态 |
| --- | --- | --- |
| `POST /api/auth/token` | 创建一枚访问令牌 | `201` |
| `PATCH /api/auth/token/status` | 启用或禁用指定令牌 | `200` |
| `PATCH /api/auth/tokens/status` | 启用或禁用当前全部令牌 | `200` |
| `POST /api/models` | 添加模型配置 | `201` |
| `GET /api/models` | 列出未删除的模型 | `200` |
| `GET /api/models/{model_id}` | 查询一个模型 | `200` |
| `PATCH /api/models/{model_id}` | 修改模型配置 | `200` |
| `DELETE /api/models/{model_id}` | 软删除模型 | `204` |
| `GET /api/llm/response-types` | 列出可用结果类型 | `200` |
| `POST /api/llm/invoke` | 调用模型并返回结构化结果 | `200` |
| `GET /api/calls` | 分页查询调用记录 | `200` |

### 创建与管理访问令牌

`POST /api/auth/token` 不使用 Bearer 令牌。请求头 `X-Setup-Secret` 必须与配置文件中的 `setup_secret` 一致。请求体：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `token` | 是 | 调用者自行生成、至少 32 个字符的共享令牌。 |
| `expires_at` | 是 | 未来的过期时间，必须含时区，例如 `2030-12-31T00:00:00Z`。 |

```json
{
  "token": "replace-with-a-random-token-at-least-32-chars",
  "expires_at": "2030-12-31T00:00:00Z"
}
```

成功时返回 `{"created_at":"2026-09-28T10:00:00Z","expires_at":"2030-12-31T00:00:00Z","enabled":true}`。每次调用都会新增一枚独立令牌，不会替换已有令牌。服务只保存令牌摘要，不生成或返回令牌原文；请自行妥善保存提交的令牌原文。相同令牌不能重复创建，重复时返回 `409 token_already_exists`。缺少或填错 `X-Setup-Secret` 返回 `403`，令牌太短或过期时间无效返回 `422`。令牌过期后仍可用初始化密钥创建新令牌。

两个状态接口都使用 `X-Setup-Secret`，**不需要 Bearer 令牌**：

- `PATCH /api/auth/token/status` 请求体为 `{"token":"令牌原文","enabled":false}`（启用时设为 `true`），只修改该令牌，返回 `created_at`、`expires_at` 和最新 `enabled`；令牌不存在返回 `404 token_not_found`。
- `PATCH /api/auth/tokens/status` 请求体为 `{"enabled":false}`（启用时设为 `true`），修改**当前已创建的全部令牌**，返回 `{"enabled":false,"updated_count":2}`；没有令牌时 `updated_count` 为 `0`。之后新建的令牌仍默认启用。

被禁用的令牌访问其他接口会收到 `401 invalid_token`；重新启用后，只有尚未过期的令牌能恢复使用。缺少或填错初始化密钥返回 `403 invalid_setup_secret`。数据库中的 `id` 仅供内部使用，接口通过令牌原文定位单个令牌。

### 模型配置

本组接口都需要 Bearer 令牌。添加模型：`POST /api/models`，请求字段如下。

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `upstream_model_id` | 是 | 发送给上游服务的模型名称，非空。 |
| `name` | 是 | 本服务显示的名称，非空。 |
| `description` | 否 | 模型描述，可为 `null`。 |
| `base_url` | 是 | OpenAI 兼容服务的 HTTP(S) API 根地址，通常以 `/v1` 结尾。 |
| `api_key` | 是 | 上游服务密钥，非空；加密保存且不会在查询响应中返回。 |

```json
{
  "upstream_model_id": "provider-model",
  "name": "内容分析模型",
  "description": "供内容分析业务调用",
  "base_url": "https://models.example.com/v1",
  "api_key": "your-provider-api-key"
}
```

`201` 响应示例；后续接口使用这里的本地 `model_id`，而不是 `upstream_model_id`：

```json
{
  "model_id": "4b63df30-9119-4e95-a8ee-e91c1cbd0d77",
  "upstream_model_id": "provider-model",
  "name": "内容分析模型",
  "description": "供内容分析业务调用",
  "base_url": "https://models.example.com/v1",
  "created_at": "2026-09-28T10:00:00Z",
  "updated_at": "2026-09-28T10:00:00Z"
}
```

- `GET /api/models` 返回上述对象组成的数组，没有模型时返回 `[]`。`GET /api/models/{model_id}` 返回一个同结构对象；找不到或已软删除时返回 `404`。
- `PATCH /api/models/{model_id}` 接收上述创建字段的任意子集，例如 `{"name":"新名称","api_key":"new-key"}`，成功返回更新后的完整模型对象。`description` 可设为 `null`；其他字段不能显式设为 `null`。不存在的模型返回 `404`，无效字段值返回 `422`。
- `DELETE /api/models/{model_id}` 成功返回 `204` 且无响应体。它只标记删除，后续不能再查询或调用该模型，历史调用记录仍可查询；不存在的模型返回 `404`。

### 结构化模型调用

`GET /api/llm/response-types` 返回可选类型名数组，当前内置 `["judgement", "text_answer"]`。业务可在服务端定义 Pydantic `BaseModel` 类并用 `llm.outputs.register_response_type("类型名", 类)` 注册更多类型；客户端不能上传任意类或 Schema。

`POST /api/llm/invoke` 的请求体三个字段均必填、非空：`model_id` 是上述本地 ID，`input` 是输入文本，`response_type` 是已注册的类型名。

```json
{
  "model_id": "4b63df30-9119-4e95-a8ee-e91c1cbd0d77",
  "input": "用一句话描述这段内容",
  "response_type": "text_answer"
}
```

`200` 响应示例：

```json
{
  "model_id": "4b63df30-9119-4e95-a8ee-e91c1cbd0d77",
  "response_type": "text_answer",
  "result": {"answer": "这段内容的简要描述。"}
}
```

服务调用 OpenAI 兼容的 Chat Completions `create()`，在 system 消息中提供服务端 Pydantic 类生成的 JSON Schema，要求上游只返回符合它的 JSON 对象；不会向上游发送 `response_format=json_schema`。收到文本后，服务端再用 Pydantic 解析、校验为 Python 类实例，并将 `result` 序列化为 JSON。提示词无法保证上游一定遵守 Schema；空响应、截断或校验失败会返回错误，不会把无效结果交给业务代码。通用接口不会自动执行 `llm/tools.py` 中的工具，业务代码可通过 `get_all_tools()` 获取它们。未知 `response_type` 返回 `422`，模型不存在返回 `404`，上游超时返回 `504`，上游拒绝、格式不符或其他调用失败返回 `502`。成功和上游调用失败均会写入调用记录；未知类型和不存在的模型不会生成记录。

### 调用记录

`GET /api/calls?limit=50&offset=0` 按时间从新到旧返回记录数组。`limit` 默认 `50`，允许 `1`～`200`；`offset` 默认 `0`，必须大于等于 `0`。参数无效返回 `422`。每条记录含 `id`、`model_id`、`input`、`output`、`response_type`、`status`、`error_code`、`created_at`；成功时 `status` 为 `success`、`output` 为结果对象、`error_code` 为 `null`，失败时 `status` 为 `error`、`output` 为 `null`、`error_code` 给出原因。即使模型已软删除，其历史记录仍会保留。

```json
[
  {
    "id": "8e50bc2d-5ddc-47e8-a1b7-cae048933cee",
    "model_id": "4b63df30-9119-4e95-a8ee-e91c1cbd0d77",
    "input": "用一句话描述这段内容",
    "output": {"answer": "这段内容的简要描述。"},
    "response_type": "text_answer",
    "status": "success",
    "error_code": null,
    "created_at": "2026-09-28T10:01:00Z"
  }
]
```

### 错误格式

业务错误使用 `{"detail":{"code":"..."}}` 格式；请求字段格式校验失败的 `422` 则使用 FastAPI 标准的 `detail` 数组。主要错误码如下：

| HTTP 状态 | `detail.code` | 含义 |
| --- | --- | --- |
| `401` | `invalid_token` | Bearer 令牌缺失、错误、被禁用或过期。 |
| `403` | `invalid_setup_secret` | 创建或管理令牌时缺少或填错 `X-Setup-Secret`。 |
| `404` | `model_not_found`、`token_not_found` | 本地模型 ID 不存在或模型已软删除；或指定的令牌 ID 不存在。 |
| `409` | `token_already_exists` | 相同令牌已经创建。 |
| `422` | `null_model_field`、`unknown_response_type` | 修改模型时将非空字段设为 `null`，或调用了未注册的结果类型。 |
| `502` | `credential_unavailable`、`empty_response`、`model_refusal`、`invalid_structured_output`、`provider_unavailable`、`provider_error`、`incomplete_response` | 模型密钥不可解密，或上游服务未能给出可用的结构化结果。 |
| `504` | `provider_timeout` | 上游模型请求超时。 |

例如令牌错误会返回 `{"detail":{"code":"invalid_token","message":"Missing, invalid, disabled, or expired token"}}`；模型不存在会返回 `{"detail":{"code":"model_not_found","model_id":"..."}}`。添加模型只保存配置，不会预先测试上游服务连通性；首次调用时才会发现上游地址、密钥或输出格式方面的问题。

上游返回 HTTP 错误时，本服务仍返回 `502`，但 `detail` 会附带 `upstream_status`（例如 `401`、`402`、`429`），便于区分鉴权、余额或限流等情况；不会向前端透传上游原始错误正文。上游 HTTP 错误统一使用 `provider_error`，本地 Pydantic 解析或校验失败使用 `invalid_structured_output`。

## 注意事项

- `settings.local.json` 和默认 SQLite 文件已加入 `.gitignore`；若改用其他路径，也要确保配置与数据库文件不会被提交或公开。
- `credential_key` 用于加密模型 API Key。重启或迁移时必须沿用同一个值；丢失后，数据库中已有的模型密钥无法解密。模型查询接口不会返回 API Key。
- SQLite 中的调用输入、输出是明文，可能包含敏感内容。应限制数据库文件和 `GET /api/calls` 的访问，并按需制定备份与保留期限。
- 可创建多枚访问令牌；持有任一有效令牌的人都可以调用模型、管理模型和查看调用记录。只有持有初始化密钥的人能创建、启用或禁用令牌。对外部署时应使用 HTTPS，并将初始化密钥留在服务端。

代码职责：`router/` 处理 HTTP，`schemas/` 定义接口数据结构，`service/` 编排业务流程，`llm/` 管理模型调用与工具，`db/` 管理 SQLite，通用时间函数在 `utils/time.py`。

## 测试

```powershell
uv run pytest
```

测试使用临时 SQLite 和模拟模型服务，不会产生真实模型调用费用。
