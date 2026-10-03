# Content Security LLM

这是一个 FastAPI 服务：在 SQLite 中管理模型配置、访问令牌和模型调用记录，通过 OpenAI 兼容接口调用模型，并把输出解析为服务端定义的 Pydantic 对象。

## 配置文件说明

仓库中的 `settings.example.json` 只是模板。复制成项目根目录的 `settings.local.json` 并替换占位值后，程序才会读取该文件：

| 字段 | 作用 | 要填什么 |
| --- | --- | --- |
| `setup_secret` | 管理员初始化密钥。创建令牌及启用、禁用令牌时放在 `X-Setup-Secret` 请求头。 | 至少 32 个字符的随机字符串；只保存在服务端。 |
| `credential_key` | 本服务加密、解密 SQLite 中模型 `api_key` 和业务库密码的密钥。它**不参与前端鉴权**，也不是模型服务的 API Key。 | 一次生成的 Fernet 密钥；重启时必须保持不变。 |
| `database_path` | SQLite 数据库文件的位置；首次启动时自动创建。 | 默认 `db/app.sqlite3` 即可；相对路径按配置文件所在目录解析。 |
| `business_api_base_url` | 管理员查询业务用户时使用的业务 HTTP 根地址，与治理 MySQL 连接独立。 | 可选，默认 `http://127.0.0.1:8000/api/v1`；使用 HTTP(S) 地址，不包含用户名、密码、查询参数或片段。 |

这几个值不要混用：`setup_secret` 负责签发与管理访问令牌；访问令牌由 `POST /api/auth/token` 的请求体提供，供调用者访问本服务；添加模型时提交的 `api_key` 则用于调用上游模型服务，由 `credential_key` 加密保存。

## 本地启动

需要 Python 3.11 和 `uv`。在项目根目录依次执行：

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
uv run fastapi dev main.py --port 8001
```

打开 `http://127.0.0.1:8001/docs` 查看和调用接口。启动时如果配置缺失或格式有误，服务会直接报错并指出相应字段。原业务后端继续占用 `8000`。

默认读取项目根目录的 `settings.local.json`，**不要求设置环境变量**。如需指定其他文件，设置 `APP_CONFIG_FILE`；相对路径从项目根目录计算。环境变量 `APP_SETUP_SECRET`、`MODEL_CREDENTIAL_KEY`、`APP_DB_PATH` 可以逐项覆盖文件中的 `setup_secret`、`credential_key`、`database_path`。数据库相对路径从配置文件所在目录计算。

## 接口说明

本地服务地址为 `http://127.0.0.1:8001`，请求体和响应体均为 JSON（`204` 响应没有响应体）。`GET /` 无需鉴权，返回 `{"message":"Content Security LLM"}`。`/api/auth/token` 下的令牌创建与状态接口使用 `X-Setup-Secret`；其余 `/api` 接口都需要 `Authorization: Bearer <访问令牌>`。可在 `/docs` 中交互式调用。

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
| `label` | 否 | 令牌持有人或用途名称，建议创建时填写；旧令牌可由管理员补充。 |

```json
{
  "token": "replace-with-a-random-token-at-least-32-chars",
  "expires_at": "2030-12-31T00:00:00Z",
  "label": "业务后端"
}
```

成功时仍返回 `created_at`、`expires_at` 和 `enabled`；管理员可通过 `/api/v1/admin/tokens` 查询 `id` 和 `label`。每次调用都会新增一枚独立令牌，不会替换已有令牌。服务只保存令牌摘要，不生成或返回令牌原文；请自行妥善保存提交的令牌原文。相同令牌不能重复创建，重复时返回 `409 token_already_exists`。缺少或填错 `X-Setup-Secret` 返回 `403`，令牌太短或过期时间无效返回 `422`。令牌过期后仍可用初始化密钥创建新令牌。

两个状态接口都使用 `X-Setup-Secret`，**不需要 Bearer 令牌**：

- `PATCH /api/auth/token/status` 请求体为 `{"token":"令牌原文","enabled":false}`（启用时设为 `true`），只修改该令牌，返回 `created_at`、`expires_at` 和最新 `enabled`；令牌不存在返回 `404 token_not_found`。
- `PATCH /api/auth/tokens/status` 请求体为 `{"enabled":false}`（启用时设为 `true`），修改**当前已创建的全部令牌**，返回 `{"enabled":false,"updated_count":2}`；没有令牌时 `updated_count` 为 `0`。之后新建的令牌仍默认启用。

管理端单枚令牌启停应使用 `PATCH /api/v1/admin/tokens/{id}/status`，请求头 `X-Setup-Secret`，请求体为 `{"enabled":false}`（启用时为 `true`）。`id` 来自 `GET /api/v1/admin/tokens`；响应的 `data` 含 `id`、`label`、`created_at`、`expires_at`、`enabled`，无需令牌原文。旧的按原文接口保留供现有调用兼容。

删除令牌使用 `DELETE /api/v1/admin/tokens/{id}`，同样只需 `X-Setup-Secret`，成功返回 `data:{"id":"...","deleted":true}`。这是软删除：令牌立即不能用于新请求，且不再出现在管理员令牌列表；历史调用记录仍能显示原令牌 ID 和标签。已删除令牌不能通过单枚或批量启用接口恢复，再次删除返回 `404`；其原文也不能重新创建为新令牌。

被禁用的令牌访问其他接口会收到 `401 invalid_token`；重新启用后，只有尚未过期的令牌能恢复使用。缺少或填错初始化密钥返回 `403 invalid_setup_secret`。旧状态接口仍通过令牌原文定位令牌。

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

`GET /api/llm/response-types` 返回可选类型名数组，当前内置类型如下：

```json
[
  "anomaly_repair",
  "full_chain_audit",
  "judgement",
  "model_risk_governance",
  "reasoning_audit",
  "semantic_risk",
  "text_answer",
  "value_score"
]
```

内置类型对应的结果结构为：

| 类型名 | 结果字段 |
| --- | --- |
| `text_answer` | `answer` |
| `judgement` | `score`, `reason` |
| `model_risk_governance` | `originalOutput`, `governedOutput`, `reason`, `riskLevel`, `reconstruction`（可为 `null`） |
| `semantic_risk` | `findings[{category, suggestedLevel, reason, ruleId, evidenceRefs}]`, `evidence[{quote, feature}]`, `primaryCategory`, `maximumSuggestedLevel` |
| `value_score` | `dimensions` 必须恰好包含文化价值、信息价值、稀缺性、可信度、代表性五个维度；每项包含 `name`, `score`, `reason`, `evidence`，另有 `tier`, `unavailableReason` |
| `anomaly_repair` | `findings[{type, field, reason, quote, ruleId}]`, `primaryType`, `source{filename, line}`, `fieldChanges[{field, before, after}]`, `reason`, `validationResults[{name, passed}]` |
| `full_chain_audit` | `conclusion`, `checks[{stage, passed, reason}]`, `gaps[{description, reason}]` |
| `reasoning_audit` | `steps[{label, detail, occurredAt, verificationState}]`, `riskNodes[{stepId, ruleRef, description, evidenceRefs}]`, `auditResult` |

业务可在服务端定义 Pydantic `BaseModel` 类并用 `llm.outputs.register_response_type("类型名", 类)` 注册更多类型；客户端不能上传任意类或 Schema。

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

`GET /api/calls?limit=50&offset=0` 只返回当前 Bearer 令牌产生的调用记录，按时间从新到旧排列。`limit` 默认 `50`，允许 `1`～`200`；`offset` 默认 `0`，必须大于等于 `0`，分页在按令牌过滤后进行。参数无效返回 `422`。每条记录含 `id`、`model_id`、`input`、`output`、`response_type`、`status`、`error_code`、`created_at`；成功时 `status` 为 `success`、`output` 为结果对象、`error_code` 为 `null`，失败时 `status` 为 `error`、`output` 为 `null`、`error_code` 给出原因。数据库保存令牌的 SHA-256 摘要以关联记录，但接口不会返回摘要。即使模型已软删除，其历史记录仍会保留。当前版本仅支持包含 `token_hash` 字段的新数据库结构，不再自动迁移旧结构。

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

## 数据治理模型任务

本服务另提供 `/api/v1` 下的三类异步模型任务，不改变上文 `/api` 接口。新增接口沿用本服务 Bearer 令牌，响应为 `{code, message, data, trace_id, timestamp}`。结果按创建任务的令牌隔离；所有结果查询只读取 SQLite 快照。

先用 `POST /api/models` 创建模型，再用管理员接口绑定三类任务的本地 `model_id`。管理员接口只接受 `X-Setup-Secret`，成功响应采用 `{code, message, data, trace_id, timestamp}`：

| 方法与路径 | 请求体或用途 |
| --- | --- |
| `PUT /api/v1/admin/governance/models/{kind}` | `{"model_id":"本地模型ID"}`；`kind` 为 `governance-value`、`governance-anomaly` 或 `governance-risk`。 |
| `GET /api/v1/admin/governance/models` | 查询当前三类任务的模型绑定。 |
| `PUT /api/v1/admin/governance/business-database` | `{"host":"主机","port":3306,"database":"库名","username":"只读用户","password":"密码"}`；保存时加密密码，不立即连库。 |
| `GET /api/v1/admin/governance/business-database` | 查询连接元数据；不返回密码或密文。未配置时 `data` 为 `null`。 |
| `GET /api/v1/admin/business-users?page=1&page_size=20&keyword=张` | 使用管理员密钥查询业务用户目录；服务端访问业务后端已有的 `/api/v1/users`。 |
| `GET /api/v1/admin/tokens` | 查询令牌 ID、持有人或用途标签及状态，不返回令牌原文或哈希。 |
| `PATCH /api/v1/admin/tokens/{id}/status` | `{"enabled":false}` 或 `true`；管理员按列表 ID 启停单枚令牌。 |
| `DELETE /api/v1/admin/tokens/{id}` | 按列表 ID 软删除令牌；立即失效并从令牌列表隐藏，保留历史调用归属。 |
| `PATCH /api/v1/admin/tokens/{id}/label` | `{"label":"持有人或用途"}`；只允许给空标签的旧令牌补一次，避免历史记录换名。 |
| `GET /api/v1/admin/calls?limit=50&offset=0` | 跨令牌查询调用记录及 `token_id`、`token_label`。 |

业务用户查询返回分页对象，用户字段限定为 `id/username/display_name/role/enabled`，业务接口 `status=normal` 映射为启用，其他状态映射为停用。请求超时为 5 秒；连接失败、上游非成功响应或格式错误返回 `502`，超时返回 `504`，空列表正常返回。管理员密钥和模型令牌不会转发给业务后端，查询失败不影响已有模型访问。

管理端选中用户后，可生成独立模型令牌，通过现有 `POST /api/auth/token` 登记，并使用 `business-user:<用户ID>` 标签标记持有人。标签不构成业务登录身份绑定，业务账号停用不会自动禁用模型令牌。令牌列表只查询元数据；单枚启停仍通过 `PATCH /api/auth/token/status` 输入令牌原文。具体管理端修改意见、接口示例及验收见 [管理端业务用户授权对接说明.md](管理端业务用户授权对接说明.md)。本项目没有修改管理端、业务前端或业务后端代码。

任务模型绑定和业务库连接均保存在本项目 SQLite。密码用 `credential_key` 加密；重启时须使用相同密钥。旧配置文件中的 `governance_model_ids` 和 `business_database_url` 暂被接受但不再生效，应通过上述接口重新配置。未配置模型或模型已删除时，创建对应任务返回 `503`。启动 API 后，另起一个进程执行：

```powershell
uv run python -m service.governance_worker
```

当前 worker 设计为单实例运行。`--once` 可领取并处理一个任务。worker 重启时会将中断的 `running` 任务重新排队，跳过已保存的逐条结果；模型调用进行中恰好中断时，该条可能再次调用。

| 任务 | 创建和轮询 | 只读结果 |
| --- | --- | --- |
| 价值分析 | `POST /api/v1/tasks`（`kind=governance-value`）；`GET /api/v1/tasks`、`GET /api/v1/tasks/{id}` | `/api/v1/data-governance/value-results/latest`、`/{id}`、`/{id}/samples` |
| 异常治理 | `POST /api/v1/data-governance/anomaly-tasks`；`GET .../anomaly-tasks/{id}` | `/api/v1/data-governance/anomaly-results/latest`、历史列表、`/{id}`、`/{id}/samples`、`/{id}/samples/{sampleId}` |
| 风险分级 | `POST /api/v1/data-governance/risk-tasks`；`GET .../risk-tasks/{id}` | `/api/v1/data-governance/risk-results/latest`、历史列表、`/{id}`、`/{id}/samples`、`/{id}/samples/{sampleId}` |

创建请求只包含 `kind`、`name`、`input: {dataset_id, version_id, language, scheme_id}`，立即返回 `pending`，不需要传样本正文。worker 使用管理员保存的只读连接访问业务后端 MySQL，核对 `datasets.version` 与请求的 `version_id`，按 `dataset_records.id` 升序读取该数据集的全部记录。每条记录的 `payload` 按字段名升序拼成多行 `字段名: 值`；嵌套值采用字段名排序的 JSON。`language` 保留在范围中，目前不筛选记录；行内无语种时模型输入使用 `unknown`。支持的方案为价值 `general-v1`/`general-v2`、异常 `anomaly-basic-v1`、风险 `risk-v1`，规则仍由 `service/governance_data.py` 提供。

本服务无需 MySQL 即可启动和创建任务。未配置连接、连接失败、数据集版本不匹配或无记录时，worker 将任务标记为 `failed`，不会调用模型。业务库的 `dataset_records` 没有版本字段，因此当前版本匹配后会读取该数据集的全部记录，包括可能在旧版本接入的记录。业务库连接建议使用只具有 `datasets` 和 `dataset_records` 查询权限的账号；连接参数仅保存在本项目 SQLite，不进入任务请求或结果。治理任务、逐条模型输出和结果快照仍写入本项目 SQLite；业务后端可通过现有只读结果接口获取。真实业务库连接尚未提供，自动化测试用模拟连接验证。

### 旧前端接入契约

已按前端提供的《数据治理三模块接口清单》对齐分析、查询和导出，逐项对应及返回字段见 [前端契约.md](前端契约.md)。三方职责及修改原因分别见 [业务后端对接说明.md](业务后端对接说明.md) 和 [前端对接说明.md](前端对接说明.md)。用户确认模型结果继续只读。错误 `kind` 返回 HTTP 400；结果类型或令牌不匹配返回 404。不传 `kind` 的已有只读调用按路由类型处理，共享方案查询和价值任务查询仍需显式传入。

本项目保持独立模型服务，不转发业务接口，不修改外部前端或原业务后端。调用方将模型请求发送到 `/llm-api/v1`，代理到 `http://127.0.0.1:8001` 并将 `/llm-api` 重写为 `/api`。例如 `/llm-api/v1/data-governance/risk-tasks` 对应本服务 `/api/v1/data-governance/risk-tasks`。部署服务器也需配置相同代理规则。

清单第一部分的数据集、版本目录、资源样本三个接口全部归原业务后端 `8000`，本项目不提供 `/api/v1/datasets` 下的资源接口。接入、处理、业务任务中心等请求也继续走原业务后端。本服务 `/api/v1/tasks` 只接收价值分析任务，不能把所有业务任务请求切过来。资源目录应提供真实数据集 ID 和当前版本；模型接入使用 `language=all`。客户端使用本服务已签发的访问令牌，不能使用业务登录令牌、管理员初始化密钥或上游模型密钥。

以下接口补齐旧治理页面的只读请求，均需 Bearer 令牌并返回治理响应包裹：

| 方法与 `/api/v1` 后的路径 | 用途 |
| --- | --- |
| `GET /data-governance/options?kind=governance-value` | 价值方案 `general-v1/general-v2`。 |
| `GET /data-governance/options?kind=governance-anomaly` | 异常方案 `anomaly-basic-v1`，只展示现有标签异常能力。 |
| `GET /data-governance/risk-options` | 风险方案 `risk-v1`、分级、筛选及只读动作；数据集由业务目录补入。 |
| `GET /kpis?kind=governance-value` | 当前令牌的价值结果统计。 |
| `GET /overview?kind=governance-anomaly` 或 `/data-governance/anomaly-results/overview` | 当前令牌的异常检测统计。 |
| `GET /data-governance/risk-overview` | 当前令牌的风险统计和结果记录。 |
| `POST /data-governance/risk-results/{id}/export` | 请求体包含 `keyword/level/status`，返回全部匹配样本，保留风险文本遮蔽。 |
| `GET /risk-knowledge?result_id=结果ID&sample_id=样本ID&keyword=关键词` | 筛选该风险样本快照中的规则。 |
| `GET /data-governance/change-sets/current` | 携带完整范围参数，返回 `data=null`，供旧只读页面初始化。 |

统计只包含当前令牌的保存结果，同一数据集版本取最近一次分析，不重复累加任务；价值均分按有效样本加权。统计不代表业务平台全部语料。异常待复核工单与已修复数量为零，本服务没有对应业务流程。

三类页面直接显示所属模型结果的保存样本，校验结果 `scope` 与请求范围，以及样本的 `dataset_id/version_id` 与所属结果一致。价值样本的两个范围字段仅在 HTTP 响应补充，不修改快照。模型模式不再调用原资源样本接口比较正文：原接口读取旧业务任务快照，不能校验本项目模型正文。历史内容按结果 ID 读取，不用当前业务正文替换；风险页保留遮蔽文本。所有结果查询和导出均读取 SQLite，不重新调用模型或读取业务 MySQL。

风险结果汇总的只读动作是 `viewTask/export`，样本不开放复核动作。异常候选审批、修改集校验/发布、风险复核的 POST 请求明确返回 `405` 和“当前模型结果仅支持分析查看”。模型结果不写入业务任务中心，不应用异常建议，也不发布业务版本。

调用方需启用真实 HTTP 请求，并在 `pending`、`running` 时持续轮询；`succeeded` 后使用 `result_id` 读取结果，`failed` 时显示 `error_message`。未启动 worker 时任务保持 `pending`；令牌无效返回 `401`，不自动回退模拟结果。使用保存结果正文、停止页面离开后的轮询和隐藏写操作由调用方处理。本项目没有修改这些前端行为。

## 自定义 PyTorch 训练

`/api/v1/training` 提供可信 Python 源码的异步训练、逐步指标查询、取消和权重下载。`GET /api/v1/training/defaults` 返回默认参数和完整请求示例；`POST /api/v1/training/tasks` 接收源码、业务数据集 ID/当前版本和张量契约，HTTP 202 返回排队任务。请求只检查源码语法，不执行源码；后台单实例训练 worker 负责独立进程中的真实训练：

```powershell
uv run python -m service.training_worker
```

该 worker 与治理 worker 独立，API、worker 共享 SQLite 和业务库管理员配置。项目使用 Python 3.11，PyTorch 官方 CUDA 12.8 构建及指标依赖由 `uv.lock` 固定。默认 80%/20% 划分、AdamW、10 个 epoch，每步记录 loss，每轮记录验证 loss 和 precision/accuracy/F1/recall；分类指标不适用时返回 null。产物在 SQLite 同目录的 `training-runs` 中，包含源码、数据/划分快照、配置、环境、日志、最后和最优权重。

具体参数、Python 函数契约、接口及现有 Vue 训推页面的字段映射见 [训练后端与前端对接说明](训练后端与前端对接说明.md)。完整分类及自定义训练请求示例通过 `GET /api/v1/training/defaults` 获取。上传代码只适用于可信开发者，训练进程不是安全沙箱。本次没有修改外部前端或业务后端；前端需按说明改用真实训练接口。
