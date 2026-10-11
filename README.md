# Content Security LLM

基于 FastAPI 的内容安全与模型服务，提供模型调用、内容分析、数据治理、训练任务和神经因果溯源。使用 SQLite 保存配置、调用记录及任务状态，通过 OpenAI 兼容接口调用上游模型。

## 主要功能

| 功能 | 说明 |
| --- | --- |
| 模型与令牌管理 | 管理模型配置、访问令牌及调用记录，加密保存上游凭据。 |
| 内容分析 | 支持结构化问答、风险判断、价值评分、生成治理和证据审计，覆盖六类业务场景。 |
| 数据治理 | 执行数据价值评估、标签异常检测和风险识别，保存结果及处理建议。 |
| 模型训练 | 支持可信 PyTorch 源码和已注册脚本任务，提供进度、日志、取消及产物下载。 |
| 神经因果溯源 | 基于本地模型和案例材料执行干预分析，输出层级评分及关键组件结果。 |

本项目负责模型服务，业务用户、文件入库等由独立业务后端提供。

## 配置与启动

需要 Python 3.11 和 `uv`，在项目根目录安装依赖：

```shell
uv sync --python 3.11
```

首次使用时，将 `settings.example.json` 复制为 `settings.local.json` 并填写配置；已有配置请勿覆盖。

| 配置项 | 说明 |
| --- | --- |
| `setup_secret` | 至少 32 个字符的随机管理员密钥。 |
| `credential_key` | Fernet 加密密钥，用于保存模型 API Key 和业务库密码；重启、迁移时保持不变。 |
| `database_path` | SQLite 路径，默认 `db/app.sqlite3`，相对配置文件所在目录解析。 |
| `business_api_base_url` | 业务 HTTP 接口根地址，默认 `http://127.0.0.1:8000/api/v1`。 |
| `governance_mock_samples` | 治理任务的空记录模拟开关，默认 `false`。 |

依次生成管理员密钥和 Fernet 密钥，将结果填入配置文件，替换模板占位值：

```shell
uv run python -c "import secrets; print(secrets.token_urlsafe(32))"
uv run python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

启动开发服务：

```shell
uv run fastapi dev main.py --host 127.0.0.1 --port 8001
```

默认读取项目根目录的 `settings.local.json`。可用 `APP_CONFIG_FILE` 指定配置文件；`APP_SETUP_SECRET`、`MODEL_CREDENTIAL_KEY`、`APP_DB_PATH` 可分别覆盖对应字段。私有配置已加入 Git 忽略规则。

## 接口使用

服务默认地址为 `http://127.0.0.1:8001`。通过 [Swagger UI](http://127.0.0.1:8001/docs) 查看和调试接口，或读取 [OpenAPI 定义](http://127.0.0.1:8001/openapi.json)。

使用流程：管理员创建访问令牌 → 配置模型 → 调用或创建任务 → 查询记录和结果。

- 管理操作使用 `X-Setup-Secret`。
- 模型调用及个人任务使用 `Authorization: Bearer <访问令牌>`。
- 模型配置写操作和训练建单需要同时提供以上两个请求头。
- 上游模型的 `api_key` 仅用于访问模型服务，与本服务访问令牌分别配置。

## 后台任务

异步任务需要对应 worker 执行。在独立终端按需启动，每类 worker 运行一个实例：

| 任务 | 启动命令 |
| --- | --- |
| 数据治理 | `uv run python -m service.governance_worker` |
| 模型训练 | `uv run python -m service.training_worker` |
| 神经因果溯源 | `.venv/Scripts/python.exe -m service.neural_trace_worker`（Windows）；`.venv/bin/python -m service.neural_trace_worker`（Linux） |

API 与 worker 使用同一配置、SQLite 数据库和任务文件目录；未启动 worker 时，任务会保持待执行状态。治理任务需配置业务数据库、数据版本及治理模型；训练按任务类型准备数据和运行环境。训练源码仅接受可信代码。

神经因果溯源还需安装 `neurotrace-run/requirements.txt` 中的依赖，准备模型清单指定的本地权重和案例材料，并使用兼容的 GPU 环境。API 启动时不会加载这些权重。

Linux 可在依赖和配置准备完成后使用管理脚本：

```bash
bash start-content-security.sh start                # 启动 API 和治理 worker
bash start-content-security.sh start training       # 按需启动训练 worker
bash start-content-security.sh start neural-trace   # 按需启动溯源 worker
bash start-content-security.sh status
bash start-content-security.sh stop
```

脚本使用 `.venv/bin/python`，日志位于 `logs/`。

## 测试

```shell
uv run python -m pytest
```

## 本地详细文档

以下文档供本地查阅，仅根目录 `README.md` 纳入版本管理；其他机器拉取代码后不会自动获得这些文档。

- [功能说明](项目功能说明.md)
- [接口请求与响应说明](项目对外接口说明.md)
- [Windows 启动说明](启动.md)
- [Linux 服务部署](Linux服务部署.md)
- [训练脚本接入](通用脚本任务接入.md)
- [神经因果溯源接口与部署](神经因果溯源接口与部署.md)
