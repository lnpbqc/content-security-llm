# Content Security LLM

基于 FastAPI 的内容安全与模型服务，提供模型调用、内容分析、数据治理、训练任务和神经因果溯源。使用 MySQL 的 `llm_` 表保存配置、调用记录及任务状态，通过 OpenAI 兼容接口调用上游模型。

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
| `data_dir` | 本地任务文件目录，默认 `db`，相对配置文件所在目录解析。 |
| `mysql_host` / `mysql_port` | 主存储 MySQL 地址和端口，默认 `127.0.0.1:3306`。 |
| `mysql_user` / `mysql_password` | MySQL 账号和密码；密码必须配置。 |
| `mysql_database` | MySQL 库名，默认 `content_safety`；本项目只使用 `llm_` 表。 |
| `business_api_base_url` | 业务 HTTP 接口根地址，默认 `http://127.0.0.1:8000/api/v1`。 |
| `governance_mock_samples` | 治理任务的空记录模拟开关，默认 `false`。 |

依次生成管理员密钥和 Fernet 密钥，将结果填入配置文件，替换模板占位值：

```shell
uv run python -c "import secrets; print(secrets.token_urlsafe(32))"
uv run python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

首次启动前先执行下节的建表命令，再启动开发服务：

```shell
uv run fastapi dev main.py --host 127.0.0.1 --port 8001
```

默认读取项目根目录的 `settings.local.json`。可用 `APP_CONFIG_FILE` 指定配置文件；`APP_SETUP_SECRET`、`MODEL_CREDENTIAL_KEY`、`APP_DATA_DIR` 可覆盖对应字段；`MYSQL_HOST`、`MYSQL_PORT`、`MYSQL_USER`、`MYSQL_PASSWORD`、`MYSQL_DATABASE` 优先于 JSON 中对应的 `mysql_*` 配置。旧 `database_path` / `APP_DB_PATH` 仅在未配置 `data_dir` / `APP_DATA_DIR` 时用于推导历史任务目录，不再读写 SQLite。私有配置已加入 Git 忽略规则。

## 初始化 MySQL 表

当前按全新开发库使用，不需要导入旧数据。先准备好 `content_safety` 数据库及连接账号，在项目根目录执行：

```shell
uv run python -m scripts.init_mysql
```

建表 SQL 位于 [db/mysql_schema.sql](db/mysql_schema.sql)，也可通过 MySQL 客户端在目标库中执行该文件。脚本复用这份 SQL，按依赖顺序创建 14 张 `llm_` 表并检查结构；使用 MySQL 8.4、InnoDB 和 utf8mb4，包含主键、唯一约束、外键、状态约束及任务查询索引。重复执行不会清空已有数据；已有不兼容的表会报错，不自动改表。

建表完成后启动 API 和所需 worker。API 和 worker 启动只检查结构，不自动建表。建表账号需有 CREATE 权限，运行账号需有 SELECT/INSERT/UPDATE/DELETE 权限。本项目使用 `llm_` 前缀，与业务后端已有的无前缀表分开。

本机 Docker MySQL 地址为 `127.0.0.1:13306`，服务器按实际地址配置（常见端口 3306）。治理数据集的业务库连接仍由管理端独立配置。

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

API 与 worker 使用同一配置、MySQL 数据库和任务文件目录；未启动 worker 时，任务会保持待执行状态。治理任务需配置业务数据库、数据版本及治理模型；训练按任务类型准备数据和运行环境。训练源码仅接受可信代码。

神经因果溯源还需安装 `neurotrace-run/requirements.txt` 中的依赖，将 Qwen3-4B 权重放在 `neurotrace-run/models/Qwen3-4B`，保留 `neurotrace-run/data/manual_review.csv` 及对应 `data/upstream/<case_id>/E1.json`、`E2.json`，GPU 需支持 CUDA 和 BF16。API 启动时不会加载这些权重，溯源 worker 领取任务后才加载。

主项目依赖同步完成后，在项目根目录安装附加依赖并检查（下面是 Linux 命令；Windows 将 Python 路径替换为 `.venv/Scripts/python.exe`）：

```bash
uv pip install --python .venv/bin/python -r neurotrace-run/requirements.txt
.venv/bin/python -m service.neural_trace_worker --check
```

`--check` 只确认模型文件和依赖存在，不验证真实 GPU 运行。附加依赖不在主项目锁文件中，安装后使用虚拟环境 Python 或下方 Linux 脚本启动 API 和各类 worker；后续执行 `uv sync` 后需重新安装附加依赖，避免 `uv run` 自动同步时移除它们。

溯源 worker 默认拒绝已有 GPU 计算进程；Windows WDDM 桌面程序触发该检查时，可先退出旧 worker，再在上述溯源启动命令末尾加 `--allow-busy-gpu`。该选项会传给三个扫描阶段，不保证共享 GPU 的显存足够。HTTP 接口与管理端调用不变，失败任务需要重新提交，详见 [溯源接口与部署说明](神经因果溯源接口与部署.md)。

Linux 可在依赖和配置准备完成后使用管理脚本：

```bash
bash start-content-security.sh start                # 启动 API 和治理 worker
bash start-content-security.sh start training       # 按需启动训练 worker
bash start-content-security.sh start neural-trace   # 按需启动溯源 worker
bash start-content-security.sh status
bash start-content-security.sh log neural-trace     # 跟随溯源日志，Ctrl+C 退出查看
bash start-content-security.sh restart neural-trace # 只重启溯源 worker
bash start-content-security.sh stop
```

脚本使用 `.venv/bin/python`，日志位于 `logs/`。默认 `start` 只启动 API 和治理，停止全部服务后需按需再次启动训练和溯源。

完整用法：`bash start-content-security.sh {start|stop|restart|status|log|fg} [api|governance|training|neural-trace]`。每条命令最多指定一个服务；省略服务时，`stop/status` 面向全部服务，`restart` 重启当前已运行的服务（没有服务运行则启动 API 和治理），`log/fg` 选择 API。`fg` 用于前台调试，需先停止后台同类服务。脚本不透传 `--allow-busy-gpu` 等 worker 选项，需直接使用虚拟环境 Python 启动。Linux 完整部署步骤见 [Linux 服务部署](Linux服务部署.md)。

## 测试

测试必须使用独立库，默认 `content_security_llm_test`，仅清理其中的 `llm_` 表。先由数据库管理员创建该库并授予测试账号权限：

```sql
CREATE DATABASE content_security_llm_test CHARACTER SET utf8mb4 COLLATE utf8mb4_bin;
GRANT ALL PRIVILEGES ON content_security_llm_test.* TO 'your_test_user'@'your_test_host';
```

测试默认从私有配置读取连接信息，可通过 `TEST_MYSQL_HOST`、`TEST_MYSQL_PORT`、`TEST_MYSQL_USER`、`TEST_MYSQL_PASSWORD` 覆盖。`TEST_MYSQL_DATABASE` 必须以 `_test` 结尾且与主库不同。测试子进程继承相同的测试连接，不使用真实服务库。

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
