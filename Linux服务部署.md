# Linux 服务部署

适用于 Bitahub 等 Linux 容器，按全新 MySQL 开发库部署，API 端口为 **8001**。本文统一说明环境准备、配置、建表、训练与神经因果溯源准备、服务启停和验收。以下命令在项目根目录的 Bash 中执行。

## 1. 准备环境

上传项目源码、`pyproject.toml`、`uv.lock` 和 `start-content-security.sh`。保留 `db/` 中的 Python 代码及 `mysql_schema.sql`；启用溯源时，还需保留 `neurotrace-run/` 代码、`requirements.txt`、`data/manual_review.csv` 和对应 `data/upstream/<case_id>/E1.json`、`E2.json`。本机虚拟环境、缓存和历史任务文件无需上传。

环境需要 MySQL 8.4、uv，以及 util-linux 提供的 `flock`、`setsid`。未安装 uv 时执行：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

安装 Linux 依赖并确认启动工具可用：

```bash
command -v flock setsid
uv sync --frozen --no-dev --python 3.11
```

## 2. 配置服务

首次部署创建私有配置；已有配置直接编辑：

```bash
test -f settings.local.json || cp settings.example.json settings.local.json
```

编辑 `settings.local.json`：

| 配置项 | 填写内容 |
| --- | --- |
| `mysql_host`、`mysql_port` | 应用可访问的 MySQL 地址，服务器端口通常为 `3306` |
| `mysql_user`、`mysql_password` | 服务器账号 `content_user` 和实际密码 |
| `mysql_database` | `content_safety`，需提前创建并授予本项目表的建表及读写权限 |
| `setup_secret` | 至少 32 字符的随机管理员密钥 |
| `credential_key` | 有效的 Fernet 密钥 |
| `data_dir` | `db`，保存任务文件、训练产物和锁文件 |
| `business_api_base_url` | 应用可访问的业务后端地址，包含 `/api/v1` |
| `governance_mock_samples` | `false`，使用真实业务数据 |

首次生成两项密钥，分别填入 `setup_secret`、`credential_key`；后续保持配置中的密钥稳定：

```bash
.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))'
.venv/bin/python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'
```

`MYSQL_HOST`、`MYSQL_PORT`、`MYSQL_USER`、`MYSQL_PASSWORD`、`MYSQL_DATABASE` 环境变量优先于文件配置。容器内的 `127.0.0.1` 指容器自身；数据库和业务后端位于其他主机或容器时，填写实际可访问地址。应用使用 TCP 连接 MySQL。

默认读取项目根目录的 `settings.local.json`，可通过 `APP_CONFIG_FILE` 指定其他配置文件。`APP_SETUP_SECRET`、`MODEL_CREDENTIAL_KEY`、`APP_DATA_DIR` 可分别覆盖管理员密钥、加密密钥和任务目录。相对 `data_dir` 按配置文件所在目录解析。API 和全部 worker 必须使用相同配置、密钥、MySQL 连接和任务目录；建议部署在同一台服务器，共享本地任务文件。

## 3. 初始化表

```bash
.venv/bin/python -m scripts.init_mysql
```

脚本执行 `db/mysql_schema.sql`，创建并检查 14 张 `llm_` 表。重复执行不会清空数据，结构不兼容时明确报错。API 和 worker 启动只检查结构，因此先完成建表。

## 4. 按需准备训练和神经因果溯源

只使用 API 和治理时，直接进入第 5 节。

### 4.1 训练环境和数据

训练 worker 同时处理可信 PyTorch 源码任务和服务器已登记的脚本任务。源码训练使用的业务数据库、数据集和版本在管理端单独配置；主存储 MySQL 连接不替代业务数据源配置。脚本任务按自身输入约定准备数据，不统一要求业务数据库。

使用 GPU 训练时检查 PyTorch 和 CUDA：

```bash
nvidia-smi
.venv/bin/python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__); print("CUDA:", torch.cuda.is_available())'
```

项目通过锁文件安装 CUDA 12.8 索引的 PyTorch 构建，实际驱动和显卡仍需在目标服务器验证。默认使用 CUDA 的任务在 CUDA 不可用时失败；支持 `device` 参数的脚本可显式指定 `cpu` 调试。

内置脚本 `cifar100-cnn` 默认运行 3 轮，batch size 为 128，学习率为 0.001，训练 10000 条、验证 2000 条。首次运行下载官方 CIFAR-100，需要网络、磁盘空间以及可写任务目录。离线部署时，先将 torchvision 校验并解压的数据目录放入 `<data_dir>/training-data/cifar100-cnn/`，建单时设置 `inputs.download=false`；缺少数据会失败，不使用随机数据替代。

自定义脚本需在 `service/training_scripts.py` 的 `SCRIPTS` 登记入口、输入和参数模型，部署入口依赖的模块及库，并重启 API。任务只快照入口文件，不复制整个脚本项目。源码及登记脚本均以服务账户权限执行，只接入可信代码。

### 4.2 溯源依赖、案例和模型下载

神经因果溯源固定使用本地 Qwen3-4B，GPU 需要支持 CUDA 和 BF16。保留以下案例材料：

```text
neurotrace-run/data/manual_review.csv
neurotrace-run/data/upstream/<case_id>/E1.json
neurotrace-run/data/upstream/<case_id>/E2.json
```

当前支持清单登记的五个人工审核案例，输入已包含上游检索文档，不需另行下载 Embedding 模型。安装附加依赖：

```bash
uv pip install --python .venv/bin/python -r neurotrace-run/requirements.txt
```

下载 Qwen3-4B 到固定目录；已有完整模型时跳过。这一步只下载，不加载到 GPU：

```bash
mkdir -p neurotrace-run/models/Qwen3-4B
export HF_HOME="$PWD/neurotrace-run/.cache/huggingface"
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-4B",
    local_dir="neurotrace-run/models/Qwen3-4B",
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
)
PY
```

默认下载模型仓库当时的主分支版本；需固定版本时，在 `snapshot_download` 中增加 `revision="完整提交 SHA"` 并记录。模型目录和下载缓存应位于持久化挂载中。任务执行使用离线模式，不会自动补下载缺失文件。

检查模型文件和附加依赖：

```bash
.venv/bin/python -m service.neural_trace_worker --check
```

`--check` 不加载模型、不领取任务，只检查配置、tokenizer、权重分片是否存在且非空，以及关键库是否安装；不验证权重内容或真实 GPU 兼容性。

### 4.3 首次 GPU 加载验证

确认 GPU 空闲后执行：

```bash
nvidia-smi
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python neurotrace-run/core/model_runtime.py
```

这一步会真正加载 Qwen3-4B，检查 CUDA 和 BF16，并执行一次简短生成，输出设备、精度、层数和答案。进程退出后释放显存，不创建溯源任务。依赖安装成功不能代替此检查；加载失败时先解决目标服务器的驱动、显存或库兼容问题。

附加依赖不在主项目锁文件中。安装后使用 `.venv/bin/python` 或下方管理脚本启动服务；执行 `uv sync` 后需重新安装附加依赖，`uv run` 自动同步也可能移除它们。

## 5. 启动服务

默认启动 API 和治理 worker：

```bash
bash start-content-security.sh start
```

按需启用训练：

```bash
bash start-content-security.sh start training
```

完成第 4 节的溯源准备后，启动并检查 GPU worker：

```bash
bash start-content-security.sh start neural-trace
bash start-content-security.sh status neural-trace
bash start-content-security.sh log neural-trace
```

`log` 持续跟随日志，`Ctrl+C` 退出查看，后台服务继续运行。API 启动和溯源 worker 空队列等待时都不会加载本地模型，领取任务后才启动实验子进程。

- 默认 `start` 不启动训练或溯源；每次停止全部服务后，需按需再次单独启动。
- 只需要 API 和溯源时，先执行 `bash start-content-security.sh start api`，再单独启动 `neural-trace`。
- 治理还需在管理端配置业务数据库、数据版本及治理模型；未启动对应 worker 时，任务保持 `pending`。
- 溯源默认拒绝已有 GPU 计算进程，忙碌时将领取的任务记为失败；训练和溯源应错开运行。

脚本使用 `.venv/bin/python`，API 监听 `0.0.0.0:8001`，日志与 PID 保存在 `logs/`。API 和全部 worker 使用同一配置及任务文件目录，每类 worker 运行一个实例。

## 6. 检查与访问

```bash
bash start-content-security.sh status
curl --fail http://127.0.0.1:8001/
bash start-content-security.sh log
```

根路径应返回 `{"message":"Content Security LLM"}`。该检查确认 API 可访问；继续通过实际请求检查业务数据源、上游模型和任务执行。

在平台开放容器端口 **8001**，接口文档地址为 `<服务地址>/docs`。前端 `/llm-api` 代理指向本服务并重写为 `/api`，例如 `/llm-api/v1/models` 对应本服务 `/api/v1/models`。

### 6.1 配置访问凭据与业务依赖

在 `/docs` 的管理接口使用 `X-Setup-Secret: <setup_secret>` 创建本模型服务的访问令牌。任务和模型调用使用 `Authorization: Bearer <访问令牌>`；模型配置写操作和训练建单需同时携带管理员密钥。业务登录令牌、管理员密钥和上游模型 API Key 不能替代本服务访问令牌。

在管理端添加上游模型、绑定治理模型，并填写业务数据库连接。根路径检查通过只表示 HTTP 服务可访问，仍需通过实际任务确认上游模型、业务数据源和 worker 能运行。

### 6.2 溯源任务验收

准备好本服务签发的访问令牌，在 Bash 中输入；以下请求会创建并实际执行一次 GPU 溯源：

```bash
read -rsp '模型服务访问令牌: ' LLM_TOKEN
echo

# 列出可运行案例
curl --fail -H "Authorization: Bearer $LLM_TOKEN" \
  http://127.0.0.1:8001/api/v1/neural-trace/cases

# 提交案例并取得任务 ID
TRACE_TASK_ID=$(curl --fail -sS \
  -H "Authorization: Bearer $LLM_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"case_id":"CASE_HIS_028_V01"}' \
  http://127.0.0.1:8001/api/v1/neural-trace/tasks \
  | .venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["data"]["task_id"])')

# 重复查询，直到 data.status 为 succeeded 或 failed
curl --fail -H "Authorization: Bearer $LLM_TOKEN" \
  "http://127.0.0.1:8001/api/v1/neural-trace/tasks/$TRACE_TASK_ID"

# 仅在 succeeded 后获取结果
curl --fail -H "Authorization: Bearer $LLM_TOKEN" \
  "http://127.0.0.1:8001/api/v1/neural-trace/tasks/$TRACE_TASK_ID/result"
```

预期任务依次完成文档干预、输入对齐、层扫描、激活差异和组件扫描，结果包含 36 个层扫描点、36 个激活差异点和 Top-3 层的 9 条组件对比记录。具体分数和答案以本次运行结果为准。未成功的任务取结果返回 409；失败时查看任务详情的 `stage`、`error_message`，以及 `<data_dir>/neural-trace-runs/<task_id>/` 中对应阶段的 `.log`。排除问题后重新提交会生成新任务 ID。

### 6.3 训练任务验收

启动训练 worker 后，在 `/docs` 调用 `GET /api/v1/training/scripts` 确认登记项及参数 Schema，再通过 `POST /api/v1/training/script-tasks` 提交内置示例，请求体为：

```json
{
  "name": "Linux CIFAR-100 验收",
  "script_id": "cifar100-cnn",
  "inputs": {"download": true},
  "parameters": {
    "epochs": 3,
    "batch_size": 128,
    "learning_rate": 0.001,
    "seed": 42,
    "device": "cuda",
    "train_samples": 10000,
    "validation_samples": 2000
  }
}
```

离线环境将 `download` 改为 `false`。建单需同时提供 `Authorization: Bearer <访问令牌>` 和 `X-Setup-Secret`；后续查询沿用建单令牌。通过 `GET /api/v1/training/tasks/{id}` 查询到 `succeeded`，检查 `/tasks/{id}/logs`、`/tasks/{id}/metrics` 和 `/tasks/{id}/artifacts`（均位于 `/api/v1/training` 下），确认有真实训练日志、指标、`best.pt`、`last.pt` 和环境记录。短训练准确率较低不能直接判定部署失败。

## 7. 日常操作

脚本名为 `start-content-security.sh`，统一用 Bash 调用，无需先执行 `chmod +x`：

```text
bash start-content-security.sh {start|stop|restart|status|log|fg} [api|governance|training|neural-trace]
```

每条命令最多指定一个服务。省略服务时：`start` 启动 API 和治理，`stop/status` 面向全部服务，`restart` 重启当前运行的服务，`log/fg` 选择 API。

```bash
bash start-content-security.sh log governance  # 查看治理日志
bash start-content-security.sh log training    # 查看训练日志
bash start-content-security.sh log neural-trace # 查看溯源日志
bash start-content-security.sh stop neural-trace # 只停止溯源 worker
bash start-content-security.sh restart neural-trace # 只重启溯源 worker
bash start-content-security.sh fg neural-trace  # 前台调试；先停止后台同类 worker
bash start-content-security.sh stop            # 停止全部服务
bash start-content-security.sh restart         # 重启当前已运行的服务
```

没有服务运行时，`restart` 默认启动 API 和治理。脚本不透传 `--check`、`--once` 或 `--allow-busy-gpu`。使用溯源 worker 选项时，先停止后台同类 worker，再在前台执行下面其中一条：

```bash
bash start-content-security.sh stop neural-trace

# 最多领取并执行一个任务后退出；空队列时直接退出
.venv/bin/python -m service.neural_trace_worker --once

# 确认允许共享 GPU 后，跳过占用拦截并常驻运行
.venv/bin/python -m service.neural_trace_worker --allow-busy-gpu
```

`--allow-busy-gpu` 同时传给层扫描、激活差异和组件扫描，不保证共享显存足够。手动启动的 worker 不由管理脚本记录 PID，用原终端的 `Ctrl+C` 停止。溯源按同一 `data_dir` 下的文件锁限制单实例，切换启动选项需先停止旧 worker。

溯源 worker 重启会将遗留 `running` 任务标记失败，保留产物，不自动重跑；正常停止会终止当前实验子进程。强制终止后用 `nvidia-smi` 检查是否仍有遗留实验进程。训练 worker 重启也不会自动续训，更新服务前先完成或取消训练任务。

更新源码或依赖前先停服务，训练和溯源任务先完成或取消。执行 `uv sync --frozen --no-dev --python 3.11` 后，启用溯源的环境必须重新安装 `neurotrace-run/requirements.txt` 并运行 `--check`，再按需启动各服务。附加依赖不在主项目锁文件中，日常启动用本脚本或 `.venv/bin/python`，避免 `uv run` 自动同步时移除它们。

项目目录及 `data_dir` 应放在可持久保存、可写且支持文件锁的挂载中。后台运行使用 `nohup`，终端断开后服务继续运行；进程崩溃或容器重启后需重新启动，脚本不提供自动重启。
