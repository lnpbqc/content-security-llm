# curl 接口测试手册

核对日期：2026-10-14。依据当前工作区 `main.py` 挂载路由、请求模型和服务实现编写，包含已有未提交实现。覆盖 **85 个项目接口（HTTP 方法＋路径）和 4 个框架文档入口**。别名分别列出；同一路径的不同方法分别计数。

默认在 **Linux 服务器 Bash** 中执行，服务直接访问地址为 `http://127.0.0.1:8001`。先复制第 1 节设置变量，再按需要复制单个接口代码块。代码块中的变量取自第 1 节或上一条响应；没有隐藏的 shell 函数或额外脚本。所有密钥均为占位符。

阅读顺序：[变量与约定](#variables) → [首次初始化](#first-run) → [每次测试前检查](#preflight) → [数据治理完整链路](#governance) → [其他可选功能](#optional) → [错误与验收](#acceptance)。

## 执行频率速查

| 阶段 | 必要动作 | 是否每次重做 |
| --- | --- | --- |
| 每个新终端 | 设置地址、令牌、管理员密钥等 shell 变量 | 是，变量不会自动恢复 |
| 首次部署 | 完成数据库建表，启动 API 和所需 worker | 建表按部署需要执行；进程停止后才需启动 |
| 首次使用/更换令牌 | 登记自备访问令牌 | 否；有效且已登记的令牌继续复用 |
| 首次使用/新增模型 | 查询上游模型（上游支持时），注册本地模型 | 否；模型配置持久化，不要反复 POST |
| 首次使用/更换治理模型 | 分别绑定价值、异常、风险模型 | 否；缺失或改变时再 PUT，正常重启后保留 |
| 每次测试前 | 查服务、查本地模型、查治理模型绑定、查所测功能方案 | 是，属于检查步骤 |
| 每次治理/源码训练前 | 确认实际数据集 ID、当前版本和真实记录，以及对应 worker 正常 | 是；本服务没有数据集目录/上传接口 |
| 每次功能测试 | 按需要调用模型、创建任务、查询状态和结果 | 按需；创建任务会新增记录并实际消耗模型/GPU 资源 |

“必要”表示本手册的联调前置步骤，不表示服务会强制要求先调用某个 GET 才能 POST。仅测试普通推理时，不需要配置三类治理绑定；测试全部数据治理功能时，三类绑定都必须可用。

<a id="variables"></a>

## 1. 变量与请求约定

服务器需要 `curl`、`jq`、Bash。`jq` 用于安全生成 JSON，密钥中的引号、反斜杠等会正确转义。先检查工具：

```bash
command -v curl jq bash
```

替换下面占位符后，在同一 Bash 会话执行。`${...}` 是 shell 引用，不需要逐条替换。访问令牌须至少 32 字符；管理员密钥使用服务实际 `setup_secret`（或 `APP_SETUP_SECRET`）。上游 API Key、管理员密钥、本服务访问令牌是三个不同用途的凭据。

字符串请用 shell 单引号赋值；值本身含单引号时可用 `read -rs VARIABLE; printf '\n'` 交互输入，避免 shell 转义错误。不要将 `<...>` 原样提交给真实服务。

```bash
BASE_URL='http://127.0.0.1:8001'
SETUP_SECRET='<服务实际管理员密钥>'
ACCESS_TOKEN='<自备且已登记的至少32字符随机访问令牌>'
TOKEN_EXPIRES_AT='2099-01-01T00:00:00+08:00'

# 上游地址含 /v1；必须是 API/worker 所在服务器可访问的地址。
UPSTREAM_BASE_URL='http://<上游主机>:<端口>/v1'
UPSTREAM_API_KEY='<上游API-Key>'
UPSTREAM_MODEL_ID='<上游返回的模型id，例如/data/models/实际模型目录名>'

# 首次注册后，从响应 model_id 填入；之后从模型列表选取。
MODEL_ID='<本服务model_id>'
# 可让三类治理使用同一模型，也可分别填写其他已注册 model_id。
VALUE_MODEL_ID="$MODEL_ID"
ANOMALY_MODEL_ID="$MODEL_ID"
RISK_MODEL_ID="$MODEL_ID"

# 从业务后端确认；DATASET_ID 必须是正整数，此占位符未替换前不能建单。
DATASET_ID='<实际正整数数据集ID>'
VERSION_ID='<数据集当前版本，例如v1.0.0>'
VALUE_SCHEME_ID='general-v1'
ANOMALY_SCHEME_ID='anomaly-basic-v1'
RISK_SCHEME_ID='risk-v1'

# 运行到相应模块后，用响应中的真实值更新；不同任务和结果不可混用。
VALUE_TASK_ID='<价值任务task_id>'
ANOMALY_TASK_ID='<异常任务task_id>'
RISK_TASK_ID='<风险任务id>'
VALUE_RESULT_ID='<价值任务完成后的result_id>'
ANOMALY_RESULT_ID='<异常任务完成后的result_id>'
RISK_RESULT_ID='<风险任务完成后的result_id>'
ANOMALY_SAMPLE_ID='<异常样本items中的id>'
RISK_SAMPLE_ID='<风险样本items中的id>'
RECORD_ID='<独立分析记录id>'
MULTILINGUAL_TASK_ID='<多语种任务task_id>'
MULTILINGUAL_RESULT_ID='<多语种任务完成后的result_id>'
TRAINING_TASK_ID='<训练任务id>'
ARTIFACT_ID='<训练产物列表中的id>'
CASE_ID='<溯源cases列表中的case_id>'
TRACE_TASK_ID='<溯源任务task_id>'
TOKEN_ID='<管理员令牌列表中的id>'
TEST_ACCESS_TOKEN='<专门用于启停测试的已登记32字符以上令牌>'
TEST_TOKEN_ID='<专用测试令牌对应id>'
TEST_MODEL_ID='<专门用于删除测试且未被业务使用的model_id>'
DOWNLOAD_PATH='./training-artifact.bin'
```

后文 `curl -i` 会同时显示 HTTP 状态、响应头和正文，不能直接整体交给 `jq`。需要提取 ID 时，把该条命令的 `-i` 去掉再接 `| jq ...`，或从正文手动复制。例如模型创建返回裸对象，取 `.model_id`；v1 接口取 `.data` 内的字段。不要为取 ID 再创建一次任务。

| 响应类型 | 判定方式 |
| --- | --- |
| `/api/auth`、`/api/models`、`/api/llm`、`/api/calls` | 旧版裸 JSON；无统一 code/data；删除模型是 204 空体 |
| `/api/v1/...` | 通常 `{"code":0,"message":"success","data":...,"trace_id":...,"timestamp":...}` |
| 训练产物下载 | 文件字节流，不能用 JSON 解析 |
| `/`、`/openapi.json`、文档网页 | 根入口 JSON、OpenAPI JSON 或 HTML；不按 v1 包裹 |

没有使用 `curl -f`，目的是让错误测试也能看到正文。因此 curl 退出码为 0 不代表业务通过，应检查 HTTP 状态和 JSON；异步建单返回 200/202 仅代表入队成功。

<a id="first-run"></a>

## 2. 首次初始化：已有有效配置时跳过写操作

前提：已按 [Linux 服务部署](Linux服务部署.md) 准备私有配置、MySQL 和业务表，并完成建表；API 和 worker 使用同一配置、数据库及任务目录。治理和源码训练直接只读查询同一 MySQL 的 `datasets`、`dataset_records`，**没有额外的数据源配置 HTTP 接口**。

以下是首次部署的 shell 前置动作，不计入接口；在项目根目录执行，已有进程不必重复启动：

```bash
.venv/bin/python -m scripts.init_mysql
bash start-content-security.sh start
bash start-content-security.sh status
```

默认 `start` 启动 API 和治理 worker；训练、溯源 worker 在相应可选章节启动。治理 worker 也处理多语种任务。没有 HTTP 接口能可靠证明 worker 存活，必须结合脚本状态和任务状态确认。

### 2.1 登记本服务访问令牌【首次/更换时必要】

`POST /api/auth/token`

预期 HTTP 201，裸 JSON 的 enabled=true，过期时间含时区且在未来。服务登记调用方自备的令牌，不自动生成、不返回令牌原文。重复登记同一令牌返回 409；若此前已登记且有效，直接复用。实际使用时按需要缩短占位过期时间。

```bash
jq -n \
  --arg access_token "${ACCESS_TOKEN}" \
  --arg token_expires_at "${TOKEN_EXPIRES_AT}" \
  '{
  "token": $access_token,
  "expires_at": $token_expires_at,
  "label": "curl-test"
}' | \
curl -sS -i -X POST "${BASE_URL}/api/auth/token" \
  -H "X-Setup-Secret: ${SETUP_SECRET}" \
  -H "Content-Type: application/json" \
  --data-binary @-
```

### 2.2 查询上游模型【首次配置模型时；上游支持则执行】

这是访问上游的 OpenAI 兼容模型目录，不属于本项目接口。本项目的 `/api/v1/models` 只列出已经注册到本地的配置，不会自动发现上游模型。

```bash
curl -sS -i "${UPSTREAM_BASE_URL}/models" \
  -H "Authorization: Bearer ${UPSTREAM_API_KEY}"
```

预期上游 HTTP 200，通常从 `data[].id` 复制完整值到 `UPSTREAM_MODEL_ID`。部分上游不提供该目录，此时使用部署者确认的实际模型 ID。本仓库 README 提到现用 xju 上游采用 `/data/models/模型名`，仍须以服务器实际返回/部署配置为准，不能把本地生成的 `model_id` 当成上游 ID。

### 2.3 注册本地模型【本地模型列表为空/新增时必要】

`POST /api/models`

预期 HTTP 201，裸对象中返回 model_id。此请求仅保存配置，不证明上游可调用；继续做第 3.6 节推理检查。同一模型重复 POST 会创建新的本地配置，已有可用模型时应跳过。

```bash
jq -n \
  --arg upstream_model_id "${UPSTREAM_MODEL_ID}" \
  --arg upstream_base_url "${UPSTREAM_BASE_URL}" \
  --arg upstream_api_key "${UPSTREAM_API_KEY}" \
  '{
  "upstream_model_id": $upstream_model_id,
  "name": "curl测试模型",
  "description": "curl接口联调",
  "base_url": $upstream_base_url,
  "api_key": $upstream_api_key
}' | \
curl -sS -i -X POST "${BASE_URL}/api/models" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "X-Setup-Secret: ${SETUP_SECRET}" \
  -H "Content-Type: application/json" \
  --data-binary @-
```

将响应中的 ID 写入当前会话。若三类治理使用不同模型，分别填写其本地 ID：

```bash
MODEL_ID='<刚返回的model_id>'
VALUE_MODEL_ID="$MODEL_ID"
ANOMALY_MODEL_ID="$MODEL_ID"
RISK_MODEL_ID="$MODEL_ID"
```

### 2.4 绑定价值评估模型【首次/缺失/更换时必要】

`PUT /api/v1/admin/governance/models/{kind}`

预期 HTTP 200、code=0，data.kind=governance-value、data.model_id 为指定本地 ID。三类绑定分别配置；绑定保存在 MySQL，普通重启无需重写。新建任务使用当前绑定，已创建任务保存原有模型选择。

```bash
jq -n \
  --arg value_model_id "${VALUE_MODEL_ID}" \
  '{
  "model_id": $value_model_id
}' | \
curl -sS -i -X PUT "${BASE_URL}/api/v1/admin/governance/models/governance-value" \
  -H "X-Setup-Secret: ${SETUP_SECRET}" \
  -H "Content-Type: application/json" \
  --data-binary @-
```

### 2.4 绑定标签异常检测模型【首次/缺失/更换时必要】

`PUT /api/v1/admin/governance/models/{kind}`

预期 HTTP 200、code=0，data.kind=governance-anomaly、data.model_id 为指定本地 ID。三类绑定分别配置；绑定保存在 MySQL，普通重启无需重写。新建任务使用当前绑定，已创建任务保存原有模型选择。

```bash
jq -n \
  --arg anomaly_model_id "${ANOMALY_MODEL_ID}" \
  '{
  "model_id": $anomaly_model_id
}' | \
curl -sS -i -X PUT "${BASE_URL}/api/v1/admin/governance/models/governance-anomaly" \
  -H "X-Setup-Secret: ${SETUP_SECRET}" \
  -H "Content-Type: application/json" \
  --data-binary @-
```

### 2.4 绑定风险识别模型【首次/缺失/更换时必要】

`PUT /api/v1/admin/governance/models/{kind}`

预期 HTTP 200、code=0，data.kind=governance-risk、data.model_id 为指定本地 ID。三类绑定分别配置；绑定保存在 MySQL，普通重启无需重写。新建任务使用当前绑定，已创建任务保存原有模型选择。

```bash
jq -n \
  --arg risk_model_id "${RISK_MODEL_ID}" \
  '{
  "model_id": $risk_model_id
}' | \
curl -sS -i -X PUT "${BASE_URL}/api/v1/admin/governance/models/governance-risk" \
  -H "X-Setup-Secret: ${SETUP_SECRET}" \
  -H "Content-Type: application/json" \
  --data-binary @-
```

上面三条是同一参数化接口的三个必要取值：`governance-value`、`governance-anomaly`、`governance-risk`。不能用 `value`、`risk` 简写。普通治理建单 JSON 不接受 `model_id`，必须先在管理员接口绑定；多语种任务则在请求中显式传 `model_id`。

<a id="preflight"></a>

## 3. 每次开始测试前：必要检查

首次初始化完成后也应执行本节。仅测试某一功能时，执行与该功能相关的方案查询。检查到缺失配置，再回第 2 节补齐；不必每次登记令牌、注册模型和重写绑定。

### 3.1 服务可达【每次必要】

`GET /`

预期 HTTP 200，正文 {"message":"Content Security LLM"}。仅证明 API 响应，不能证明上游模型、业务数据和 worker 都可用。

```bash
curl -sS -i -X GET "${BASE_URL}/"
```

### 3.2 查询有哪些本地模型【每次必要】

`GET /api/v1/models`

预期 HTTP 200、code=0，data 是模型数组。选取 model_id 更新 MODEL_ID，核对 upstream_model_id、base_url；空数组表示需要首次注册。本请求也验证访问令牌有效；401 时检查登记、启用状态和过期时间。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/models" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### 3.3 确认选中模型详情【每次必要】

`GET /api/v1/models/{model_id}`

预期 HTTP 200、code=0，data.model_id 与所选 ID 一致。若三种治理选了其他模型，也应分别将 URL 的 MODEL_ID 换成对应变量检查其配置。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/models/${MODEL_ID}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### 3.4 查询三类治理模型绑定【治理测试前必要】

`GET /api/v1/admin/governance/models`

预期 HTTP 200、code=0。data 是 kind → model_id 的字典；完整治理测试要求包含 governance-value、governance-anomaly、governance-risk，且 ID 存在于本地模型列表。缺少某个键时回第 2.4 节绑定。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/admin/governance/models" \
  -H "X-Setup-Secret: ${SETUP_SECRET}"
```

绑定示意（不是服务器固定 ID）：`{"governance-value":"<本地ID>","governance-anomaly":"<本地ID>","governance-risk":"<本地ID>"}`。管理员 GET 只返回已配置项；HTTP 200 本身不能证明三项齐全。

### 3.5 查询通用推理支持的响应类型【推理测试前必要】

`GET /api/v1/llm/response-types`

预期 HTTP 200、code=0，data 是字符串数组。下节使用实际支持的 text_answer；其他 response_type 应从此列表选择。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/llm/response-types" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### 3.6 普通推理连通性检查【本次含模型调用时必要】

`POST /api/v1/llm/invoke`

预期 HTTP 200、code=0，data.result.answer 有实际回答。此请求会真实调用上游、写调用记录；用于确认配置和结构化输出链路可用，不能替代逐类治理输出验证。服务上游超时为 120 秒，可能返回 502/504。

```bash
jq -n \
  --arg model_id "${MODEL_ID}" \
  '{
  "model_id": $model_id,
  "input": "请用一句中文回答：2加3等于多少？",
  "response_type": "text_answer"
}' | \
curl -sS -i -X POST "${BASE_URL}/api/v1/llm/invoke" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  --data-binary @-
```

### 3.7 查询价值/异常治理方案【对应治理测试前必要】

`GET /api/v1/data-governance/options`

预期 HTTP 200、code=0，data.schemes 包含 general-v1/general-v2。把所选 id 写入 VALUE_SCHEME_ID。此接口不返回实际业务数据集目录。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/data-governance/options" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "kind=governance-value"
```

异常治理使用同一接口，但 kind 不同：

```bash
curl -sS -i --get "${BASE_URL}/api/v1/data-governance/options" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --data-urlencode 'kind=governance-anomaly'
```

预期 data.schemes 包含 `anomaly-basic-v1`，将 id 用作 ANOMALY_SCHEME_ID；当前模型适用规则为标签异常，不要把选项中未开放的业务修复当作已实现功能。

### 3.8 查询风险治理方案【风险治理测试前必要】

`GET /api/v1/data-governance/risk-options`

预期 HTTP 200、code=0，data.schemes 包含 risk-v1，写入 RISK_SCHEME_ID。datasets=[] 是当前设计；default_scope 的 dataset_id=0/version_id="" 只是空占位，不能原样建单。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/data-governance/risk-options" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### 3.9 确认数据集与 worker【治理/训练前必要；不是 HTTP 接口】

本服务没有上传文件、查询业务数据集目录或发布数据版本的接口。通过业务后端/业务库确认 DATASET_ID 存在、VERSION_ID 等于 `datasets.version`，`dataset_records` 有该数据集的真实 payload；不虚构业务后端 URL。worker 执行时才校验数据集版本和读取记录，因此建单 200 也可能随后 failed。

普通治理的 language 字段必须提交，但当前不会过滤业务记录，示例统一用 `all`；查最新结果时，dataset_id/version_id/language/scheme_id 四项必须与建单完全一致。多语种功能使用 languages 数组，确实按语种过滤。正式验收保持 `governance_mock_samples=false`；结果若出现 data_source=mock，不计为真实数据验收通过。

```bash
bash start-content-security.sh status
```

<a id="governance"></a>

## 4. 数据治理：按需测试，选定类型后完成建单→状态→结果链路

此节是功能测试，不要求每次启动都创建任务。请先选小型真实数据集，避免对大数据集反复建单。一个任务可能逐样本调用模型。

治理任务状态：pending → running → succeeded/failed。轮询下面的 GET 即可，不要反复 POST。succeeded 后取 result_id，填写对应 RESULT_ID；failed 查看 error_message，pending 长时间不变检查治理 worker。部分样本失败时仍可能产生结果，验收需检查汇总中的 valid_count、unavailable_count、failed_count 及实际样本，不能只看 succeeded。

### 4.value.1 创建价值评估任务

`POST /api/v1/tasks`

预期 HTTP 200、code=0，状态初始 pending。取 data.task_id 填入 VALUE_TASK_ID；这是任务 ID，不是结果 ID。建单不立即执行模型。

```bash
jq -n \
  --argjson dataset_id "${DATASET_ID}" \
  --arg version_id "${VERSION_ID}" \
  --arg value_scheme_id "${VALUE_SCHEME_ID}" \
  '{
  "kind": "governance-value",
  "name": "curl价值评估测试",
  "input": {
    "dataset_id": $dataset_id,
    "version_id": $version_id,
    "language": "all",
    "scheme_id": $value_scheme_id
  }
}' | \
curl -sS -i -X POST "${BASE_URL}/api/v1/tasks" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  --data-binary @-
```

### 4.value.2 查询价值任务列表

`GET /api/v1/tasks`

预期 HTTP 200、code=0，data.items 为当前令牌和完整范围内的任务；data.total/page/total_pages 为分页信息。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/tasks" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "kind=governance-value" \
  --data-urlencode "dataset_id=${DATASET_ID}" \
  --data-urlencode "version_id=${VERSION_ID}" \
  --data-urlencode "language=all" \
  --data-urlencode "scheme_id=${VALUE_SCHEME_ID}" \
  --data-urlencode "page=1" \
  --data-urlencode "page_size=10"
```

### 4.value.3 轮询价值评估任务

`GET /api/v1/tasks/{task_id}`

预期 HTTP 200、code=0。确认 data.status=succeeded，取 data.result_id 填入 VALUE_RESULT_ID。未完成继续查询此接口，失败读取 error_message。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/tasks/${VALUE_TASK_ID}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "kind=governance-value"
```

### 4.value.4 查询最新价值评估结果

`GET /api/v1/data-governance/value-results/latest`

预期 HTTP 200、code=0，data 是汇总；没有该令牌和完整范围匹配的已完成结果时 data=null，属于正常空结果，不代表任务已通过。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/data-governance/value-results/latest" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "dataset_id=${DATASET_ID}" \
  --data-urlencode "version_id=${VERSION_ID}" \
  --data-urlencode "language=all" \
  --data-urlencode "scheme_id=${VALUE_SCHEME_ID}"
```

### 4.value.6 查询指定价值评估结果

`GET /api/v1/data-governance/value-results/{result_id}`

预期 HTTP 200、code=0，data 为该结果汇总。ID 不存在、类型不符或换了访问令牌时返回 404。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/data-governance/value-results/${VALUE_RESULT_ID}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### 4.value.7 查询价值评估样本分页

`GET /api/v1/data-governance/value-results/{result_id}/samples`

预期 HTTP 200、code=0，data.items/total/total_pages 返回分页。样本 ID 取 items[].id。价值 tier 支持 all/high/medium/low/unavailable；bin 可为 0～4；sort_by 支持 score/tier，必须与 sort_order=asc/desc 成对。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/data-governance/value-results/${VALUE_RESULT_ID}/samples" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "page=1" \
  --data-urlencode "page_size=10" \
  --data-urlencode "tier=all" \
  --data-urlencode "sort_by=score" \
  --data-urlencode "sort_order=desc"
```

### 4.anomaly.1 创建异常检测任务

`POST /api/v1/data-governance/anomaly-tasks`

预期 HTTP 200、code=0，状态初始 pending。取 data.task_id 填入 ANOMALY_TASK_ID；这是任务 ID，不是结果 ID。建单不立即执行模型。

```bash
jq -n \
  --argjson dataset_id "${DATASET_ID}" \
  --arg version_id "${VERSION_ID}" \
  --arg anomaly_scheme_id "${ANOMALY_SCHEME_ID}" \
  '{
  "kind": "governance-anomaly",
  "name": "curl异常检测测试",
  "input": {
    "dataset_id": $dataset_id,
    "version_id": $version_id,
    "language": "all",
    "scheme_id": $anomaly_scheme_id
  }
}' | \
curl -sS -i -X POST "${BASE_URL}/api/v1/data-governance/anomaly-tasks" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  --data-binary @-
```

### 4.anomaly.3 轮询异常检测任务

`GET /api/v1/data-governance/anomaly-tasks/{task_id}`

预期 HTTP 200、code=0。确认 data.status=succeeded，取 data.result_id 填入 ANOMALY_RESULT_ID。未完成继续查询此接口，失败读取 error_message。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/data-governance/anomaly-tasks/${ANOMALY_TASK_ID}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "kind=governance-anomaly"
```

### 4.anomaly.4 查询最新异常检测结果

`GET /api/v1/data-governance/anomaly-results/latest`

预期 HTTP 200、code=0，data 是汇总；没有该令牌和完整范围匹配的已完成结果时 data=null，属于正常空结果，不代表任务已通过。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/data-governance/anomaly-results/latest" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "dataset_id=${DATASET_ID}" \
  --data-urlencode "version_id=${VERSION_ID}" \
  --data-urlencode "language=all" \
  --data-urlencode "scheme_id=${ANOMALY_SCHEME_ID}"
```

### 4.anomaly.5 查询异常检测结果历史

`GET /api/v1/data-governance/anomaly-results`

预期 HTTP 200、code=0，data 是结果汇总数组（非分页对象）；没有结果为 []。价值分析没有单独的结果历史接口，可查价值任务列表。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/data-governance/anomaly-results" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "dataset_id=${DATASET_ID}" \
  --data-urlencode "version_id=${VERSION_ID}" \
  --data-urlencode "language=all" \
  --data-urlencode "scheme_id=${ANOMALY_SCHEME_ID}"
```

### 4.anomaly.6 查询指定异常检测结果

`GET /api/v1/data-governance/anomaly-results/{result_id}`

预期 HTTP 200、code=0，data 为该结果汇总。ID 不存在、类型不符或换了访问令牌时返回 404。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/data-governance/anomaly-results/${ANOMALY_RESULT_ID}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### 4.anomaly.7 查询异常检测样本分页

`GET /api/v1/data-governance/anomaly-results/{result_id}/samples`

预期 HTTP 200、code=0，data.items/total/total_pages 返回分页。样本 ID 取 items[].id。可用 keyword 查询；其他筛选值从实际结果/方案选取。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/data-governance/anomaly-results/${ANOMALY_RESULT_ID}/samples" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "page=1" \
  --data-urlencode "page_size=10"
```

### 4.anomaly.8 查询单条异常检测样本

`GET /api/v1/data-governance/anomaly-results/{result_id}/samples/{sample_id}`

预期 HTTP 200、code=0，data 为实际样本详情。先从上一条 items[].id 填写样本变量，不能使用列表序号代替。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/data-governance/anomaly-results/${ANOMALY_RESULT_ID}/samples/${ANOMALY_SAMPLE_ID}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### 4.risk.1 创建风险识别任务

`POST /api/v1/data-governance/risk-tasks`

预期 HTTP 200、code=0，状态初始 pending。取 data.id 填入 RISK_TASK_ID；这是任务 ID，不是结果 ID。建单不立即执行模型。

```bash
jq -n \
  --argjson dataset_id "${DATASET_ID}" \
  --arg version_id "${VERSION_ID}" \
  --arg risk_scheme_id "${RISK_SCHEME_ID}" \
  '{
  "kind": "governance-risk",
  "name": "curl风险识别测试",
  "input": {
    "dataset_id": $dataset_id,
    "version_id": $version_id,
    "language": "all",
    "scheme_id": $risk_scheme_id
  }
}' | \
curl -sS -i -X POST "${BASE_URL}/api/v1/data-governance/risk-tasks" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  --data-binary @-
```

### 4.risk.3 轮询风险识别任务

`GET /api/v1/data-governance/risk-tasks/{task_id}`

预期 HTTP 200、code=0。确认 data.status=succeeded，取 data.result_id 填入 RISK_RESULT_ID。未完成继续查询此接口，失败读取 error_message。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/data-governance/risk-tasks/${RISK_TASK_ID}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "kind=governance-risk"
```

### 4.risk.4 查询最新风险识别结果

`GET /api/v1/data-governance/risk-results/latest`

预期 HTTP 200、code=0，data 是汇总；没有该令牌和完整范围匹配的已完成结果时 data=null，属于正常空结果，不代表任务已通过。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/data-governance/risk-results/latest" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "dataset_id=${DATASET_ID}" \
  --data-urlencode "version_id=${VERSION_ID}" \
  --data-urlencode "language=all" \
  --data-urlencode "scheme_id=${RISK_SCHEME_ID}"
```

### 4.risk.5 查询风险识别结果历史

`GET /api/v1/data-governance/risk-results`

预期 HTTP 200、code=0，data 是结果汇总数组（非分页对象）；没有结果为 []。价值分析没有单独的结果历史接口，可查价值任务列表。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/data-governance/risk-results" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "dataset_id=${DATASET_ID}" \
  --data-urlencode "version_id=${VERSION_ID}" \
  --data-urlencode "language=all" \
  --data-urlencode "scheme_id=${RISK_SCHEME_ID}"
```

### 4.risk.6 查询指定风险识别结果

`GET /api/v1/data-governance/risk-results/{result_id}`

预期 HTTP 200、code=0，data 为该结果汇总。ID 不存在、类型不符或换了访问令牌时返回 404。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/data-governance/risk-results/${RISK_RESULT_ID}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### 4.risk.7 查询风险识别样本分页

`GET /api/v1/data-governance/risk-results/{result_id}/samples`

预期 HTTP 200、code=0，data.items/total/total_pages 返回分页。样本 ID 取 items[].id。可用 keyword 查询；其他筛选值从实际结果/方案选取。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/data-governance/risk-results/${RISK_RESULT_ID}/samples" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "page=1" \
  --data-urlencode "page_size=10" \
  --data-urlencode "sort_by=maximum_suggested_level" \
  --data-urlencode "sort_order=desc"
```

### 4.risk.8 查询单条风险识别样本

`GET /api/v1/data-governance/risk-results/{result_id}/samples/{sample_id}`

预期 HTTP 200、code=0，data 为实际样本详情。先从上一条 items[].id 填写样本变量，不能使用列表序号代替。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/data-governance/risk-results/${RISK_RESULT_ID}/samples/${RISK_SAMPLE_ID}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

## 5. 治理统计、规则与导出【可选】

先完成对应治理任务。统计只覆盖当前令牌的结果；同一数据集版本取最近一次已保存结果，不累加重复检测。

### 5.1 价值 KPI

`GET /api/v1/kpis`

预期 HTTP 200、code=0，data 为 KPI 数组；无结果时数值可为 0。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/kpis" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "kind=governance-value"
```

### 5.2 异常概览

`GET /api/v1/data-governance/anomaly-results/overview`

预期 HTTP 200、code=0，data.cards/definitions。复核、修复数量为 0，因为本服务只提供检测建议。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/data-governance/anomaly-results/overview" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "kind=governance-anomaly"
```

### 5.3 异常概览兼容别名

`GET /api/v1/overview`

预期同 5.2；这是同功能的另一路径。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/overview" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "kind=governance-anomaly"
```

### 5.4 风险概览

`GET /api/v1/data-governance/risk-overview`

预期 HTTP 200、code=0，data.cards/definitions/records；建议复核数不代表真实人工复核工单数。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/data-governance/risk-overview" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### 5.5 导出风险样本为 JSON

`POST /api/v1/data-governance/risk-results/{result_id}/export`

预期 HTTP 200、code=0，data 是全部匹配样本数组，不受分页限制；返回 JSON，不是 CSV/Excel 下载。空筛选导出全部保存样本，不再次调用模型。

```bash
jq -n \
  '{
  "keyword": "",
  "level": "",
  "status": ""
}' | \
curl -sS -i -X POST "${BASE_URL}/api/v1/data-governance/risk-results/${RISK_RESULT_ID}/export" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  --data-binary @-
```

### 5.6 检索指定风险样本的规则

`GET /api/v1/risk-knowledge`

预期 HTTP 200、code=0，data 是样本已保存规则的匹配数组，不发起新的模型分析；无匹配可以是 []。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/risk-knowledge" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "result_id=${RISK_RESULT_ID}" \
  --data-urlencode "sample_id=${RISK_SAMPLE_ID}" \
  --data-urlencode "keyword="
```

### 5.7 只读当前修改集

`GET /api/v1/data-governance/change-sets/current`

预期 HTTP 200、code=0、data=null；本服务没有业务修改集，此结果是设计行为。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/data-governance/change-sets/current" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "dataset_id=${DATASET_ID}" \
  --data-urlencode "version_id=${VERSION_ID}" \
  --data-urlencode "language=all" \
  --data-urlencode "scheme_id=${ANOMALY_SCHEME_ID}"
```

<a id="optional"></a>

## 6. 独立场景分析、生成治理与审计【可选】

本组是同步分析接口，无需治理 worker，也不依赖三类治理绑定；请求显式提供 MODEL_ID，使用当前令牌隔离分析记录。先查 options，再选择类型。每次 POST 都会实际调用上游模型。

### 6.1 查询分析类型与场景

`GET /api/v1/analysis/options`

预期 HTTP 200、code=0，data.analysis_types/audit_types/scenarios 返回支持值和输入要求。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/analysis/options" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### 6.2 场景分析示例：舆情

`POST /api/v1/analysis/scenarios`

预期 HTTP 200、code=0、data.status=succeeded，保存 data.id 为 RECORD_ID。核对结果引用、结论与提交材料一致。只有本例这一个接口，不同 scenario_code 使用不同场景契约。

```bash
jq -n \
  --arg model_id "${MODEL_ID}" \
  '{
  "model_id": $model_id,
  "scenario_code": "public_opinion",
  "content": "社区发布活动公告，部分居民希望增加线上报名方式，请分析主要诉求。",
  "language": "zh",
  "materials": [
    {
      "id": "m1",
      "title": "活动反馈",
      "content": "居民反馈：希望增加线上报名方式。"
    }
  ]
}' | \
curl -sS -i -X POST "${BASE_URL}/api/v1/analysis/scenarios" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  --data-binary @-
```

同一接口还支持 `hot_events`、`cross_cultural`、`ethnic_governance`、`smart_government`、`cyber_security`，替换 scenario_code 并提供相应材料即可；不要沿用无关测试正文。只有 cross_cultural 可加 target_language（如 en）。

### 6.3 模型生成治理

`POST /api/v1/analysis/model-governance`

预期 HTTP 200、code=0、data.status=succeeded；保存 data.id 为 RECORD_ID。该同步功能不等同于数据集治理任务，也不需要管理员绑定。

```bash
jq -n \
  --arg model_id "${MODEL_ID}" \
  '{
  "model_id": $model_id,
  "prompt": "请用客观、中性的语言介绍跨文化交流时如何避免误解。",
  "temperature": 0.2,
  "max_tokens": 512
}' | \
curl -sS -i -X POST "${BASE_URL}/api/v1/analysis/model-governance" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  --data-binary @-
```

### 6.4 推理审计

`POST /api/v1/analysis/audits`

预期 HTTP 200、code=0、data.status=succeeded。audit_type 也支持 full_chain，同一请求材料可改此字段测试另一类型；结果 verification_state=unverified 表示材料分析，不能视为业务事实已核验。保存 data.id 为 RECORD_ID。

```bash
jq -n \
  --arg model_id "${MODEL_ID}" \
  '{
  "model_id": $model_id,
  "audit_type": "reasoning",
  "subject": {
    "id": "curl-subject-1",
    "type": "answer",
    "version": "v1"
  },
  "steps": [
    {
      "id": "step-1",
      "label": "生成答复",
      "detail": "依据公告说明活动需提前报名。",
      "evidence_refs": [
        "e1"
      ],
      "rule_refs": [
        "r1"
      ]
    }
  ],
  "evidence": [
    {
      "id": "e1",
      "content": "活动公告：本次活动需提前报名。"
    }
  ],
  "rules": [
    {
      "id": "r1",
      "text": "答复中关于报名的说明应与公告一致。"
    }
  ]
}' | \
curl -sS -i -X POST "${BASE_URL}/api/v1/analysis/audits" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  --data-binary @-
```

### 6.5 分析记录列表

`GET /api/v1/analysis/records`

预期 HTTP 200、code=0，data 为分页记录；可另加 kind=scenario/model_governance/audit、scenario_code 或 status=running/succeeded/failed。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/analysis/records" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "page=1" \
  --data-urlencode "page_size=10"
```

### 6.6 分析记录详情

`GET /api/v1/analysis/records/{record_id}`

预期 HTTP 200、code=0，data 为完整记录，含结果与调用信息。分析执行失败时 POST 可能返回非 2xx 且 data 仍有失败记录；应保留其 id，用此接口检查 error_code/error_message。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/analysis/records/${RECORD_ID}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

## 7. 多语种语义权重与风险分层【可选】

需要真实业务记录、治理 worker；请求显式选模型，不通过三类治理模型绑定。所选语种必须有真实样本；不接受 mock 数据。analysis_mode 为 semantic_weighting 或 risk_tiering，切换模式时历史和 latest 查询也应同步修改。

### 7.1 创建多语种任务

`POST /api/v1/analysis/ethnic-multilingual/tasks`

预期 HTTP 200、code=0，data.task_id 填入 MULTILINGUAL_TASK_ID。示例选 zh/en；按数据集实际语种替换本节建单和查询的 languages，不能重复。

```bash
jq -n \
  --arg model_id "${MODEL_ID}" \
  --argjson dataset_id "${DATASET_ID}" \
  --arg version_id "${VERSION_ID}" \
  '{
  "model_id": $model_id,
  "input": {
    "dataset_id": $dataset_id,
    "version_id": $version_id,
    "languages": [
      "zh",
      "en"
    ],
    "analysis_mode": "semantic_weighting",
    "scheme_id": "multilingual-v1"
  }
}' | \
curl -sS -i -X POST "${BASE_URL}/api/v1/analysis/ethnic-multilingual/tasks" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  --data-binary @-
```

### 7.2 轮询多语种任务

`GET /api/v1/analysis/ethnic-multilingual/tasks/{task_id}`

预期 HTTP 200、code=0。等待 data.status=succeeded 后取 data.result_id 填入 MULTILINGUAL_RESULT_ID；失败查看 error_message。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/analysis/ethnic-multilingual/tasks/${MULTILINGUAL_TASK_ID}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### 7.3 最新多语种结果

`GET /api/v1/analysis/ethnic-multilingual/results/latest`

预期 HTTP 200、code=0。URL 的 languages 重复传递两次，表示数组，不能拼成 zh,en；范围与任务一致，无结果 data=null。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/analysis/ethnic-multilingual/results/latest" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "dataset_id=${DATASET_ID}" \
  --data-urlencode "version_id=${VERSION_ID}" \
  --data-urlencode "languages=zh" \
  --data-urlencode "languages=en" \
  --data-urlencode "analysis_mode=semantic_weighting" \
  --data-urlencode "scheme_id=multilingual-v1"
```

### 7.4 多语种结果历史

`GET /api/v1/analysis/ethnic-multilingual/results`

预期 HTTP 200、code=0，data 为匹配范围的结果数组。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/analysis/ethnic-multilingual/results" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "dataset_id=${DATASET_ID}" \
  --data-urlencode "version_id=${VERSION_ID}" \
  --data-urlencode "languages=zh" \
  --data-urlencode "languages=en" \
  --data-urlencode "analysis_mode=semantic_weighting" \
  --data-urlencode "scheme_id=multilingual-v1"
```

### 7.5 多语种结果详情

`GET /api/v1/analysis/ethnic-multilingual/results/{result_id}`

预期 HTTP 200、code=0，data 为汇总；检查真实样本数量、有效性和失败数。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/analysis/ethnic-multilingual/results/${MULTILINGUAL_RESULT_ID}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### 7.6 多语种样本分页

`GET /api/v1/analysis/ethnic-multilingual/results/{result_id}/samples`

HTTP 200；v1 接口还需确认 code=0。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/analysis/ethnic-multilingual/results/${MULTILINGUAL_RESULT_ID}/samples" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "page=1" \
  --data-urlencode "page_size=10"
```

## 8. 训练任务【可选】

仅在测试训练时启动训练 worker；默认 API/治理启动命令不会启动训练：

```bash
bash start-content-security.sh start training
```

### 8.1 查询源码训练默认配置和完整示例

`GET /api/v1/training/defaults`

预期 HTTP 200、code=0，data.example_request 含可提交的 python_source、张量 contract，data.example_record 说明业务记录格式。源码训练需要真实数据与示例格式兼容，不能直接用普通文本记录跑分类示例。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/training/defaults" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### 8.2 查询服务器登记的训练脚本

`GET /api/v1/training/scripts`

预期 HTTP 200、code=0，data 是脚本数组，核对 id、inputs_schema、parameters_schema；当前内置 cifar100-cnn。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/training/scripts" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### 8.3 创建脚本训练任务：小规模 CIFAR-100

`POST /api/v1/training/script-tasks`

预期 HTTP 202、code=0，data.id 填入 TRAINING_TASK_ID。这是实际训练，会首次下载完整 CIFAR-100 并占磁盘；这里只缩小训练样本量。离线且已准备缓存时改 download=false。脚本任务不要求业务数据集 ID。

```bash
jq -n \
  '{
  "name": "curl-CIFAR100-小规模测试",
  "script_id": "cifar100-cnn",
  "inputs": {
    "download": true
  },
  "parameters": {
    "epochs": 1,
    "batch_size": 16,
    "learning_rate": 0.001,
    "seed": 42,
    "device": "cpu",
    "train_samples": 100,
    "validation_samples": 100
  }
}' | \
curl -sS -i -X POST "${BASE_URL}/api/v1/training/script-tasks" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "X-Setup-Secret: ${SETUP_SECRET}" \
  -H "Content-Type: application/json" \
  --data-binary @-
```

### 8.4 创建源码训练任务

`POST /api/v1/training/tasks`；需要访问令牌和管理员密钥。先确认第 8.1 节的示例源码可信，且业务记录 payload 形如 `{"features":[0.2,0.8],"label":1}`，至少有足够记录拆分训练/验证集。本示例从 defaults 取完整源码与 contract，避免手抄遗漏。

预期 HTTP 202、code=0，data.id 填入 TRAINING_TASK_ID。下面先 GET 再 POST；如果 defaults 请求失败，`jq -e` 不输出建单正文，后续 POST 会校验失败。

```bash
curl -sS "${BASE_URL}/api/v1/training/defaults" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" | \
  jq -e --argjson dataset_id "${DATASET_ID}" --arg version_id "${VERSION_ID}" '
    if .code == 0 and .data.example_request != null then
      .data.example_request
      | .name = "curl源码训练测试"
      | .dataset_id = $dataset_id
      | .version_id = $version_id
      | .epochs = 1
      | .batch_size = 2
      | .device = "cpu"
    else error("训练默认配置读取失败") end
  ' | \
  curl -sS -i -X POST "${BASE_URL}/api/v1/training/tasks" \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    -H "X-Setup-Secret: ${SETUP_SECRET}" \
    -H 'Content-Type: application/json' \
    --data-binary @-
```

### 8.5 训练任务列表

`GET /api/v1/training/tasks`

HTTP 200；v1 接口还需确认 code=0。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/training/tasks" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "page=1" \
  --data-urlencode "page_size=10"
```

### 8.6 训练任务状态

`GET /api/v1/training/tasks/{task_id}`

预期 HTTP 200、code=0。状态含 pending/running/canceling/succeeded/failed/canceled；终态为 succeeded 才算训练完成，失败检查 error_message 与 logs。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/training/tasks/${TRAINING_TASK_ID}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### 8.7 增量训练指标

`GET /api/v1/training/tasks/{task_id}/metrics`

预期 HTTP 200、code=0，返回增量指标；下一次使用已收到的最大指标 id 作为 after_id，未产生指标可为空。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/training/tasks/${TRAINING_TASK_ID}/metrics" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "after_id=0" \
  --data-urlencode "limit=200"
```

### 8.8 增量训练日志

`GET /api/v1/training/tasks/{task_id}/logs`

预期 HTTP 200、code=0，data.text/next_after_offset/has_more；下次把 next_after_offset 作为 after_offset，此游标是字节偏移，不是行号。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/training/tasks/${TRAINING_TASK_ID}/logs" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "after_offset=0" \
  --data-urlencode "limit=65536"
```

### 8.9 训练产物列表

`GET /api/v1/training/tasks/{task_id}/artifacts`

预期 HTTP 200、code=0。从返回的产物列表选取真实 id 填入 ARTIFACT_ID；任务未完成可能为空，不要猜文件名。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/training/tasks/${TRAINING_TASK_ID}/artifacts" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### 8.10 下载训练产物

`GET /api/v1/training/tasks/{task_id}/artifacts/{artifact_id}/download`；预期 HTTP 200 和文件字节流。替换 ARTIFACT_ID，并将 DOWNLOAD_PATH 设置为期望的新文件位置；`-D -` 显示响应头，`-o` 仅写正文，避免把 HTTP 头写进产物。返回非 200 时不要把错误 JSON 当作模型文件。

```bash
curl -sS -D - --output "${DOWNLOAD_PATH}" \
  --write-out '\nHTTP %{http_code}\n' \
  "${BASE_URL}/api/v1/training/tasks/${TRAINING_TASK_ID}/artifacts/${ARTIFACT_ID}/download" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### 8.11 取消专用训练任务【改变任务状态】

`POST /api/v1/training/tasks/{task_id}/cancel`

预期 HTTP 200、code=0。仅在希望终止该测试任务时调用；pending 可直接 canceled，running 通常先 canceling 后由 worker 结束为 canceled。完成后的任务不会被重新取消。

```bash
curl -sS -i -X POST "${BASE_URL}/api/v1/training/tasks/${TRAINING_TASK_ID}/cancel" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

## 9. 神经因果溯源【可选】

固定使用服务器本地 Qwen3-4B 模型，不使用 MODEL_ID 或治理绑定。先按 [溯源接口与部署](神经因果溯源接口与部署.md) 准备权重、审核案例、依赖及支持 CUDA/BF16 的 GPU。检查和启动命令：

```bash
.venv/bin/python -m service.neural_trace_worker --check
bash start-content-security.sh start neural-trace
```

`--check` 只验证文件和依赖，不证明真实 GPU 运行成功。建单会实际占用 GPU，完成时间由案例和硬件决定；不要重复提交同一测试以代替轮询。

### 9.1 查询可运行溯源案例

`GET /api/v1/neural-trace/cases`

预期 HTTP 200、code=0，data 为正式审核的案例数组；复制 data[].case_id 到 CASE_ID，不接收任意文件路径。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/neural-trace/cases" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### 9.2 创建溯源任务

`POST /api/v1/neural-trace/tasks`

预期 HTTP 202、code=0，data.task_id 填入 TRACE_TASK_ID；不存在或未正式审核的案例返回 404。

```bash
jq -n \
  --arg case_id "${CASE_ID}" \
  '{
  "case_id": $case_id
}' | \
curl -sS -i -X POST "${BASE_URL}/api/v1/neural-trace/tasks" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  --data-binary @-
```

### 9.3 溯源任务列表

`GET /api/v1/neural-trace/tasks`

预期 HTTP 200、code=0，data.items/total。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/neural-trace/tasks" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "limit=20" \
  --data-urlencode "offset=0"
```

### 9.4 轮询溯源任务

`GET /api/v1/neural-trace/tasks/{task_id}`

预期 HTTP 200、code=0，data.status 为 pending/running/succeeded/failed；查看 stage 和 error_message，等 succeeded 后取结果。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/neural-trace/tasks/${TRACE_TASK_ID}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### 9.5 查询溯源结果

`GET /api/v1/neural-trace/tasks/{task_id}/result`

成功任务预期 HTTP 200、code=0，data 包含 document_trace/layer_scan/activation_difference/component_scan 等结果；任务尚未成功时预期 409 trace_result_unavailable。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/neural-trace/tasks/${TRACE_TASK_ID}/result" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

## 10. 管理端与模型维护【可选】

只测试需要的操作。启停、删除使用专用测试令牌/模型变量；全量启停会影响所有已登记令牌，不能当作日常检查执行。模型与治理绑定是服务共享配置，任务、结果和个人调用记录则按访问令牌隔离。

### 10.1 查询业务用户目录

`GET /api/v1/admin/business-users`

预期 HTTP 200、code=0，读取服务配置 business_api_base_url 指向的业务后端。业务后端不可用/超时可能返回 502/504；不影响已经登记令牌的普通模型调用。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/admin/business-users" \
  -H "X-Setup-Secret: ${SETUP_SECRET}" \
  --get \
  --data-urlencode "page=1" \
  --data-urlencode "page_size=10" \
  --data-urlencode "keyword="
```

### 10.2 管理员查询令牌元数据

`GET /api/v1/admin/tokens`

预期 HTTP 200、code=0，data 含 id/label/enabled/expires_at，不返回令牌原文；从中选择专用测试令牌 ID，填入 TEST_TOKEN_ID。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/admin/tokens" \
  -H "X-Setup-Secret: ${SETUP_SECRET}"
```

### 10.3 按 ID 启用专用测试令牌

`PATCH /api/v1/admin/tokens/{token_id}/status`

预期 HTTP 200、code=0，data.enabled=true。要测试禁用可改 false，再用该专用令牌请求模型列表应为 401；最后用管理员密钥恢复 true。

```bash
jq -n \
  '{
  "enabled": true
}' | \
curl -sS -i -X PATCH "${BASE_URL}/api/v1/admin/tokens/${TEST_TOKEN_ID}/status" \
  -H "X-Setup-Secret: ${SETUP_SECRET}" \
  -H "Content-Type: application/json" \
  --data-binary @-
```

### 10.4 为尚无标签的令牌设置标签

`PATCH /api/v1/admin/tokens/{token_id}/label`

仅选择 label 为空的令牌；预期 HTTP 200、code=0。标签只允许设置一次；已标注令牌预期 409 token_already_labeled，第 2.1 节已带标签的令牌不能再改。

```bash
jq -n \
  '{
  "label": "curl-label-test"
}' | \
curl -sS -i -X PATCH "${BASE_URL}/api/v1/admin/tokens/${TOKEN_ID}/label" \
  -H "X-Setup-Secret: ${SETUP_SECRET}" \
  -H "Content-Type: application/json" \
  --data-binary @-
```

### 10.5 删除专用测试令牌【最后执行】

`DELETE /api/v1/admin/tokens/{token_id}`

预期 HTTP 200、code=0，data.deleted=true；软删除后该令牌不可调用，历史调用仍可关联元数据。仅对可废弃测试令牌执行。

```bash
curl -sS -i -X DELETE "${BASE_URL}/api/v1/admin/tokens/${TEST_TOKEN_ID}" \
  -H "X-Setup-Secret: ${SETUP_SECRET}"
```

### 10.6 管理员跨令牌调用记录

`GET /api/v1/admin/calls`

预期 HTTP 200、code=0，data 含调用记录以及 token_id/token_label。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/admin/calls" \
  -H "X-Setup-Secret: ${SETUP_SECRET}" \
  --get \
  --data-urlencode "limit=50" \
  --data-urlencode "offset=0"
```

### 10.7 当前令牌调用记录

`GET /api/v1/calls`

预期 HTTP 200、code=0，仅当前令牌记录；完成推理后应能查到对应模型调用。

```bash
curl -sS -i -X GET "${BASE_URL}/api/v1/calls" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "limit=50" \
  --data-urlencode "offset=0"
```

### 10.8 更新本地模型配置

`PATCH /api/models/{model_id}`

预期 HTTP 200，裸对象 updated_at 更新；只提交要修改的字段。模型名称/description 也可修改。更新后重新执行第 3.6 节；不要用 null 清空必需字段。

```bash
jq -n \
  --arg upstream_model_id "${UPSTREAM_MODEL_ID}" \
  --arg upstream_base_url "${UPSTREAM_BASE_URL}" \
  --arg upstream_api_key "${UPSTREAM_API_KEY}" \
  '{
  "upstream_model_id": $upstream_model_id,
  "base_url": $upstream_base_url,
  "api_key": $upstream_api_key
}' | \
curl -sS -i -X PATCH "${BASE_URL}/api/models/${MODEL_ID}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "X-Setup-Secret: ${SETUP_SECRET}" \
  -H "Content-Type: application/json" \
  --data-binary @-
```

### 10.9 删除专用测试模型【最后执行】

`DELETE /api/models/{model_id}`

预期 HTTP 204、空正文。仅删除不再需要且未用于业务的测试配置；删除被治理绑定引用的模型会使后续建单不可用，需要重新绑定有效模型。

```bash
curl -sS -i -X DELETE "${BASE_URL}/api/models/${TEST_MODEL_ID}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "X-Setup-Secret: ${SETUP_SECRET}"
```

## 11. 旧版裸响应接口与兼容入口【可选】

如客户端只使用 v1 包裹，无需重复测旧版推理；下列用于确认兼容性。模型配置写入与令牌登记本来就在旧版路径，已在前文给出。

### 11.1 旧版本地模型列表

`GET /api/models`

预期 HTTP 200，直接返回模型数组，没有 code/data 包裹。

```bash
curl -sS -i -X GET "${BASE_URL}/api/models" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### 11.2 旧版模型详情

`GET /api/models/{model_id}`

预期 HTTP 200，直接返回模型对象。

```bash
curl -sS -i -X GET "${BASE_URL}/api/models/${MODEL_ID}" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### 11.3 旧版响应类型列表

`GET /api/llm/response-types`

预期 HTTP 200，直接返回字符串数组。

```bash
curl -sS -i -X GET "${BASE_URL}/api/llm/response-types" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### 11.4 旧版通用推理

`POST /api/llm/invoke`

预期 HTTP 200，直接返回 model_id/response_type/result；会额外真实调用一次上游。

```bash
jq -n \
  --arg model_id "${MODEL_ID}" \
  '{
  "model_id": $model_id,
  "input": "请回答2加3等于多少。",
  "response_type": "text_answer"
}' | \
curl -sS -i -X POST "${BASE_URL}/api/llm/invoke" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  --data-binary @-
```

### 11.5 旧版调用记录

`GET /api/calls`

预期 HTTP 200，直接返回当前令牌的调用记录数组。

```bash
curl -sS -i -X GET "${BASE_URL}/api/calls" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  --get \
  --data-urlencode "limit=50" \
  --data-urlencode "offset=0"
```

### 11.6 按令牌原文启停专用测试令牌

`PATCH /api/auth/token/status`

预期 HTTP 200，裸 JSON enabled=true；改 false 可测试禁用，再恢复 true。此接口通过原文定位，不接收 token_id。

```bash
jq -n \
  --arg test_access_token "${TEST_ACCESS_TOKEN}" \
  '{
  "token": $test_access_token,
  "enabled": true
}' | \
curl -sS -i -X PATCH "${BASE_URL}/api/auth/token/status" \
  -H "X-Setup-Secret: ${SETUP_SECRET}" \
  -H "Content-Type: application/json" \
  --data-binary @-
```

### 11.7 全量启用所有令牌【仅隔离环境按需执行】

`PATCH /api/auth/tokens/status`

预期 HTTP 200，裸 JSON 含 enabled/updated_count。影响所有非删除令牌；改 false 会禁用全部令牌，因此不要在共享服务上随手测试。

```bash
jq -n \
  '{
  "enabled": true
}' | \
curl -sS -i -X PATCH "${BASE_URL}/api/auth/tokens/status" \
  -H "X-Setup-Secret: ${SETUP_SECRET}" \
  -H "Content-Type: application/json" \
  --data-binary @-
```

## 12. 明确拒绝业务写入的接口【可选错误测试】

下面三条在鉴权成功后固定返回 **HTTP 405**，message 表明“当前模型结果仅支持分析查看，业务写操作请使用业务后端”。405 是本组测试通过的标准；不能把这些接口写成复核/修复/发布成功。action 为路径占位参数，代码未定义可执行动作，示例 reject/submit 只用于验证拒绝写入行为。

### 12.1 风险人工复核兼容入口

`POST /api/v1/data-governance/risk-results/{result_id}/samples/{sample_id}/reviews`

预期 HTTP 405、v1 code=405；不执行人工复核，无需提交请求体。

```bash
curl -sS -i -X POST "${BASE_URL}/api/v1/data-governance/risk-results/${RISK_RESULT_ID}/samples/${RISK_SAMPLE_ID}/reviews" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### 12.2 异常候选变更兼容入口

`POST /api/v1/data-governance/anomaly-results/{result_id}/samples/{sample_id}/candidates/{action}`

预期 HTTP 405、v1 code=405；不执行候选变更，无需提交请求体。

```bash
curl -sS -i -X POST "${BASE_URL}/api/v1/data-governance/anomaly-results/${ANOMALY_RESULT_ID}/samples/${ANOMALY_SAMPLE_ID}/candidates/reject" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### 12.3 修改集动作兼容入口

`POST /api/v1/data-governance/change-sets/{action}`

预期 HTTP 405、v1 code=405；不发布业务版本，无需提交请求体。

```bash
curl -sS -i -X POST "${BASE_URL}/api/v1/data-governance/change-sets/submit" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

## 13. 框架文档入口【可选，无需鉴权】

这四个入口不计入 85 个项目接口。Swagger/ReDoc 返回 HTML，curl 只检查可达；在浏览器查看交互界面。

### OpenAPI 定义

`GET /openapi.json`

HTTP 200，JSON 含 openapi/info/paths/components；可与当前服务器版本核对路径及请求字段。

```bash
curl -sS -i -X GET "${BASE_URL}/openapi.json"
```

### Swagger UI

`GET /docs`

HTTP 200，text/html；浏览器页面引用的外部资源还取决于服务器网络。

```bash
curl -sS -i -X GET "${BASE_URL}/docs"
```

### Swagger OAuth 辅助页

`GET /docs/oauth2-redirect`

HTTP 200，text/html；本项目并未因此提供业务 OAuth 登录接口。

```bash
curl -sS -i -X GET "${BASE_URL}/docs/oauth2-redirect"
```

### ReDoc

`GET /redoc`

HTTP 200，text/html。

```bash
curl -sS -i -X GET "${BASE_URL}/redoc"
```

<a id="acceptance"></a>

## 14. 可选错误测试与问题定位

### 14.1 缺少访问令牌，应返回 401

```bash
curl -sS -i "${BASE_URL}/api/v1/models"
```

预期 HTTP 401、code=401、error_code=invalid_token。若有效令牌也返回 401，检查是否已登记、是否启用、是否过期，不能用业务后端登录令牌或上游 API Key 替代。

### 14.2 管理员密钥错误，应返回 403

```bash
curl -sS -i "${BASE_URL}/api/v1/admin/governance/models" \
  -H 'X-Setup-Secret: intentionally-invalid-secret'
```

预期 HTTP 403、error_code=invalid_setup_secret。

### 14.3 请求体缺少必填字段，应返回 422

```bash
curl -sS -i -X POST "${BASE_URL}/api/v1/tasks" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H 'Content-Type: application/json' \
  --data '{}'
```

预期 HTTP 422、error_code=validation_error；不创建有效任务。

### 14.4 不存在的模型，应返回 404

```bash
curl -sS -i "${BASE_URL}/api/v1/models/curl-deliberately-missing-model" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

预期 HTTP 404、error_code=model_not_found（前提是服务中没有这个测试 ID）。

| 现象 | 优先检查 |
| --- | --- |
| curl 无法连接/Connection refused | BASE_URL、端口、API 进程、容器网络 |
| 401 | 本服务令牌登记、启用和过期时间 |
| 403 | X-Setup-Secret 是否与实际服务配置一致；模型写入和训练建单是否同时带两个头 |
| 503 model_not_configured | 对应 kind 的治理绑定缺失，或绑定模型已删除 |
| 建单 422 invalid_scope | 治理 kind 与 scheme_id 是否匹配；无效字段通常为 validation_error |
| 建单 400 invalid_kind | 路径与 kind 不一致，或价值任务查询遗漏/误写 kind |
| 任务长期 pending | 对应 worker 未启动/不存活，API 与 worker 是否连到同一个库 |
| 治理 failed，提示业务数据库读取失败 | 共享 MySQL 中业务表是否存在、SELECT 权限是否满足、配置是否一致 |
| 治理 failed，提示版本不匹配/无记录 | datasets 当前版本与 VERSION_ID、dataset_records 真实记录 |
| 模型 502/504 或所有样本失败 | 上游地址、API Key、upstream_model_id、超时、模型结构化输出能力；继续查调用记录 |
| 429 analysis_busy | 同步分析并发槽已满；稍后查询记录或重试，避免反复建单 |
| latest 返回 data=null | 尚无已完成结果，或令牌/完整 scope 与原任务不一致 |
| 样本/任务/结果 404 | ID 是否取自正确模块、是否用同一访问令牌读取 |
| 溯源结果 409 | 任务尚未 succeeded，先查状态和 error_message |
| 三个业务写入口 405 | 设计行为，业务复核/应用变更/发布需调用业务后端 |

对 v1 普通业务错误，检查 HTTP 状态、code/message/error_code；独立分析执行失败可能仍在 data 中附完整失败记录。网关错误或未捕获的 500 不保证符合 JSON 包裹。

## 15. 验收记录建议

- 必要初始化：有效令牌可查询模型；本地模型配置正确；所测治理类型均已绑定有效本地 ID；方案已确认。
- 每次前置检查：根入口 200、模型列表可读、绑定符合预期、实际数据版本与 worker 状态确认；本次含模型调用时推理响应有效。
- 数据治理：选定类型至少完成一次真实数据任务，status=succeeded、有 result_id，汇总及样本可读取，确认有效/失败/无法评估数量，不能用 mock 结果替代。
- 可选模块：仅记录本次实际测试的场景、多语种、训练、溯源、管理员或兼容功能；未执行的标记“未测”，不要默认通过。
- 负向测试：401/403/422/404 和固定 405 按各条预期判定；下载确认 HTTP 200 且文件内容有效。

建议记录每项的 HTTP 状态、trace_id、task_id/result_id/record_id、执行时间与错误信息。记录中不保存真实密钥。参数和路由已经与本地实现静态核对；本文生成时未连接真实服务器、MySQL、上游或 GPU 执行上述请求，服务器联调需按本文逐项运行。

文档检查结果：85 个项目接口和 4 个框架入口均有对应命令；104 个 Bash 代码块通过 `bash -n`；23 组正常请求体（含动态取得的源码训练示例）通过实际请求模型校验；36 个变量的引用与定义已核对。以上属于本地静态验证，不替代真实服务器联调。

本文件是新建的本地文档；仓库当前 `.gitignore` 默认忽略新 Markdown，因此复制到服务器时需显式上传本文件。此任务没有修改忽略规则或业务代码。

