# 书生 InternLM3-8B 部署

适用：Linux、单张 A100 80GB，与 Qwen3-8B 共用 GPU 0。先启动书生，确认 1221 就绪后再启动 Qwen。

| 配置 | 默认值 |
|---|---|
| 模型目录 | `/data/models/internlm3-8b-instruct` |
| 工作目录 / 端口 | `~/internlm` / `1221` |
| `SESSION_LEN` | `32768` |
| `CACHE` | `0.2` |
| `BATCH` / `MAXCONC` | `4` / `4` |

## 1. 安装环境

```bash
nvidia-smi
conda create -n internlm python=3.12 -y
conda activate internlm
pip install lmdeploy modelscope
lmdeploy check_env
```

## 2. 下载模型

```bash
modelscope download \
  --model Shanghai_AI_Laboratory/internlm3-8b-instruct \
  --local_dir /data/models/internlm3-8b-instruct
```

## 3. 启动

将 [start-internlm.sh](./start-internlm.sh) 放入 `~/internlm/`，执行：

```bash
conda activate internlm
cd ~/internlm
sed -i 's/\r$//' start-internlm.sh
chmod +x start-internlm.sh
./start-internlm.sh start
./start-internlm.sh status
```

脚本保留 `--trust-remote-code`，监听 `0.0.0.0:1221`。平台开放 1221 端口，业务地址使用平台生成的 HTTPS 地址加 `/v1`。

Key 文件：`~/internlm/api.key`。启停、日志及覆盖参数见 [启动说明](./start-internlm-docs.md)。

## 4. 检查

```bash
curl -fsS --max-time 10 http://127.0.0.1:1221/v1/models \
  -H "Authorization: Bearer $(cat ~/internlm/api.key)"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
```

两模型就绪后，用实际业务样本检查非空正文、JSON 解析和原有业务校验。显存以实测为准，不保证四条满长请求同时执行。

## 5. 结构化业务请求

在业务调用中传入对应的 Pydantic 结果类型 `response_model`，不要只依赖提示词：

```python
response = client.chat.completions.create(
    model=model_id,
    messages=messages,
    response_format={"type": "json_schema", "json_schema": {
        "name": response_model.__name__, "schema": response_model.model_json_schema(),
    }},
    max_tokens=4096,
)
```

`client`、`model_id`、`messages` 使用业务已有值；4096 是输出预算起步值。普通文本生成不加 Schema；返回后保留字段、规则和原文证据校验，不自动降级。

Schema 是请求参数，修改文档或重启模型不会让调用方自动携带它。修改业务请求代码需同步并重启对应 API、worker。

## 6. 长文本与显存

32K 包含完整提示词、输入和输出，不等于 32K 字符。业务接口上限保持不变，长材料需要另配模型上下文。

书生 128K 单请求配置：

```bash
cd ~/internlm
SESSION_LEN=131072 BATCH=1 MAXCONC=1 ./start-internlm.sh restart
```

先用最长业务输入实测上下文和 KV 容量。`CACHE` 控制缓存预留，`BATCH` 控制推理批量上限，`MAXCONC` 控制服务并发上限；参数均可由环境变量覆盖。调整启动参数后需重启模型。

## 7. 常见问题

| 现象 | 操作 |
|---|---|
| `CUDA out of memory` | 降低缓存预留、批量或并发，重启后检查实际峰值 |
| KV 容量不足 | 减少并发，或在显存预算内增加 `CACHE`；不要继续缩小缓存 |
| 输入超长 | 按 token 核对输入和输出总预算，使用长文本配置 |
| `empty_response` | 检查结束原因、正文字段与模型服务日志，不把空返回当成功 |
| `invalid_structured_output` | 检查该业务 Schema、字段及证据校验，不放宽校验 |
| Schema 请求报 400/500 | 核对 LMDeploy 版本与 grammar 日志，不自动退回无约束请求 |
| 外网 502 / 401 | 检查端口监听、平台映射及对应模型 Key |

参考：[LMDeploy 服务](https://lmdeploy.readthedocs.io/en/latest/llm/api_server.html)、[结构化输出](https://lmdeploy.readthedocs.io/en/latest/advance/structed_output.html)。
