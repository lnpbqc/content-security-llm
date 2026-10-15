# Qwen3-8B 部署

适用：Linux、单张 A100 80GB，与书生共用 GPU 0。先启动书生，确认 1221 就绪后再启动 Qwen。

| 配置 | 默认值 |
|---|---|
| 模型目录 | `/data/models/Qwen3-8B` |
| 工作目录 / 端口 | `~/qwen` / `1222` |
| `SESSION_LEN` | `32768` |
| `CACHE` | `0.3` |
| `BATCH` / `MAXCONC` | `4` / `4` |

## 1. 安装环境

```bash
nvidia-smi
conda create -n qwen python=3.12 -y
conda activate qwen
pip install lmdeploy modelscope
lmdeploy check_env
```

## 2. 下载模型

```bash
modelscope download --model Qwen/Qwen3-8B --local_dir /data/models/Qwen3-8B
```

## 3. 启动

将 [start-qwen.sh](./start-qwen.sh) 放入 `~/qwen/`，执行：

```bash
conda activate qwen
cd ~/qwen
sed -i 's/\r$//' start-qwen.sh
chmod +x start-qwen.sh
./start-qwen.sh start
./start-qwen.sh status
```

脚本不加 `--trust-remote-code`，监听 `0.0.0.0:1222`。平台开放 1222 端口，业务地址使用平台生成的 HTTPS 地址加 `/v1`。

Key 文件：`~/qwen/api.key`。启停、日志及覆盖参数见 [启动说明](./start-qwen-docs.md)。

### 解析器版本

脚本保留原默认值 `REASONING_PARSER=default`。在模型环境核对安装版本及支持的名称：

```bash
python -c 'import lmdeploy; print(lmdeploy.__version__)'
lmdeploy serve api_server -h
```

当前官方文档列出 `qwen3`；确认安装版本支持后，可覆盖默认值：

```bash
REASONING_PARSER=qwen3 ./start-qwen.sh restart
```

后续启停继续使用同一覆盖值。解析器只拆分思考与正文，不会关闭思考。[官方说明](https://lmdeploy.readthedocs.io/en/latest/llm/api_server_reasoning.html)

## 4. 检查

```bash
curl -fsS --max-time 10 http://127.0.0.1:1222/v1/models \
  -H "Authorization: Bearer $(cat ~/qwen/api.key)"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
```

两模型就绪后，用实际业务样本检查非空正文、JSON 解析和原有业务校验。显存以实测为准，不保证四条满长请求同时执行。

## 5. 结构化业务与思考开关

针对支持思考开关的 Qwen3-8B，非思考结构化请求同时携带业务 Schema 和模板参数：

```python
response = client.chat.completions.create(
    model=model_id,
    messages=messages,
    response_format={"type": "json_schema", "json_schema": {
        "name": response_model.__name__, "schema": response_model.model_json_schema(),
    }},
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    max_tokens=4096,
)
```

`client`、`model_id`、`messages` 使用业务已有值，`response_model` 使用该业务的 Pydantic 类型；4096 是输出预算起步值。普通文本生成不加 Schema，思考设置单独验证。

调用方必须实际发送上述参数；修改文档或仅重启 Qwen 不会自动添加。修改业务请求代码需同步并重启对应 API、worker。保留返回后的字段、规则和原文证据校验，不自动降级。

## 6. 上下文与显存

32K 包含完整提示词、输入和输出，不等于 32K 字符。业务接口上限保持不变，超长材料另行安排长文本模型配置。

不要直接套用书生的 128K。Qwen3-8B 原生上下文为 40960；超过原生范围需另行验证模型及上下文扩展配置。

`CACHE` 控制缓存预留，`BATCH` 控制推理批量上限，`MAXCONC` 控制服务并发上限；参数均可由环境变量覆盖。调整启动参数或解析器后需重启模型。

## 7. 常见问题

| 现象 | 操作 |
|---|---|
| `CUDA out of memory` | 降低缓存预留、批量或并发，重启后检查实际峰值 |
| KV 容量不足 | 减少并发，或在显存预算内增加 `CACHE`；不要继续缩小缓存 |
| 输入超长 | 按 token 核对输入和输出总预算，不直接改成 128K |
| 解析器名称不认识 | 按安装版本核对 `REASONING_PARSER` 并覆盖默认值 |
| `empty_response` / `Model returned no content` | 比较相同请求默认思考与关闭思考的结果，记录结束原因、正文及思考长度；根因仍需服务日志确认 |
| 正文为空但有 `reasoning_content` | 核对思考开关、解析器与输出预算，不用思考字段代替业务结果 |
| `invalid_structured_output` | 检查业务 Schema、字段及证据校验，不放宽校验 |
| Schema 请求报 400/500 | 核对 LMDeploy 版本与 grammar 日志，不自动退回无约束请求 |
| 外网 502 / 401 | 检查端口监听、平台映射及对应模型 Key |

参考：[Qwen 思考开关](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html)、[LMDeploy 结构化输出](https://lmdeploy.readthedocs.io/en/latest/advance/structed_output.html)。
