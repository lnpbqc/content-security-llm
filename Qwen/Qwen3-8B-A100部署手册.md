# Qwen3-8B A100 部署手册（与书生同机）

| 项目 | 说明 |
|---|---|
| 适用硬件 | NVIDIA A100 80GB，与书生 InternLM3-8B 共用这一张卡 |
| 适用系统 | Linux（Ubuntu 20.04/22.04、CentOS 7+） |
| 目标模型 | Qwen3-8B |
| 推荐引擎 | LMDeploy + TurboMind |
| 访问方式 | 平台「开放端口」（服务名 + 端口），公网 HTTPS 由平台提供 |
| 主要调用方 | `content-security-llm`（要 JSON，输入上限 20 万字符） |

路径统一用 `/data/models`，端口统一用 `1222`，替换成你自己的。

**和书生那份的关系**：命令结构一一对应，只换脚本名和端口。书生那份是 `start-internlm.sh` / 1221 / 加 `--trust-remote-code`，这份是 `start-qwen.sh` / 1222 / 不加。全文凡写「同书生 §N」的，去那份手册看细节。

两个服务都在 GPU 0 上，不分卡。**先把 §2.1 的显存分账和 §10.2 的启动顺序看完再动手**——这份手册的服务是**第二个**启动的。

| | 书生 InternLM3-8B | Qwen3-8B |
|---|---|---|
| 端口 | 1221 | **1222** |
| 平台服务名 | 控制台登记的那个 | `qwen3web` |
| 模型目录 | `/data/models/internlm3-8b-instruct` | `/data/models/Qwen3-8B` |
| 工作目录 / Key | `~/internlm` / `api.key` | `~/qwen` / `api.key` |
| tmux 会话 | `internlm` | `qwen` |
| 脚本 | `start-internlm.sh` | `start-qwen.sh` |
| 上下文 / 批 / 并发 | 131072 / 32 / 16 | 32768 / 32 / 16 |
| `--cache-max-entry-count` | 0.4 | 0.6 |
| 启动顺序 | 先起 | 后起 |
| 绑卡 | `GPU=0` | `GPU=0` |
| `--trust-remote-code` | 必加 | **不要加** |
| 思考模式 | 无 | 有，见 §7 |

---

## 0. 决策表

| 你的场景 | 模型 | 启动要点 |
|---|---|---|
| 1×A100 80GB，与书生同卡（本案） | Qwen3-8B，BF16 | tp=1，GPU 0，session-len 32768，cache 0.6，**第二个启动** |
| 1×A100 80GB，整卡只跑 Qwen | Qwen3-8B，BF16 | session-len 32768，cache 0.8，batch 可提到 64 |
| 同卡，并发还要更高 | Qwen3-8B，AWQ 4bit | 官方 AWQ 权重 5.68 GiB，省出来的全进 KV 池，见 §11 |
| 要接 20 万字符的长输入 | 见 §2 末尾 | **Qwen3-8B 原生装不下**，要 YaRN 或换型号 |

---

## 1. 上机体检

```bash
nvidia-smi
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # 卡有没有被占

free -g          # 建议 ≥ 32GB
df -h /data      # 两个模型权重合计约 34GB，留 100GB 以上
python3 --version   # 3.10~3.12
```

和书生同一套体检，多看一眼的是**卡上还有谁**：两个服务都在这张卡上。

| 现象 | 处理 |
|---|---|
| `compute_cap` 是 8.0 | 确认 A100，本文全部适用 |
| `memory.total` 是 81920 MiB | 80GB 卡，同卡跑两个服务够用，按 §2.1 分 |
| `memory.total` 是 40960 MiB | 40GB 卡，光两个 8B 权重就 31.7 GiB，同卡放不下，得加卡或给两边都上 AWQ |
| `nvidia-smi -L` 出现 MIG 字样 | 卡被切分，按实例显存重算，见 §11 |
| 驱动 < 525 | 不支持 CUDA 12，联系运维升级 |

---

## 2. 显存账

Qwen3-8B 的真实配置（读自仓库 `config.json`）：

```
36 层 ｜ hidden 4096 ｜ 32 个 attention head ｜ 8 个 KV head ｜ head_dim 128
词表 151936 ｜ 原生 40960 上下文（rope_scaling: null）｜ bfloat16
参数量约 8.19B，BF16 权重 16,381,470,720 字节 = 15.26 GiB（5 个 safetensors，文件大小之和 16,381,516,776，含 46,056 字节文件头）
```

KV 开销（**Qwen3-8B 是书生的 3 倍**，这是全文最重要的一条）：

```
Qwen3-8B   每 token = 2 × 36 层 × 8 KV head × 128 head_dim × 2 字节 = 144 KiB
书 生      每 token = 2 × 48 层 × 2 KV head × 128 head_dim × 2 字节 =  48 KiB
```

| 上下文 | Qwen 单请求 KV | 书生单请求 KV |
|---|---|---|
| 8K | 1.13 GiB | 384 MiB |
| 32K | 4.5 GiB | 1.5 GiB |
| 131072 | 18 GiB | 6 GiB |

### 2.1 一张 A100 80GB 上两个服务怎么分

`--cache-max-entry-count` 是**权重加载完之后剩余空闲显存的比例**（官方定义：percentage of free gpu memory occupied by the k/v cache, excluding weights），TurboMind 启动时按这个比例把 KV block 池一次划走、常驻显存。谁先起谁按整卡的空闲算，所以比例和启动顺序是配死的。

| 项 | GiB |
|---|---|
| 书生 BF16 权重 | 16.40 |
| Qwen3-8B BF16 权重 | 15.26 |
| 两个进程的 CUDA context + 工作区（每个留 2） | 4 |
| 书生 KV 池（cache 0.4，先启动） | 约 25 |
| Qwen KV 池（cache 0.6，后启动） | 约 13 |
| **合计** | **约 74 / 80 GiB，余 6** |

Qwen 拿到的这 13 GiB 能装几路：

| 单请求上下文 | 单请求 KV | 13 GiB 池 |
|---|---|---|
| 8K | 1.13 GiB | 11 路 |
| 32K（本服务的满长） | 4.5 GiB | 2 路（第 3 路要 13.5 GiB） |

顺序：**先起书生，等 1221 监听，再起本服务**。反过来的话本服务的 0.6 会从约 63 GiB 的空闲里划走约 38 GiB，书生再起时只剩约 6 GiB、0.4 只能划出 2~3 GiB 的 KV 池，满长请求装不下。

起完对账：

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
```

书生那个进程 41~44 GiB，Qwen 29~31 GiB，合计 70~75 GiB，空闲 ≥ 5 GiB。对不上看 §12。

整卡只跑 Qwen（不部署书生）时，cache 给 0.8：权重 15.26 + KV 约 51 GiB。

**长上下文并发只有书生的三分之一**：同样大小的池子，Qwen 装得下的路数就是书生的三分之一，所以 `--max-concurrent-requests` 别比书生高。

**别照抄书生的 `--session-len 131072`。** Qwen3-8B 原生训练长度只有 **40960**，写 131072 不报错，但超出部分质量没有保证，而且单请求 KV 就 18 GiB——同卡时 Qwen 那份池子只有 13 GiB，一个满长请求都装不下。要接那批 20 万字符的长输入，三条路：

| 方案 | 怎么配 | 代价 |
|---|---|---|
| 只给 Qwen 短输入 | 维持 `--session-len 32768` | 最省事，但没法在同一批长文本上对比两个模型 |
| YaRN 外推到 131072 | 配 rope scaling 与 `--rope-scaling-factor`，池子同时要抬到 20 GiB 以上 | 只有整卡单跑 Qwen（cache 0.8）才装得下 18 + 15.26 GiB；外推质量必须实测 |
| 换原生长上下文型号 | `Qwen/Qwen3-30B-A3B-Instruct-2507`：262144 上下文，每 token 只要 96 KiB | 它是**纯非思考**模型，和 Qwen3-8B 不是一类 |

---

## 3. 环境准备

```bash
conda create -n qwen python=3.12 -y
conda activate qwen
pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install lmdeploy -i https://pypi.tuna.tsinghua.edu.cn/simple

# 自检
lmdeploy check_env
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

期望看到 `True` 和 `NVIDIA A100-...`。

**单开一个环境，别和书生那个共用。** 两边都是 lmdeploy，版本升级节奏不同，一个环境的升级把另一个服务搞挂，排查起来很费时间。

---

## 4. 下载权重

```bash
pip install modelscope
modelscope download \
  --model Qwen/Qwen3-8B \
  --local_dir /data/models/Qwen3-8B
```

内网机器离线搬运，到达后必须校验（半个文件要到加载阶段才暴露）：

```bash
du -sh /data/models/Qwen3-8B                      # 约 16G（15.26 GiB 权重 + tokenizer 等）
ls /data/models/Qwen3-8B/*.safetensors | wc -l    # 5

python - <<'EOF'
import json, os, struct
p = '/data/models/Qwen3-8B'
idx = json.load(open(f'{p}/model.safetensors.index.json'))
total = idx['metadata']['total_size']; data = 0
for f in os.listdir(p):
    if f.endswith('.safetensors'):
        fp = os.path.join(p, f)
        with open(fp, 'rb') as fh: n = struct.unpack('<Q', fh.read(8))[0]
        data += os.path.getsize(fp) - 8 - n
print('index:', total, '| 权重数据:', data, '|', 'OK' if total == data else '不一致，重传')
EOF
```

`data` 和 `total` 都应该是 **16381470720**。

**别拿「文件大小之和」比 `total_size`**——safetensors 每个文件前面有 JSON 头，这 5 个分片的头加起来 46,056 字节，文件大小之和是 16,381,516,776。直接比会在权重完好的情况下报「不一致」，白重传 16GB。上面这段减掉头再比。

---

## 5. 不需要 `--trust-remote-code`

书生那份是「必加」，**Qwen 反过来：不用加**。

| | 书生 InternLM3-8B | Qwen3-8B |
|---|---|---|
| `config.json` 有 `auto_map` | 有（三份 `.py`） | **无** |
| 必须 `--trust-remote-code` | 是 | 否 |

书生不加会直接报 `ValueError: ... contains custom code`，因为三份实现随权重放在仓库里，transformers 主干没有。Qwen3-8B 的 `config.json` 里**没有 `auto_map`**，`Qwen3ForCausalLM` 在 transformers 主干里，**4.51.0 起原生支持**（仓库 `config.json` 里的 `transformers_version` 就是 `4.51.0`）。

**各入口写法**：LMDeploy / vLLM / SGLang 都不用加；Python 代码里 `trust_remote_code=False`（默认）。

**安全**：这个参数会让模型目录的 Python 以服务进程权限执行。加上去不报错，只是白开一个口子——没必要开的口子就别开。

报 `Unrecognized model` 是 transformers 太旧，升级它，不是加这个参数。

---

## 6. 启动服务

先冒烟，确认能加载：

```bash
lmdeploy chat /data/models/Qwen3-8B --backend turbomind
```

**这里必须绑 `0.0.0.0`，不能绑 `127.0.0.1`。** 平台的端口映射在容器外做转发，只监听回环地址的话外网流量进不来，平台会返回 502。

`--server-port` 填你在平台上开放的那个端口。下面示例用 `1222`。**同卡部署时这条命令第二个跑**，书生先起（§2.1）。

```bash
CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server /data/models/Qwen3-8B \
  --server-name 0.0.0.0 \
  --server-port 1222 \
  --backend turbomind \
  --tp 1 \
  --session-len 32768 \
  --cache-max-entry-count 0.6 \
  --max-batch-size 32 \
  --max-concurrent-requests 16 \
  --reasoning-parser default \
  --api-keys 换成你的长随机串 \
  --disable-fastapi-docs \
  --log-level INFO
```

抽出来就是附件 [start-qwen.sh](./start-qwen.sh)（`./start-qwen.sh start`，GPU 默认 0，KV 比例默认 0.6）。

**和书生那份命令的四处差别**：

| 差别 | 说明 |
|---|---|
| `--server-port 1222` | 书生占着 1221 |
| 没有 `--trust-remote-code` | 见 §5 |
| 多了 `--reasoning-parser default` | 思考内容拆到 `reasoning_content`，见 §7 |
| `--session-len 32768` | 原生只有 40960，别抄书生的 131072，见 §2 |

整卡只跑 Qwen 时 `--cache-max-entry-count` 可以提到 0.8、`--max-batch-size` 提到 64；和书生同卡就按 §2.1 用 0.6，别在这里贪大。

### 参数速查

| 参数 | 作用 | 调整方向 |
|---|---|---|
| `--server-name` | 监听地址 | **必须 `0.0.0.0`**，绑回环地址平台转发不进来 |
| `--server-port` | 监听端口 | 与平台上开放的端口号一致（这里 1222） |
| `--tp` | 张量并行卡数 | 单卡装不下权重时才加，8B 用不上 |
| `--session-len` | 单请求最大上下文 | **不超过 40960**，见 §2 |
| `--cache-max-entry-count` | KV 占空闲显存比例 | 同卡固定 0.6（本服务后启动）；整卡单跑可到 0.8；OOM 再降到 0.4，见 §2.1 |
| `--max-batch-size` | 单批最大请求数 | 长上下文场景别贪大，和书生一样 32 |
| `--max-concurrent-requests` | 并发上限，超出排队 | KV 贵 3 倍，比书生只低不高 |
| `--reasoning-parser` | 思考内容拆到 `reasoning_content` | Qwen3 填 `default`，见 §7.1 |
| `--api-keys` | 服务鉴权 | **公网可达，必开** |
| `--disable-fastapi-docs` | 关掉 Swagger UI | **公网可达，必开** |
| `--quant-policy` | **KV** 量化：4=int4，8=int8 | 并发不够时用 8 |

`--quant-policy` 只管 KV cache，不管权重。

### 本地验证

```bash
curl -s http://127.0.0.1:1222/v1/models -H "Authorization: Bearer 你的KEY" | python -m json.tool
```

返回的 `id` 就是模型目录路径 `/data/models/Qwen3-8B`，客户端的 `model` 字段填它。

流式：

```bash
curl -N http://127.0.0.1:1222/v1/chat/completions \
  -H "Authorization: Bearer 你的KEY" -H "Content-Type: application/json" \
  -d '{"model":"/data/models/Qwen3-8B","messages":[{"role":"user","content":"数到十"}],"stream":true}'
```

---

## 7. 思考模式与结构化输出（业务系统要 JSON 时必看）

书生没有思考模式，这一节是 Qwen 独有的；但最后一小节要 JSON 的部分和书生 §7 完全对得上。

### 7.1 思考开关

Qwen3 回答前先生成一段思考。启动时加了 `--reasoning-parser default`，思考内容会单独走 `reasoning_content` 字段，正文走 `content`，客户端不用自己切 `<think>` 标签。

**硬开关**（请求体里传模板参数，服务端在拼 prompt 时就按非思考模式拼）：

```json
{"messages": [{"role": "user", "content": "数到十"}],
 "chat_template_kwargs": {"enable_thinking": false}}
```

`enable_thinking` 默认是 `true`。**软开关**（`/think`、`/no_think`）只是提示，模板照样开 `<think>` 块，模型可能什么都不想而已——要**保证**不思考只能用硬开关。

### 7.2 采样参数

| 模式 | temperature | top_p | top_k | min_p |
|---|---|---|---|---|
| 思考（默认） | 0.6 | 0.95 | 20 | 0 |
| 非思考 | 0.7 | 0.8 | 20 | 0 |

思考那一行就是仓库 `generation_config.json` 里的值。**官方明确说不要用贪心解码**（`temperature=0`、`top_k=1`），会导致性能退化和无限重复——上生产习惯把 temperature 调 0 的人最容易踩。

`top_k` 和 `min_p` 不是 OpenAI 协议字段，要传得走 `extra_body`，能不能生效看服务端。按 temperature + top_p 调也够用。

多轮对话**不要把历史思考内容塞回 `messages`**，模板只认最后一轮的，历史轮塞了白烧输入 token（[qwen-chat.html](./qwen-chat.html) 已按这个处理）。

### 7.3 结构化输出：支持情况

和书生 §7.1 同一套，一条都不用改：

| 问题 | 答案 |
|---|---|
| 要额外装 xgrammar 吗 | **不用**，随 lmdeploy 自动安装 |
| TurboMind 支持吗 | **支持**，两个后端都支持 |
| 要加启动参数吗 | **不用**，`response_format` 是请求级参数 |
| 支持哪些格式 | `json_schema`、`regex_schema`、`structural_tag` |

### 7.4 Qwen 多一条硬规则：要 JSON 就先关思考

请求里必须带上 `chat_template_kwargs: {"enable_thinking": false}`。

理由：书生 §7.4 记过一个历史 bug（Issue #4347），根因是**模型的 CoT 文本跑进了不该去的字段**；Qwen3 默认就产 CoT，撞上的概率比书生高。关掉思考既省输出 token，也避开这类边界。

### 7.5 客户端必须配合改的三处

| 位置 | 取值 |
|---|---|
| 请求参数 | `response_format={"type":"json_schema","json_schema":{"name":类名,"schema":model_json_schema()}}` |
| `max_tokens` | **4096** |
| `temperature` | **0.1** |

再加上 7.4 的 `chat_template_kwargs`。

### 7.6 上线前必须验证 schema 能不能编译

把书生 §7.3 的 `verify_schema.py` 拿过来，改两处：`base_url` 换成 Qwen 的平台地址、`create()` 里加上 `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`。

结果怎么读（同书生）：

| 现象 | 说明 | 下一步 |
|---|---|---|
| `✅ schema 约束生效` | 约束解码工作正常 | 只改客户端代码 |
| 无围栏、无废话，但字段报错 | schema 编译成功，是内容问题 | 查业务校验器 |
| 返回带 ``` 围栏或前后有解释 | **`response_format` 被忽略了** | `pip install -U lmdeploy` |
| 400/500，日志说 grammar 编译失败 | xgrammar 啃不动这个 schema | 退化成 `{"type": "json_object"}`，靠客户端容错 |
| TurboMind 报错 | 后端问题 | 换 `--backend pytorch` 再测一次 |

**别同时开投机解码**（Issue #4551：和引导解码同时用行为未定义）。本文配置没开。

---

## 8. 使用服务名进行访问

平台机制和书生 §8 完全一样（「开放端口」登记服务名 + 容器内端口，平台生成 `https://<实例ID>-<服务名>.svc.bitahub.com:20443`），操作步骤照抄，只换值：

| 位置 | 值 |
|---|---|
| 平台登记的端口号 | **1222** |
| `--server-port` | **1222** |
| `--server-name` | `0.0.0.0` |

**先在平台开放端口，再起服务**，顺序反了就是 502。服务名填 `qwen3web`；书生占的是 1221，两个服务在平台上各登记一条，端口别填重。

```bash
curl -sS <平台地址>/v1/models -H "Authorization: Bearer 你的KEY"
```

## 9. Web 界面

和书生 §9 一样，`api_server` 起的是纯 API，不带界面。

### 9.1 方案一：单文件 HTML 页面（零安装，最快）

附件 [qwen-chat.html](./qwen-chat.html) 是自带流式输出、Markdown 渲染、多轮上下文的单文件聊天页，另外还有 Qwen 专有的**思考开关**和**推理过程折叠显示**：

1. 把 `qwen-chat.html` 下载到你自己电脑
2. 双击用浏览器打开（Chrome / Edge 都行）
3. 展开顶部「连接设置」，填**服务地址**（平台控制台复制的完整地址）和 **API Key**；模型名默认 `/data/models/Qwen3-8B`
4. 点「测试连接」确认通了，然后就能聊

配置存在浏览器 localStorage 里（key 是 `qwen-cfg`，和书生那份的 `internlm-cfg` 不冲突，两个页面可以在同一个浏览器里共存）。

### 9.2 方案二：Open WebUI（多用户、有历史、功能全）

装法同书生 §9.2（单独一个 conda 环境，别装进 lmdeploy 那个）。起来后在「设置 → 连接」里加：

```
http://127.0.0.1:1222/v1
```

**两个模型可以都加进去**：再加一条 `http://127.0.0.1:1221/v1`（书生），key 各填各的，界面上直接切模型。

### 9.3 只想调试接口

把 `--disable-fastapi-docs` 去掉重启，访问平台地址会看到 Swagger UI。**调完记得加回去。**

---

## 10. 后台运行

### 10.1 tmux（推荐）

```bash
tmux new -s qwen
# 在新会话里跑启动命令，看到日志滚出来说明在加载
# 按 Ctrl+B，松手，再按 D —— 脱离会话，进程继续跑
```

| 操作 | 命令 |
|---|---|
| 回到会话 | `tmux attach -t qwen` |
| 看有哪些会话 | `tmux ls` |
| 关掉会话 | `tmux kill-session -t qwen` |

### 10.2 管理脚本（一条命令）

附件 [start-qwen.sh](./start-qwen.sh) 封装了 nohup 启动、PID 管理、日志跟随，命令和书生那份完全一样：

```bash
chmod +x start-qwen.sh
sed -i 's/\r$//' start-qwen.sh     # 从 Windows 传过去的话先转换行

./start-qwen.sh start          # 后台启动
./start-qwen.sh status         # 状态（显存占用、端口监听、接口自测、整卡占用）
./start-qwen.sh log            # 跟日志，Ctrl+C 不会停服务
./start-qwen.sh stop           # 停止
./start-qwen.sh restart        # 重启
./start-qwen.sh key            # 打印 API Key
./start-qwen.sh fg             # 前台跑，调试用
```

脚本里定好的同卡参数：`GPU=0`、`CACHE=0.6`、`SESSION_LEN=32768`、`PORT=1222`。`start` 和 `fg` 之前会先看一眼卡上空闲显存：不足 20 GiB 直接退出；整卡几乎全空（说明书生还没起）会提醒你先起书生；空闲不到 30 GiB 会提示本次 KV 池会变小。

两个服务的启动顺序：

```bash
cd ~/internlm && ./start-internlm.sh start     # 先起书生，等日志里 1221 开始监听
cd ~/qwen     && ./start-qwen.sh start         # 再起本服务
```

**两个服务都绑 GPU 0**（`GPU=0` 会写进 `CUDA_VISIBLE_DEVICES`），不要分卡：一张 80GB 卡塞得下两个 8B，分卡只会让两边都变小。

**API Key 不要手写进文档或脚本。** 脚本会自动生成一个 40 位随机串存在 `~/qwen/api.key`（权限 600），用 `./start-qwen.sh key` 取。也可以直接改这个文件手动指定 key。

改模型路径、端口、上下文长度、绑卡，编辑脚本开头的配置区，或者临时覆盖：

```bash
MODEL=/data/models/Qwen3-8B PORT=1222 GPU=0 CACHE=0.6 SESSION_LEN=16384 ./start-qwen.sh start
```

### 10.3 supervisor / systemd / 监控

照抄书生 §10.3 ~ §10.6，全部替换成 Qwen 的值：

| 位置 | 书生 | Qwen |
|---|---|---|
| 程序名 | `internlm` | `qwen` |
| lmdeploy 路径 | `/opt/conda/envs/internlm/bin/lmdeploy` | `/opt/conda/envs/qwen/bin/lmdeploy` |
| 模型目录 | `/data/models/internlm3-8b-instruct` | `/data/models/Qwen3-8B` |
| 端口 | 1221 | 1222 |
| 卡号 | `CUDA_VISIBLE_DEVICES="0"` | `"0"`（同一张卡） |
| `--cache-max-entry-count` | 0.4 | **0.6** |
| supervisor `priority` | 10 | **20** |
| `--trust-remote-code` | 有 | **删掉** |
| `--reasoning-parser` | 无 | **加上 `default`** |
| `--session-len` | 131072 | **32768** |

书生那份的 `--max-batch-size 32 --max-concurrent-requests 16` 照抄保留（限流靠它兜底）。

`priority` 数字小的先起，10/20 这一对保证书生先拿到它那 25 GiB 的 KV 池（§2.1）。systemd 那份（书生 §10.5）给 qwen 的 unit 加一行 `After=internlm.service`，同样是保顺序。

**开机自启**同书生 §10.4：容器里没有 init 系统，别在这上面花时间。

---

## 11. 补充

**MIG 切分**：`nvidia-smi -L` 出现 `MIG 3g.20gb` 说明卡被切分，按实例显存重算，且 `CUDA_VISIBLE_DEVICES` 要填 MIG UUID 而不是卡号。

**权重 4bit（AWQ）**：Qwen 官方直接发布了 AWQ 权重，**先下官方的，别自己量化**：

```bash
modelscope download --model Qwen/Qwen3-8B-AWQ --local_dir /data/models/Qwen3-8B-AWQ   # 5.68 GiB，2 个分片

CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server /data/models/Qwen3-8B-AWQ \
  --model-format awq --backend turbomind --tp 1 \
  --server-name 0.0.0.0 --server-port 1222 \
  --session-len 32768 --cache-max-entry-count 0.7 \
  --max-batch-size 32 --max-concurrent-requests 16 \
  --reasoning-parser default \
  --api-keys 换成你的长随机串 --disable-fastapi-docs
```

权重从 15.26 GiB 掉到 5.68 GiB，省出来的约 9.6 GiB 全进 KV 池：同卡时 Qwen 那份从约 13 GiB 涨到约 22 GiB，32K 并发从 2 路涨到 4 路，整卡仍余约 7 GiB（按 §2.1 的表算）。**注意**：量化会影响结构化输出的稳定性，开了 `response_format` 之后要重跑一遍 §7.6 的验证（和书生 §11 的提醒一致）。

`lmdeploy lite auto_awq` 对 Qwen3（2504）能不能跑没查到确凿依据，对 Qwen3.5 有已知报错，官方有现成的就别自己量化。

**FP8 别在 A100 上用**：A100 是 sm80，没有 FP8 计算单元，官方 FP8 权重要 compute capability > 8.9，A100 上只能走 vLLM 的降级路径。要省显存用 AWQ。

**工具调用**：chat template 自带 `<tool_call>` 协议，服务端加 `--tool-call-parser` 就能接 OpenAI 的 `tools` 字段，取值以 `lmdeploy serve api_server -h` 为准。

**换 vLLM**：`CUDA_VISIBLE_DEVICES=0 vllm serve /data/models/Qwen3-8B --port 1222 --enable-reasoning --reasoning-parser qwen3 --gpu-memory-utilization 0.45`。`--gpu-memory-utilization` 是**整张卡**的比例，同卡时必须按分给自己那份算（0.45 × 80 ≈ 36 GiB，vLLM 把权重、激活、KV 一起算，比 §2.1 里 Qwen 那份 15.26 权重 + 约 13 KV + 2 工作区 ≈ 30 GiB 多留约 6 GiB），填 0.9 会把书生那份一起吃掉。parser 名字和 LMDeploy 不一样（vLLM 是 `qwen3`，LMDeploy 是 `default`），平台侧端口映射完全一样。

**压测**：上线前用 lmdeploy 源码里的 `benchmark/profile_restful_api.py` 跑一次，记输出吞吐、请求吞吐、TTFT、TPOT。**思考/非思考两档要分开测**，思考模式下输出 token 数翻好几倍，混在一起测出来的数没参考价值。

**两个服务共用一张卡的算力**：显存是分开划的，SM 是抢的。单服务压出来的吞吐加不到一起，两个一起压一遍才是真实基线；高峰期让延迟敏感的那个优先。

---

## 12. 故障排查

| 症状 | 原因 | 处理 |
|---|---|---|
| 正文里混着 `<think>` 标签 | 没加 reasoning parser | 加 `--reasoning-parser default`，见 §7.1 |
| `--reasoning-parser default` 报不认识 | LMDeploy 版本旧，注册名不同 | `REASONING_PARSER=qwen3 ./start-qwen.sh start`（新版叫 `default`） |
| 关不掉思考 | 用了软开关 | `/no_think` 只是提示，用 §7.1 的硬开关 |
| 输出复读、停不下来 | 用了贪心解码 | `temperature=0` / `top_k=1` 官方明令禁止，按 §7.2 给参数 |
| 传 `chat_template_kwargs` 没反应 | lmdeploy 版本旧 | 0.7.3 及更早不支持 `enable_thinking`，升级 |
| 传了 `response_format` 但返回带 ``` 围栏 | 服务端忽略了该参数（版本旧） | `pip install -U lmdeploy`，见 §7.6 |
| 传了 `response_format` 报 400/500，日志提到 grammar | xgrammar 编译不了这个 schema | 退化成 `{"type":"json_object"}` + 客户端容错，见 §7.6 |
| JSON 被截断、字段缺一半 | 客户端没传 `max_tokens` | 显式传 4096，见 §7.5 |
| 结构化输出时好时坏 | `temperature` 太高 | 抽取任务降到 0.1，见 §7.5 |
| `CUDA out of memory` | KV 是书生的 3 倍 | `--cache-max-entry-count` 降到 0.4，`--session-len` 砍半，`--max-concurrent-requests` 降到 8 |
| 起本服务时显存分配失败 | 书生还没起（顺序反了），或两个 CACHE 之和超了 | 停掉两个，先起书生（0.4）再起 Qwen（0.6），见 §2.1 |
| 两个服务都在跑，整卡只剩几百 MiB | 比例配大了 | 把 Qwen 的 `CACHE` 降到 0.4 重启它（Qwen 那份是后算的，改它不影响书生） |
| 长材料请求直接 400 / `INPUT_LENGTH_ERROR` | 输入超过原生 40960 | 见 §2，调大 `--session-len` 不解决这个问题 |
| `Unrecognized model` | transformers 太旧 | `pip install -U transformers`（要 ≥ 4.51），**不是**加 `--trust-remote-code` |
| `Unsupported model` | lmdeploy 版本旧 | `pip install -U lmdeploy` |
| 两个服务都要显存 | 同一张卡上的两个服务，正常 | 按 §2.1 的比例和顺序启动；不够就降 `CACHE` / `SESSION_LEN` |
| 平台地址返回 502 | 转发的端口上没人监听 | ① 服务在跑吗 ② 端口号和平台登记的一致吗（1222） ③ `--server-name` 是 `0.0.0.0` 吗 |
| 平台地址打不开 / 超时 | 服务没起来，或端口没在平台开放 | `ss -lntp \| grep 1222` 确认在监听 |
| `Address already in use` | 端口被占 | `ss -lntp \| grep 1222`；确认没把 Qwen 也配成 1221 |
| 流式输出变成一次性返回 | 客户端没按 SSE 处理 | 先用 `curl -N` 验证；curl 正常就是前端问题 |
| 外网 401 | key 没传或不对 | 两个服务的 key 不同，别混用 |

---

## 13. 上线验收清单

**服务本身**

- [ ] `nvidia-smi` 正常，两个服务都在 GPU 0 上，没有互相挤
- [ ] `torch.cuda.is_available()` 返回 True
- [ ] 权重 `du -sh` 约 16G，5 个分片，§4 校验脚本 `data` 和 `total` 都是 16381470720
- [ ] 启动命令**没有** `--trust-remote-code`、**有** `--reasoning-parser default`、`--cache-max-entry-count` 是 0.6
- [ ] 启动顺序是书生在前、Qwen 在后
- [ ] `lmdeploy chat` 冒烟通过，中文正常
- [ ] 思考开关两档都试过：开（有 `reasoning_content`）、关（直接出正文）
- [ ] `--session-len` 按调用方最长输入定过，且不超过 40960
- [ ] 一次 32K 上下文请求正常返回
- [ ] 并发压测 30 分钟无 OOM，显存曲线不单调上涨

**同卡（两个服务在一张 80GB 上）**

- [ ] 两个进程都在 GPU 0 上，合计 70~75 GiB，空闲 ≥ 5 GiB
- [ ] 1221 / 1222 同时监听，两条平台地址都能流式返回
- [ ] 单独重启本服务不会失败（重启后按当时的空闲重算，池子比 13 GiB 略有出入属正常）

**结构化输出**（调用方要 JSON 时）

- [ ] §7.6 的验证脚本跑通，`ScenarioOutput` 能解析成功，且是**关思考**跑的
- [ ] 返回内容无 ``` 围栏、无前后解释文字
- [ ] 客户端已显式传 `max_tokens`（4096）和 `temperature`（0.1）
- [ ] 用一批真实业务样本批量跑过，统计失败率可接受
- [ ] 客户端解析器有容错兜底，服务端退化的预案就位

**安全**（平台给的地址没有登录态，这段最不能跳）

- [ ] `--api-keys` 已开，key 是长随机串，**且和书生的不同**（一个泄露不牵连另一个）
- [ ] `--disable-fastapi-docs` 已开
- [ ] Qwen 和书生各开各的端口（1222 / 1221），不用的端口没在平台开放
- [ ] `--max-concurrent-requests` 已设，不比书生的 16 高
- [ ] key 不在前端代码里、不在公开仓库里、没在群里发过的截图上

**运维**

- [ ] 平台生成的两个地址都能通，流式输出是逐字的
- [ ] 从容器外访问返回的不是 502
- [ ] 容器重启后两个服务都能快速拉起来，顺序没反（脚本名 / 会话名 / 启动顺序记在手边）
- [ ] 压测基线已归档，含完整启动命令和卡号
- [ ] 回滚方案就位（上一版模型目录和启动脚本都留着）

---

## 14. 参考

- [Qwen3 Quickstart（采样参数、thinking 开关）](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html)
- [LMDeploy Reasoning Outputs](https://lmdeploy.readthedocs.io/zh-cn/latest/llm/api_server_reasoning.html)
- [LMDeploy 结构化输出](https://lmdeploy.readthedocs.io/zh-cn/latest/advance/structed_output.html)
- [LMDeploy 命令行参数全集](https://lmdeploy.readthedocs.io/zh-cn/latest/api/cli.html)
- [LMDeploy KV 量化](https://lmdeploy.readthedocs.io/zh-cn/latest/quantization/kv_quant.html)
- [LMDeploy 支持模型列表](https://lmdeploy.readthedocs.io/zh-cn/latest/supported_models/supported_models.html)
- [ModelScope 权重仓库 Qwen/Qwen3-8B](https://www.modelscope.cn/models/Qwen/Qwen3-8B/summary)
- [ModelScope 权重仓库 Qwen/Qwen3-8B-AWQ](https://www.modelscope.cn/models/Qwen/Qwen3-8B-AWQ/summary)
- [Issue #4551 投机解码不支持引导解码](https://github.com/InternLM/lmdeploy/issues/4551)
- [Issue #4347 response_format 返回空结果](https://github.com/InternLM/lmdeploy/issues/4347)
- [Issue #2732 一个实例上跑多个服务时，按比例分配 KV 会受启动顺序影响](https://github.com/InternLM/lmdeploy/issues/2732)

---

*权重字节数、`config.json`、`generation_config.json`、chat template 均读自 ModelScope 仓库实际文件；KV 占用按 `2 × 层数 × KV head × head_dim × 2 字节` 计算。参数名以 `lmdeploy serve api_server -h` 实时输出为准。*
