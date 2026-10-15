# 书生 InternLM3-8B A100 80GB 部署手册（与 Qwen3-8B 同卡）

| 项目 | 说明 |
|---|---|
| 适用硬件 | NVIDIA A100 80GB，与 Qwen3-8B 共用这一张卡 |
| 适用系统 | Linux（Ubuntu 20.04/22.04、CentOS 7+） |
| 目标模型 | InternLM3-8B-Instruct |
| 推荐引擎 | LMDeploy + TurboMind |
| 访问方式 | 平台「开放端口」（服务名 + 端口），公网 HTTPS 由平台提供 |
| 主要调用方 | `content-security-llm`（要 JSON，输入上限 20 万字符） |
| 同卡伙伴 | Qwen3-8B，端口 1222，脚本 `start-qwen.sh` |

路径统一用 `/data/models`，端口统一用 `1221`，替换成你自己的。

两个服务都在 GPU 0 上，不分卡。**先把 §2.2 的显存分账和 §10.2 的启动顺序看完再动手**——这两件事定了，后面全是照抄命令。

---

## 0. 决策表

| 你的场景 | 模型 | 启动要点 |
|---|---|---|
| 1×A100 80GB，与 Qwen3-8B 同卡（本案） | InternLM3-8B，BF16 | tp=1，GPU 0，session-len 131072，cache 0.4，**第一个启动** |
| 1×A100 80GB，整卡只跑书生 | InternLM3-8B，BF16 | session-len 131072，cache 0.8，batch 可提到 64 |
| 同卡，并发还要更高 | InternLM3-8B，AWQ 4bit | 权重降到约 5GB，省出来的全进 KV 池，见 §11 |
| 要接 20 万字符的长输入 | 见 §7.2 | 中文按约 1 字 1 token 算，131072 装的是 10 万字符，业务上限要跟着降 |

---

## 1. 上机体检

```bash
nvidia-smi
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # 卡上有没有别的进程，两个服务都在这张卡上

free -g          # 建议 ≥ 32GB
df -h /data      # 两个模型的权重合计约 34GB，留 100GB 以上
python3 --version   # 3.10~3.12
```

| 现象 | 处理 |
|---|---|
| `compute_cap` 是 8.0 | 确认 A100，本文全部适用 |
| `memory.total` 是 81920 MiB | 80GB 卡，同卡跑两个服务够用，按 §2.2 分 |
| `memory.total` 是 40960 MiB | 40GB 卡，光两个 8B 权重就 31.7 GiB，同卡放不下，得加卡或给两边都上 AWQ |
| `nvidia-smi -L` 出现 MIG 字样 | 卡被切分，按实例显存重算，见 §11 |
| 驱动 < 525 | 不支持 CUDA 12，联系运维升级 |

---

## 2. 显存账

### 2.1 书生自己占多少

InternLM3-8B-Instruct 的真实配置（读自仓库 `config.json`）：

```
48 层 ｜ hidden 4096 ｜ 32 个 attention head ｜ 2 个 KV head ｜ head_dim 128
词表 128512 ｜ 原生 32K 上下文（dynamic RoPE factor 6.0，可外推到 196608）｜ bfloat16
参数量约 8.8B，BF16 权重 17,608,482,816 字节 = 16.40 GiB（两个 safetensors：9.25 + 7.15 GiB）
```

KV 开销（48 层但只有 2 个 KV head，GQA 压得很狠）：

```
每 token = 2 × 48 层 × 2 KV head × 128 head_dim × 2 字节 = 48 KiB
```

| 上下文 | 单请求 KV |
|---|---|
| 8K | 384 MiB |
| 32K | 1.5 GiB |
| 131072 | 6 GiB |

### 2.2 一张 A100 80GB 上两个服务怎么分

`--cache-max-entry-count` 是**权重加载完之后剩余空闲显存的比例**（官方定义：percentage of free gpu memory occupied by the k/v cache, excluding weights），TurboMind 启动时按这个比例把 KV block 池一次划走、常驻显存。谁先起谁按整卡的空闲算，比例和顺序配错了，后起的那个就分不到。

| 项 | GiB |
|---|---|
| 书生 BF16 权重 | 16.40 |
| Qwen3-8B BF16 权重 | 15.26 |
| 两个进程的 CUDA context + 工作区（每个留 2） | 4 |
| 书生 KV 池（cache 0.4，先启动） | 约 25 |
| Qwen KV 池（cache 0.6，后启动） | 约 13 |
| **合计** | **约 74 / 80 GiB，余 6** |

按这个分法，两边各能扛：

| 单请求上下文 | 书生 KV | 书生池 25 GiB | Qwen KV | Qwen 池 13 GiB |
|---|---|---|---|---|
| 8K | 384 MiB | 16 路（撞 `--max-concurrent-requests` 上限） | 1.13 GiB | 11 路 |
| 32K | 1.5 GiB | 16 路（同上） | 4.5 GiB | 2 路（第 3 路要 13.5 GiB） |
| 满长（书生 131072 / Qwen 32768） | 6 GiB | 4 路 | 4.5 GiB | 2 路 |

顺序：**先起书生（cache 0.4），等 1221 监听，再起 Qwen（cache 0.6）**。反过来的话 Qwen 的 0.6 会从约 63 GiB 的空闲里划走约 38 GiB，书生再起时只剩约 6 GiB，0.4 只能划出 2~3 GiB 的 KV 池——进程起得来，但满长请求装不下，等于白起。

起完对账：

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
```

书生那个进程 41~44 GiB，Qwen 29~31 GiB，合计 70~75 GiB，空闲 ≥ 5 GiB。对不上看 §12。

整卡只跑书生（不部署 Qwen）时，cache 给 0.8：权重 16.40 + KV 约 50 GiB。

---

## 3. 环境准备

```bash
conda create -n internlm python=3.12 -y
conda activate internlm
pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install lmdeploy -i https://pypi.tuna.tsinghua.edu.cn/simple

# 自检
lmdeploy check_env
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

期望看到 `True` 和 `NVIDIA A100-...`。

结构化输出所需的 `xgrammar` 已随 lmdeploy 自动安装。

**Qwen 那边另开一个 conda 环境，别和这个共用。** 两边都是 lmdeploy，版本升级节奏不同，一个环境的升级把另一个服务搞挂，排查很费时间。

---

## 4. 下载权重

```bash
pip install modelscope
modelscope download \
  --model Shanghai_AI_Laboratory/internlm3-8b-instruct \
  --local_dir /data/models/internlm3-8b-instruct
```

内网机器离线搬运，到达后必须校验（半个文件要到加载阶段才暴露）：

```bash
du -sh /data/models/internlm3-8b-instruct          # 约 17G（16.40 GiB 权重 + tokenizer 等）
ls /data/models/internlm3-8b-instruct/*.safetensors | wc -l   # 2

python - <<'EOF'
import json, os, struct
p = '/data/models/internlm3-8b-instruct'
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

`data` 和 `total` 都应该是 **17608482816**。

**别拿「文件大小之和」比 `total_size`**——safetensors 每个文件前面有 JSON 头，这两个分片的头加起来 50,624 字节，文件大小之和是 17,608,533,440。直接比会在权重完好的情况下报「不一致」，白重传 17GB。上面这段减掉头再比。

---

## 5. `--trust-remote-code` 必加

不加会直接报：

```
ValueError: The repository /data/models/internlm3-8b-instruct contains custom code
which must be executed to correctly load the model.
Please pass the argument `trust_remote_code=True` to allow custom code to be run.
```

**原因**：InternLM3 把三份实现随权重放在仓库里，transformers 主干没有：

| 文件 | 声明位置 |
|---|---|
| `tokenization_internlm3.py` | `tokenizer_config.json` → `auto_map.AutoTokenizer` |
| `configuration_internlm3.py` | `config.json` → `auto_map.AutoConfig` |
| `modeling_internlm3.py` | `config.json` → `auto_map.AutoModelForCausalLM` |

`from_pretrained()` 要执行的类不在自己代码库里，出于安全默认拒绝。LMDeploy 早期版本默认当 True，[后来改成显式开关](https://github.com/InternLM/lmdeploy/pull/4511)，所以老教程不写也能跑，新版本就报错。

**各入口写法**：LMDeploy / vLLM / SGLang 都是 `--trust-remote-code`；Python 代码里是 `trust_remote_code=True`。

**安全**：这个参数会让模型目录的 Python 以服务进程权限执行。官方仓库权重可以放心开，来路不明的先看一眼那几个 `.py`。

---

## 6. 启动服务

先冒烟，确认能加载、模板对得上：

```bash
lmdeploy chat /data/models/internlm3-8b-instruct --backend turbomind --trust-remote-code
```

**这里必须绑 `0.0.0.0`，不能绑 `127.0.0.1`。** 平台的端口映射在容器外做转发，只监听回环地址的话外网流量进不来，平台会返回 502。

`--server-port` 填你在平台上开放的那个端口。下面示例用 `1221`，换成你自己登记的那个号。**同卡部署时这条命令第一个跑**，Qwen 等它监听 1221 之后再起（§2.2）。

```bash
CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server /data/models/internlm3-8b-instruct \
  --server-name 0.0.0.0 \
  --server-port 1221 \
  --backend turbomind \
  --tp 1 \
  --session-len 131072 \
  --cache-max-entry-count 0.4 \
  --max-batch-size 32 \
  --max-concurrent-requests 16 \
  --trust-remote-code \
  --api-keys gzhu-yyds \
  --disable-fastapi-docs \
  --log-level INFO
```

整卡只跑书生时 `--cache-max-entry-count` 提到 0.8、`--max-batch-size` 提到 64；和 Qwen 同卡就按 §2.2 用 0.4，别在这里贪大。

### 参数速查

| 参数 | 作用 | 调整方向 |
|---|---|---|
| `--server-name` | 监听地址 | **必须 `0.0.0.0`**，绑回环地址平台转发不进来 |
| `--server-port` | 监听端口 | 与平台上开放的端口号一致 |
| `--tp` | 张量并行卡数 | 单卡装不下权重时才加 |
| `--session-len` | 单请求最大上下文 | **按调用方最长输入定**，本案 131072；OOM 时先砍它 |
| `--cache-max-entry-count` | KV 占空闲显存比例 | 同卡固定 0.4（本服务先启动）；整卡单跑可到 0.8；OOM 再降到 0.3，见 §2.2 |
| `--max-batch-size` | 单批最大请求数 | 长上下文场景别贪大 |
| `--max-concurrent-requests` | 并发上限，超出排队 | 没有反向代理限流，靠它兜底 |
| `--api-keys` | 服务鉴权 | **公网可达，必开** |
| `--disable-fastapi-docs` | 关掉 Swagger UI | **公网可达，必开** |
| `--quant-policy` | **KV** 量化：4=int4，8=int8 | 并发不够时用 8 |

`--quant-policy` 只管 KV cache，不管权重。

### 本地验证

```bash
curl -s http://127.0.0.1:1221/v1/models -H "Authorization: Bearer 你的KEY" | python -m json.tool
```

---

## 7. 结构化输出（业务系统要 JSON 时必看）

调用方传 `response_format` 要 JSON（`content-security-llm` 就是），这一节决定成败。

### 7.1 支持情况

| 问题 | 答案 |
|---|---|
| 要额外装 xgrammar 吗 | **不用**，随 lmdeploy 自动安装 |
| TurboMind 支持吗 | **支持** |
| 要加启动参数吗 | **不用**，`response_format` 是请求级参数 |
| 支持哪些格式 | `json_schema`、`regex_schema`、`structural_tag` |

### 7.2 `--session-len`

取值 **131072** —— 模型卡 RULER 验证过的上限。

131072 个 token 按中文约 1 字 1 token 算，装得下 10 万字符的输入加 4096 的输出；英文按约 4 字符一个 token 算，20 万字符也进得来。建议把 `analysis_contracts.py` 里的 `200000` 上限降到 `100000`，中英混排的输入才稳。

KV 占用：48 KiB/token × 131072 = **6 GiB/满长请求**，同卡部署时 25 GiB 的池子能同时装 4 路（§2.2）。

### 7.3 上线前必须验证 schema 能不能编译

`ScenarioOutput` 是最难的情况：discriminated union、两层嵌套、`Literal` 枚举、`extra="forbid"`。**必须实测。**

在你自己电脑上跑（打服务器地址）：

```python
# verify_schema.py
import json, sys
from openai import OpenAI

sys.path.insert(0, r"D:\Projects\Pycharm\content-security-llm")
from llm.analysis_contracts import ScenarioOutput

schema = ScenarioOutput.model_json_schema()
print("schema 大小:", len(json.dumps(schema)), "字节 | $defs 数量:", len(schema.get("$defs", {})))

client = OpenAI(api_key="你的KEY", base_url="https://<平台地址>/v1")
model = client.models.list().data[0].id

resp = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "你是内容安全分析助手，按给定的 JSON Schema 输出。"},
        {"role": "user", "content": "分析这条消息：某地发生群体性事件，网上流传大量不实信息。"},
    ],
    temperature=0.1,
    max_tokens=4096,
    response_format={"type": "json_schema",
                     "json_schema": {"name": "ScenarioOutput", "schema": schema}},
)

choice = resp.choices[0]
content = choice.message.content
print("finish_reason:", choice.finish_reason, "| 返回长度:", len(content))

try:
    obj = ScenarioOutput.model_validate_json(content)
    print("✅ schema 约束生效，解析成功")
    print(json.dumps(obj.model_dump(), ensure_ascii=False)[:400])
except Exception as e:
    print("❌ 解析失败:", type(e).__name__)
    print("原始返回前 600 字:\n", content[:600])
```

结果怎么读：

| 现象 | 说明 | 下一步 |
|---|---|---|
| `✅ schema 约束生效` | 约束解码工作正常 | 只改客户端代码 |
| 无围栏、无废话，但字段报错 | schema 编译成功，是内容问题 | 查业务校验器 |
| 返回带 ``` 围栏或前后有解释 | **`response_format` 被忽略了** | `pip install -U lmdeploy` |
| 400/500，日志说 grammar 编译失败 | xgrammar 啃不动这个 schema | 退化成 `{"type": "json_object"}`（只保证是合法 JSON），靠客户端解析器容错 |
| TurboMind 报错 | 后端问题 | 换 `--backend pytorch` 再测一次，两边都支持 |

### 7.4 两个已知限制

**① 投机解码不支持引导解码。** [Issue #4551](https://github.com/InternLM/lmdeploy/issues/4551) 明确说明两者同时使用会有未定义行为。本文配置没开投机解码，保持现状即可。

**② `response_format` 历史上有边角 bug。** [Issue #4347](https://github.com/InternLM/lmdeploy/issues/4347) 是 `client.beta.chat.completions.parse` 配 `response_format` 返回空结果的案例，根因是模型的 CoT 文本跑进了不该去的字段。场景不同（那是 `parse`，本案用的是 `create`），但说明这条链路不能盲信 —— §7.3 的验证不能省。

### 7.5 客户端必须配合改的三处

| 位置 | 取值 |
|---|---|
| 请求参数 | `response_format={"type":"json_schema","json_schema":{"name":类名,"schema":model_json_schema()}}` |
| `max_tokens` | **4096** |
| `temperature` | **0.1** |

---

## 8. 使用服务名进行访问

![image-20261010160234111](./InternLM-A100部署手册/image-20261010160234111.png)

平台用的是「开放端口」机制：给容器实例登记**服务名**和**容器内端口**，平台自动生成一个公网 HTTPS 地址，格式是

```
https://<实例ID>-<服务名>.svc.bitahub.com:20443
```

`20443` 是平台统一的对外端口。TLS 由平台终止、证书也是平台的，不用自己申请。

给 InternLM 新开一个：服务名 `internlm3web`、端口 `1221`。**服务名 + 端口这两项组合起来唯一确定一个对外入口**，端口别和实例上其它服务撞车（Qwen 占的是 1222）。

## 9. Web 界面

LMDeploy 本身**不带聊天界面**。`lmdeploy serve api_server` 起的是一个纯 API 服务。

> 老版本有 `lmdeploy serve gradio`，**当前版本已删除**，`lmdeploy serve` 只剩 `api_server` 和 `proxy` 两个子命令，照老教程敲会直接报错。

### 9.1 方案一：单文件 HTML 页面（零安装，最快）

附件 [internlm-chat.html](./internlm-chat.html) 是一个自带流式输出、Markdown 渲染、多轮上下文的单文件聊天页。没有任何依赖，也不用部署：

1. 把 `internlm-chat.html` 下载到你自己电脑
2. 双击用浏览器打开（Chrome / Edge 都行）
3. 展开顶部「连接设置」，填三样：
   - **服务地址**：平台控制台复制的完整地址，形如 `https://<实例ID>-<服务名>.svc.bitahub.com:20443`
   - **API Key**：启动 LMDeploy 时 `--api-keys` 设的那个
   - **模型名**：先随便填，点「测试连接」它会告诉你真实的名字
4. 点「测试连接」确认通了，然后就能聊

配置存在浏览器 localStorage 里，下次打开不用重填；key 只发给你自己的服务地址，不经过任何第三方。

**要给同事用**：把这个 html 丢到任意静态托管（对象存储、GitHub Pages、任意 web 服务器）就行。它只做前端，真实请求是浏览器直连平台地址。

### 9.2 方案二：Open WebUI（多用户、有历史、功能全）

要正经的产品级体验（多用户、会话历史、知识库、模型切换）就上这个。[InternLM 官方推荐的就是它](https://internlm.readthedocs.io/zh-cn/latest/application/openwebui.html)。

容器实例里通常没有 Docker 守护进程，所以用 pip 装：

```bash
conda create -n webui python=3.11 -y
conda activate webui
pip install open-webui -i https://pypi.tuna.tsinghua.edu.cn/simple

# 绑 0.0.0.0，另开一个端口（比如 1227），记得去平台开放它
open-webui serve --host 0.0.0.0 --port 1227
```

**别装进 lmdeploy 那个环境。** Open WebUI 对 fastapi、pydantic 的版本要求和 lmdeploy 会打架，单开一个 conda 环境最省心。

起来之后在平台开放 1227，用平台给的外网地址访问，再到 Open WebUI 的「设置 → 连接」里把 OpenAI API 地址填成：

```
http://127.0.0.1:1221/v1      # 书生
http://127.0.0.1:1222/v1      # Qwen，key 各填各的，界面上直接切模型
```

同一台机器上两个服务走回环地址互通，不用绕公网。首次启动会下载一些文件，比较慢，多等一会儿。

### 9.3 只想调试接口

那就把 `--disable-fastapi-docs` 去掉，重启后访问平台地址会看到 Swagger UI，可以直接在网页上发 `/v1/chat/completions`。

**调完记得加回去**——公网上挂着接口文档等于把攻击面白送。

---

## 10. 后台运行

### 10.1 tmux（推荐）

```bash
tmux new -s internlm
# 在新会话里跑启动命令，看到日志滚出来说明在加载
# 按 Ctrl+B，松手，再按 D —— 脱离会话，进程继续跑
```

| 操作 | 命令 |
|---|---|
| 回到会话 | `tmux attach -t internlm` |
| 看有哪些会话 | `tmux ls` |
| 关掉会话 | `tmux kill-session -t internlm` |

好处是随时能回去看日志，排查最方便。

### 10.2 管理脚本（一条命令）

附件 [start-internlm.sh](./start-internlm.sh) 封装了 nohup 启动、PID 管理、日志跟随：

```bash
chmod +x start-internlm.sh
sed -i 's/\r$//' start-internlm.sh     # 从 Windows 传过去的话先转换行

./start-internlm.sh start      # 后台启动
./start-internlm.sh status     # 状态（显存占用、端口监听、接口自测、整卡占用）
./start-internlm.sh log        # 跟日志，Ctrl+C 不会停服务
./start-internlm.sh stop       # 停止
./start-internlm.sh restart    # 重启
./start-internlm.sh key        # 打印 API Key
./start-internlm.sh fg         # 前台跑，调试用
```

脚本里定好的同卡参数：`GPU=0`、`CACHE=0.4`、`SESSION_LEN=131072`、`PORT=1221`。`start` 和 `fg` 之前会先看一眼卡上空闲显存：不足 20 GiB 直接退出，空闲低于 60 GiB（说明卡上已经有别的进程）会提醒你确认启动顺序。

两个服务的启动顺序：

```bash
cd ~/internlm && ./start-internlm.sh start     # 先起这个，等日志里 1221 开始监听
cd ~/qwen     && ./start-qwen.sh start         # 再起 Qwen
```

**API Key 不要手写进文档或脚本。** 脚本会自动生成一个 40 位随机串存在 `~/internlm/api.key`（权限 600），用 `./start-internlm.sh key` 取。也可以直接改这个文件手动指定 key。

改模型路径、端口、上下文长度、绑卡，编辑脚本开头的配置区，或者临时覆盖：

```bash
MODEL=/data/models/internlm3-8b-instruct PORT=1221 SESSION_LEN=131072 CACHE=0.4 ./start-internlm.sh start
```

### 10.3 supervisor（要进程崩了自动拉起）

上面两种方式进程挂了不会自动重启。要这个能力就上 supervisor：

```bash
pip install supervisor
mkdir -p /etc/supervisor/conf.d /root/internlm
```

`/etc/supervisor/conf.d/internlm.conf`：

```ini
[program:internlm]
command=/opt/conda/envs/internlm/bin/lmdeploy serve api_server /data/models/internlm3-8b-instruct --server-name 0.0.0.0 --server-port 1221 --backend turbomind --tp 1 --session-len 131072 --cache-max-entry-count 0.4 --max-batch-size 32 --max-concurrent-requests 16 --trust-remote-code --api-keys 换成你的KEY --disable-fastapi-docs
directory=/root
autostart=true
autorestart=true
priority=10
startsecs=120
startretries=3
stopasgroup=true
killasgroup=true
environment=CUDA_VISIBLE_DEVICES="0"
stdout_logfile=/root/internlm/out.log
stderr_logfile=/root/internlm/err.log
```

```bash
supervisord -c /etc/supervisor/supervisord.conf     # 启动守护进程
supervisorctl status
supervisorctl restart internlm
```

`startsecs=120` 是关键。加载权重要几分钟，这段时间进程还没进入稳定状态，默认 1 秒会让 supervisor 误判启动失败然后疯狂重启。

`priority=10` 和 Qwen 那份 conf 里的 `priority=20` 配套：数字小的先起，保证书生先拿到它那 25 GiB 的 KV 池（§2.2）。

supervisor 自己也不是 PID 1，容器重启后同样要手动拉起。它的价值是**进程崩溃时自动重启**，不是开机自启。

### 10.4 关于"开机自启"

容器实例没有 init 系统，也就没有传统意义的开机自启。三个现实选项：

1. 把启动命令塞进平台自定义镜像的入口脚本（如果平台支持）
2. 容器重启后手动 `./start-internlm.sh start`，十秒钟的事
3. 平台外挂探活脚本，检测端口不通就 SSH 进来拉起

别在这上面花太多时间，容器本来就不是为常驻服务设计的。

### 10.5 附：迁到虚拟机后的 systemd 配置

哪天从容器迁到正经的物理机 / 虚拟机，这份可以用。**第一行必须是 `[Unit]`**，漏了就会报 `Assignment outside of section. Ignoring.`：

```ini
[Unit]
Description=InternLM3-8B API Server (LMDeploy)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=appuser
Group=appuser
EnvironmentFile=-/etc/internlm.env
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="PATH=/opt/conda/envs/internlm/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/opt/conda/envs/internlm/bin/lmdeploy serve api_server \
  /data/models/internlm3-8b-instruct \
  --server-name 0.0.0.0 --server-port 1221 \
  --backend turbomind --tp 1 \
  --session-len 131072 --cache-max-entry-count 0.4 \
  --max-batch-size 32 --max-concurrent-requests 16 \
  --trust-remote-code \
  --api-keys $INTERNLM_API_KEY \
  --disable-fastapi-docs \
  --log-level INFO
Restart=always
RestartSec=15
LimitNOFILE=65535
TimeoutStartSec=900

[Install]
WantedBy=multi-user.target
```

`/etc/internlm.env`（`chmod 600`）：

```
INTERNLM_API_KEY=换成你的长随机串
```

```bash
sudo chmod 600 /etc/internlm.env
sudo systemctl daemon-reload
sudo systemctl enable --now internlm
journalctl -u internlm -f
```

两个坑：

- `TimeoutStartSec=900` 不能省。加载权重几分钟属正常，systemd 默认 90 秒会把它杀掉然后无限重启。
- `$INTERNLM_API_KEY` 由 systemd 展开（不是 shell），变量必须来自 `Environment=` 或 `EnvironmentFile=`。

### 10.6 监控

```bash
curl -s http://127.0.0.1:1221/metrics | head -50
```

| 指标 | 判断 |
|---|---|
| `lmdeploy:gpu_cache_usage_perc` 长期贴 1.0 | 该加卡或开 KV 量化 |
| 队列持续排队 | 并发打满 |

两个服务各有自己的 `/metrics`（1221 / 1222），同卡时两个都要看，别只看一个。

官方 Prometheus + Grafana 套件在 lmdeploy 源码的 `lmdeploy/monitoring`，`docker compose up` 起。

---

## 11. 补充

**MIG 切分**：`nvidia-smi -L` 出现 `MIG 3g.20gb` 说明卡被切分，按实例显存重算，且 `CUDA_VISIBLE_DEVICES` 要填 MIG UUID 而不是卡号。

**权重 4bit（AWQ）**：权重从 16.40 GiB 降到约 5GB，省出来的全进 KV 池。同卡时按下面命令里的 cache 0.5 启动，本服务那份从约 25 GiB 涨到约 37 GiB，满长（131072）并发从 4 路涨到 6 路；后起的 Qwen 那份会从约 13 GiB 变成约 11 GiB，还是 2 路 32K 的量级。

```bash
lmdeploy lite auto_awq /data/models/internlm3-8b-instruct \
  --work-dir /data/models/internlm3-8b-awq \
  --calib-dataset wikitext2 --calib-samples 128 \
  --batch-size 1 --trust-remote-code

CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server /data/models/internlm3-8b-awq \
  --model-format awq --backend turbomind --tp 1 \
  --server-name 0.0.0.0 --server-port 1221 \
  --session-len 131072 --cache-max-entry-count 0.5 --trust-remote-code \
  --api-keys 换成你的长随机串 --disable-fastapi-docs
```

量化后拿真实业务问题跟 BF16 版对比一遍再上线。**注意**：量化会影响结构化输出的稳定性，开了 `response_format` 之后要重新跑一遍 §7.3 的验证。

**换 vLLM**：`--gpu-memory-utilization` 是**整张卡**的比例，同卡部署时必须按分给本服务的那份算，不能照抄教程里的 0.9：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /data/models/internlm3-8b-instruct \
  --served-model-name internlm3-8b --max-model-len 131072 \
  --gpu-memory-utilization 0.5 --trust-remote-code \
  --api-key 换成你的长随机串 \
  --host 0.0.0.0 --port 1221
```

0.5 × 80 GiB = 40 GiB（权重 16.4 + 激活 + KV 约 22），和 §2.2 里书生那份是一个量级；填 0.9 会把 Qwen 那份一起吃掉。平台侧的端口映射完全一样。

**压测**：上线前用 lmdeploy 源码里的 `benchmark/profile_restful_api.py --backend lmdeploy --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json` 跑一次，把输出吞吐、请求吞吐、TTFT、TPOT 四个数记下来，作为以后调参的对比基线。

**两个服务共用一张卡的算力**：显存是分开划的，SM 是抢的。单服务压出来的吞吐加不到一起，两个一起压一遍才是真实基线；高峰期让延迟敏感的那个优先。

---

## 12. 故障排查

| 症状 | 原因 | 处理 |
|---|---|---|
| `ValueError: ... contains custom code` | 没授权执行仓库里的自定义代码 | 加 `--trust-remote-code`，见 §5 |
| `CUDA out of memory` | KV 比例高 / 上下文长 / 并发多 | `--cache-max-entry-count` 降到 0.3，`--session-len` 砍半；同卡时先看 §2.2 的分账 |
| 第二个服务起不来，日志里是显存分配失败 | CACHE 之和超过整卡空闲，或者启动顺序反了 | 停掉两个，先起书生（0.4）再起 Qwen（0.6），见 §2.2 |
| 两个服务都在跑，整卡只剩几百 MiB | 比例配大了 | 把 Qwen 的 `CACHE` 降到 0.4 重启它（Qwen 那份是后算的，改它不影响书生） |
| 启动卡在加载阶段几分钟 | 正常，首次尤其慢 | `./start-internlm.sh log` 或 `tmux attach` 看进度输出 |
| `Unsupported model` | lmdeploy 版本旧 | `pip install -U lmdeploy` |
| 输出复读、格式错乱 | chat 模板没匹配上 | 模型目录名保持规范；或显式 `--chat-template` |
| 中文回答变差 | 没传 system prompt | 客户端补上官方 system prompt |
| 平台地址返回 502 | 转发的端口上没人监听 | ① 服务在跑吗 ② 端口号和平台登记的一致吗 ③ `--server-name` 是 `0.0.0.0` 吗 |
| 平台地址打不开 / 超时 | 服务没起来，或端口没在平台开放 | `ss -lntp \| grep 端口号` 确认在监听 |
| 流式输出变成一次性返回 | 客户端没按 SSE 处理 | 先用 `curl -N` 验证；curl 正常就是前端问题 |
| 浏览器控制台报 CORS | `--allow-origins` 被改窄了 | 保持默认不限，或把你的页面来源加进白名单 |
| 外网 401 | key 没传或不对 | 检查 `Authorization: Bearer` 头 |
| `Address already in use` | 端口被占 | `ss -lntp \| grep 1221` |
| 多卡 NCCL 超时 | 网卡 / NVLink / 防火墙 | `export NCCL_IB_DISABLE=1`，`nvidia-smi topo -m` 查拓扑 |
| 响应慢但显存没满 | 请求被串行化 | 调大 `--max-concurrent-requests` |
| 传了 `response_format` 但返回带 ``` 围栏 | 服务端忽略了该参数（版本旧） | `pip install -U lmdeploy`，见 §7.3 |
| 传了 `response_format` 报 400/500，日志提到 grammar | xgrammar 编译不了这个 schema | 退化成 `{"type":"json_object"}` + 客户端容错解析，见 §7.3 |
| 长材料请求直接 400 / `INPUT_LENGTH_ERROR` | `--session-len` 太小 | 开到 131072，并把业务输入上限降到 10 万字符，见 §7.2 |
| JSON 被截断、字段缺一半 | 客户端没传 `max_tokens` | 显式传 4096，见 §7.5 |
| 结构化输出时好时坏 | `temperature` 太高 | 抽取任务降到 0.1，见 §7.5 |

---

## 13. 上线验收清单

**服务本身**

- [ ] `nvidia-smi` 正常，卡片是 80GB，无意外 MIG 切分
- [ ] `torch.cuda.is_available()` 返回 True
- [ ] 权重 `du -sh` 约 17G，index 校验输出 `17608482816` 一致
- [ ] 启动命令带 `--trust-remote-code`，`--cache-max-entry-count` 是 0.4
- [ ] `lmdeploy chat` 冒烟通过，中文正常
- [ ] `--session-len` 已按调用方最长输入设定（本案 131072）
- [ ] 一次满长上下文请求（接近 12 万 token）正常返回
- [ ] 并发压测 30 分钟无 OOM，显存曲线不单调上涨

**同卡（书生 + Qwen 在一张 80GB 上）**

- [ ] 启动顺序是书生在前、Qwen 在后，容器重启后也按这个顺序拉
- [ ] 两个进程都在 GPU 0 上，`nvidia-smi` 里合计 70~75 GiB，空闲 ≥ 5 GiB
- [ ] 1221 / 1222 同时监听，两个平台地址都能流式返回
- [ ] 单独重启其中一个不会失败（重启后面那个池子会按当时的空闲重算，比 §2.2 的数小一点，属正常）

**结构化输出**（调用方要 JSON 时）

- [ ] §7.3 的验证脚本跑通，`ScenarioOutput` 能解析成功
- [ ] 返回内容无 ``` 围栏、无前后解释文字
- [ ] 客户端已显式传 `max_tokens`（4096）和 `temperature`（0.1）
- [ ] 用一批真实业务样本批量跑过，统计失败率可接受
- [ ] 客户端解析器有容错兜底（剥围栏、提取首个完整 JSON），服务端退化的预案就位

**安全**（平台给的地址没有登录态，这段最不能跳）

- [ ] `--api-keys` 已开，key 是长随机串，不是 `123456`
- [ ] `--disable-fastapi-docs` 已开（Swagger 挂在公网上等于把接口清单送给别人）
- [ ] 不用的端口没在平台开放；下线的服务记得把端口映射删掉
- [ ] `--max-concurrent-requests` 已设，防止单个客户端占满显卡
- [ ] key 不在前端代码里、不在公开仓库里、没在群里发过的截图上

**运维**

- [ ] 平台生成的地址能通，流式输出是逐字的
- [ ] 从容器外访问返回的不是 502
- [ ] 容器重启后两个服务都能快速拉起来，且顺序没反（脚本 / tmux 会话名 / 启动顺序记在手边）
- [ ] 压测基线已归档，含完整启动命令
- [ ] 回滚方案就位（上一版模型目录和启动脚本都留着）

---

## 14. 参考

- [InternLM3 模型卡](https://internlm.readthedocs.io/zh-cn/latest/model_card/InternLM3.html)
- [LMDeploy 安装](https://lmdeploy.readthedocs.io/zh-cn/latest/get_started/installation.html)
- [LMDeploy OpenAI 服务](https://lmdeploy.readthedocs.io/zh-cn/latest/llm/api_server.html)
- [LMDeploy 命令行参数全集](https://lmdeploy.readthedocs.io/zh-cn/latest/api/cli.html)
- [LMDeploy 结构化输出](https://lmdeploy.readthedocs.io/zh-cn/latest/advance/structed_output.html)
- [LMDeploy KV 量化](https://lmdeploy.readthedocs.io/zh-cn/latest/quantization/kv_quant.html)
- [LMDeploy 支持模型列表](https://lmdeploy.readthedocs.io/zh-cn/latest/supported_models/supported_models.html)
- [ModelScope 权重仓库](https://www.modelscope.cn/models/Shanghai_AI_Laboratory/internlm3-8b-instruct/summary)
- [PR #3965 TurboMind 引导解码](https://github.com/InternLM/lmdeploy/pull/3965)
- [Issue #4551 投机解码不支持引导解码](https://github.com/InternLM/lmdeploy/issues/4551)
- [Issue #4347 response_format 返回空结果](https://github.com/InternLM/lmdeploy/issues/4347)
- [Issue #2732 一个实例上跑多个服务时，按比例分配 KV 会受启动顺序影响](https://github.com/InternLM/lmdeploy/issues/2732)

---

*权重大小与 config 参数取自 ModelScope 仓库实际文件清单，KV 占用按官方 config 计算。参数名以 `lmdeploy serve api_server -h` 实时输出为准。*
