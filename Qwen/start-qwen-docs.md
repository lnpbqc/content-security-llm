##  后台运行

### tmux

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

### 管理脚本

附件 [start-qwen.sh](./start-qwen.sh) 封装了 nohup 启动、PID 管理、日志跟随，命令和书生那份完全一样：

```bash
chmod +x start-qwen.sh
sed -i 's/\r$//' start-qwen.sh     # 从 Windows 传过去的话先转换行

./start-qwen.sh start      # 后台启动
./start-qwen.sh status     # 状态（显存占用、端口监听、接口自测、整卡占用）
./start-qwen.sh log        # 跟日志，Ctrl+C 不会停服务
./start-qwen.sh stop       # 停止
./start-qwen.sh restart    # 重启
./start-qwen.sh key        # 打印 API Key
./start-qwen.sh fg         # 前台跑，调试用
```

脚本会自动生成一个 40 位随机串存在 `~/qwen/api.key`（权限 600），用 `./start-qwen.sh key` 取。可以改 `~/qwen/api.key` 来手动指定 apikey。

改模型路径、端口、上下文长度、KV 比例，编辑脚本开头的「配置区」，或者用环境变量临时覆盖：

```bash
MODEL=/data/models/Qwen3-8B PORT=1222 SESSION_LEN=16384 CACHE=0.6 ./start-qwen.sh start
```

### 和书生同卡

两个服务同住一张 A100 80GB，都绑 GPU 0，不分卡。端口、KV 比例、启动顺序是配死的：

| | 书生（start-internlm.sh） | Qwen（本脚本） |
|---|---|---|
| 端口 | 1221 | 1222 |
| GPU | 0 | 0 |
| `CACHE` | 0.4 | 0.6 |
| 启动顺序 | 先起 | **后起** |

```bash
cd ~/internlm && ./start-internlm.sh start     # 先起书生，等日志里 1221 开始监听
cd ~/qwen     && ./start-qwen.sh start         # 再起 Qwen
```

`start` 和 `fg` 之前脚本会先看一眼卡上的空闲显存：不足 20 GiB 直接退出；整卡几乎全空（说明书生还没起）会提醒你先起书生；空闲不到 30 GiB 会提示本次 KV 池会变小。顺序反了的话，Qwen 会按 `CACHE=0.6` 划走约 38 GiB，书生随后只划得到 2~3 GiB，满长请求装不下。

起来之后对账，`status` 末尾也会打这两样：

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
```

两个进程合计 70~75 GiB，整卡空闲 ≥ 5 GiB。比例怎么来的见手册 §2.1。
