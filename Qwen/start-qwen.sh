#!/usr/bin/env bash
# Qwen3-8B 后台管理脚本（适用于没有 systemd 的容器环境）
# 与 start-internlm.sh 同一套命令；两个服务同住一张 A100 80GB，都在 GPU 0 上。
#
# 用法:
#   ./start-qwen.sh start     后台启动
#   ./start-qwen.sh stop      停止
#   ./start-qwen.sh restart   重启
#   ./start-qwen.sh status    看状态
#   ./start-qwen.sh log       跟日志 (Ctrl+C 退出，不会停服务)
#   ./start-qwen.sh key       打印 API Key
#   ./start-qwen.sh fg        前台跑（调试用，Ctrl+C 直接停）
#
# 如果报 "bad interpreter: ...^M"，说明文件是 Windows 换行，先执行:
#   sed -i 's/\r$//' start-qwen.sh
#
# 同卡部署（一张 A100 80GB 上同时跑书生和 Qwen）:
#   先起书生（start-internlm.sh start），等 1221 端口监听之后再起本服务（1222）。
#   参数、启动顺序与验收见 Qwen3-8B-A100部署手册.md。

set -uo pipefail

# ==================== 配置区（按需修改） ====================
MODEL="${MODEL:-/data/models/Qwen3-8B}"
PORT="${PORT:-1222}"
HOME_DIR="${HOME:-/root}"
WORKDIR="${WORKDIR:-$HOME_DIR/qwen}"

# 单卡 80GB 上两个服务都绑 0，不要分卡；机器上有别的卡时才改
GPU="${GPU:-0}"

# 上下文包含输入与输出；显存紧张时可用 SESSION_LEN=16384 覆盖
# Qwen3-8B 原生上下文为 40960，长文本不要直接套用书生的 128K 配置
SESSION_LEN="${SESSION_LEN:-32768}"
# 同卡省显存起步值；实际 KV 容量随启动时空闲显存变化
CACHE="${CACHE:-0.3}"
BATCH="${BATCH:-4}"
MAXCONC="${MAXCONC:-4}"
TP="${TP:-1}"
# 启动前要求的最低空闲显存（MiB），不够就直接退出，别等它 OOM
MIN_FREE_MIB="${MIN_FREE_MIB:-20480}"

# 推理内容（<think>...</think>）从 content 里拆出来，单独走 reasoning_content 字段
# 名称按安装版本核对，可用 REASONING_PARSER 覆盖；此参数不会关闭思考
REASONING_PARSER="${REASONING_PARSER:-default}"

# lmdeploy 可执行文件；留空则自动从 PATH 里找
LMDEPLOY="${LMDEPLOY:-}"
# ===========================================================

PIDFILE="$WORKDIR/qwen.pid"
LOGFILE="$WORKDIR/qwen.log"
KEYFILE="$WORKDIR/api.key"

mkdir -p "$WORKDIR"

die() { echo "错误: $*" >&2; exit 1; }

find_lmdeploy() {
  if [ -n "$LMDEPLOY" ]; then
    [ -x "$LMDEPLOY" ] || die "LMDEPLOY 指向的文件不可执行: $LMDEPLOY"
    return
  fi
  LMDEPLOY="$(command -v lmdeploy || true)"
  [ -n "$LMDEPLOY" ] || die "PATH 里找不到 lmdeploy。先 conda activate 你的环境，或设置 LMDEPLOY=/完整/路径/lmdeploy"
}

gpu_field() {  # $1 = nvidia-smi 字段名，取绑定卡的值
  if [ -n "$GPU" ]; then
    nvidia-smi -i "$GPU" --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' '
  else
    nvidia-smi --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' '
  fi
}

check_gpu_free() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  [ "$TP" = "1" ] || return 0
  local free; free="$(gpu_field memory.free)"
  case "$free" in ''|*[!0-9]*) return 0 ;; esac
  echo "GPU ${GPU:-全部} 空闲显存: $((free / 1024)) GiB"
  [ "$free" -ge "$MIN_FREE_MIB" ] || die "空闲显存不足 $((MIN_FREE_MIB / 1024)) GiB：先停掉占卡的进程（nvidia-smi 看 pid），或降低 CACHE / SESSION_LEN。"
  if [ "$free" -gt 71680 ]; then
    echo "提示: 整卡几乎全空，书生还没起。先执行 ~/internlm/start-internlm.sh start，"
    echo "      等 1221 监听之后再起本服务；启动顺序会影响两边实际可用的 KV 容量。"
  elif [ "$free" -lt 30720 ]; then
    echo "提示: 空闲显存偏紧；降低 CACHE 可减少预留，长请求还需核对 KV 容量及 SESSION_LEN。"
  fi
}

ensure_key() {
  if [ ! -s "$KEYFILE" ]; then
    head -c 64 /dev/urandom | base64 | tr -d '/+=\n' | cut -c 1-40 > "$KEYFILE"
    chmod 600 "$KEYFILE"
    echo "已生成新的 API Key -> $KEYFILE"
  fi
  API_KEY="$(cat "$KEYFILE")"
}

is_running() {
  [ -f "$PIDFILE" ] || return 1
  local pid; pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null
}

build_cmd() {
  # Qwen3 是 transformers 原生架构，不需要 --trust-remote-code
  # （书生那边必须加，见 InternLM-A100部署手册.md）
  CMD=(
    "$LMDEPLOY" serve api_server "$MODEL"
    --server-name 0.0.0.0
    --server-port "$PORT"
    --backend turbomind
    --tp "$TP"
    --session-len "$SESSION_LEN"
    --cache-max-entry-count "$CACHE"
    --max-batch-size "$BATCH"
    --max-concurrent-requests "$MAXCONC"
    --reasoning-parser "$REASONING_PARSER"
    --api-keys "$API_KEY"
    --disable-fastapi-docs
    --log-level INFO
  )
}

cmd_start() {
  find_lmdeploy; ensure_key
  if is_running; then
    echo "已经在跑了 (PID $(cat "$PIDFILE"))，要重启用: $0 restart"
    return 0
  fi
  [ -f "$MODEL/config.json" ] || die "模型目录不对: $MODEL"
  check_gpu_free
  build_cmd

  echo "启动中... 日志: $LOGFILE"
  if [ -n "$GPU" ]; then
    echo "绑定 GPU: $GPU (CUDA_VISIBLE_DEVICES)"
    CUDA_VISIBLE_DEVICES="$GPU" nohup "${CMD[@]}" >>"$LOGFILE" 2>&1 &
  else
    nohup "${CMD[@]}" >>"$LOGFILE" 2>&1 &
  fi
  echo $! > "$PIDFILE"
  sleep 2

  if is_running; then
    echo "已启动，PID $(cat "$PIDFILE")"
    echo "监听: 0.0.0.0:$PORT"
    echo "Key : $(cat "$KEYFILE")"
    echo
    echo "加载权重要几分钟，用这个看进度:  $0 log"
  else
    echo "启动失败，日志尾部："
    tail -n 30 "$LOGFILE"
    rm -f "$PIDFILE"
    return 1
  fi
}

cmd_stop() {
  if ! is_running; then
    echo "没在跑"
    rm -f "$PIDFILE"
    return 0
  fi
  local pid; pid="$(cat "$PIDFILE")"
  echo "停止 PID $pid ..."
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "还没退出，强制杀"
    kill -9 "$pid" 2>/dev/null || true
    sleep 1
  fi
  rm -f "$PIDFILE"
  echo "已停止"
}

card_summary() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local total used free
  total="$(gpu_field memory.total)"; used="$(gpu_field memory.used)"; free="$(gpu_field memory.free)"
  case "$total" in ''|*[!0-9]*) return 0 ;; esac
  echo "整卡: 已用 $((used / 1024)) GiB / 共 $((total / 1024)) GiB，空闲 $((free / 1024)) GiB"
  local apps
  apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true)"
  if [ -n "$apps" ]; then
    echo "卡上进程 (pid, 显存):"
    printf '%s\n' "$apps" | sed 's/^/  /'
  else
    echo "卡上进程: 无"
  fi
}

cmd_status() {
  if is_running; then
    local pid; pid="$(cat "$PIDFILE")"
    echo "运行中  PID=$pid  端口=$PORT"
    echo "启动时长: $(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')"
    echo "显存占用:"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep "^$pid" || echo "  (还没加载完，或没用上 GPU)"
    if command -v ss >/dev/null 2>&1; then
      echo "端口监听:"
      ss -lntp 2>/dev/null | grep ":$PORT " || echo "  端口 $PORT 尚未监听（还在加载？）"
    fi
    if command -v curl >/dev/null 2>&1; then
      if [ -s "$KEYFILE" ]; then
        echo -n "接口自测: "
        curl -s -m 3 -o /dev/null -w "HTTP %{http_code}\n" \
          -H "Authorization: Bearer $(cat "$KEYFILE")" \
          "http://127.0.0.1:$PORT/v1/models" || echo "连不上"
      else
        echo "接口自测: 跳过（找不到 $KEYFILE）"
      fi
    fi
    card_summary
  else
    echo "未运行"
    card_summary
  fi
}

cmd_log() { tail -f "$LOGFILE"; }
cmd_key() { ensure_key; cat "$KEYFILE"; echo; }

cmd_fg() {
  find_lmdeploy; ensure_key
  [ -f "$MODEL/config.json" ] || die "模型目录不对: $MODEL"
  check_gpu_free
  build_cmd
  echo "前台运行，Ctrl+C 停止"
  if [ -n "$GPU" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
    echo "绑定 GPU: $GPU (CUDA_VISIBLE_DEVICES)"
  fi
  exec "${CMD[@]}"
}

case "${1:-}" in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  restart) cmd_stop; cmd_start ;;
  status)  cmd_status ;;
  log)     cmd_log ;;
  key)     cmd_key ;;
  fg)      cmd_fg ;;
  *)
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
