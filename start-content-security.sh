#!/usr/bin/env bash
# Bitahub 容器后台管理：bash start-content-security.sh start|stop|restart|status|log|fg [服务]
# 服务：api、governance、training、neural-trace；默认启动 API + 治理，其余按需。
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$ROOT" || exit 1
PYTHON="$ROOT/.venv/bin/python"
PORT=8001
export APP_CONFIG_FILE="${APP_CONFIG_FILE:-$ROOT/settings.local.json}"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
umask 077
mkdir -p "$ROOT/logs"
SERVICES=(api governance training neural-trace)
declare -A MODULES=([api]=uvicorn [governance]=service.governance_worker
                   [training]=service.training_worker [neural-trace]=service.neural_trace_worker)

die() { echo "错误：$*" >&2; exit 1; }
usage() {
  echo "用法：bash $0 {start|stop|restart|status|log|fg} [api|governance|training|neural-trace]"
  echo "start 默认启动 API + 治理；stop/status 默认全部；restart 重启已运行服务。"
  echo "log/fg 默认选择 API。训练、神经溯源需要单独 start。"
}

is_running() {
  local service="$1" pid args
  read -r pid 2>/dev/null < "$ROOT/logs/$service.pid" || return 1
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  [[ "$(readlink "/proc/$pid/cwd" 2>/dev/null)" == "$ROOT" ]] || return 1
  args="$(tr '\0' ' ' 2>/dev/null < "/proc/$pid/cmdline")" || return 1
  [[ "$args" == *"$PYTHON -m ${MODULES[$service]} "* ]]
}

build_cmd() {
  CMD=("$PYTHON" -m "${MODULES[$1]}")
  if [[ "$1" == api ]]; then
    CMD+=(main:app --host 0.0.0.0 --port "$PORT" --workers 1)
  fi
}

check_environment() {
  [[ -x "$PYTHON" ]] || die "缺少 Linux 虚拟环境，请先执行 uv sync --frozen --no-dev --python 3.11"
  "$PYTHON" -c 'from config import Settings; Settings.load()' || die "配置校验失败，请检查 settings.local.json"
}

check_port() {
  "$PYTHON" -c 'import socket,sys; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("0.0.0.0", int(sys.argv[1]))); s.close()' "$PORT" \
    || die "端口 $PORT 已被占用，请先停止原 API"
}

start_service() {
  local service="$1" pid
  if is_running "$service"; then
    echo "$service 已运行，PID=$(cat "$ROOT/logs/$service.pid")"
    return 0
  fi
  [[ "$service" != api ]] || check_port
  build_cmd "$service"
  nohup setsid "${CMD[@]}" >> "$ROOT/logs/$service.log" 2>&1 < /dev/null 9>&- &
  pid=$!
  echo "$pid" > "$ROOT/logs/$service.pid"
  sleep 1
  if is_running "$service"; then
    echo "$service 已启动，PID=$pid，日志：$ROOT/logs/$service.log"
  else
    echo "$service 启动失败，日志尾部：" >&2
    tail -n 30 "$ROOT/logs/$service.log" >&2
    rm -f -- "$ROOT/logs/$service.pid"
    return 1
  fi
}

stop_service() {
  local service="$1" pid pgid i
  if ! is_running "$service"; then
    echo "$service 未运行"
    rm -f -- "$ROOT/logs/$service.pid"
    return 0
  fi
  pid="$(cat "$ROOT/logs/$service.pid")"
  pgid="$(ps -o pgid= -p "$pid")"
  [[ "${pgid//[[:space:]]/}" == "$pid" ]] || die "$service 不是本脚本启动的独立进程组，请先按原方式停止"
  kill -TERM -- "-$pid" 2>/dev/null || true
  for ((i=0; i<150; i++)); do
    is_running "$service" || break
    sleep 1
  done
  # 父进程退出后，也清理同组尚未退出的训练/实验子进程。
  kill -KILL -- "-$pid" 2>/dev/null || true
  rm -f -- "$ROOT/logs/$service.pid"
  echo "$service 已停止"
}

action="${1:-start}"
target="${2:-}"
[[ "$#" -le 2 ]] || { usage; exit 1; }
case "$action" in start|stop|restart|status|log|fg) ;; *) usage; exit 1 ;; esac
if [[ -n "$target" ]]; then
  case "$target" in api|governance|training|neural-trace) ;; *) usage; exit 1 ;; esac
  selected=("$target")
else
  case "$action" in
    start) selected=(api governance) ;;
    log|fg) selected=(api) ;;
    *) selected=("${SERVICES[@]}") ;;
  esac
fi

case "$action" in
  start|stop|restart|fg)
    command -v flock >/dev/null || die "缺少 flock，请在容器镜像中安装 util-linux"
    exec 9> "$ROOT/logs/launcher.lock"
    flock -n 9 || die "另一条启动/停止命令正在执行，请稍后重试"
    ;;
esac

case "$action" in
  start)
    check_environment
    for service in "${selected[@]}"; do start_service "$service" || exit 1; done
    ;;
  stop)
    for service in "${selected[@]}"; do stop_service "$service"; done
    ;;
  restart)
    check_environment
    if [[ -z "$target" ]]; then
      selected=()
      for service in "${SERVICES[@]}"; do is_running "$service" && selected+=("$service"); done
      [[ "${#selected[@]}" -gt 0 ]] || selected=(api governance)
    fi
    for service in "${selected[@]}"; do stop_service "$service"; done
    for service in "${selected[@]}"; do start_service "$service" || exit 1; done
    ;;
  status)
    for service in "${selected[@]}"; do
      if is_running "$service"; then
        echo "$service 运行中，PID=$(cat "$ROOT/logs/$service.pid")"
      else
        echo "$service 未运行"
      fi
    done
    ;;
  log)
    [[ -f "$ROOT/logs/${selected[0]}.log" ]] || die "暂无日志，请先启动 ${selected[0]}"
    exec tail -n 50 -f "$ROOT/logs/${selected[0]}.log"
    ;;
  fg)
    check_environment
    service="${selected[0]}"
    is_running "$service" && die "$service 已在后台运行，请先停止"
    [[ "$service" != api ]] || check_port
    build_cmd "$service"
    echo "$service 前台运行，Ctrl+C 停止"
    exec "${CMD[@]}" 9>&-
    ;;
esac
