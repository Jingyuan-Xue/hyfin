#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh"

stop_pid_file() {
  local name="$1" pid_file="$2" pid cmdline
  if [[ ! -f "$pid_file" ]]; then
    echo "[STOP] $name already stopped"
    return 0
  fi
  pid="$(tr -dc '0-9' < "$pid_file")"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$pid_file"
    echo "[STOP] $name removed stale PID"
    return 0
  fi
  cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
  if [[ "$cmdline" != *"$ROOT"* ]]; then
    echo "[WARN] Refusing to stop unowned PID $pid for $name" >&2
    return 1
  fi
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in {1..50}; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.1
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$pid_file"
  echo "[STOP] $name stopped"
}

stop_pid_file "Frontend gateway" "$DEMO_PID_DIR/web.pid"
stop_pid_file "Risk exposure" "$DEMO_PID_DIR/risk.pid"

if [[ -f "$FINGLMQA_RUNTIME_DIR/service_state.json" ]]; then
  "$FINGLMQA_PHASE10_PYTHON" -m finglmqa.service_control stop
else
  echo "[STOP] FinGLMQA already stopped"
fi
