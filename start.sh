#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh"

mkdir -p "$DEMO_PID_DIR" "$DEMO_LOG_DIR" "$FINGLMQA_RUNTIME_DIR" "$A2RAG_OUTPUT_ROOT"

require_file() {
  [[ -f "$1" ]] || { echo "[FAIL] Missing required file: $1" >&2; exit 1; }
}
require_command() {
  command -v "$1" >/dev/null 2>&1 || { echo "[FAIL] Missing command: $1" >&2; exit 1; }
}
port_in_use() {
  (echo >"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1
}
pid_owned() {
  local pid_file="$1" pid cmdline
  [[ -f "$pid_file" ]] || return 1
  pid="$(tr -dc '0-9' < "$pid_file")"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
  [[ "$cmdline" == *"$ROOT"* ]]
}
wait_http() {
  local name="$1" url="$2" attempts="${3:-90}"
  for ((attempt=1; attempt<=attempts; attempt++)); do
    if curl -fsS --max-time 3 "$url" >/dev/null 2>&1; then
      echo "[READY] $name"
      return 0
    fi
    sleep 1
  done
  echo "[FAIL] $name did not become ready: $url" >&2
  return 1
}
start_process() {
  local name="$1" port="$2" pid_file="$3" log_file="$4"
  shift 4
  if pid_owned "$pid_file"; then
    echo "[READY] $name already running (PID $(cat "$pid_file"))"
    return 0
  fi
  if port_in_use "$port"; then
    echo "[FAIL] Port $port is occupied by a process not owned by this final_demo." >&2
    echo "       Stop the old service or override DEMO_*_PORT before running start.sh." >&2
    return 1
  fi
  (
    cd "$ROOT"
    exec setsid "$@"
  ) >"$log_file" 2>&1 &
  echo "$!" >"$pid_file"
  echo "[START] $name (PID $!, port $port)"
}

on_error() {
  local code=$?
  echo "[FAIL] Startup aborted. Inspect logs under $DEMO_LOG_DIR" >&2
  for log in "$DEMO_LOG_DIR"/*.log; do
    [[ -f "$log" ]] || continue
    echo "----- $(basename "$log") -----" >&2
    tail -n 20 "$log" >&2 || true
  done
  exit "$code"
}
trap on_error ERR

require_command curl
require_command setsid
require_file "$FINGLMQA_PHASE10_PYTHON"
require_file "$FINGLMQA_A2RAG_PYTHON"
require_file "$FINGLMQA_A2RAG_WORKER"
require_file "$FINGLMQA_ROOT/runs/phase_10/immutable_inputs_manifest.json"
require_file "$RISK_EXPOSURE_RUN_DIR/03_dataset/dataset_summary.json"
require_file "$ROOT/icdm_demo/index.html"

echo "[1/4] Starting FinGLMQA (text + table retrieval + online LLM)…"
"$FINGLMQA_PHASE10_PYTHON" -m finglmqa.service_control start
wait_http "FinGLMQA" "$DEMO_QA_URL/health/ready" 10

echo "[2/4] Starting risk-exposure artifact service…"
start_process "Risk exposure" "$DEMO_RISK_PORT" "$DEMO_PID_DIR/risk.pid" "$DEMO_LOG_DIR/risk.log" \
  "$FINGLMQA_PHASE10_PYTHON" "$ROOT/risk_exposure_method/serve_api.py"
wait_http "Risk exposure" "$DEMO_RISK_URL/health/ready" 30

echo "[3/4] Starting frontend gateway…"
start_process "Frontend gateway" "$DEMO_WEB_PORT" "$DEMO_PID_DIR/web.pid" "$DEMO_LOG_DIR/web.log" \
  "$FINGLMQA_PHASE10_PYTHON" -m uvicorn icdm_demo.stable_backend:app \
  --host "$DEMO_WEB_HOST" --port "$DEMO_WEB_PORT" --no-access-log
wait_http "Frontend gateway" "$DEMO_WEB_URL/api/health" 30

echo "[4/4] Running integrated self-check…"
"$ROOT/selfcheck.sh"

trap - ERR
echo
echo "HyFin demo is ready: $DEMO_WEB_URL/"
echo "Logs: $DEMO_LOG_DIR"
