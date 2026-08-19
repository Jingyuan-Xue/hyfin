#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env/finglmqa.phase10.env"
export PYTHONPATH="$ROOT/src"

"$ROOT/scripts/start_finglmqa.sh" >/dev/null
exec "$FINGLMQA_PHASE10_PYTHON" "$ROOT/scripts/eval_phase_10_http.py" "$@"
