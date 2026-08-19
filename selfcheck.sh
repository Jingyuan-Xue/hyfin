#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh"
exec "$FINGLMQA_PHASE10_PYTHON" "$ROOT/scripts/selfcheck.py" "$@"
