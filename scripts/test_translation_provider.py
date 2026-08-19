#!/usr/bin/env python3
"""Safe Tencent translation smoke test; never prints credentials or source text."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from icdm_demo.translation_service import TranslationService


def main() -> int:
    service = TranslationService(ROOT)
    if not service.configured:
        print("[FAIL] Tencent translation credentials are not configured.")
        return 2
    translated = service.translate("企业经营风险", "zh", "en")
    if not translated or re.search(r"[\u3400-\u9fff]", translated):
        print("[FAIL] Tencent returned no usable English translation.")
        return 1
    reverse = service.translate("operating risk", "en", "zh")
    if not re.search(r"[\u3400-\u9fff]", reverse):
        print("[FAIL] Tencent returned no usable Chinese reverse translation.")
        return 1
    health = service.health()
    print(f"[PASS] Tencent {health['action']} zh→en and en→zh")
    print(f"[PASS] Translation cache ready · provider requests {health['stats']['requests']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
