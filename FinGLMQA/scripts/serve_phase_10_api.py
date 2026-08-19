#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import uvicorn  # noqa: E402

from finglmqa.service_api import create_app  # noqa: E402


if __name__ == "__main__":
    uvicorn.run(
        create_app(),
        host=os.environ.get("FINGLMQA_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("FINGLMQA_API_PORT", "8010")),
        workers=1,
        access_log=False,
        server_header=False,
        log_level="critical",
    )
