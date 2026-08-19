#!/usr/bin/env python3
"""Launch the risk-exposure artifact service."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from risk_exposure_method.service_api import app


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.environ.get("RISK_EXPOSURE_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("RISK_EXPOSURE_API_PORT", "8012")),
        workers=1,
        access_log=False,
        server_header=False,
    )
