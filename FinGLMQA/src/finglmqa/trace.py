"""Deterministic trace and non-deterministic telemetry helpers."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from .contracts import SCHEMA_TELEMETRY, SCHEMA_TRACE, semantic_sha256, validate_qa_trace


def dense_score_text(score: int | float | str | Decimal) -> str:
    value = Decimal(str(score)).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)
    return format(value, ".8f")


def finalize_trace(payload: dict[str, Any]) -> dict[str, Any]:
    trace = dict(payload)
    trace["schema_version"] = SCHEMA_TRACE
    trace.pop("trace_hash", None)
    trace["trace_hash"] = semantic_sha256(trace)
    validate_qa_trace(trace)
    return trace


class TelemetryRecorder:
    """Record runtime-only measurements that never enter the trace hash."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.started_at = datetime.now(timezone.utc)
        self.started_monotonic = time.monotonic()

    def finish(self, *, runtime: dict[str, Any] | None = None) -> dict[str, Any]:
        finished = datetime.now(timezone.utc)
        return {
            "schema_version": SCHEMA_TELEMETRY,
            "request_id": self.request_id,
            "started_at_utc": self.started_at.isoformat(),
            "finished_at_utc": finished.isoformat(),
            "elapsed_seconds": round(time.monotonic() - self.started_monotonic, 6),
            "process_id": os.getpid(),
            "runtime": runtime or {},
        }
