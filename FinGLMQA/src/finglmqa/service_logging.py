"""Redacted rotating JSONL telemetry for Phase 10."""

from __future__ import annotations

import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class ServiceTelemetryLogger:
    def __init__(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"finglmqa.phase10.{target.name}.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        handler = RotatingFileHandler(
            target, maxBytes=100 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)

    @staticmethod
    def request_id_hash(request_id: str | None) -> str | None:
        if not request_id:
            return None
        return hashlib.sha256(request_id.encode("utf-8")).hexdigest()

    def event(self, event: str, **fields: Any) -> None:
        payload = {"event": event, **fields}
        self.logger.info(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


__all__ = ["ServiceTelemetryLogger"]
