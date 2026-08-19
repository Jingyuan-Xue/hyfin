"""One-time immutable release verification and cheap readiness stat probes."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import semantic_sha256


class ManifestVerificationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class VerifiedManifest:
    semantic_hash: str
    stats: dict[str, tuple[int, int]]


class ImmutableManifestVerifier:
    def __init__(self, root: str | Path, manifest_path: str | Path) -> None:
        self.root = Path(root).resolve()
        self.path = Path(manifest_path).resolve()
        self.verified: VerifiedManifest | None = None
        self._last_stat_check = 0.0
        self._last_stat_result = False

    def verify_full(self) -> VerifiedManifest:
        try:
            manifest = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ManifestVerificationError("immutable manifest is unavailable") from exc
        if manifest.get("schema_version") != "finglmqa.phase10.immutable_inputs_manifest.v1":
            raise ManifestVerificationError("immutable manifest schema is unsupported")
        expected_semantic = manifest.get("manifest_semantic_sha256")
        unhashed = dict(manifest)
        unhashed.pop("manifest_semantic_sha256", None)
        if expected_semantic != semantic_sha256(unhashed):
            raise ManifestVerificationError("immutable manifest semantic hash mismatch")
        stats: dict[str, tuple[int, int]] = {}
        for row in manifest.get("entries", []):
            label = row.get("path")
            if not isinstance(label, str) or label.startswith("/") or ".." in Path(label).parts:
                raise ManifestVerificationError("immutable manifest contains a non-portable path")
            path = self.root / label
            try:
                stat = path.stat()
            except OSError as exc:
                raise ManifestVerificationError(f"immutable input missing: {label}") from exc
            if stat.st_size != row.get("size") or stat.st_mtime_ns != row.get("mtime_ns"):
                raise ManifestVerificationError(f"immutable input stat mismatch: {label}")
            if sha256_file(path) != row.get("sha256"):
                raise ManifestVerificationError(f"immutable input hash mismatch: {label}")
            stats[label] = (stat.st_size, stat.st_mtime_ns)
        if not stats:
            raise ManifestVerificationError("immutable manifest has no entries")
        self.verified = VerifiedManifest(expected_semantic, stats)
        self._last_stat_check = time.monotonic()
        self._last_stat_result = True
        return self.verified

    def cheap_check(self, *, minimum_interval_seconds: float = 10.0) -> bool:
        if self.verified is None:
            return False
        now = time.monotonic()
        if now - self._last_stat_check < minimum_interval_seconds:
            return self._last_stat_result
        result = True
        for label, expected in self.verified.stats.items():
            try:
                stat = (self.root / label).stat()
            except OSError:
                result = False
                break
            if (stat.st_size, stat.st_mtime_ns) != expected:
                result = False
                break
        self._last_stat_check = now
        self._last_stat_result = result
        return result


__all__ = [
    "ImmutableManifestVerifier", "ManifestVerificationError", "VerifiedManifest",
    "sha256_file",
]
