"""Server-side translation with Tencent TC3 signing and persistent caching."""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


_CJK = re.compile(r"[\u3400-\u9fff]")


def _load_local_environment(path: Path) -> None:
    if not path.is_file():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip("'\"")


_ASCII_WORD = re.compile(r"[A-Za-z]{2,}")
_MARKER = re.compile(r"\[\[\[T(\d{4})\]\]\]")


class TranslationError(RuntimeError):
    """A provider, authentication, quota, or response error."""


class TranslationService:
    """Translate display text without mutating canonical Chinese artifacts."""

    def __init__(self, root: Path) -> None:
        self.root = root
        _load_local_environment(root / ".env")
        self.provider = os.environ.get("TRANSLATION_PROVIDER", "tencent_mps").strip().lower()
        self.secret_id = os.environ.get("TENCENTCLOUD_SECRET_ID", "").strip()
        self.secret_key = os.environ.get("TENCENTCLOUD_SECRET_KEY", "").strip()
        self.host = os.environ.get("TENCENT_TRANSLATION_HOST", "mps.tencentcloudapi.com").strip()
        self.action = os.environ.get("TENCENT_TRANSLATION_ACTION", "TextTranslation").strip()
        self.version = os.environ.get("TENCENT_TRANSLATION_VERSION", "2019-06-12").strip()
        self.service = os.environ.get("TENCENT_TRANSLATION_SERVICE", "mps").strip()
        self.region = os.environ.get("TENCENT_TRANSLATION_REGION", "ap-beijing").strip()
        self.project_id = int(os.environ.get("TENCENT_TRANSLATION_PROJECT_ID", "0"))
        self.timeout = float(os.environ.get("TRANSLATION_TIMEOUT_SECONDS", "12"))
        self.max_chars = max(200, min(int(os.environ.get("TRANSLATION_MAX_CHARS", "1800")), 1900))
        self.min_interval = 1.0 / max(float(os.environ.get("TRANSLATION_QPS", "4.5")), 0.1)
        self.strict = os.environ.get("TRANSLATION_STRICT", "0").lower() in {"1", "true", "yes"}
        cache_value = os.environ.get("TRANSLATION_CACHE_PATH", "runtime/translation_cache.sqlite3")
        cache_path = Path(cache_value)
        self.cache_path = cache_path if cache_path.is_absolute() else root / cache_path
        static_value = os.environ.get("TRANSLATION_STATIC_PATH", "i18n/static_zh_en.json")
        static_path = Path(static_value)
        self.static_path = static_path if static_path.is_absolute() else root / static_path
        self._provider_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._last_request = 0.0
        self._stats = {
            "requests": 0,
            "translated_characters": 0,
            "cache_hits": 0,
            "static_hits": 0,
            "errors": 0,
        }
        self._static = self._load_static()
        self._init_cache()

    @property
    def configured(self) -> bool:
        return self.provider in {"tencent_mps", "tencent_tmt"} and bool(self.secret_id and self.secret_key)

    def _load_static(self) -> dict[str, dict[str, str]]:
        if not self.static_path.is_file():
            return {"zh_en": {}, "en_zh": {}}
        try:
            value = json.loads(self.static_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"zh_en": {}, "en_zh": {}}
        return {
            "zh_en": dict(value.get("zh_en") or {}),
            "en_zh": dict(value.get("en_zh") or {}),
        }

    def _init_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.cache_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS translations (
                    cache_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    source_language TEXT NOT NULL,
                    target_language TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    target_text TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    def health(self) -> dict[str, Any]:
        with self._stats_lock:
            stats = dict(self._stats)
        return {
            "provider": self.provider,
            "configured": self.configured,
            "ready": self.configured,
            "host": self.host,
            "action": self.action,
            "version": self.version,
            "max_chars": self.max_chars,
            "cache_path": str(self.cache_path.relative_to(self.root)),
            "static_entries": sum(len(values) for values in self._static.values()),
            "stats": stats,
        }

    def _key(self, text: str, source: str, target: str) -> str:
        raw = "\0".join((self.provider, source, target, text)).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _cache_get(self, text: str, source: str, target: str) -> str | None:
        cache_key = self._key(text, source, target)
        with sqlite3.connect(self.cache_path) as connection:
            row = connection.execute(
                "SELECT target_text FROM translations WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row:
            with self._stats_lock:
                self._stats["cache_hits"] += 1
            return str(row[0])
        return None

    def _cache_set(self, text: str, translated: str, source: str, target: str) -> None:
        with sqlite3.connect(self.cache_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO translations
                (cache_key, provider, source_language, target_language, source_text, target_text, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (self._key(text, source, target), self.provider, source, target, text, translated, time.time()),
            )

    def _static_get(self, text: str, source: str, target: str) -> str | None:
        direction = "zh_en" if source == "zh" and target == "en" else "en_zh"
        value = self._static.get(direction, {}).get(text)
        if value:
            with self._stats_lock:
                self._stats["static_hits"] += 1
            return str(value)
        return None

    @staticmethod
    def _requires_translation(text: str, source: str, target: str) -> bool:
        if not text.strip() or source == target:
            return False
        if source == "zh":
            return bool(_CJK.search(text))
        if source == "en":
            return bool(_ASCII_WORD.search(text))
        return True

    def _throttle(self) -> None:
        remaining = self.min_interval - (time.monotonic() - self._last_request)
        if remaining > 0:
            time.sleep(remaining)
        self._last_request = time.monotonic()

    def _request_tencent(self, text: str, source: str, target: str) -> str:
        if not self.configured:
            raise TranslationError("Tencent translation credentials are not configured")
        timestamp = int(time.time())
        date = dt.datetime.fromtimestamp(timestamp, dt.timezone.utc).strftime("%Y-%m-%d")
        payload_data: dict[str, Any] = {
            "SourceText": text,
            "Source": source,
            "Target": target,
        }
        if self.provider == "tencent_tmt":
            payload_data["ProjectId"] = self.project_id
        payload = json.dumps(
            payload_data,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        canonical_headers = (
            "content-type:application/json; charset=utf-8\n"
            f"host:{self.host}\n"
            f"x-tc-action:{self.action.lower()}\n"
        )
        signed_headers = "content-type;host;x-tc-action"
        hashed_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        canonical_request = "\n".join(
            ("POST", "/", "", canonical_headers, signed_headers, hashed_payload)
        )
        credential_scope = f"{date}/{self.service}/tc3_request"
        string_to_sign = "\n".join(
            (
                "TC3-HMAC-SHA256",
                str(timestamp),
                credential_scope,
                hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
            )
        )

        def sign(key: bytes, message: str) -> bytes:
            return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()

        secret_date = sign(("TC3" + self.secret_key).encode("utf-8"), date)
        secret_service = sign(secret_date, self.service)
        secret_signing = sign(secret_service, "tc3_request")
        signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
        authorization = (
            "TC3-HMAC-SHA256 "
            f"Credential={self.secret_id}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        headers = {
            "Authorization": authorization,
            "Content-Type": "application/json; charset=utf-8",
            "Host": self.host,
            "X-TC-Action": self.action,
            "X-TC-Timestamp": str(timestamp),
            "X-TC-Version": self.version,
        }
        if self.provider == "tencent_tmt":
            headers["X-TC-Region"] = self.region
        request = urllib.request.Request(
            f"https://{self.host}",
            data=payload.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with self._provider_lock:
            self._throttle()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read())
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")
                raise TranslationError(f"Tencent HTTP {exc.code}: {detail[:500]}") from exc
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                raise TranslationError(f"Tencent translation request failed: {exc}") from exc
        response = result.get("Response") or {}
        if response.get("Error"):
            error = response["Error"]
            raise TranslationError(f"{error.get('Code', 'TencentError')}: {error.get('Message', '')}")
        translated = str(response.get("TargetText") or "")
        if not translated:
            raise TranslationError("Tencent returned an empty translation")
        with self._stats_lock:
            self._stats["requests"] += 1
            self._stats["translated_characters"] += len(text)
        return translated

    def _provider_translate(self, text: str, source: str, target: str) -> str:
        if self.provider in {"tencent_mps", "tencent_tmt"}:
            return self._request_tencent(text, source, target)
        raise TranslationError(f"Unsupported translation provider: {self.provider}")

    def _translate_one(self, text: str, source: str, target: str) -> str:
        static = self._static_get(text, source, target)
        if static is not None:
            return static
        cached = self._cache_get(text, source, target)
        if cached is not None:
            return cached
        try:
            translated = self._provider_translate(text, source, target)
        except TranslationError:
            with self._stats_lock:
                self._stats["errors"] += 1
            if self.strict:
                raise
            return text
        self._cache_set(text, translated, source, target)
        return translated

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self.max_chars:
            return [text]
        pieces = re.split(r"(?<=[。！？!?；;])|\n+", text)
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            if not piece:
                continue
            if len(piece) > self.max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(piece[index : index + self.max_chars] for index in range(0, len(piece), self.max_chars))
            elif len(current) + len(piece) <= self.max_chars:
                current += piece
            else:
                chunks.append(current)
                current = piece
        if current:
            chunks.append(current)
        return chunks

    def translate(self, text: Any, source: str = "zh", target: str = "en") -> str:
        value = str(text or "")
        if not self._requires_translation(value, source, target):
            return value
        static = self._static_get(value, source, target)
        if static is not None:
            return static
        cached = self._cache_get(value, source, target)
        if cached is not None:
            return cached
        chunks = self._split_text(value)
        if len(chunks) == 1:
            return self._translate_one(value, source, target)
        translated = "".join(self._translate_one(chunk, source, target) for chunk in chunks)
        if translated != value:
            self._cache_set(value, translated, source, target)
        return translated

    def translate_many(
        self,
        values: Iterable[Any],
        source: str = "zh",
        target: str = "en",
    ) -> list[str]:
        texts = [str(value or "") for value in values]
        output = list(texts)
        pending: list[tuple[int, str]] = []
        for index, text in enumerate(texts):
            if not self._requires_translation(text, source, target):
                continue
            static = self._static_get(text, source, target)
            if static is not None:
                output[index] = static
                continue
            cached = self._cache_get(text, source, target)
            if cached is not None:
                output[index] = cached
                continue
            if len(text) > self.max_chars // 2:
                output[index] = self.translate(text, source, target)
            else:
                pending.append((index, text))

        groups: list[list[tuple[int, str]]] = []
        current: list[tuple[int, str]] = []
        current_size = 0
        for item in pending:
            marker_size = 16
            if current and current_size + len(item[1]) + marker_size > self.max_chars:
                groups.append(current)
                current = []
                current_size = 0
            current.append(item)
            current_size += len(item[1]) + marker_size
        if current:
            groups.append(current)

        for group in groups:
            if len(group) == 1:
                index, text = group[0]
                output[index] = self._translate_one(text, source, target)
                continue
            packed = "".join(f"\n[[[T{index:04d}]]]\n{text}" for index, text in group)
            try:
                translated = self._provider_translate(packed, source, target)
                matches = list(_MARKER.finditer(translated))
                parsed: dict[int, str] = {}
                for position, match in enumerate(matches):
                    start = match.end()
                    end = matches[position + 1].start() if position + 1 < len(matches) else len(translated)
                    parsed[int(match.group(1))] = translated[start:end].strip()
                expected = {index for index, _ in group}
                if set(parsed) != expected:
                    raise TranslationError("Provider changed translation segment markers")
                for index, text in group:
                    output[index] = parsed[index]
                    self._cache_set(text, parsed[index], source, target)
            except TranslationError:
                for index, text in group:
                    output[index] = self._translate_one(text, source, target)
        return output

    def translate_markdown_table(self, value: Any, source: str = "zh", target: str = "en") -> str:
        text = str(value or "")
        lines = text.splitlines()
        cell_locations: list[tuple[int, int]] = []
        rows: list[list[str]] = []
        values: list[str] = []
        for line_index, line in enumerate(lines):
            if not (line.strip().startswith("|") and line.strip().endswith("|")):
                rows.append([])
                continue
            cells = [cell.strip() for cell in line.strip()[1:-1].split("|")]
            rows.append(cells)
            if all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells):
                continue
            for cell_index, cell in enumerate(cells):
                if _CJK.search(cell):
                    cell_locations.append((line_index, cell_index))
                    values.append(cell)
        translated = self.translate_many(values, source, target)
        for (line_index, cell_index), result in zip(cell_locations, translated):
            rows[line_index][cell_index] = result
        rebuilt: list[str] = []
        for line, row in zip(lines, rows):
            rebuilt.append("|" + "|".join(row) + "|" if row else line)
        return "\n".join(rebuilt)


def translation_metadata(service: TranslationService, language: str) -> dict[str, Any]:
    health = service.health()
    return {
        "language": language,
        "provider": health["provider"],
        "configured": health["configured"],
        "status": "translated" if language == "en" and health["configured"] else ("source" if language == "zh" else "unavailable"),
    }
