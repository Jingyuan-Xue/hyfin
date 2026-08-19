"""ASGI wrapper for the public demo port.

It keeps the API/static FastAPI app unchanged, but disables browser caching while
the demo is being iterated and closes stale Vite HMR WebSocket connections that
may survive in VS Code's forwarded-port browser tab.

When DEMO_ACCESS_TOKEN is set it additionally gates every request behind a shared
token and applies per-IP rate limits. The demo holds metered credentials (the
online LLM key and the Tencent translation secrets), so it must not be reachable
by anyone who merely finds the URL. With the variable unset the gateway behaves
exactly as before, which keeps local development and ./selfcheck.sh unchanged.
"""

from __future__ import annotations

import hmac
import json
import os
import time
from collections import defaultdict, deque
from http.cookies import SimpleCookie
from urllib.parse import parse_qs

from icdm_demo.final_backend import app as fastapi_app


COOKIE_NAME = "demo_access"

# Readiness probing must stay open: start.sh polls this before the token is known.
OPEN_PATHS = frozenset({"/api/health"})

# Routes that spend money on every call — online generation and metered translation.
EXPENSIVE_PREFIXES = (
    "/api/finglmqa/qa",
    "/api/finglmqa/consolidate",
    "/api/translation/qa",
)

PRUNE_INTERVAL_SECONDS = 600


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _is_expensive(path: str) -> bool:
    if path.startswith(EXPENSIVE_PREFIXES):
        return True
    # POST /api/demo/cases/{case_id}/runs replays a pipeline stage by stage.
    return path.startswith("/api/demo/cases/") and path.endswith("/runs")


class SlidingWindow:
    """Per-key request counter over a fixed window.

    The demo runs as a single uvicorn process (start.sh passes no --workers), so
    in-process counters see every request and need no shared store.
    """

    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._last_prune = time.monotonic()

    def retry_after(self, key: str) -> int:
        """Record a hit and return 0 when allowed, else seconds until a slot frees."""
        now = time.monotonic()
        self._prune(now)
        bucket = self._hits[key]
        while bucket and now - bucket[0] >= self.window:
            bucket.popleft()
        if len(bucket) >= self.limit:
            return max(1, int(self.window - (now - bucket[0])) + 1)
        bucket.append(now)
        return 0

    def _prune(self, now: float) -> None:
        if now - self._last_prune < PRUNE_INTERVAL_SECONDS:
            return
        self._last_prune = now
        for key in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window]:
            del self._hits[key]


async def _send_json(send, status: int, payload: dict, extra_headers=()) -> None:
    body = json.dumps(payload).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
        (b"cache-control", b"no-store"),
    ]
    headers.extend(extra_headers)
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


async def _send_html(send, status: int, title: str, message: str, extra_headers=()) -> None:
    body = (
        "<!doctype html><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{title}</title>"
        "<style>body{font:16px/1.6 system-ui,sans-serif;max-width:34rem;"
        "margin:20vh auto;padding:0 1.5rem;color:#222}h1{font-size:1.25rem}"
        "@media(prefers-color-scheme:dark){body{background:#111;color:#eee}}</style>"
        f"<h1>{title}</h1><p>{message}</p>"
    ).encode("utf-8")
    headers = [
        (b"content-type", b"text/html; charset=utf-8"),
        (b"content-length", str(len(body)).encode("ascii")),
        (b"cache-control", b"no-store"),
    ]
    headers.extend(extra_headers)
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class DemoGateway:
    def __init__(self, inner):
        self.inner = inner
        self.token = os.environ.get("DEMO_ACCESS_TOKEN", "").strip()
        self.trust_proxy = os.environ.get("DEMO_TRUST_PROXY", "").strip() == "1"
        self.expensive = SlidingWindow(_int_env("DEMO_RATE_LIMIT_EXPENSIVE", 20), 3600)
        self.general = SlidingWindow(_int_env("DEMO_RATE_LIMIT_GENERAL", 240), 60)

    def _client_ip(self, scope, headers: dict) -> str:
        if self.trust_proxy:
            forwarded = headers.get(b"x-forwarded-for", b"").decode("latin-1")
            first = forwarded.split(",")[0].strip()
            if first:
                return first
        client = scope.get("client")
        return client[0] if client else "unknown"

    def _presented_token(self, scope, headers: dict) -> tuple[str, bool]:
        """Return the token the caller presented and whether it came from the URL."""
        header = headers.get(b"x-demo-token", b"").decode("latin-1").strip()
        if header:
            return header, False

        query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
        from_url = (query.get("t") or [""])[0].strip()
        if from_url:
            return from_url, True

        raw_cookie = headers.get(b"cookie", b"").decode("latin-1")
        if raw_cookie:
            jar = SimpleCookie()
            try:
                jar.load(raw_cookie)
            except Exception:
                return "", False
            morsel = jar.get(COOKIE_NAME)
            if morsel:
                return morsel.value.strip(), False
        return "", False

    def _cookie_header(self, headers: dict) -> tuple[bytes, bytes]:
        secure = ""
        if headers.get(b"x-forwarded-proto", b"").decode("latin-1").strip() == "https":
            secure = " Secure;"
        value = (
            f"{COOKIE_NAME}={self.token}; Path=/; HttpOnly; SameSite=Lax;{secure} Max-Age=86400"
        )
        return (b"set-cookie", value.encode("latin-1"))

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1000})
            return

        if scope["type"] != "http":
            await self.inner(scope, receive, send)
            return

        path = scope.get("path", "")
        headers = dict(scope.get("headers") or [])
        wants_html = b"text/html" in headers.get(b"accept", b"")
        extra_headers: list[tuple[bytes, bytes]] = []

        if self.token and path not in OPEN_PATHS:
            presented, from_url = self._presented_token(scope, headers)
            if not hmac.compare_digest(presented, self.token):
                if wants_html:
                    await _send_html(
                        send,
                        401,
                        "Access token required",
                        "Open the demo using the full link you were sent, "
                        "including its <code>?t=</code> token.",
                    )
                else:
                    await _send_json(send, 401, {"error": "access token required"})
                return
            if from_url:
                # Assets loaded by index.html carry no query string, so persist
                # the token as a cookie on the way back.
                extra_headers.append(self._cookie_header(headers))

        if path.startswith("/api/"):
            client_ip = self._client_ip(scope, headers)
            if _is_expensive(path):
                window, scope_name = self.expensive, "expensive"
            else:
                window, scope_name = self.general, "general"
            retry_after = window.retry_after(f"{scope_name}:{client_ip}")
            if retry_after:
                await _send_json(
                    send,
                    429,
                    {
                        "error": "rate limit exceeded",
                        "scope": scope_name,
                        "retry_after_seconds": retry_after,
                    },
                    [(b"retry-after", str(retry_after).encode("ascii"))],
                )
                return

        async def send_no_cache(message):
            if message["type"] == "http.response.start":
                response_headers = list(message.get("headers", []))
                response_headers.extend(
                    [
                        (b"cache-control", b"no-store, no-cache, must-revalidate, max-age=0"),
                        (b"pragma", b"no-cache"),
                        (b"expires", b"0"),
                        (b"x-icdm-demo-version", b"2026-07-25.5"),
                    ]
                )
                response_headers.extend(extra_headers)
                message["headers"] = response_headers
            await send(message)

        await self.inner(scope, receive, send_no_cache)


app = DemoGateway(fastapi_app)
