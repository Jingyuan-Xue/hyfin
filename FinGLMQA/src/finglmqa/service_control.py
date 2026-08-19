"""Safe idempotent start/status/stop controller for the local Phase 10 API."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .contracts import canonical_json_bytes, semantic_sha256
from .service_contracts import SERVICE_VERSION


ROOT = Path(__file__).resolve().parents[2]


def _paths() -> tuple[Path, Path, Path]:
    root = Path(os.environ.get("FINGLMQA_ROOT", ROOT)).resolve()
    runtime = Path(os.environ.get("FINGLMQA_RUNTIME_DIR", root / "runs/phase_10/runtime"))
    return root, runtime / "service_state.json", runtime / "control.lock"


def _start_ticks(pid: int) -> int | None:
    try:
        return int((Path("/proc") / str(pid) / "stat").read_text().split()[21])
    except (OSError, ValueError, IndexError):
        return None


def _cmdline(pid: int) -> str:
    try:
        return (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (OSError, UnicodeDecodeError):
        return ""


def _owned(state: dict[str, Any]) -> bool:
    pid = state.get("pid")
    ticks = state.get("process_start_ticks")
    return bool(
        isinstance(pid, int)
        and isinstance(ticks, int)
        and _start_ticks(pid) == ticks
        and "serve_phase_10_api.py" in _cmdline(pid)
    )


def _read_state(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_state(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    os.replace(temporary, path)


def _config() -> dict[str, Any]:
    keys = (
        "FINGLMQA_API_HOST", "FINGLMQA_API_PORT", "FINGLMQA_QUEUE_CAPACITY",
        "FINGLMQA_QUEUE_TIMEOUT_SECONDS", "FINGLMQA_EXECUTION_TIMEOUT_SECONDS",
        "FINGLMQA_WORKER_STARTUP_TIMEOUT_SECONDS", "FINGLMQA_SHUTDOWN_GRACE_SECONDS",
        "FINGLMQA_MAX_REQUEST_BYTES", "FINGLMQA_BREAKER_WINDOW_SECONDS",
        "FINGLMQA_BREAKER_THRESHOLD", "FINGLMQA_EVIDENCE_DEVICE",
        "FINGLMQA_SUPPLEMENTAL_FACTS_ENABLED",
    )
    return {key: os.environ.get(key) for key in keys}


def _health(host: str, port: int, timeout: float = 2.0) -> tuple[bool, dict[str, Any] | None]:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health/ready", timeout=timeout) as response:
            value = json.loads(response.read())
            return response.status == 200 and value.get("ready") is True, value
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False, None


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


def _terminate(state: dict[str, Any], grace: float) -> None:
    if not _owned(state):
        raise RuntimeError("refusing to signal an unowned or PID-reused process")
    pid = state["pid"]
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while _start_ticks(pid) == state["process_start_ticks"] and time.monotonic() < deadline:
        time.sleep(0.1)
    if _start_ticks(pid) == state["process_start_ticks"]:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def start() -> int:
    root, state_path, lock_path = _paths()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        host = os.environ.get("FINGLMQA_API_HOST", "127.0.0.1")
        port = int(os.environ.get("FINGLMQA_API_PORT", "8010"))
        fingerprint = semantic_sha256(_config())
        state = _read_state(state_path)
        if state and _owned(state):
            ready, value = _health(host, port)
            if ready and state.get("config_fingerprint") == fingerprint:
                print(json.dumps({"status": "already_ready", "health": value}, sort_keys=True))
                return 0
            raise RuntimeError("owned service exists with mismatched config or health")
        if state_path.exists():
            state_path.unlink()
        if _port_in_use(host, port):
            raise RuntimeError("API port is occupied by an unowned process")
        python = Path(os.environ.get("FINGLMQA_PHASE10_PYTHON", root / ".venv-phase10/bin/python"))
        script = root / "scripts/serve_phase_10_api.py"
        if not python.is_file() or not script.is_file():
            raise RuntimeError("Phase 10 API executable is missing")
        log_dir = Path(os.environ.get("FINGLMQA_LOG_DIR", root / "logs/phase_10"))
        log_dir.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(
            [str(python), str(script)], cwd=root, env=os.environ.copy(),
            # Uvicorn diagnostics are outside the log allow-list and can
            # contain paths.  ServiceTelemetryLogger owns the only persisted
            # runtime stream and rotates it at 100 MiB with five backups.
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True,
        )
        ticks = _start_ticks(process.pid)
        if ticks is None:
            process.kill()
            raise RuntimeError("could not identify started process")
        state = {
            "schema_version": "finglmqa.phase10.service_state.v1",
            "service_version": SERVICE_VERSION,
            "pid": process.pid,
            "process_start_ticks": ticks,
            "config_fingerprint": fingerprint,
            "host": host,
            "port": port,
        }
        _write_state(state_path, state)
        deadline = time.monotonic() + float(os.environ.get("FINGLMQA_WORKER_STARTUP_TIMEOUT_SECONDS", "300"))
        while time.monotonic() < deadline:
            if process.poll() is not None:
                state_path.unlink(missing_ok=True)
                raise RuntimeError("Phase 10 service exited during startup")
            ready, value = _health(host, port)
            if ready:
                print(json.dumps({"status": "ready", "health": value}, sort_keys=True))
                return 0
            time.sleep(0.5)
        _terminate(state, 5)
        state_path.unlink(missing_ok=True)
        raise RuntimeError("Phase 10 service startup timed out")


def status() -> int:
    _, state_path, _ = _paths()
    state = _read_state(state_path)
    if not state or not _owned(state):
        print(json.dumps({"status": "stopped"}, sort_keys=True))
        return 1
    ready, value = _health(state["host"], state["port"])
    print(json.dumps({"status": "ready" if ready else "degraded", "health": value}, sort_keys=True))
    return 0 if ready else 1


def stop() -> int:
    _, state_path, lock_path = _paths()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = _read_state(state_path)
        if not state:
            print(json.dumps({"status": "already_stopped"}, sort_keys=True))
            return 0
        if not _owned(state):
            raise RuntimeError("stale state does not identify an owned process")
        _terminate(state, float(os.environ.get("FINGLMQA_SHUTDOWN_GRACE_SECONDS", "30")))
        state_path.unlink(missing_ok=True)
        print(json.dumps({"status": "stopped"}, sort_keys=True))
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("start", "status", "stop"))
    command = parser.parse_args().command
    try:
        return {"start": start, "status": status, "stop": stop}[command]()
    except Exception as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
