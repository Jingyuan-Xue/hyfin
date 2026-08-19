"""Bounded FIFO and hard process isolation for the Phase 10 QA worker."""

from __future__ import annotations

import asyncio
import json
import os
import selectors
import signal
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .service_contracts import SERVICE_VERSION, WORKER_PROTOCOL
from .service_logging import ServiceTelemetryLogger
from .service_manifest import ImmutableManifestVerifier, ManifestVerificationError


ROOT = Path(__file__).resolve().parents[2]


class WorkerProtocolError(RuntimeError):
    pass


class ServiceRequestFailure(RuntimeError):
    def __init__(self, status_code: int, failure_code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.failure_code = failure_code
        self.message = message


@dataclass(frozen=True)
class ServiceConfig:
    root: Path
    python: Path
    worker_script: Path
    manifest_path: Path
    trace_dir: Path
    log_dir: Path
    queue_capacity: int = 32
    queue_timeout_seconds: float = 30.0
    execution_timeout_seconds: float = 90.0
    startup_timeout_seconds: float = 300.0
    shutdown_grace_seconds: float = 30.0
    breaker_window_seconds: float = 300.0
    breaker_threshold: int = 3
    max_request_bytes: int = 65536

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        root = Path(os.environ.get("FINGLMQA_ROOT", ROOT)).resolve()
        return cls(
            root=root,
            python=Path(os.environ.get("FINGLMQA_PHASE10_PYTHON", root / ".venv-phase10/bin/python")),
            worker_script=root / "scripts/serve_phase_10_worker.py",
            manifest_path=root / "runs/phase_10/immutable_inputs_manifest.json",
            trace_dir=Path(os.environ.get("FINGLMQA_TRACE_DIR", root / "runs/phase_10/service/traces")),
            log_dir=Path(os.environ.get("FINGLMQA_LOG_DIR", root / "logs/phase_10")),
            queue_capacity=int(os.environ.get("FINGLMQA_QUEUE_CAPACITY", "32")),
            queue_timeout_seconds=float(os.environ.get("FINGLMQA_QUEUE_TIMEOUT_SECONDS", "30")),
            execution_timeout_seconds=float(os.environ.get("FINGLMQA_EXECUTION_TIMEOUT_SECONDS", "90")),
            startup_timeout_seconds=float(os.environ.get("FINGLMQA_WORKER_STARTUP_TIMEOUT_SECONDS", "300")),
            shutdown_grace_seconds=float(os.environ.get("FINGLMQA_SHUTDOWN_GRACE_SECONDS", "30")),
            breaker_window_seconds=float(os.environ.get("FINGLMQA_BREAKER_WINDOW_SECONDS", "300")),
            breaker_threshold=int(os.environ.get("FINGLMQA_BREAKER_THRESHOLD", "3")),
            max_request_bytes=int(os.environ.get("FINGLMQA_MAX_REQUEST_BYTES", "65536")),
        )


class QAWorkerProcess:
    def __init__(self, config: ServiceConfig, generation: int) -> None:
        self.config = config
        self.generation = generation
        self.process: subprocess.Popen[str] | None = None
        self.sequence = 0
        self.ready_payload: dict[str, Any] | None = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _read(self, timeout: float) -> dict[str, Any]:
        process = self.process
        if process is None or process.stdout is None:
            raise WorkerProtocolError("worker stdout unavailable")
        selector = selectors.DefaultSelector()
        try:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                raise TimeoutError("worker response timed out")
            raw = process.stdout.readline()
        finally:
            selector.close()
        if not raw:
            raise WorkerProtocolError("worker exited before response")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise WorkerProtocolError("worker emitted invalid JSON") from exc
        if not isinstance(value, dict) or value.get("protocol_version") != WORKER_PROTOCOL:
            raise WorkerProtocolError("worker protocol mismatch")
        return value

    def start(self) -> None:
        if self.running:
            return
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        python_paths = [
            str(self.config.root / "src"),
            str(self.config.root.parent / "A2RAG" / "src"),
        ]
        inherited_pythonpath = os.environ.get("PYTHONPATH")
        if inherited_pythonpath:
            python_paths.append(inherited_pythonpath)
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(python_paths)}
        self.process = subprocess.Popen(
            [str(self.config.python), str(self.config.worker_script)],
            cwd=self.config.root,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Child runtimes may print model paths or source text.  Only the
            # structured, redacted telemetry stream is retained by Phase 10.
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        try:
            message = self._read(self.config.startup_timeout_seconds)
            ready = message.get("ready")
            if message.get("type") != "ready" or message.get("request_id") is not None:
                raise WorkerProtocolError("worker READY envelope is invalid")
            if not isinstance(ready, dict) or ready.get("concurrency") != 1:
                raise WorkerProtocolError("worker concurrency is not one")
            if ready.get("a2rag_preheated") is not True:
                raise WorkerProtocolError("worker did not preheat A2RAG")
            self.ready_payload = ready
        except Exception:
            self.force_stop()
            raise

    def _exchange(self, message_type: str, **values: Any) -> dict[str, Any]:
        process = self.process
        if process is None or process.stdin is None or not self.running:
            raise WorkerProtocolError("worker is not running")
        self.sequence += 1
        request_id = f"phase10-worker-{self.generation:04d}-{self.sequence:08d}"
        payload = {
            "protocol_version": WORKER_PROTOCOL,
            "type": message_type,
            "request_id": request_id,
            **values,
        }
        try:
            process.stdin.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise WorkerProtocolError("worker input pipe failed") from exc
        response = self._read(self.config.execution_timeout_seconds + 5)
        if response.get("request_id") != request_id:
            raise WorkerProtocolError("worker request_id mismatch")
        if response.get("type") == "error":
            raise WorkerProtocolError("worker rejected command")
        return response

    def ping(self) -> None:
        if self._exchange("ping").get("type") != "pong":
            raise WorkerProtocolError("worker ping failed")

    def query(self, request: Mapping[str, Any]) -> dict[str, Any]:
        response = self._exchange("query", request=dict(request))
        if response.get("type") != "result" or not isinstance(response.get("result"), dict):
            raise WorkerProtocolError("worker result envelope is invalid")
        return response["result"]

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        try:
            if self.running:
                try:
                    response = self._exchange("shutdown")
                    if response.get("type") != "shutdown_ack":
                        raise WorkerProtocolError("worker shutdown ack invalid")
                except Exception:
                    self.force_stop()
                    return
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.force_stop()
            return
        finally:
            self._cleanup_streams()

    def force_stop(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        self._cleanup_streams()

    def _cleanup_streams(self) -> None:
        process = self.process
        if process is not None:
            for stream in (process.stdin, process.stdout):
                if stream is not None and not stream.closed:
                    stream.close()
        self.process = None
        self.ready_payload = None


@dataclass
class QueueItem:
    request: dict[str, Any]
    request_id_hash: str
    future: asyncio.Future[dict[str, Any]]
    started: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: bool = False
    enqueued_monotonic: float = field(default_factory=time.monotonic)


class ServiceSupervisor:
    def __init__(
        self,
        config: ServiceConfig | None = None,
        *,
        worker_factory: Callable[[ServiceConfig, int], QAWorkerProcess] | None = None,
        manifest_verifier: ImmutableManifestVerifier | None = None,
    ) -> None:
        self.config = config or ServiceConfig.from_env()
        self.queue: asyncio.Queue[QueueItem] = asyncio.Queue(maxsize=self.config.queue_capacity)
        self.telemetry = ServiceTelemetryLogger(self.config.log_dir / "service.telemetry.jsonl")
        self.manifest = manifest_verifier or ImmutableManifestVerifier(
            self.config.root, self.config.manifest_path
        )
        self.worker_factory = worker_factory or QAWorkerProcess
        self.manifest_verified = False
        self.generation = 0
        self.worker: QAWorkerProcess | None = None
        self.consumer: asyncio.Task[None] | None = None
        self.failure_events: deque[float] = deque()
        self.breaker_open = False
        self.shutting_down = False
        self.active = False

    async def start(self) -> None:
        verified = await asyncio.to_thread(self.manifest.verify_full)
        self.manifest_verified = True
        self.telemetry.event("manifest_verified", manifest_semantic_sha256=verified.semantic_hash)
        try:
            await self._start_worker()
        except Exception as exc:
            self._record_failure("startup_failure")
            self.telemetry.event("worker_start_failed", error_type=type(exc).__name__)
            raise
        self.consumer = asyncio.create_task(self._consume(), name="phase10-qa-consumer")

    async def _start_worker(self) -> None:
        next_generation = self.generation + 1
        worker = self.worker_factory(self.config, next_generation)
        await asyncio.to_thread(worker.start)
        self.worker = worker
        self.generation = next_generation
        self.telemetry.event("worker_ready", generation=self.generation)

    def _record_failure(self, kind: str) -> None:
        now = time.monotonic()
        self.failure_events.append(now)
        cutoff = now - self.config.breaker_window_seconds
        while self.failure_events and self.failure_events[0] < cutoff:
            self.failure_events.popleft()
        if len(self.failure_events) >= self.config.breaker_threshold:
            self.breaker_open = True
        self.telemetry.event(
            "breaker_event", kind=kind, event_count=len(self.failure_events), breaker_open=self.breaker_open,
        )

    def ready(self) -> bool:
        return bool(
            self.manifest_verified
            and not self.shutting_down
            and not self.breaker_open
            and self.worker is not None
            and self.worker.running
            and self.worker.ready_payload is not None
            and self.worker.ready_payload.get("a2rag_preheated") is True
            and self.manifest.cheap_check(minimum_interval_seconds=10)
        )

    def _evidence_channel_state(self) -> dict[str, Any]:
        """Echo the worker's measured retrieval state, defaulting to closed.

        When the worker has not reported ready these stay false/empty, so a
        caller can never read a stale or optimistic value for the table channel.
        """
        payload = self.worker.ready_payload if self.worker else None
        payload = payload or {}
        return {
            "tabgr_ready": payload.get("tabgr_ready") is True,
            "tabgr_document_count": int(payload.get("tabgr_document_count") or 0),
            "evidence_channels": list(payload.get("evidence_channels") or []),
            "evidence_provider_version": payload.get("evidence_provider_version") or "",
            "evidence_provider_fingerprint": payload.get("evidence_provider_fingerprint") or "",
        }

    def readiness(self) -> dict[str, Any]:
        return {
            "service_version": SERVICE_VERSION,
            "ready": self.ready(),
            "manifest_verified": self.manifest_verified,
            "worker_ready": bool(self.worker and self.worker.running and self.worker.ready_payload),
            "a2rag_preheated": bool(self.worker and self.worker.ready_payload and self.worker.ready_payload.get("a2rag_preheated")),
            **self._evidence_channel_state(),
            "breaker_open": self.breaker_open,
            "generation": self.generation,
            "queue_depth": self.queue.qsize(),
            "active": self.active,
        }

    async def submit(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if not self.ready():
            raise ServiceRequestFailure(503, "SERVICE_NOT_READY", "QA worker is not ready")
        loop = asyncio.get_running_loop()
        request_id = request.get("request_id") if isinstance(request.get("request_id"), str) else None
        item = QueueItem(
            request=dict(request),
            request_id_hash=self.telemetry.request_id_hash(request_id) or "",
            future=loop.create_future(),
        )
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull as exc:
            self.telemetry.event("queue_full", request_id_hash=item.request_id_hash)
            raise ServiceRequestFailure(429, "SERVICE_QUEUE_FULL", "QA queue is full") from exc
        try:
            await asyncio.wait_for(item.started.wait(), timeout=self.config.queue_timeout_seconds)
        except TimeoutError as exc:
            item.cancelled = True
            self.telemetry.event("queue_timeout", request_id_hash=item.request_id_hash)
            raise ServiceRequestFailure(504, "SERVICE_TIMEOUT", "QA request exceeded queue timeout") from exc
        try:
            return await asyncio.shield(item.future)
        except asyncio.CancelledError:
            item.cancelled = True
            raise

    async def _consume(self) -> None:
        while True:
            item = await self.queue.get()
            try:
                if item.cancelled:
                    continue
                if not self.ready():
                    if not item.future.done():
                        item.future.set_exception(ServiceRequestFailure(
                            503, "SERVICE_NOT_READY", "QA worker became unavailable"
                        ))
                    continue
                item.started.set()
                # Give a queue-timeout waiter one event-loop turn to mark the
                # item cancelled at the exact deadline.  A timed-out queued
                # request must never cross into execution.
                await asyncio.sleep(0)
                if item.cancelled:
                    continue
                self.active = True
                started = time.monotonic()
                try:
                    assert self.worker is not None
                    result = await asyncio.wait_for(
                        asyncio.to_thread(self.worker.query, item.request),
                        timeout=self.config.execution_timeout_seconds,
                    )
                except TimeoutError:
                    self._record_failure("execution_timeout")
                    if self.worker is not None:
                        await asyncio.to_thread(self.worker.force_stop)
                    failure = ServiceRequestFailure(504, "SERVICE_TIMEOUT", "QA execution timed out")
                    if not item.future.done():
                        item.future.set_exception(failure)
                    await self._restart_unless_open()
                    continue
                except Exception as exc:
                    self._record_failure("worker_crash_or_protocol")
                    if self.worker is not None:
                        await asyncio.to_thread(self.worker.force_stop)
                    self.telemetry.event("worker_failed", error_type=type(exc).__name__)
                    failure = ServiceRequestFailure(
                        503, "SERVICE_WORKER_RESTARTED", "QA worker failed and was restarted"
                    )
                    if not item.future.done():
                        item.future.set_exception(failure)
                    await self._restart_unless_open()
                    continue
                elapsed = round(time.monotonic() - started, 6)
                telemetry = result.get("telemetry") if isinstance(result, dict) else None
                answer = result.get("answer") if isinstance(result, dict) else None
                contract_drift = bool(
                    isinstance(answer, dict)
                    and any(row.get("failure_code") == "INVALID_REQUEST" for row in answer.get("errors", []))
                )
                self.telemetry.event(
                    "query_complete",
                    request_id_hash=item.request_id_hash,
                    generation=self.generation,
                    queue_wait_seconds=round(started - item.enqueued_monotonic, 6),
                    execution_seconds=elapsed,
                    status=answer.get("status") if isinstance(answer, dict) else "invalid",
                    trace_hash=(result.get("trace") or {}).get("trace_hash") if isinstance(result, dict) else None,
                    contract_drift_detected=contract_drift,
                    worker_telemetry_present=isinstance(telemetry, dict),
                )
                if not item.future.done():
                    item.future.set_result(result)
            finally:
                self.active = False
                self.queue.task_done()

    async def _restart_unless_open(self) -> None:
        self.worker = None
        if self.breaker_open or self.shutting_down:
            return
        try:
            await self._start_worker()
        except Exception as exc:
            self._record_failure("startup_failure")
            self.telemetry.event("worker_restart_failed", error_type=type(exc).__name__)

    async def stop(self) -> None:
        self.shutting_down = True
        try:
            await asyncio.wait_for(self.queue.join(), timeout=self.config.shutdown_grace_seconds)
        except TimeoutError:
            pass
        if self.consumer is not None:
            self.consumer.cancel()
            try:
                await self.consumer
            except asyncio.CancelledError:
                pass
        if self.worker is not None:
            await asyncio.to_thread(self.worker.close)
        self.worker = None
        self.telemetry.event("service_stopped", generation=self.generation)


__all__ = [
    "QAWorkerProcess", "ServiceConfig", "ServiceRequestFailure", "ServiceSupervisor",
    "WorkerProtocolError",
]
