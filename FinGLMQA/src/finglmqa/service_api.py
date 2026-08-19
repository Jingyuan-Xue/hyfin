"""Local-only FastAPI projection of the frozen Phase 8 QA pipeline."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .contracts import canonical_json_bytes
from .demo_api import (
    DemoDocumentCatalog,
    DemoRequestError,
    build_demo_wire_request,
    demo_examples,
    demo_metadata,
)
from .service_contracts import (
    ServiceContractError,
    build_service_error,
    build_service_projection,
    persist_trace,
    read_trace,
    validate_wire_request,
)
from .service_supervisor import ServiceConfig, ServiceRequestFailure, ServiceSupervisor


def canonical_response(value: object, status_code: int = 200) -> Response:
    return Response(
        content=canonical_json_bytes(value),
        status_code=status_code,
        media_type="application/json",
    )


async def _limited_body(request: Request, maximum: int) -> bytes:
    header = request.headers.get("content-length")
    if header is not None:
        try:
            if int(header) > maximum:
                raise ServiceRequestFailure(
                    413, "SERVICE_PAYLOAD_TOO_LARGE", "request body exceeds 64 KiB"
                )
        except ValueError as exc:
            raise ServiceRequestFailure(
                400, "SERVICE_PAYLOAD_INVALID", "Content-Length is invalid"
            ) from exc
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > maximum:
            raise ServiceRequestFailure(
                413, "SERVICE_PAYLOAD_TOO_LARGE", "request body exceeds 64 KiB"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def create_app(
    config: ServiceConfig | None = None,
    *,
    supervisor_instance: ServiceSupervisor | None = None,
) -> FastAPI:
    runtime_config = config or ServiceConfig.from_env()
    supervisor = supervisor_instance or ServiceSupervisor(runtime_config)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await supervisor.start()
        try:
            yield
        finally:
            await supervisor.stop()

    app = FastAPI(
        title="FinGLMQA Phase 10",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.supervisor = supervisor

    web_root = Path(runtime_config.root) / "web"
    assets_root = web_root / "assets"
    document_catalog = DemoDocumentCatalog(
        Path(runtime_config.root) / "data/corpus_package/company_year_index.jsonl"
    )
    if assets_root.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_root), name="demo-assets")

    async def execute(checked: dict) -> dict:
        trace_delivery = checked.get("trace_delivery", "reference")
        run = await supervisor.submit(checked)
        answer = run.get("answer") if isinstance(run, dict) else None
        trace = run.get("trace") if isinstance(run, dict) else None
        if not isinstance(answer, dict) or not isinstance(trace, dict):
            raise ServiceRequestFailure(
                503, "SERVICE_WORKER_RESTARTED", "worker returned an invalid PipelineRun"
            )
        persist_trace(trace, runtime_config.trace_dir)
        return build_service_projection(answer, trace, trace_delivery=trace_delivery)

    async def read_json_body(request: Request) -> object:
        body = await _limited_body(request, runtime_config.max_request_bytes)
        try:
            return json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ServiceRequestFailure(
                400, "SERVICE_PAYLOAD_INVALID", "request body is not valid JSON"
            ) from exc

    @app.get("/")
    async def demo_page() -> Response:
        page = web_root / "index.html"
        if not page.is_file():
            return canonical_response(
                build_service_error("SERVICE_NOT_READY", "demo page is unavailable"), 503
            )
        return FileResponse(
            page,
            media_type="text/html",
            headers={
                "Cache-Control": "no-cache",
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/api/v1/meta")
    async def meta() -> Response:
        return canonical_response(demo_metadata(supervisor.readiness()))

    @app.get("/api/v1/examples")
    async def examples() -> Response:
        return canonical_response(demo_examples())

    @app.get("/api/v1/documents")
    async def documents(request: Request) -> Response:
        try:
            query = request.query_params.get("query", "")
            raw_year = request.query_params.get("year")
            report_year = int(raw_year) if raw_year is not None else None
            return canonical_response(document_catalog.response(
                query=query,
                report_year=report_year,
            ))
        except (DemoRequestError, TypeError, ValueError):
            return canonical_response(
                build_service_error(
                    "SERVICE_PAYLOAD_INVALID",
                    "document catalog query is invalid",
                ),
                400,
            )

    @app.get("/health/live")
    async def live() -> Response:
        return canonical_response({"service_version": "phase10-service-v1", "live": True})

    @app.get("/health/ready")
    async def ready() -> Response:
        value = supervisor.readiness()
        return canonical_response(value, 200 if value["ready"] else 503)

    @app.post("/v1/qa")
    async def qa(request: Request) -> Response:
        try:
            raw = await read_json_body(request)
            try:
                checked = validate_wire_request(raw)
            except ServiceContractError as exc:
                raise ServiceRequestFailure(
                    400, "SERVICE_PAYLOAD_INVALID", "request violates QARequest schema"
                ) from exc
            return canonical_response(await execute(checked))
        except ServiceRequestFailure as exc:
            return canonical_response(
                build_service_error(exc.failure_code, exc.message), exc.status_code
            )
        except ServiceContractError:
            return canonical_response(
                build_service_error("SERVICE_NOT_READY", "service projection failed closed"), 503
            )

    @app.post("/api/v1/qa")
    async def demo_qa(request: Request) -> Response:
        try:
            raw = await read_json_body(request)
            try:
                checked = validate_wire_request(build_demo_wire_request(raw))
            except (DemoRequestError, ServiceContractError) as exc:
                raise ServiceRequestFailure(
                    400,
                    "SERVICE_PAYLOAD_INVALID",
                    "request violates the demo QA contract",
                ) from exc
            return canonical_response(await execute(checked))
        except ServiceRequestFailure as exc:
            return canonical_response(
                build_service_error(exc.failure_code, exc.message), exc.status_code
            )
        except ServiceContractError:
            return canonical_response(
                build_service_error("SERVICE_NOT_READY", "service projection failed closed"), 503
            )

    @app.get("/v1/traces/{trace_hash}")
    async def trace(trace_hash: str) -> Response:
        try:
            value = read_trace(trace_hash, runtime_config.trace_dir)
        except ServiceContractError:
            return canonical_response(
                build_service_error("SERVICE_PAYLOAD_INVALID", "trace reference is invalid"), 404
            )
        return canonical_response(value)

    return app


__all__ = ["canonical_response", "create_app"]
