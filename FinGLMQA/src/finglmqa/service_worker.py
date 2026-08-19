"""Warm single-concurrency Phase 10 QA worker JSONL protocol."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .evidence_executor import EvidenceExecutor
from .evidence_provider import A2RAGWarmWorkerTransport, DocumentScopedEvidenceProvider
from .hybrid_evidence_provider import A2RAGTabGRHybridEvidenceProvider
from .pipeline import PIPELINE_VERSION, Phase8Pipeline
from .qwen_shadow import QwenShadowGenerator
from .repositories import FactRepository
from .service_contracts import WORKER_PROTOCOL
from .supplement_store import SupplementAwareFactRepository


ROOT = Path(__file__).resolve().parents[2]


def _message(message_type: str, request_id: str | None, **values: Any) -> dict[str, Any]:
    return {
        "protocol_version": WORKER_PROTOCOL,
        "type": message_type,
        "request_id": request_id,
        **values,
    }


def _emit(value: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _online_generator() -> QwenShadowGenerator | None:
    base_url = os.environ.get("A2RAG_CHAT_BASE_URL") or os.environ.get("FINGLMQA_CHAT_BASE_URL")
    model = os.environ.get("A2RAG_CHAT_MODEL") or os.environ.get("FINGLMQA_CHAT_MODEL")
    api_key = os.environ.get("A2RAG_API_KEY") or os.environ.get("FINGLMQA_CHAT_API_KEY")
    explicit = os.environ.get("FINGLMQA_LLM_ENABLED")
    if explicit is None:
        enabled = bool(base_url and model and api_key)
    else:
        normalized = explicit.strip().lower()
        if normalized not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
            raise RuntimeError("FINGLMQA_LLM_ENABLED must be a boolean flag")
        enabled = normalized in {"1", "true", "yes", "on"}
    if not enabled:
        return None
    if not base_url or not model or not api_key:
        raise RuntimeError("online generator configuration is incomplete")
    generator = QwenShadowGenerator(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_seconds=float(os.environ.get("FINGLMQA_LLM_TIMEOUT_SECONDS", "130")),
    )
    generator.ping()
    return generator


def _flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
        raise RuntimeError(f"{name} must be a boolean flag")
    return normalized in {"1", "true", "yes", "on"}


def _build() -> tuple[
    Phase8Pipeline, A2RAGWarmWorkerTransport,
    A2RAGTabGRHybridEvidenceProvider, bool, bool,
]:
    python = Path(os.environ.get(
        "FINGLMQA_A2RAG_PYTHON", ROOT / "refs/a2rag_runtime/.venv/bin/python"
    ))
    script = Path(os.environ.get(
        "FINGLMQA_A2RAG_WORKER", ROOT / "scripts/query_type3_evidence.py"
    ))
    device = os.environ.get("FINGLMQA_EVIDENCE_DEVICE", "cpu")
    model_cache = os.environ.get("FINGLMQA_EVIDENCE_MODEL_CACHE") or None
    transport = A2RAGWarmWorkerTransport(
        python_executable=python,
        worker_script=script,
        device=device,
        model_cache=model_cache,
        timeout_seconds=60,
    )
    text_provider = DocumentScopedEvidenceProvider(transport)
    provider = A2RAGTabGRHybridEvidenceProvider(text_provider)
    transport.ping()
    generator = _online_generator()
    supplemental_enabled = _flag("FINGLMQA_SUPPLEMENTAL_FACTS_ENABLED")
    fact_repository = (
        SupplementAwareFactRepository() if supplemental_enabled else FactRepository()
    )
    pipeline = Phase8Pipeline(
        fact_repository=fact_repository,
        evidence_executor=EvidenceExecutor(provider, generator=generator),
    )
    return (
        pipeline, transport, provider,
        generator is not None, supplemental_enabled,
    )


def serve() -> int:
    pipeline: Phase8Pipeline | None = None
    transport: A2RAGWarmWorkerTransport | None = None
    try:
        (
            pipeline, transport, provider,
            generator_online, supplemental_enabled,
        ) = _build()
        # Every field below is read off the provider that was actually built.
        # Nothing here may be a literal: a text-only provider must report
        # tabgr_ready false rather than silently claiming hybrid retrieval.
        tabgr_document_count = getattr(provider, "tabgr_document_count", 0)
        tabgr_ready = tabgr_document_count > 0
        _emit(_message("ready", None, ready={
            "concurrency": 1,
            "commands": ["ping", "query", "shutdown"],
            "tabgr_ready": tabgr_ready,
            "tabgr_document_count": tabgr_document_count,
            "evidence_channels": ["a2rag", "tabgr"] if tabgr_ready else ["a2rag"],
            "pipeline_version": PIPELINE_VERSION,
            "a2rag_preheated": True,
            "qwen_online": generator_online,
            "evidence_provider_version": getattr(provider, "provider_version", "unknown"),
            "evidence_provider_fingerprint": provider.provider_fingerprint,
            "supplemental_facts_enabled": supplemental_enabled,
        }))
        for raw in sys.stdin:
            request_id: str | None = None
            try:
                message = json.loads(raw)
                if not isinstance(message, dict):
                    raise ValueError("worker input must be an object")
                if message.get("protocol_version") != WORKER_PROTOCOL:
                    raise ValueError("worker protocol mismatch")
                request_id = message.get("request_id")
                if not isinstance(request_id, str) or not request_id:
                    raise ValueError("worker request_id is invalid")
                message_type = message.get("type")
                if message_type == "ping":
                    _emit(_message("pong", request_id))
                    continue
                if message_type == "shutdown":
                    _emit(_message("shutdown_ack", request_id))
                    break
                if message_type != "query" or not isinstance(message.get("request"), dict):
                    raise ValueError("worker command is invalid")
                run = pipeline.run(message["request"])
                _emit(_message("result", request_id, result=run.as_dict()))
            except Exception as exc:
                _emit(_message("error", request_id, error_type=type(exc).__name__))
    except Exception as exc:
        print(
            json.dumps({"event": "worker_start_failed", "error_type": type(exc).__name__}, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )
        return 1
    finally:
        if transport is not None:
            transport.close()
    return 0


__all__ = ["serve"]
