"""Small, honest browser-facing contract for the stable QA service."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


DEMO_API_VERSION = "finglmqa.demo_api.v1"
MAX_DEMO_QUESTION_CHARS = 1000

DEMO_EXAMPLES: tuple[dict[str, str], ...] = (
    {
        "id": "fact-revenue",
        "category": "财务事实",
        "question": "2019年飞亚达营业收入是多少？",
    },
    {
        "id": "formula-growth",
        "category": "公式计算",
        "question": "一品红2020年相比2019年的营业收入增长率是多少？",
    },
    {
        "id": "narrative-risk",
        "category": "叙述证据",
        "question": "飞亚达2019年面临哪些经营风险？",
    },
    {
        "id": "metadata-code",
        "category": "公司信息",
        "question": "2019年飞亚达的股票代码是什么？",
    },
)


class DemoRequestError(ValueError):
    """The simplified browser request is invalid."""


class DemoDocumentCatalog:
    """Safe public projection of the immutable company/year index."""

    def __init__(self, index_path: str | Path) -> None:
        rows: list[dict[str, Any]] = []
        path = Path(index_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                try:
                    source = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise DemoRequestError(
                        f"document catalog row {line_number} is invalid"
                    ) from exc
                required = {
                    "document_id", "stock_code", "stock_name", "company_full",
                    "report_year", "aliases", "status",
                }
                if not isinstance(source, dict) or not required.issubset(source):
                    raise DemoRequestError(
                        f"document catalog row {line_number} is incomplete"
                    )
                if source["status"] != "unique":
                    continue
                aliases = source["aliases"]
                if not isinstance(aliases, list) or not all(
                    isinstance(value, str) for value in aliases
                ):
                    raise DemoRequestError(
                        f"document catalog row {line_number} aliases are invalid"
                    )
                try:
                    report_year = int(source["report_year"])
                except (TypeError, ValueError) as exc:
                    raise DemoRequestError(
                        f"document catalog row {line_number} year is invalid"
                    ) from exc
                projected = {
                    "document_id": str(source["document_id"]),
                    "stock_code": str(source["stock_code"]),
                    "stock_name": str(source["stock_name"]),
                    "company_full": str(source["company_full"]),
                    "report_year": report_year,
                }
                search_values = [
                    projected["document_id"],
                    projected["stock_code"],
                    projected["stock_name"],
                    projected["company_full"],
                    *aliases,
                ]
                projected["_search"] = _search_text(" ".join(search_values))
                rows.append(projected)
        if not rows:
            raise DemoRequestError("document catalog is empty")
        rows.sort(key=lambda row: (
            row["report_year"], row["stock_code"], row["document_id"]
        ))
        if len({row["document_id"] for row in rows}) != len(rows):
            raise DemoRequestError("document catalog contains duplicate document IDs")
        self._rows = tuple(rows)
        self.years = tuple(sorted({row["report_year"] for row in rows}))

    def response(
        self,
        *,
        query: str = "",
        report_year: int | None = None,
    ) -> dict[str, Any]:
        if not isinstance(query, str) or len(query) > 100:
            raise DemoRequestError("document query is invalid")
        if report_year is not None and (
            isinstance(report_year, bool)
            or not isinstance(report_year, int)
            or not 1900 <= report_year <= 2200
        ):
            raise DemoRequestError("document report year is invalid")
        needle = _search_text(query)
        selected = [
            row for row in self._rows
            if (not needle or needle in row["_search"])
            and (report_year is None or row["report_year"] == report_year)
        ]
        return {
            "api_version": DEMO_API_VERSION,
            "corpus_total": len(self._rows),
            "total": len(selected),
            "years": list(self.years),
            "documents": [
                {key: value for key, value in row.items() if key != "_search"}
                for row in selected
            ],
        }


def _search_text(value: str) -> str:
    return "".join(value.casefold().split())


def _online_generation_enabled() -> bool:
    explicit = os.environ.get("FINGLMQA_LLM_ENABLED")
    if explicit is not None:
        return explicit.strip().lower() in {"1", "true", "yes", "on"}
    return bool(
        (os.environ.get("A2RAG_CHAT_BASE_URL") or os.environ.get("FINGLMQA_CHAT_BASE_URL"))
        and (os.environ.get("A2RAG_CHAT_MODEL") or os.environ.get("FINGLMQA_CHAT_MODEL"))
        and (os.environ.get("A2RAG_API_KEY") or os.environ.get("FINGLMQA_CHAT_API_KEY"))
    )


def demo_metadata(runtime_state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Describe the running engine.

    ``runtime_state`` is the supervisor readiness payload. The retrieval fields
    are reported from it rather than hardcoded, so that a service running
    without a usable TabGR index advertises ``a2rag`` only instead of claiming
    hybrid retrieval it cannot perform. Callers that pass nothing get the
    conservative view: the table channel is reported closed until measured.
    """
    generator_online = _online_generation_enabled()
    state = runtime_state or {}
    tabgr_online = state.get("tabgr_ready") is True
    channels = [str(name) for name in (state.get("evidence_channels") or [])]
    if not channels:
        channels = ["a2rag", "tabgr"] if tabgr_online else ["a2rag"]
    return {
        "api_version": DEMO_API_VERSION,
        "name": "FinGLMQA",
        "description": "面向上市公司年报的可审计问答演示",
        "engine": {
            "release": "phase8-grounded-llm" if generator_online else "phase8-deterministic",
            "online_evidence": "+".join(channels),
            "tabgr_online": tabgr_online,
            "tabgr_document_count": int(state.get("tabgr_document_count") or 0),
            "evidence_provider_version": state.get("evidence_provider_version") or "",
            "qwen_online": generator_online,
        },
        "request_limits": {
            "question_max_characters": MAX_DEMO_QUESTION_CHARS,
            "locale": "zh-CN",
        },
        "answer_statuses": [
            "ok",
            "partial",
            "needs_clarification",
            "not_found",
            "unsupported",
            "fallback_required",
            "blocked",
            "error",
        ],
    }


def demo_examples() -> dict[str, Any]:
    return {
        "api_version": DEMO_API_VERSION,
        "examples": [dict(row) for row in DEMO_EXAMPLES],
    }


def build_demo_wire_request(payload: Any) -> dict[str, Any]:
    """Convert the deliberately small public request into a frozen QARequest."""

    if not isinstance(payload, dict):
        raise DemoRequestError("request JSON root must be an object")
    allowed = {"question", "company", "report_year"}
    if set(payload) - allowed:
        raise DemoRequestError("request contains unsupported fields")

    question = payload.get("question")
    if not isinstance(question, str):
        raise DemoRequestError("question must be a string")
    question = question.strip()
    if not question:
        raise DemoRequestError("question must not be empty")
    if len(question) > MAX_DEMO_QUESTION_CHARS:
        raise DemoRequestError("question is too long")

    request: dict[str, Any] = {
        "schema_version": "finglmqa.phase8.qa_request.v1",
        "request_id": f"demo_{uuid4().hex}",
        "question": question,
        "locale": "zh-CN",
        "trace_delivery": "reference",
    }

    if "company" in payload:
        company = payload["company"]
        if not isinstance(company, str) or not company.strip():
            raise DemoRequestError("company must be a non-empty string")
        if len(company.strip()) > 100:
            raise DemoRequestError("company is too long")
        request["company"] = company.strip()

    if "report_year" in payload:
        report_year = payload["report_year"]
        if isinstance(report_year, bool) or not isinstance(report_year, int):
            raise DemoRequestError("report_year must be an integer")
        if not 1900 <= report_year <= 2200:
            raise DemoRequestError("report_year is outside the supported range")
        request["report_year"] = report_year

    return request


__all__ = [
    "DEMO_API_VERSION",
    "DEMO_EXAMPLES",
    "MAX_DEMO_QUESTION_CHARS",
    "DemoDocumentCatalog",
    "DemoRequestError",
    "build_demo_wire_request",
    "demo_examples",
    "demo_metadata",
]
