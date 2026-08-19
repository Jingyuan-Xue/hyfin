#!/usr/bin/env python3
"""Fast read-only end-to-end checks for the packaged demo."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


WEB = os.environ.get("DEMO_WEB_URL", "http://127.0.0.1:4173").rstrip("/")
QA = os.environ.get("DEMO_QA_URL", "http://127.0.0.1:8010").rstrip("/")
RISK = os.environ.get("DEMO_RISK_URL", "http://127.0.0.1:8012").rstrip("/")


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 15.0) -> Any:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    token = os.environ.get("DEMO_ACCESS_TOKEN", "").strip()
    if token:
        headers["X-Demo-Token"] = token
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status // 100 != 2:
            raise RuntimeError(f"HTTP {response.status}")
        return json.loads(response.read())


def require(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(detail or name)
    print(f"[PASS] {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-translation",
        action="store_true",
        default=os.environ.get("DEMO_REQUIRE_TRANSLATION", "").lower() in {"1", "true", "yes"},
        help="fail unless Tencent translation is configured and ready",
    )
    parser.add_argument(
        "--full-qa",
        action="store_true",
        default=os.environ.get("DEMO_FULL_QA_SMOKE", "").lower() in {"1", "true", "yes"},
        help="also issue one LLM-backed narrative QA request",
    )
    args = parser.parse_args()
    try:
        qa_health = request_json(f"{QA}/health/ready")
        require(
            "FinGLMQA ready",
            qa_health.get("ready") is True
            and qa_health.get("manifest_verified") is True
            and qa_health.get("a2rag_preheated") is True,
        )

        # The table channel is verified against the worker's measured state:
        # tabgr_document_count is len(retriever.document_ids), so a missing or
        # unreadable TabGR index reports 0 here instead of passing on a flag.
        require(
            "TabGR index loaded",
            qa_health.get("tabgr_ready") is True
            and int(qa_health.get("tabgr_document_count", 0)) == 170,
            f"expected 170 TabGR documents, worker reported "
            f"{qa_health.get('tabgr_document_count')!r} (tabgr_ready="
            f"{qa_health.get('tabgr_ready')!r})",
        )
        require(
            "Hybrid evidence provider active",
            qa_health.get("evidence_provider_version") == "a2rag-tabgr-hybrid-provider-v1"
            and len(str(qa_health.get("evidence_provider_fingerprint") or "")) == 64,
            f"unexpected evidence provider: "
            f"{qa_health.get('evidence_provider_version')!r}",
        )

        qa_meta = request_json(f"{QA}/api/v1/meta")
        engine = qa_meta.get("engine") or {}
        require(
            "QA hybrid retrieval",
            engine.get("online_evidence") == "a2rag+tabgr"
            and engine.get("tabgr_online") is True,
        )
        require("Online LLM generation", engine.get("qwen_online") is True)

        documents = request_json(f"{QA}/api/v1/documents")
        require("QA document catalog", int(documents.get("corpus_total", 0)) == 170)

        risk_health = request_json(f"{RISK}/health/ready")
        coverage = risk_health.get("coverage") or {}
        require(
            "Risk exposure ready",
            risk_health.get("ready") is True
            and int(coverage.get("available_company_count", 0)) >= 24,
        )

        web_health = request_json(f"{WEB}/api/health")
        require(
            "Frontend gateway ready",
            web_health.get("status") == "ok"
            and int(web_health.get("case_count", 0)) == 26,
        )

        translation = request_json(f"{WEB}/api/translation/health")
        require(
            "Translation layer initialized",
            translation.get("provider") in {"tencent_mps", "tencent_tmt"}
            and int(translation.get("static_entries", 0)) >= 60,
        )
        if args.require_translation:
            require("Tencent translation ready", translation.get("ready") is True)
        elif translation.get("ready") is not True:
            print("[WARN] Tencent translation credentials are not configured; dynamic English evidence remains unavailable.")

        english_cases = request_json(f"{WEB}/api/demo/cases?lang=en")
        require(
            "Static English company catalog",
            bool(english_cases)
            and english_cases[0].get("name") == "Gree Electric"
            and english_cases[0].get("name_original") == "格力电器",
        )

        case = request_json(f"{WEB}/api/demo/cases/gree-2021")
        require(
            "Industry evidence carousel data",
            len((case.get("text_evidence") or {}).get("items") or []) >= 2
            and len((case.get("table_evidence") or {}).get("items") or []) >= 2,
        )

        proxy_meta = request_json(f"{WEB}/api/finglmqa/meta")
        require(
            "Frontend → QA proxy",
            (proxy_meta.get("engine") or {}).get("online_evidence") == "a2rag+tabgr",
        )
        proxy_risk = request_json(f"{WEB}/api/risk/health")
        require("Frontend → risk proxy", proxy_risk.get("ready") is True)

        if args.full_qa:
            result = request_json(
                f"{WEB}/api/finglmqa/qa",
                {
                    "question": "What operating risks did FIYTA face in 2019?",
                    "display_question": "What operating risks did FIYTA face in 2019?",
                    "canonical_question_zh": "飞亚达2019年面临哪些经营风险？",
                    "question_language": "en",
                    "response_language": "en",
                    "company": "飞亚达",
                    "report_year": 2019,
                },
                timeout=180.0,
            )
            answer = result.get("answer")
            if isinstance(answer, dict):
                answer_text = answer.get("answer_text") or answer.get("text")
                answer_status = answer.get("status") or result.get("status")
            else:
                answer_text = answer
                answer_status = result.get("status")
            require("Canonical Chinese QA retrieval", result.get("canonical_question_zh") == "飞亚达2019年面临哪些经营风险？")
            require(
                "LLM-backed narrative QA",
                answer_status in {"ok", "partial"}
                and isinstance(answer_text, str)
                and bool(answer_text.strip()),
            )
            # Channel mix of the citations that actually reached the answer.
            # Which channel wins is question-dependent, so only the absence of
            # any grounding fails; the split is reported for inspection.
            citations = [c for c in (result.get("citations") or []) if isinstance(c, dict)]
            chunk_ids = [
                str((c.get("provenance") or {}).get("evidence_chunk_id") or "")
                for c in citations
            ]
            table_cited = sum(1 for value in chunk_ids if value.startswith("tabgr:"))
            require("Answer carries grounded citations", bool(chunk_ids))
            print(
                f"[INFO] Citation channels: {table_cited} TabGR table row(s), "
                f"{len(chunk_ids) - table_cited} A2RAG text chunk(s)."
            )
            if not table_cited:
                print(
                    "[WARN] No TabGR row was cited for this narrative question. "
                    "The index is loaded (checked above); the composer selected "
                    "text evidence only."
                )

        print("[OK] All integrated checks passed.")
        return 0
    except (AssertionError, OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
