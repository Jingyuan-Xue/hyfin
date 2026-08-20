"""Document-scoped A2RAG + TabGR evidence provider for the demo service."""

from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .evidence_provider import EVIDENCE_PROVIDER_RESULT_SCHEMA
from .type3_tabgr_retriever import Type3TabGRRetriever


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABGR_INDEX = ROOT / "data/indexes/type3/annual_reports_170_v1/tabgr"
HYBRID_PROVIDER_VERSION = "a2rag-tabgr-hybrid-provider-v1"

# Table rows the fused evidence set may hold, whatever the question asks. The
# quota used to depend on a literal list of Chinese numeric terms, which gave an
# untranslated English question the lower quota for no defensible reason.
TABLE_QUOTA = 2


class A2RAGTabGRHybridEvidenceProvider:
    """Fuse document-local text chunks and structured table rows.

    Cross-channel scores are deliberately replaced with a deterministic
    interleaving rank because A2RAG similarity and TabGR lexical scores are not
    calibrated to the same numeric scale.
    """

    def __init__(
        self,
        text_provider: Any,
        *,
        index_dir: str | Path = DEFAULT_TABGR_INDEX,
        corpus_id: str = "annual_reports_170_v1",
    ) -> None:
        if not callable(getattr(text_provider, "retrieve", None)):
            raise TypeError("text_provider must implement retrieve")
        self.text_provider = text_provider
        self.index_dir = Path(index_dir)
        self.tabgr = Type3TabGRRetriever(
            self.index_dir,
            expected_corpus_id=corpus_id,
        )
        manifest_bytes = (self.index_dir / "manifest.json").read_bytes()
        fingerprint = {
            "version": HYBRID_PROVIDER_VERSION,
            "a2rag": str(getattr(text_provider, "provider_fingerprint", "unknown")),
            "tabgr_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        }
        self.provider_fingerprint = hashlib.sha256(
            json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        # Reported through /health/ready and /api/v1/meta so callers can verify
        # that the table channel is really loaded instead of trusting a
        # hardcoded flag. Reading document_ids forces the index open here.
        self.provider_version = HYBRID_PROVIDER_VERSION
        self.tabgr_document_count = len(self.tabgr.document_ids)

    @staticmethod
    def _table_quota(top_k: int) -> int:
        if top_k <= 1:
            return 0
        return min(TABLE_QUOTA, top_k - 1)

    @staticmethod
    def _table_chunk(candidate: Any, identity: Mapping[str, Any], rank: int) -> dict[str, Any]:
        return {
            "chunk_id": "tabgr:" + candidate.evidence_id,
            "document_chunk_ordinal": 10_000_000 + rank,
            "score": "0.00000000",
            "document_id": identity["document_id"],
            "company": identity["company"],
            "stock_code": identity["stock_code"],
            "report_year": identity["report_year"],
            "section_path": list(candidate.heading_path) or ["年报表格"],
            "semantic_tags": ["tabgr", "table_row", candidate.table_id],
            "line_range": list(candidate.line_range),
            "source_markdown": candidate.source_markdown,
            # Keep the exact table-row rendering. Existing numeric gates decide
            # which values may appear in a final claim.
            "content": candidate.display_text,
        }

    @staticmethod
    def _interleave(text_chunks: list[dict[str, Any]], table_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index in range(max(len(text_chunks), len(table_chunks))):
            if index < len(text_chunks):
                rows.append(dict(text_chunks[index]))
            if index < len(table_chunks):
                rows.append(dict(table_chunks[index]))
        for rank, row in enumerate(rows):
            row["score"] = format(Decimal("1") - Decimal(rank) / Decimal("1000000"), ".8f")
        return rows

    def retrieve(self, request: Mapping[str, Any]) -> dict[str, Any]:
        question = str(request.get("question") or "").strip()
        document_id = str(request.get("document_id") or "").strip()
        top_k = request.get("top_k")
        if not question or not document_id or isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("hybrid evidence request is invalid")

        table_quota = self._table_quota(top_k)
        text_quota = top_k - table_quota
        text_result = self.text_provider.retrieve({
            "document_id": document_id,
            "question": question,
            "top_k": text_quota,
        })
        identity = {
            key: text_result[key]
            for key in ("document_id", "company", "stock_code", "report_year")
        }
        candidates = self.tabgr.retrieve(
            question,
            document_id=document_id,
            top_k=table_quota,
        ) if table_quota else []
        table_chunks = [
            self._table_chunk(candidate, identity, rank)
            for rank, candidate in enumerate(candidates, 1)
            if candidate.display_text.strip()
        ]
        chunks = self._interleave(list(text_result["chunks"]), table_chunks)[:top_k]
        return {
            "schema_version": EVIDENCE_PROVIDER_RESULT_SCHEMA,
            "status": "ok",
            **identity,
            "retrieval_method": HYBRID_PROVIDER_VERSION,
            "provider_fingerprint": self.provider_fingerprint,
            "chunks": chunks,
        }


__all__ = [
    "A2RAGTabGRHybridEvidenceProvider",
    "DEFAULT_TABGR_INDEX",
    "HYBRID_PROVIDER_VERSION",
    "TABLE_QUOTA",
]
