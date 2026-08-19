#!/usr/bin/env python3
"""Independent black-box validation for the table + Qwen experiment.

This module deliberately does not import the experiment runner, its generator,
or its acceptance gate.  Accepted claims are joined back to the frozen Phase 7
text chunks and the experimental table-evidence index and are then checked from
first principles.  The 1.2 GiB table index is streamed and only referenced
fragment IDs are materialized.

The validator has two independent conclusions:

* ``integrity_status`` is fail-closed for schema, scope, citation, extractive,
  numeric, and repeatability violations.
* ``promotion_ready`` additionally considers automatic/manual relevance and
  the public benchmark score.  A higher score can never excuse an integrity
  violation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RESULT_SCHEMA = "finglmqa.experimental.table_qwen_result.v1"
REPORT_SCHEMA = "finglmqa.experimental.table_qwen_independent_validation.v1"

ALLOWED_STATUS = {"ok", "not_found", "error"}
ALLOWED_OUTCOMES = {
    "accepted",
    "generator_refused",
    "generator_rejected_by_gate",
    "generator_invalid_output",
    "retrieval_error",
}
ALLOWED_SOURCE_KINDS = {"a2rag_text", "mixed_narrative", "table_row"}
RESULT_FIELDS = {
    "schema_version",
    "case_id",
    "question",
    "scope",
    "status",
    "generator_outcome",
    "answer",
    "accepted_claim_projection",
    "citation_projection",
    "rejections",
    "source_snapshot",
    "authorization_snapshot",
    "model_config",
    "result_fingerprint",
}
SCOPE_FIELDS = {"document_id", "company", "stock_code", "report_year"}
CLAIM_FIELDS = {
    "ordinal",
    "text",
    "evidence_id",
    "citation_id",
    "document_id",
    "company",
    "stock_code",
    "report_year",
    "source_kind",
}
CITATION_FIELDS = {
    "citation_id",
    "evidence_id",
    "document_id",
    "company",
    "stock_code",
    "report_year",
    "source_kind",
    "source_markdown",
    "line_range",
    "content_sha256",
}
SOURCE_FIELDS = {
    "evidence_id",
    "source_kind",
    "document_id",
    "company",
    "stock_code",
    "report_year",
    "content",
    "content_sha256",
    "source_markdown",
    "line_range",
    "numeric_authorization",
}
AUTHORIZATION_FIELDS = {"evidence_id", "numeric_authorization"}

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
ARABIC_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])[+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)"
    r"(?:\.[0-9]+)?(?:[%％]|(?:万|亿|千|百)?(?:亿元|万元|元|股|人|次|倍))?"
    r"(?![A-Za-z0-9])"
)
A2RAG_ID_RE = re.compile(br'"evidence_chunk_id"\s*:\s*"([^"]+)"')
TABLE_ID_RE = re.compile(br'"fragment_id"\s*:\s*"([^"]+)"')
FORBIDDEN_RUNTIME_KEYS = {
    "created_at",
    "timestamp",
    "time_ns",
    "latency",
    "latency_ms",
    "duration",
    "duration_ms",
    "elapsed",
    "pid",
    "process_id",
    "worker_generation",
    "generation_id",
    "device",
    "gpu",
    "runtime_path",
    "temporary_path",
    "temp_path",
    "request_id",
    "telemetry",
}
GENERIC_QUESTION_TERMS = {
    "根据", "年度", "报告", "年报", "公司", "该公", "情况", "什么", "哪些", "为何",
    "为什么", "如何", "请问", "简要", "分析", "概述", "介绍", "说明", "主要", "相关",
}
RELEVANCE_CONCEPTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("原因", "为何", "为什么"), ("由于", "主要系", "导致", "影响", "得益于", "同比")),
    (("客户", "供应商"), ("客户", "供应商", "采购", "销售", "集中")),
    (("员工", "人员", "职工"), ("员工", "人员", "职工", "人才")),
    (("研发", "专利", "技术"), ("研发", "专利", "技术", "创新")),
    (("风险", "挑战"), ("风险", "挑战", "压力", "不确定", "波动")),
    (("分红", "利润分配"), ("分红", "利润分配", "现金红利", "股利")),
    (("诉讼", "仲裁"), ("诉讼", "仲裁")),
    (("资产", "股权"), ("资产", "股权", "出售", "转让", "过户")),
    (("主营", "业务"), ("主营", "业务", "经营", "产品", "服务")),
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_json_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def semantic_sha256(value: Any) -> str:
    # Match the repository's frozen semantic-hash convention: canonical JSON
    # including exactly one final newline.  The function is reimplemented here
    # instead of imported from the experiment code.
    return sha256_bytes(canonical_json_bytes(value))


def normalize_verbatim(value: str) -> str:
    """NFKC plus whitespace folding; punctuation and wording remain exact."""

    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def iter_jsonl(path: Path, *, require_canonical: bool = False) -> Iterable[dict[str, Any]]:
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL line")
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            if require_canonical and raw != canonical_json_bytes(value):
                raise ValueError(f"{path}:{line_number}: non-canonical JSONL bytes")
            yield value


def read_jsonl(path: Path, *, require_canonical: bool = False) -> list[dict[str, Any]]:
    return list(iter_jsonl(path, require_canonical=require_canonical))


def _has_forbidden_runtime_data(value: Any, *, parent_key: str | None = None) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized_key = str(key).lower()
            if normalized_key in FORBIDDEN_RUNTIME_KEYS or _has_forbidden_runtime_data(
                child, parent_key=normalized_key
            ):
                return True
        return False
    if isinstance(value, list):
        return any(_has_forbidden_runtime_data(child, parent_key=parent_key) for child in value)
    # Absolute host paths are runtime/environment data.  Relative source paths
    # remain allowed and are independently checked against frozen sources.
    # Do not mistake an evidence sentence beginning with '/' for a host path.
    return (
        isinstance(value, str)
        and value.startswith("/")
        and parent_key in {"model", "model_path", "binary", "cache_path", "executable"}
    )


def _line_range_valid(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        and value[0] >= 1
        and value[1] >= value[0]
    )


def _frozen_type3_oracle(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    order: list[str] = []
    result: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        source = row.get("source")
        if not isinstance(source, dict) or source.get("benchmark_type") != "3-1":
            continue
        case_id = row.get("case_id")
        projection = row.get("expected_planning_projection")
        if not isinstance(case_id, str) or not isinstance(projection, dict):
            raise ValueError("Type3 oracle contains an invalid row")
        scope = projection.get("scope", {})
        entities = scope.get("entities", []) if isinstance(scope, dict) else []
        plan = projection.get("plan", {})
        subplans = plan.get("subplans", []) if isinstance(plan, dict) else []
        if len(entities) != 1 or len(subplans) != 1:
            raise ValueError(f"{case_id}: expected one single-document scope")
        identity = entities[0]["identity"]
        documents = entities[0]["documents"]
        payload = subplans[0]["payload"]
        if len(documents) != 1 or payload["document_id"] != documents[0]["document_id"]:
            raise ValueError(f"{case_id}: inconsistent frozen document scope")
        expected = {
            "case_id": case_id,
            "question": source["question"],
            "document_id": documents[0]["document_id"],
            "company_aliases": {identity["company_full"], identity["stock_name"]},
            "stock_code": identity["stock_code"],
            "report_year": documents[0]["report_year"],
        }
        order.append(case_id)
        result[case_id] = expected
    return order, result


def _source_id_from_raw(raw: bytes, pattern: re.Pattern[bytes]) -> str | None:
    # Both indexes are canonical UTF-8 and IDs are ASCII.  Full JSON decoding
    # happens solely for wanted IDs, keeping the 1.2 GiB scan low-allocation.
    match = pattern.search(raw)
    return match.group(1).decode("ascii") if match else None


def _project_external_source(record: Mapping[str, Any], source_kind: str) -> dict[str, Any]:
    if source_kind == "a2rag_text":
        evidence_id = record.get("evidence_chunk_id")
        numeric_authorization = "not_authorized_for_answer"
    else:
        evidence_id = record.get("fragment_id")
        provenance = record.get("provenance")
        numeric_authorization = (
            provenance.get("numeric_authorization") if isinstance(provenance, dict) else None
        )
    content = record.get("content")
    return {
        "evidence_id": evidence_id,
        "source_kind": source_kind,
        "document_id": record.get("document_id"),
        "company_aliases": {
            value for value in (record.get("company_name"), record.get("company_full"))
            if isinstance(value, str) and value
        },
        "stock_code": record.get("stock_code"),
        "report_year": record.get("report_year"),
        "content": content,
        "content_sha256": sha256_text(content) if isinstance(content, str) else None,
        "source_markdown": record.get("source_markdown"),
        "line_range": record.get("line_range") if source_kind == "a2rag_text" else record.get("source_line_range"),
        "numeric_authorization": numeric_authorization,
    }


def stream_external_sources(
    *,
    a2rag_path: Path,
    table_path: Path,
    wanted: Mapping[str, str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Resolve wanted IDs without loading either source index into memory."""

    found: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    groups = {
        "a2rag_text": (a2rag_path, A2RAG_ID_RE),
        "table": (table_path, TABLE_ID_RE),
    }
    wanted_by_group = {
        "a2rag_text": {key for key, kind in wanted.items() if kind == "a2rag_text"},
        "table": {key for key, kind in wanted.items() if kind in {"mixed_narrative", "table_row"}},
    }
    for group, (path, pattern) in groups.items():
        remaining = set(wanted_by_group[group])
        if not remaining:
            continue
        if not path.is_file():
            issues.append(f"source_index_missing:{path.name}")
            continue
        with path.open("rb") as handle:
            for raw in handle:
                source_id = _source_id_from_raw(raw, pattern)
                if source_id not in remaining:
                    continue
                try:
                    record = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    issues.append(f"source_record_invalid_json:{source_id}")
                    remaining.remove(source_id)
                    continue
                kind = "a2rag_text" if group == "a2rag_text" else record.get("fragment_kind")
                if kind != wanted[source_id]:
                    issues.append(f"source_kind_mismatch:{source_id}")
                if source_id in found:
                    issues.append(f"duplicate_external_source_id:{source_id}")
                else:
                    found[source_id] = _project_external_source(record, str(kind))
                remaining.remove(source_id)
                if not remaining:
                    break
        issues.extend(f"unknown_evidence_id:{source_id}" for source_id in sorted(remaining))
    return found, issues


def _question_bigrams(question: str) -> set[str]:
    text = normalize_verbatim(question)
    text = re.sub(r"[A-Za-z0-9，。！？、：；,.!?:;（）()\[\]《》]", "", text)
    for term in GENERIC_QUESTION_TERMS:
        text = text.replace(term, "")
    return {text[index:index + 2] for index in range(max(0, len(text) - 1))}


def automatic_relevance(question: str, claim: str) -> bool:
    normalized_claim = normalize_verbatim(claim)
    if any(
        any(trigger in question for trigger in triggers)
        and any(term in normalized_claim for term in terms)
        for triggers, terms in RELEVANCE_CONCEPTS
    ):
        return True
    return bool(_question_bigrams(question) & {
        normalized_claim[index:index + 2]
        for index in range(max(0, len(normalized_claim) - 1))
    })


def _record_issue(issues: list[dict[str, str]], case_id: str, code: str) -> None:
    issues.append({"case_id": case_id, "code": code})


def _validate_source_snapshot(
    case_id: str,
    snapshot: Mapping[str, Any],
    external: Mapping[str, Any] | None,
    issues: list[dict[str, str]],
) -> None:
    if set(snapshot) != SOURCE_FIELDS:
        _record_issue(issues, case_id, "source_snapshot_schema_invalid")
        return
    evidence_id = snapshot["evidence_id"]
    if external is None:
        _record_issue(issues, case_id, "source_snapshot_unknown_evidence_id")
        return
    for key in (
        "source_kind", "document_id", "stock_code", "report_year", "content",
        "content_sha256", "line_range", "numeric_authorization",
    ):
        if snapshot[key] != external[key]:
            _record_issue(issues, case_id, f"source_snapshot_{key}_mismatch")
    if snapshot["company"] not in external["company_aliases"]:
        _record_issue(issues, case_id, "source_snapshot_company_mismatch")
    if not isinstance(snapshot["content"], str) or sha256_text(snapshot["content"]) != snapshot["content_sha256"]:
        _record_issue(issues, case_id, "source_snapshot_content_hash_mismatch")
    if not _line_range_valid(snapshot["line_range"]):
        _record_issue(issues, case_id, "source_snapshot_line_range_invalid")
    source_markdown = snapshot["source_markdown"]
    if not isinstance(source_markdown, str) or Path(source_markdown).is_absolute() or ".." in Path(source_markdown).parts:
        _record_issue(issues, case_id, "source_snapshot_path_not_relative")
    elif Path(source_markdown).name != Path(str(external["source_markdown"])).name:
        # The Phase 7 worker emits the portable basename while the frozen
        # corpus package records ``refs/source_markdown/<basename>``.  The
        # document_id and independently matched basename jointly pin the file.
        _record_issue(issues, case_id, "source_snapshot_source_markdown_mismatch")
    if not isinstance(evidence_id, str):
        _record_issue(issues, case_id, "source_snapshot_evidence_id_invalid")


def _fingerprint_valid(row: Mapping[str, Any]) -> bool:
    fingerprint = row.get("result_fingerprint")
    payload = {key: value for key, value in row.items() if key != "result_fingerprint"}
    return isinstance(fingerprint, str) and HEX64_RE.fullmatch(fingerprint) is not None and semantic_sha256(payload) == fingerprint


def _validate_result_row(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    external_sources: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    case_id = str(row.get("case_id", "<missing>"))
    issues: list[dict[str, str]] = []
    metrics = {"accepted_claims": 0, "auto_relevant_claims": 0}
    if set(row) != RESULT_FIELDS or row.get("schema_version") != RESULT_SCHEMA:
        _record_issue(issues, case_id, "result_schema_invalid")
        return issues, metrics
    if _has_forbidden_runtime_data(row):
        _record_issue(issues, case_id, "runtime_or_telemetry_in_semantic_result")
    if not _fingerprint_valid(row):
        _record_issue(issues, case_id, "result_fingerprint_mismatch")
    if row["question"] != expected["question"]:
        _record_issue(issues, case_id, "question_not_frozen_oracle")
    scope = row["scope"]
    if not isinstance(scope, dict) or set(scope) != SCOPE_FIELDS:
        _record_issue(issues, case_id, "scope_schema_invalid")
        return issues, metrics
    if scope["document_id"] != expected["document_id"]:
        _record_issue(issues, case_id, "scope_document_mismatch")
    if scope["company"] not in expected["company_aliases"]:
        _record_issue(issues, case_id, "scope_company_mismatch")
    if scope["stock_code"] != expected["stock_code"]:
        _record_issue(issues, case_id, "scope_stock_code_mismatch")
    if scope["report_year"] != expected["report_year"]:
        _record_issue(issues, case_id, "scope_report_year_mismatch")

    status = row["status"]
    outcome = row["generator_outcome"]
    if status not in ALLOWED_STATUS:
        _record_issue(issues, case_id, "status_not_terminal")
    if outcome not in ALLOWED_OUTCOMES:
        _record_issue(issues, case_id, "generator_outcome_unclassified")
    claims = row["accepted_claim_projection"]
    citations = row["citation_projection"]
    sources = row["source_snapshot"]
    authorizations = row["authorization_snapshot"]
    if not all(isinstance(value, list) for value in (claims, citations, sources, authorizations, row["rejections"])):
        _record_issue(issues, case_id, "projection_container_invalid")
        return issues, metrics
    if not isinstance(row["answer"], str) or not isinstance(row["model_config"], dict):
        _record_issue(issues, case_id, "answer_or_model_config_invalid")
        return issues, metrics

    if outcome == "accepted":
        if status != "ok" or not row["answer"] or not claims or not citations:
            _record_issue(issues, case_id, "accepted_outcome_state_inconsistent")
    else:
        if row["answer"] or claims or citations or status == "ok":
            _record_issue(issues, case_id, "nonaccepted_outcome_state_inconsistent")
        expected_status = "error" if outcome in {"generator_invalid_output", "retrieval_error"} else "not_found"
        if outcome in ALLOWED_OUTCOMES and status != expected_status:
            _record_issue(issues, case_id, "nonaccepted_status_inconsistent")

    source_by_id: dict[str, Mapping[str, Any]] = {}
    for snapshot in sources:
        if not isinstance(snapshot, dict):
            _record_issue(issues, case_id, "source_snapshot_not_object")
            continue
        evidence_id = snapshot.get("evidence_id")
        if not isinstance(evidence_id, str) or evidence_id in source_by_id:
            _record_issue(issues, case_id, "source_snapshot_id_invalid_or_duplicate")
            continue
        source_by_id[evidence_id] = snapshot
        _validate_source_snapshot(case_id, snapshot, external_sources.get(evidence_id), issues)

    auth_by_id: dict[str, Any] = {}
    for authorization in authorizations:
        if not isinstance(authorization, dict) or set(authorization) != AUTHORIZATION_FIELDS:
            _record_issue(issues, case_id, "authorization_snapshot_schema_invalid")
            continue
        evidence_id = authorization["evidence_id"]
        if not isinstance(evidence_id, str) or evidence_id in auth_by_id:
            _record_issue(issues, case_id, "authorization_snapshot_id_invalid_or_duplicate")
            continue
        auth_by_id[evidence_id] = authorization["numeric_authorization"]
        source = source_by_id.get(evidence_id)
        if source is None or source.get("numeric_authorization") != authorization["numeric_authorization"]:
            _record_issue(issues, case_id, "authorization_snapshot_not_source_backed")
    if set(auth_by_id) != set(source_by_id):
        _record_issue(issues, case_id, "authorization_snapshot_not_complete")

    citation_by_id: dict[str, Mapping[str, Any]] = {}
    for citation in citations:
        if not isinstance(citation, dict) or set(citation) != CITATION_FIELDS:
            _record_issue(issues, case_id, "citation_schema_invalid")
            continue
        citation_id = citation["citation_id"]
        if not isinstance(citation_id, str) or citation_id in citation_by_id:
            _record_issue(issues, case_id, "citation_id_invalid_or_duplicate")
            continue
        citation_by_id[citation_id] = citation
        evidence_id = citation["evidence_id"]
        source = source_by_id.get(evidence_id)
        if source is None:
            _record_issue(issues, case_id, "citation_unknown_evidence_id")
            continue
        for key in (
            "document_id", "company", "stock_code", "report_year", "source_kind",
            "source_markdown", "line_range", "content_sha256",
        ):
            if citation[key] != source[key]:
                _record_issue(issues, case_id, f"citation_{key}_not_source_backed")
        for key in ("document_id", "company", "stock_code", "report_year"):
            if citation[key] != scope[key]:
                _record_issue(issues, case_id, f"citation_cross_scope_{key}")

    normalized_claim_texts: list[str] = []
    referenced_citations: set[str] = set()
    for expected_ordinal, claim in enumerate(claims):
        metrics["accepted_claims"] += 1
        if not isinstance(claim, dict) or set(claim) != CLAIM_FIELDS:
            _record_issue(issues, case_id, "claim_schema_invalid")
            continue
        if claim["ordinal"] != expected_ordinal:
            _record_issue(issues, case_id, "claim_ordinal_not_canonical")
        citation = citation_by_id.get(claim["citation_id"])
        source = source_by_id.get(claim["evidence_id"])
        if citation is None or source is None:
            _record_issue(issues, case_id, "claim_unknown_citation_or_evidence")
            continue
        referenced_citations.add(claim["citation_id"])
        if citation["evidence_id"] != claim["evidence_id"]:
            _record_issue(issues, case_id, "claim_citation_evidence_mismatch")
        for key in ("document_id", "company", "stock_code", "report_year", "source_kind"):
            expected_value = source[key] if key == "source_kind" else scope[key]
            if claim[key] != expected_value:
                _record_issue(issues, case_id, f"claim_cross_scope_{key}")
        text = claim["text"]
        if not isinstance(text, str) or not normalize_verbatim(text):
            _record_issue(issues, case_id, "claim_text_invalid")
            continue
        normalized_claim = normalize_verbatim(text)
        normalized_source = normalize_verbatim(str(source["content"]))
        if normalized_claim not in normalized_source:
            _record_issue(issues, case_id, "claim_not_normalized_verbatim_substring")
        if source["source_kind"] == "table_row":
            _record_issue(issues, case_id, "table_row_claim_not_authorized")
        numeric_tokens = ARABIC_NUMBER_RE.findall(unicodedata.normalize("NFKC", text))
        if numeric_tokens and auth_by_id.get(claim["evidence_id"]) != "authorized_for_answer":
            _record_issue(issues, case_id, "claim_contains_unauthorized_number")
        normalized_claim_texts.append(text)
        if automatic_relevance(row["question"], text):
            metrics["auto_relevant_claims"] += 1

    if set(citation_by_id) != referenced_citations:
        _record_issue(issues, case_id, "citation_projection_has_unreferenced_entries")
    if outcome == "accepted" and row["answer"] != "\n".join(normalized_claim_texts):
        _record_issue(issues, case_id, "answer_contains_nonclaim_generation")
    return issues, metrics


def _collect_wanted_sources(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], list[dict[str, str]]]:
    wanted: dict[str, str] = {}
    issues: list[dict[str, str]] = []
    for row in rows:
        case_id = str(row.get("case_id", "<missing>"))
        snapshots = row.get("source_snapshot")
        if not isinstance(snapshots, list):
            continue
        for snapshot in snapshots:
            if not isinstance(snapshot, dict):
                continue
            evidence_id = snapshot.get("evidence_id")
            source_kind = snapshot.get("source_kind")
            if not isinstance(evidence_id, str) or source_kind not in ALLOWED_SOURCE_KINDS:
                _record_issue(issues, case_id, "source_snapshot_identity_invalid")
                continue
            previous = wanted.get(evidence_id)
            if previous is not None and previous != source_kind:
                _record_issue(issues, case_id, "evidence_id_kind_collision")
            wanted[evidence_id] = source_kind
    return wanted, issues


def accepted_semantic_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    """Projection independently used for the fixed-30 repeatability gate."""

    return {
        "case_id": row.get("case_id"),
        "status": row.get("status"),
        "generator_outcome": row.get("generator_outcome"),
        "answer": row.get("answer"),
        "accepted_claim_projection": row.get("accepted_claim_projection"),
        "citation_projection": row.get("citation_projection"),
    }


def _score_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    try:
        overall = report["scores"]["bge_m3"]["overall"]
        return {
            "average_score": float(overall["average_score"]),
            "total_score": float(overall["total_score"]),
            "count": int(overall["count"]),
            "report_sha256": sha256_bytes(path.read_bytes()),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid score report: {path}") from exc


def _manual_audit(path: Path | None, accepted_case_ids: set[str]) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"available": False, "count": 0, "relevant_rate": None, "citation_sufficient_rate": None}
    rows = read_jsonl(path, require_canonical=True)
    seen: set[str] = set()
    relevant = 0
    sufficient = 0
    for row in rows:
        if set(row) - {"case_id", "relevant", "citation_sufficient", "notes", "reviewer"}:
            raise ValueError("manual audit has unknown fields")
        case_id = row.get("case_id")
        if (
            not isinstance(case_id, str) or case_id not in accepted_case_ids or case_id in seen
            or not isinstance(row.get("relevant"), bool)
            or not isinstance(row.get("citation_sufficient"), bool)
        ):
            raise ValueError("manual audit contains an invalid row")
        seen.add(case_id)
        relevant += int(row["relevant"])
        sufficient += int(row["citation_sufficient"])
    count = len(rows)
    return {
        "available": True,
        "count": count,
        "relevant_rate": format(relevant / count, ".8f") if count else "0.00000000",
        "citation_sufficient_rate": format(sufficient / count, ".8f") if count else "0.00000000",
    }


def validate_experiment(
    *,
    results_path: Path,
    repeat_results_path: Path | None,
    oracle_path: Path,
    a2rag_path: Path,
    table_path: Path,
    expected_count: int = 260,
    repeat_count: int = 30,
    baseline_score_path: Path | None = None,
    candidate_score_path: Path | None = None,
    manual_audit_path: Path | None = None,
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    try:
        rows = read_jsonl(results_path, require_canonical=True)
    except ValueError as exc:
        return {
            "schema_version": REPORT_SCHEMA,
            "integrity_status": "failed",
            "promotion_ready": False,
            "issues": [{"case_id": "<file>", "code": str(exc)}],
        }
    oracle_order, oracle = _frozen_type3_oracle(oracle_path)
    if len(oracle_order) != expected_count:
        _record_issue(issues, "<oracle>", "oracle_count_mismatch")
    case_ids = [str(row.get("case_id", "<missing>")) for row in rows]
    if len(rows) != expected_count:
        _record_issue(issues, "<results>", "result_count_mismatch")
    if len(set(case_ids)) != len(case_ids):
        _record_issue(issues, "<results>", "duplicate_case_id")
    if case_ids != oracle_order:
        _record_issue(issues, "<results>", "result_order_or_coverage_mismatch")

    wanted, identity_issues = _collect_wanted_sources(rows)
    issues.extend(identity_issues)
    external_sources, external_issues = stream_external_sources(
        a2rag_path=a2rag_path, table_path=table_path, wanted=wanted,
    )
    issues.extend({"case_id": "<sources>", "code": code} for code in external_issues)

    totals = Counter()
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id not in oracle:
            _record_issue(issues, str(case_id), "case_id_not_in_oracle")
            continue
        row_issues, metrics = _validate_result_row(row, oracle[case_id], external_sources)
        issues.extend(row_issues)
        totals.update(metrics)
        if row.get("generator_outcome") in ALLOWED_OUTCOMES:
            totals[f"outcome:{row['generator_outcome']}"] += 1

    repeat_checks: list[dict[str, Any]] = []
    if repeat_results_path is None or not repeat_results_path.is_file():
        _record_issue(issues, "<repeat>", "repeat_results_required")
    else:
        try:
            repeats = read_jsonl(repeat_results_path, require_canonical=True)
        except ValueError as exc:
            repeats = []
            _record_issue(issues, "<repeat>", str(exc))
        repeat_by_id = {row.get("case_id"): row for row in repeats if isinstance(row.get("case_id"), str)}
        fixed_ids = oracle_order[:repeat_count]
        if len(repeats) != repeat_count or set(repeat_by_id) != set(fixed_ids):
            _record_issue(issues, "<repeat>", "fixed_repeat_set_mismatch")
        first_by_id = {row.get("case_id"): row for row in rows}
        for case_id in fixed_ids:
            first = first_by_id.get(case_id)
            second = repeat_by_id.get(case_id)
            identical = bool(first and second) and canonical_json_bytes(
                accepted_semantic_projection(first)
            ) == canonical_json_bytes(accepted_semantic_projection(second))
            repeat_checks.append({"case_id": case_id, "accepted_projection_byte_identical": identical})
            if not identical:
                _record_issue(issues, case_id, "accepted_projection_not_repeatable")

    accepted_ids = {
        str(row.get("case_id")) for row in rows if row.get("generator_outcome") == "accepted"
    }
    manual = _manual_audit(manual_audit_path, accepted_ids)
    baseline = _score_summary(baseline_score_path)
    candidate = _score_summary(candidate_score_path)
    score_comparison: dict[str, Any] = {"baseline": baseline, "candidate": candidate, "available": bool(baseline and candidate)}
    if baseline and candidate:
        score_comparison["average_score_delta"] = format(
            candidate["average_score"] - baseline["average_score"], ".8f"
        )
        score_comparison["total_score_delta"] = format(
            candidate["total_score"] - baseline["total_score"], ".8f"
        )

    issue_counts = Counter(item["code"].split(":", 1)[0] for item in issues)
    claim_count = totals["accepted_claims"]
    accepted_case_count = totals["outcome:accepted"]
    relevance_rate = totals["auto_relevant_claims"] / claim_count if claim_count else 0.0
    integrity_passed = not issues
    quality_checks = {
        "automatic_relevance_at_least_90pct": claim_count > 0 and relevance_rate >= 0.90,
        "manual_relevance_and_citation_at_least_90pct": bool(
            manual["available"] and manual["count"] >= 20
            and float(manual["relevant_rate"]) >= 0.90
            and float(manual["citation_sufficient_rate"]) >= 0.90
        ),
        "score_comparison_available": score_comparison["available"],
        "candidate_score_not_below_no_llm_v4": bool(
            baseline and candidate and candidate["average_score"] >= baseline["average_score"]
        ),
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "integrity_status": "passed" if integrity_passed else "failed",
        "promotion_ready": integrity_passed and all(quality_checks.values()),
        "counts": {
            "results": len(rows),
            "terminal_classified": sum(
                row.get("status") in ALLOWED_STATUS and row.get("generator_outcome") in ALLOWED_OUTCOMES
                for row in rows
            ),
            "outcomes": {
                outcome: totals[f"outcome:{outcome}"] for outcome in sorted(ALLOWED_OUTCOMES)
            },
            "accepted_claims": claim_count,
            "accepted_cases": accepted_case_count,
            "accepted_case_rate": format(accepted_case_count / len(rows), ".8f") if rows else "0.00000000",
            "external_sources_loaded": len(external_sources),
            "issues": len(issues),
        },
        "safety_checks": {
            "all_260_terminal_and_classified": len(rows) == expected_count and all(
                row.get("status") in ALLOWED_STATUS and row.get("generator_outcome") in ALLOWED_OUTCOMES
                for row in rows
            ),
            "no_unknown_or_cross_scope_citation": not any(
                "unknown" in item["code"] or "cross_scope" in item["code"] for item in issues
            ),
            "all_claims_single_source_normalized_verbatim": not any(
                item["code"] == "claim_not_normalized_verbatim_substring" for item in issues
            ),
            "no_unauthorized_numeric_or_table_row_claim": not any(
                item["code"] in {"claim_contains_unauthorized_number", "table_row_claim_not_authorized"}
                for item in issues
            ),
            "no_runtime_telemetry_in_semantic_output": not any(
                item["code"] == "runtime_or_telemetry_in_semantic_result" for item in issues
            ),
            "fixed_repeat_projection_byte_identical": len(repeat_checks) == repeat_count and all(
                item["accepted_projection_byte_identical"] for item in repeat_checks
            ),
        },
        "quality": {
            "automatic_relevance": {
                "relevant_claims": totals["auto_relevant_claims"],
                "claim_count": claim_count,
                "rate": format(relevance_rate, ".8f"),
                "informational_not_safety_override": True,
            },
            "manual_audit": manual,
            "score_comparison": score_comparison,
            "checks": quality_checks,
        },
        "repeatability": repeat_checks,
        "issue_counts": dict(sorted(issue_counts.items())),
        "issues": issues[:1000],
        "inputs": {
            "results_sha256": sha256_bytes(results_path.read_bytes()),
            "repeat_results_sha256": (
                sha256_bytes(repeat_results_path.read_bytes())
                if repeat_results_path is not None and repeat_results_path.is_file() else None
            ),
            "oracle_sha256": sha256_bytes(oracle_path.read_bytes()),
            "a2rag_index": a2rag_path.name,
            "table_index": table_path.name,
        },
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "runs/table_qwen_experiment/results.jsonl")
    parser.add_argument("--repeat-results", type=Path, default=ROOT / "runs/table_qwen_experiment/results_repeat.jsonl")
    parser.add_argument("--oracle", type=Path, default=ROOT / "runs/phase_08/benchmark_decomposition_oracle.jsonl")
    parser.add_argument("--a2rag-index", type=Path, default=ROOT / "data/corpus_package/evidence_chunks.jsonl")
    parser.add_argument("--table-index", type=Path, default=ROOT / "runs/table_evidence_experiment/table_evidence_fragments.jsonl")
    parser.add_argument("--baseline-score", type=Path, default=ROOT / "runs/type3_no_llm_experiment_v4/scoring/score_report.json")
    parser.add_argument("--candidate-score", type=Path, default=ROOT / "runs/table_qwen_experiment/scoring/score_report.json")
    parser.add_argument("--manual-audit", type=Path, default=ROOT / "runs/table_qwen_experiment/manual_audit.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "runs/table_qwen_experiment/independent_validation.json")
    parser.add_argument("--expected-count", type=int, default=260)
    parser.add_argument("--repeat-count", type=int, default=30)
    args = parser.parse_args(argv)
    report = validate_experiment(
        results_path=args.results,
        repeat_results_path=args.repeat_results,
        oracle_path=args.oracle,
        a2rag_path=args.a2rag_index,
        table_path=args.table_index,
        expected_count=args.expected_count,
        repeat_count=args.repeat_count,
        baseline_score_path=args.baseline_score,
        candidate_score_path=args.candidate_score,
        manual_audit_path=args.manual_audit,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(report))
    temporary.replace(args.report)
    print(canonical_json({
        "integrity_status": report["integrity_status"],
        "promotion_ready": report["promotion_ready"],
        "issue_counts": report.get("issue_counts", {}),
    }))
    return 0 if report["integrity_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
