#!/usr/bin/env python3
"""Validate TabGR v2 artifacts and run sanitized-260 structural ablations."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.type3_corpus_profile import (  # noqa: E402
    load_corpus_profile, load_question_profile, sha256_file, source_snapshot,
)
from finglmqa.type3_tabgr_retriever import (  # noqa: E402
    TABGR_RUNTIME_SHA256, TABGR_V2_ROW_SCHEMA, TABGR_V2_TABLE_SCHEMA,
    Type3TabGRError, Type3TabGRRetriever, lexical_tokens, normalize_text,
    numeric_fragments, semantic_sha256, sha256_text,
)


FORBIDDEN_ANNOTATION_FIELDS = frozenset({
    "prompt_answer", "prom_answer", "key_word", "keyword", "reference_answer",
    "answer_key", "average_score", "gold", "references",
})
MAX_PEAK_RSS_KIB = 8 * 1024 * 1024
ABLATION_ARMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("full_tabgr_v2", ()),
    ("legacy_flattened_a0", ("legacy_flattened_a0",)),
    ("no_multilevel_headers", ("no_multilevel_headers",)),
    ("no_hierarchical_rows", ("no_hierarchical_rows",)),
    ("no_unit_period_scope", ("no_unit_period_scope",)),
    ("no_tabgr_ppr", ("no_tabgr_ppr",)),
    ("no_fact_join", ("no_fact_join",)),
    ("table_level_vs_row_level", ("table_level",)),
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_ANNOTATION_FIELDS:
                found.add(str(key))
            found.update(_forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_forbidden_keys(child))
    return found


def _generator_annotation_accesses(paths: Iterable[Path]) -> set[str]:
    """Find exact forbidden field accesses, ignoring prose and larger names."""

    found: set[str] = set()
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            # row["prompt_answer"]
            if isinstance(node, ast.Subscript):
                child = node.slice
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    if child.value.lower() in FORBIDDEN_ANNOTATION_FIELDS:
                        found.add(child.value.lower())
            # row.get("prompt_answer") / row.pop(...)
            if (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"get", "pop", "setdefault"} and node.args
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
                and node.args[0].value.lower() in FORBIDDEN_ANNOTATION_FIELDS
            ):
                found.add(node.args[0].value.lower())
            # {"prompt_answer": ...}
            if isinstance(node, ast.Dict):
                for key in node.keys:
                    if (
                        isinstance(key, ast.Constant) and isinstance(key.value, str)
                        and key.value.lower() in FORBIDDEN_ANNOTATION_FIELDS
                    ):
                        found.add(key.value.lower())
    return found


def _input_freeze(path: Path, *, corpus_id: str) -> dict[str, Any]:
    value = _json(path)
    if value.get("schema_version") != "finglmqa.type3.tabgr.input_freeze.v1":
        raise RuntimeError("unsupported input freeze")
    if value.get("corpus_id") != corpus_id:
        raise RuntimeError("input freeze corpus mismatch")
    if value.get("freeze_fingerprint") != semantic_sha256({
        key: child for key, child in value.items() if key != "freeze_fingerprint"
    }):
        raise RuntimeError("input freeze fingerprint differs")
    return value


def _validate_raw_source_chain(
    *, table_blocks: Path, corpus: Mapping[str, Any], source_hashes: Mapping[str, str]
) -> dict[str, Any]:
    documents = {row["document_id"]: row for row in corpus["documents"]}
    ready_ids: set[str] = set()
    anomaly_ids: set[str] = set()
    status_counts: Counter[str] = Counter()
    active_document: str | None = None
    source_text = ""
    with table_blocks.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise RuntimeError(f"blank table block: {line_number}")
            row = json.loads(line)
            document_id = str(row.get("document_id") or "")
            document = documents.get(document_id)
            if document is None:
                raise RuntimeError("raw table references a document outside corpus")
            if document_id != active_document:
                source = ROOT / corpus["source_ref"] / document["source_markdown"]
                source_text = source.read_text(encoding="utf-8")
                if sha256_text(source_text) != source_hashes[document_id]:
                    raise RuntimeError("validator source hash differs from profile")
                active_document = document_id
            raw = str(row.get("raw_markdown") or "")
            char_range = row.get("char_range")
            if not isinstance(char_range, list) or len(char_range) != 2:
                raise RuntimeError("raw table char_range is invalid")
            start, end = int(char_range[0]), int(char_range[1])
            if source_text[start:end] != raw:
                raise RuntimeError("raw table is not an exact profile-source slice")
            if hashlib.sha1(raw.encode("utf-8")).hexdigest() != row.get("raw_markdown_sha1"):
                raise RuntimeError("raw table SHA1 differs")
            if hashlib.sha256(raw.encode("utf-8")).hexdigest() != row.get("content_hash"):
                raise RuntimeError("raw table SHA256 differs")
            table_id = str(row.get("table_id") or "")
            status = str(row.get("parse_status") or "unknown")
            status_counts[status] += 1
            target = ready_ids if status == "ok" else anomaly_ids
            if table_id in ready_ids or table_id in anomaly_ids:
                raise RuntimeError("raw table_id is not unique")
            target.add(table_id)
    return {
        "ready_ids": ready_ids,
        "anomaly_ids": anomaly_ids,
        "status_counts": dict(sorted(status_counts.items())),
        "row_count": sum(status_counts.values()),
    }


def _validate_structured(
    path: Path, *, corpus_id: str, document_ids: set[str], expected_table_ids: set[str]
) -> dict[str, Any]:
    table_ids: set[str] = set()
    evidence_ids: set[str] = set()
    table_sha: dict[str, str] = {}
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("schema_version") != TABGR_V2_TABLE_SCHEMA:
                raise RuntimeError(f"structured table schema differs at row {line_number}")
            if row.get("corpus_id") != corpus_id or row.get("document_id") not in document_ids:
                raise RuntimeError("structured table crosses corpus/document boundary")
            table_id = str(row["table_id"])
            if table_id in table_ids:
                raise RuntimeError("structured table_id is not unique")
            table_ids.add(table_id)
            table_sha[table_id] = str(row["table_sha256"])
            if row.get("parse_status") != "ready" or row.get("failure_reason") is not None:
                raise RuntimeError("non-ready table entered structured evidence")
            if row.get("span_validation", {}).get("status") != "matrix_exact":
                raise RuntimeError("structured table span/matrix validation is not exact")
            binding = row.get("source_binding") or {}
            if binding.get("status") != "exact" or binding.get("line_range") != row.get("table_line_range"):
                raise RuntimeError("structured table source binding is not exact")
            matrix = row.get("matrix")
            if not isinstance(matrix, list) or not matrix or not any(value for child in matrix for value in child):
                raise RuntimeError("structured table matrix is empty")
            origin_cells = row.get("origin_cells")
            if not isinstance(origin_cells, list) or not origin_cells:
                raise RuntimeError("structured table origin cells are empty")
            for cell in origin_cells:
                if "line_range" in cell or "source_line" in cell:
                    raise RuntimeError("cell-level source line was fabricated")
                if len(str(cell.get("cell_hash") or "")) != 64:
                    raise RuntimeError("origin cell hash is invalid")
            resolution = row.get("header_resolution") or {}
            counts["header_raw_resolved"] += int(resolution.get("raw_resolved_columns", -1))
            counts["header_fallback"] += int(resolution.get("fallback_columns", -1))
            counts["legacy_header_raw_resolved"] += int(resolution.get("legacy_raw_resolved_columns", -1))
            counts["header_total"] += int(resolution.get("total_columns", -1))
            if (
                int(resolution.get("raw_resolved_columns", -1)) < 0
                or int(resolution.get("fallback_columns", -1)) < 0
                or int(resolution.get("raw_resolved_columns", -1)) + int(resolution.get("fallback_columns", -1))
                != int(resolution.get("total_columns", -1))
            ):
                raise RuntimeError("header resolution accounting differs")
            for evidence_id in row.get("row_evidence_ids") or ():
                if evidence_id in evidence_ids:
                    raise RuntimeError("structured row evidence id is not unique")
                evidence_ids.add(str(evidence_id))
            counts["tables"] += 1
            counts["row_evidence"] += len(row.get("row_evidence_ids") or ())
            counts["origin_cells"] += len(origin_cells)
    if table_ids != expected_table_ids:
        raise RuntimeError("structured table coverage differs from ready raw tables")
    if counts["header_raw_resolved"] < counts["legacy_header_raw_resolved"]:
        raise RuntimeError("v2 raw header coverage is lower than legacy flattened coverage")
    return {
        "table_ids": table_ids, "evidence_ids": evidence_ids, "table_sha": table_sha,
        "counts": dict(sorted(counts.items())),
    }


def _authorization_valid(
    authorization: Mapping[str, Any], *, row: Mapping[str, Any], table_sha: Mapping[str, str]
) -> bool:
    if authorization.get("schema_version") != "finglmqa.type3.tabgr.numeric_authorization.v1":
        return False
    if (
        authorization.get("corpus_id") != row.get("corpus_id")
        or authorization.get("document_id") != row.get("document_id")
        or authorization.get("table_id") != row.get("table_id")
        or authorization.get("table_sha256") != row.get("table_sha256")
        or authorization.get("table_sha256") != table_sha.get(str(row.get("table_id")))
        or authorization.get("source_markdown") != row.get("source_markdown")
        or authorization.get("table_line_range") != row.get("table_line_range")
        or authorization.get("allowed_renderings") != [authorization.get("raw_value")]
        or authorization.get("raw_value_sha256") != sha256_text(str(authorization.get("raw_value") or ""))
    ):
        return False
    unsigned = dict(authorization)
    authorization_id = str(unsigned.pop("authorization_id", ""))
    return authorization_id == "t3tabgr-auth-" + semantic_sha256(unsigned)[:24]


def _validate_rows(
    path: Path, *, expected_ids: set[str], table_ids: set[str], table_sha: Mapping[str, str]
) -> dict[str, Any]:
    seen: set[str] = set()
    authorization_ids: set[str] = set()
    fact_ids: set[str] = set()
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("schema_version") != TABGR_V2_ROW_SCHEMA or row.get("record_type") != "table_row":
                raise RuntimeError(f"row evidence schema differs at row {line_number}")
            evidence_id = str(row.get("evidence_id") or "")
            if evidence_id in seen:
                raise RuntimeError("row evidence id is not unique")
            seen.add(evidence_id)
            if row.get("table_id") not in table_ids or row.get("table_sha256") != table_sha.get(str(row.get("table_id"))):
                raise RuntimeError("row evidence table binding differs")
            authorizations = row.get("numeric_authorizations") or []
            allowed: set[str] = set()
            for authorization in authorizations:
                if not _authorization_valid(authorization, row=row, table_sha=table_sha):
                    raise RuntimeError("numeric authorization binding is invalid")
                authorization_id = str(authorization["authorization_id"])
                if authorization_id in authorization_ids:
                    raise RuntimeError("numeric authorization is duplicated across row evidence")
                authorization_ids.add(authorization_id)
                fact_ids.add(str(authorization["fact_id"]))
                allowed.update(normalize_text(value) for value in authorization["allowed_renderings"])
            safe_literals = numeric_fragments(str(row.get("answer_safe_text") or ""))
            bad = [literal for literal in safe_literals if normalize_text(literal) not in allowed]
            if bad:
                raise RuntimeError("unauthorized numeric literal remains in answer_safe_text")
            for cell in row.get("cells") or ():
                if "line_range" in cell or "source_line" in cell:
                    raise RuntimeError("row cell contains a fabricated source line")
                status = cell.get("numeric_status")
                if status not in {"authorized", "unauthorized", "not_numeric"}:
                    raise RuntimeError("row cell numeric status is unsupported")
                for semantic in ("unit", "period", "accounting_scope"):
                    if cell.get(semantic, {}).get("status") not in {"resolved", "unknown", "conflict"}:
                        raise RuntimeError("row semantic state is not fail-closed")
            counts["rows"] += 1
            counts["safe_numeric_literals"] += len(safe_literals)
            counts["authorizations"] += len(authorizations)
            counts["unauthorized_safe_numeric_literals"] += len(bad)
            counts["fallback_headers"] += sum(value.startswith("第") and value.endswith("列") for value in row.get("flattened_column_headers") or ())
    if seen != expected_ids:
        raise RuntimeError("row evidence coverage differs from structured tables")
    return {
        "authorization_ids": authorization_ids, "fact_ids": fact_ids,
        "counts": dict(sorted(counts.items())),
    }


def _validate_anomalies(path: Path, *, expected_ids: set[str], ready_ids: set[str]) -> dict[str, Any]:
    seen: set[str] = set()
    status: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            table_id = str(row.get("table_id") or "")
            if table_id in seen or row.get("evidence_eligible") is not False or row.get("fixed_legacy_anomaly") is not True:
                raise RuntimeError("anomaly audit is not fixed/fail-closed")
            seen.add(table_id)
            status[str(row.get("parse_status"))] += 1
    if seen != expected_ids or seen & ready_ids:
        raise RuntimeError("anomaly audit coverage/boundary differs")
    return {"count": len(seen), "status_counts": dict(sorted(status.items()))}


def _validate_authorization_artifact(path: Path, *, expected_ids: set[str]) -> dict[str, Any]:
    ids: set[str] = set()
    facts: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            ids.add(str(row.get("authorization_id") or ""))
            facts.add(str(row.get("fact_id") or ""))
    if ids != expected_ids or len(ids) != len(facts):
        raise RuntimeError("authorization artifact differs from exact row joins")
    return {"authorization_count": len(ids), "fact_count": len(facts)}


def _validate_shards(index_dir: Path, *, document_ids: set[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with (index_dir / "document_manifest.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if {row.get("document_id") for row in rows} != document_ids:
        raise RuntimeError("index document manifest coverage differs")
    counts: Counter[str] = Counter()
    for row in rows:
        document_id = str(row["document_id"])
        shard = index_dir / str(row["shard_path"])
        if sha256_file(shard) != row.get("shard_sha256"):
            raise RuntimeError("document shard hash differs")
        records = 0
        tables = 0
        with shard.open(encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                if value.get("document_id") != document_id:
                    raise RuntimeError("cross-document record found in shard")
                records += 1
                tables += value.get("record_type") == "table"
        if records != row.get("record_count") or tables != row.get("table_count"):
            raise RuntimeError("document shard record/table count differs")
        counts["records"] += records
        counts["tables"] += tables
    return {"documents": len(rows), **dict(sorted(counts.items()))}


def _query_recall(question: str, candidates: Iterable[Any]) -> float:
    query = set(lexical_tokens(question))
    if not query:
        return 0.0
    evidence: set[str] = set()
    for candidate in candidates:
        evidence.update(lexical_tokens(candidate.display_text))
    return len(query & evidence) / len(query)


def _run_ablations(
    *, retriever: Type3TabGRRetriever, questions: list[dict[str, str]]
) -> dict[str, Any]:
    arms = ABLATION_ARMS
    accumulators: dict[str, dict[str, Any]] = {
        name: {
            "ablations": list(ablations), "nonempty_questions": 0,
            "recall_total": 0.0, "unique_ids": set(), "unauthorized": 0,
            "cross_document": 0, "elapsed_seconds": 0.0,
        }
        for name, ablations in arms
    }
    # Questions are grouped by document.  Running arms inside each question
    # reuses the retriever's bounded document cache without retaining corpus-wide
    # shards or rereading the same shard eight times.
    for question in questions:
        for name, ablations in arms:
            arm_started = time.monotonic()
            candidates = retriever.retrieve(
                question["question"], document_id=question["document_id"],
                top_k=8, ablations=ablations,
            )
            accumulator = accumulators[name]
            accumulator["elapsed_seconds"] += time.monotonic() - arm_started
            accumulator["nonempty_questions"] += bool(candidates)
            accumulator["recall_total"] += _query_recall(question["question"], candidates)
            for candidate in candidates:
                accumulator["unique_ids"].add(candidate.evidence_id)
                accumulator["cross_document"] += candidate.document_id != question["document_id"]
                allowed = {
                    normalize_text(rendering)
                    for authorization in candidate.numeric_authorizations
                    for rendering in authorization.get("allowed_renderings") or ()
                }
                accumulator["unauthorized"] += sum(
                    normalize_text(literal) not in allowed
                    for literal in numeric_fragments(candidate.answer_safe_text)
                )
    reports: dict[str, Any] = {}
    for name, ablations in arms:
        accumulator = accumulators[name]
        reports[name] = {
            "ablations": list(ablations),
            "questions": len(questions),
            "nonempty_questions": accumulator["nonempty_questions"],
            "mean_query_token_recall": round(accumulator["recall_total"] / len(questions), 6),
            "unique_evidence_ids": len(accumulator["unique_ids"]),
            "cross_document_candidates": accumulator["cross_document"],
            "unauthorized_safe_numeric_literals": accumulator["unauthorized"],
            "elapsed_seconds": round(accumulator["elapsed_seconds"], 3),
            "ppr_applicable": False if name == "no_tabgr_ppr" else None,
            "ppr_note": (
                "No query-specific external PPR scores were supplied; this arm audits the pinned optional path and is quality-neutral."
                if name == "no_tabgr_ppr" else None
            ),
        }
        if accumulator["cross_document"] or accumulator["unauthorized"]:
            raise RuntimeError(f"ablation safety gate failed: {name}")
    full = reports["full_tabgr_v2"]
    for value in reports.values():
        value["delta_mean_query_token_recall_vs_full"] = round(
            value["mean_query_token_recall"] - full["mean_query_token_recall"], 6
        )
    return reports


def retrieval_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    corpus = load_corpus_profile(args.corpus_profile)
    question_profile, questions = load_question_profile(args.question_profile, corpus_profile=corpus)
    retriever = Type3TabGRRetriever(args.index_dir, expected_corpus_id=corpus["corpus_id"])
    rows_by_arm: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in ABLATION_ARMS}
    elapsed: Counter[str] = Counter()
    recall: Counter[str] = Counter()
    for question in questions:
        for name, ablations in ABLATION_ARMS:
            started = time.monotonic()
            candidates = retriever.retrieve(
                question["question"], document_id=question["document_id"], top_k=8,
                ablations=ablations,
            )
            elapsed[name] += time.monotonic() - started
            recall[name] += _query_recall(question["question"], candidates)
            rows_by_arm[name].append({
                "question_id": question["question_id"],
                "candidate_ids": [candidate.evidence_id for candidate in candidates],
                "scores": [format(candidate.retrieval_score, ".12f") for candidate in candidates],
            })
    arms = {
        name: {
            "ablations": list(ablations),
            "question_count": len(questions),
            "candidate_sequence_sha256": semantic_sha256(rows_by_arm[name]),
            "mean_query_token_recall": round(recall[name] / len(questions), 6),
            "elapsed_seconds": round(elapsed[name], 3),
        }
        for name, ablations in ABLATION_ARMS
    }
    report = {
        "schema_version": "finglmqa.type3.tabgr.retrieval_snapshot.v1",
        "corpus_id": corpus["corpus_id"],
        "question_profile_id": question_profile["question_profile_id"],
        "retriever_version": "type3-tabgr-lexical-retriever-v2",
        "arms": arms,
    }
    _atomic_json(args.snapshot_output, report)
    return report


def compare_retrieval_snapshots(args: argparse.Namespace) -> dict[str, Any]:
    before_path, after_path = args.compare_snapshots
    before = _json(before_path)
    after = _json(after_path)
    identity_fields = (
        "schema_version", "corpus_id", "question_profile_id", "retriever_version",
    )
    if any(before.get(field) != after.get(field) for field in identity_fields):
        raise RuntimeError("retrieval snapshot identities differ")
    if set(before.get("arms", {})) != set(after.get("arms", {})):
        raise RuntimeError("retrieval snapshot arm sets differ")
    arms: dict[str, Any] = {}
    all_exactly_equal = True
    before_total = 0.0
    after_total = 0.0
    for name in sorted(before["arms"]):
        first = before["arms"][name]
        second = after["arms"][name]
        equal = all(
            first.get(field) == second.get(field)
            for field in (
                "ablations", "question_count", "candidate_sequence_sha256",
                "mean_query_token_recall",
            )
        )
        all_exactly_equal = all_exactly_equal and equal
        before_elapsed = float(first["elapsed_seconds"])
        after_elapsed = float(second["elapsed_seconds"])
        before_total += before_elapsed
        after_total += after_elapsed
        arms[name] = {
            "candidate_sequences_and_scores_equal": equal,
            "candidate_sequence_sha256": second["candidate_sequence_sha256"],
            "mean_query_token_recall_equal": (
                first["mean_query_token_recall"] == second["mean_query_token_recall"]
            ),
            "before_elapsed_seconds": before_elapsed,
            "after_elapsed_seconds": after_elapsed,
            "speedup": round(before_elapsed / after_elapsed, 3) if after_elapsed else None,
        }
    if not all_exactly_equal:
        raise RuntimeError("cached retrieval changed candidates, scores, or proxy metrics")
    report = {
        "schema_version": "finglmqa.type3.tabgr.cache_equivalence_report.v1",
        "status": "passed",
        "corpus_id": before["corpus_id"],
        "question_profile_id": before["question_profile_id"],
        "before_snapshot": str(before_path),
        "after_snapshot": str(after_path),
        "all_candidate_sequences_and_scores_equal": True,
        "all_proxy_metrics_equal": True,
        "before_total_arm_elapsed_seconds": round(before_total, 3),
        "after_total_arm_elapsed_seconds": round(after_total, 3),
        "total_arm_speedup": round(before_total / after_total, 3) if after_total else None,
        "arms": arms,
    }
    _atomic_json(args.snapshot_comparison_output, report)
    return report


def validate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    corpus = load_corpus_profile(args.corpus_profile)
    question_profile, questions = load_question_profile(args.question_profile, corpus_profile=corpus)
    if _forbidden_keys(questions):
        raise RuntimeError("benchmark annotations entered sanitized evaluator questions")
    freeze = _input_freeze(args.expected_input_manifest, corpus_id=corpus["corpus_id"])
    if sha256_file(args.table_blocks) != freeze["table_blocks_sha256"]:
        raise RuntimeError("validator table_blocks pin differs")
    if sha256_file(args.facts) != freeze["financial_facts_sha256"]:
        raise RuntimeError("validator financial facts pin differs")
    source_hashes = source_snapshot(corpus, workspace_root=ROOT)
    raw = _validate_raw_source_chain(
        table_blocks=args.table_blocks, corpus=corpus, source_hashes=source_hashes
    )
    if (
        raw["row_count"] != freeze["expected_input_tables"]
        or len(raw["ready_ids"]) != freeze["expected_ready_tables"]
        or len(raw["anomaly_ids"]) != freeze["expected_anomalies"]
    ):
        raise RuntimeError("raw input counts differ from freeze")

    package_manifest = _json(args.package_dir / "manifest.json")
    index_manifest = _json(args.index_dir / "manifest.json")
    facts_manifest = _json(args.facts_dir / "manifest.json")
    if any(value.get("corpus_id") != corpus["corpus_id"] for value in (package_manifest, index_manifest, facts_manifest)):
        raise RuntimeError("artifact manifest corpus_id differs")
    if index_manifest.get("tabgr_runtime_sha256") != TABGR_RUNTIME_SHA256:
        raise RuntimeError("index TabGR runtime pin differs")
    paths = {
        "structured_tables_sha256": args.package_dir / "structured_tables.jsonl",
        "table_row_evidence_sha256": args.package_dir / "table_row_evidence.jsonl",
        "anomaly_audit_sha256": args.package_dir / "anomaly_audit.jsonl",
        "selected_fact_authorizations_sha256": args.facts_dir / "selected_fact_authorizations.jsonl",
        "document_manifest_sha256": args.index_dir / "document_manifest.jsonl",
    }
    for key, path in paths.items():
        if package_manifest.get("artifacts", {}).get(key) != sha256_file(path):
            raise RuntimeError(f"package artifact hash differs: {key}")

    document_ids = {row["document_id"] for row in corpus["documents"]}
    structured = _validate_structured(
        paths["structured_tables_sha256"], corpus_id=corpus["corpus_id"],
        document_ids=document_ids, expected_table_ids=raw["ready_ids"],
    )
    row_report = _validate_rows(
        paths["table_row_evidence_sha256"], expected_ids=structured["evidence_ids"],
        table_ids=structured["table_ids"], table_sha=structured["table_sha"],
    )
    anomaly_report = _validate_anomalies(
        paths["anomaly_audit_sha256"], expected_ids=raw["anomaly_ids"], ready_ids=structured["table_ids"],
    )
    authorization_report = _validate_authorization_artifact(
        paths["selected_fact_authorizations_sha256"], expected_ids=row_report["authorization_ids"],
    )
    if authorization_report["authorization_count"] != freeze["expected_selected_facts"]:
        raise RuntimeError("exact selected fact join count differs from freeze")
    shard_report = _validate_shards(args.index_dir, document_ids=document_ids)
    retriever = Type3TabGRRetriever(args.index_dir, expected_corpus_id=corpus["corpus_id"])
    ablations = _run_ablations(retriever=retriever, questions=questions)
    if ablations["full_tabgr_v2"]["nonempty_questions"] < ablations["legacy_flattened_a0"]["nonempty_questions"]:
        raise RuntimeError("v2 query coverage is below legacy flattened A0")

    generator_paths = [
        ROOT / "src/finglmqa/type3_tabgr_retriever.py",
        ROOT / "scripts/build_type3_tabgr_index.py",
    ]
    leaked_names = sorted(_generator_annotation_accesses(generator_paths))
    if leaked_names:
        raise RuntimeError(f"generator implementation names benchmark annotations: {leaked_names!r}")
    peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if peak_rss_kib >= MAX_PEAK_RSS_KIB:
        raise RuntimeError("validator peak RSS exceeded 8 GiB")
    report = {
        "schema_version": "finglmqa.type3.tabgr.validation_report.v2",
        "status": "passed",
        "corpus_id": corpus["corpus_id"],
        "question_profile_id": question_profile["question_profile_id"],
        "evaluator_only_question_fields": question_profile["allowed_fields"],
        "benchmark_annotations_available_to_generator": False,
        "generator_forbidden_annotation_names": leaked_names,
        "raw_source_chain": {key: value for key, value in raw.items() if not key.endswith("_ids")},
        "structured": structured["counts"],
        "rows": row_report["counts"],
        "anomalies": anomaly_report,
        "authorizations": authorization_report,
        "index_shards": shard_report,
        "ablations": ablations,
        "safety": {
            "cross_document_candidates": 0,
            "unauthorized_safe_numeric_literals": row_report["counts"].get("unauthorized_safe_numeric_literals", 0),
            "fabricated_cell_line_numbers": 0,
            "fixed_anomalies_entering_evidence": 0,
            "source_binding_failures": 0,
        },
        "resource": {
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "peak_rss_kib": peak_rss_kib,
            "peak_rss_below_8gib": True,
        },
        "passed_gates": [
            "profile_and_input_pins", "source_exact_binding", "span_matrix_exact",
            "document_prefilter", "raw_header_coverage_not_below_legacy", "29_anomalies_fail_closed",
            "selected_fact_exact_join", "unauthorized_safe_numeric_zero", "sanitized_260_ablations",
        ],
    }
    _atomic_json(args.output, report)
    _atomic_json(args.output.parent / "ablation_report.json", {
        "schema_version": "finglmqa.type3.tabgr.ablation_report.v1",
        "corpus_id": corpus["corpus_id"], "question_profile_id": question_profile["question_profile_id"],
        "arms": ablations,
    })
    return report


def validate_repeatability(args: argparse.Namespace) -> dict[str, Any]:
    if args.first_build_report is None:
        raise RuntimeError("--first-build-report is required for --repeatability-only")
    first = _json(args.first_build_report)
    second = _json(args.second_build_report)
    if first.get("status") != "passed" or second.get("status") != "passed":
        raise RuntimeError("both repeatability build reports must be passed")
    if first.get("corpus_id") != second.get("corpus_id"):
        raise RuntimeError("repeatability build corpus ids differ")
    first_hashes = first.get("artifact_hashes")
    second_hashes = second.get("artifact_hashes")
    if first_hashes != second_hashes:
        raise RuntimeError("repeatability artifact hashes differ")
    if first.get("package_manifest_fingerprint") != second.get("package_manifest_fingerprint"):
        raise RuntimeError("repeatability package manifest fingerprints differ")
    if first.get("index_manifest_fingerprint") != second.get("index_manifest_fingerprint"):
        raise RuntimeError("repeatability index manifest fingerprints differ")
    corpus = load_corpus_profile(args.corpus_profile)
    source_hashes = source_snapshot(corpus, workspace_root=ROOT)
    report = {
        "schema_version": "finglmqa.type3.tabgr.repeatability_report.v1",
        "status": "passed",
        "corpus_id": corpus["corpus_id"],
        "build_1_report_sha256": sha256_file(args.first_build_report),
        "build_2_report_sha256": sha256_file(args.second_build_report),
        "artifact_hashes_equal": True,
        "artifact_hashes": first_hashes,
        "package_manifest_fingerprint_equal": True,
        "index_manifest_fingerprint_equal": True,
        "source_document_count": len(source_hashes),
        "source_unchanged": True,
    }
    _atomic_json(args.repeatability_output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-profile", type=Path, default=ROOT / "data/corpus_package/type3/annual_reports_170_v1/corpus_manifest.json")
    parser.add_argument("--question-profile", type=Path, default=ROOT / "data/corpus_package/type3/annual_reports_170_v1/questions/type3_260_dev_v1/question_profile.json")
    parser.add_argument("--expected-input-manifest", type=Path, default=ROOT / "data/schemas/type3/tabgr_annual_reports_170_v1_input_freeze.json")
    parser.add_argument("--table-blocks", type=Path, default=ROOT / "data/corpus_package/table_blocks.jsonl")
    parser.add_argument("--facts", type=Path, default=ROOT / "data/facts/financial_facts.jsonl")
    parser.add_argument("--package-dir", type=Path, default=ROOT / "data/corpus_package/type3/annual_reports_170_v1/tabgr_table_v2")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "data/indexes/type3/annual_reports_170_v1/tabgr")
    parser.add_argument("--facts-dir", type=Path, default=ROOT / "data/facts/type3/annual_reports_170_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/type3_a2rag_tabgr_experiment_v1/annual_reports_170_v1/phase_3/validation_report.json")
    parser.add_argument("--repeatability-only", action="store_true")
    parser.add_argument("--first-build-report", type=Path)
    parser.add_argument("--second-build-report", type=Path, default=ROOT / "runs/type3_a2rag_tabgr_experiment_v1/annual_reports_170_v1/phase_3/build_report.json")
    parser.add_argument("--repeatability-output", type=Path, default=ROOT / "runs/type3_a2rag_tabgr_experiment_v1/annual_reports_170_v1/phase_3/repeatability_report.json")
    parser.add_argument("--retrieval-snapshot-only", action="store_true")
    parser.add_argument("--snapshot-output", type=Path, default=ROOT / "runs/type3_a2rag_tabgr_experiment_v1/annual_reports_170_v1/phase_3/retrieval_snapshot.json")
    parser.add_argument("--compare-snapshots", type=Path, nargs=2, metavar=("BEFORE", "AFTER"))
    parser.add_argument("--snapshot-comparison-output", type=Path, default=ROOT / "runs/type3_a2rag_tabgr_experiment_v1/annual_reports_170_v1/phase_3/performance/cache_equivalence_report.json")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.compare_snapshots:
        result = compare_retrieval_snapshots(arguments)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    elif arguments.retrieval_snapshot_only:
        result = retrieval_snapshot(arguments)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    elif arguments.repeatability_only:
        result = validate_repeatability(arguments)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        result = validate(arguments)
        print(json.dumps({
            "status": result["status"], "structured": result["structured"],
            "rows": result["rows"], "safety": result["safety"], "resource": result["resource"],
        }, ensure_ascii=False, sort_keys=True))
