#!/usr/bin/env python3
"""Validate Phase 2 A2RAG artifacts and run retrieval ablations.

Evaluation reads only the sanitized Phase 1 question records plus v10's
already-selected source-evidence audit.  The builder/retriever remain isolated
from benchmark answers, keywords, and scores.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.type3_a2rag_retriever import (  # noqa: E402
    ATOM_SCHEMA,
    BUILDER_VERSION,
    EMBEDDING_DIMENSION,
    INDEX_SCHEMA,
    UNIT_SCHEMA,
    Type3A2RAGError,
    Type3A2RAGRetriever,
    canonical_json_bytes,
    read_json,
    read_jsonl,
    is_formula_only_block,
    semantic_sha256,
    sha256_file,
)
from finglmqa.type3_corpus_profile import (  # noqa: E402
    load_question_profile,
    source_snapshot,
    validate_corpus_profile,
)


MAX_OVERSIZE_SINGLE_ATOM_CHARS = 4096


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(dict(value)))
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(dict(row)))
    temporary.replace(path)


def _compact(value: str) -> str:
    return "".join(value.split())


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def structural_validation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    profile = validate_corpus_profile(read_json(args.corpus_manifest))
    package_manifest = read_json(args.package_dir / "manifest.json")
    index_manifest = read_json(args.index_dir / "index_manifest.json")
    checks: dict[str, bool] = {
        "package_builder_version": package_manifest.get("builder_version") == BUILDER_VERSION,
        "index_schema": index_manifest.get("schema_version") == INDEX_SCHEMA,
        "index_builder_version": index_manifest.get("builder_version") == BUILDER_VERSION,
        "corpus_id_match": package_manifest.get("corpus_id")
        == index_manifest.get("corpus_id") == profile["corpus_id"],
        "profile_hash_match": package_manifest.get("corpus_profile_sha256")
        == index_manifest.get("corpus_profile_sha256") == profile["profile_sha256"],
        "source_snapshot_match": index_manifest.get("source_snapshot_sha256")
        == semantic_sha256(source_snapshot(profile, workspace_root=ROOT)),
        "document_prefilter_required": index_manifest.get("document_prefilter_required") is True,
        "cross_document_scoring_forbidden": index_manifest.get("cross_document_scoring_allowed") is False,
        "embedding_dimension_1024": index_manifest.get("embedding_model", {}).get("dimension")
        == EMBEDDING_DIMENSION,
    }
    package_artifacts = package_manifest["artifacts"]
    package_paths = {
        "text_atoms.jsonl": args.package_dir / "text_atoms.jsonl",
        "retrieval_units.jsonl": args.package_dir / "retrieval_units.jsonl",
        "document_summaries.jsonl": args.package_dir / "document_summaries.jsonl",
    }
    for name, path in package_paths.items():
        checks[f"package_hash:{name}"] = sha256_file(path) == package_artifacts[name]

    documents = {str(row["document_id"]): row for row in profile["documents"]}
    source_root = (ROOT / str(profile["source_ref"])).resolve()
    source_text: dict[str, str] = {}
    source_bytes: dict[str, bytes] = {}
    for document_id, document in documents.items():
        raw = (source_root / str(document["source_markdown"])).read_bytes()
        text = raw.decode("utf-8", errors="strict")
        source_text[document_id] = text
        source_bytes[document_id] = raw

    atom_ids: set[str] = set()
    atom_projection: dict[str, tuple[str, tuple[int, int], str]] = {}
    atom_count = 0
    atom_alignment_failures = 0
    table_html_atom_count = 0
    for atom in read_jsonl(package_paths["text_atoms.jsonl"]):
        atom_count += 1
        if atom.get("schema_version") != ATOM_SCHEMA:
            atom_alignment_failures += 1
            continue
        atom_id = str(atom["atom_id"])
        document_id = str(atom["document_id"])
        if atom_id in atom_ids or document_id not in documents:
            atom_alignment_failures += 1
            continue
        atom_ids.add(atom_id)
        start, end = atom["char_range"]
        byte_start, byte_end = atom["byte_range"]
        content = str(atom["content"])
        if (
            source_text[document_id][start:end] != content
            or source_bytes[document_id][byte_start:byte_end] != content.encode("utf-8")
            or atom["source_sha256"] != documents[document_id]["source_sha256"]
        ):
            atom_alignment_failures += 1
        if "<table" in content.lower() or "</table" in content.lower():
            table_html_atom_count += 1
        atom_projection[atom_id] = (document_id, (start, end), content)

    unit_count = 0
    unit_alignment_failures = 0
    cross_document_units = 0
    missing_atom_references = 0
    unit_table_html_count = 0
    oversize_rows: list[dict[str, Any]] = []
    for unit in read_jsonl(package_paths["retrieval_units.jsonl"]):
        unit_count += 1
        if unit.get("schema_version") != UNIT_SCHEMA:
            unit_alignment_failures += 1
            continue
        document_id = str(unit["document_id"])
        start, end = unit["char_range"]
        byte_start, byte_end = unit["byte_range"]
        content = str(unit["content"])
        if (
            document_id not in documents
            or source_text[document_id][start:end] != content
            or source_bytes[document_id][byte_start:byte_end] != content.encode("utf-8")
        ):
            unit_alignment_failures += 1
        children = [atom_projection.get(str(value)) for value in unit["atom_ids"]]
        if any(value is None for value in children):
            missing_atom_references += 1
        elif any(value[0] != document_id for value in children if value is not None):
            cross_document_units += 1
        if "<table" in content.lower() or "</table" in content.lower():
            unit_table_html_count += 1
        char_count = end - start
        if char_count > 1800:
            pseudo_markers = [
                marker for marker in ("<img", "data:image", "![", "<svg", "<table")
                if marker in content.lower()
            ]
            if is_formula_only_block(content):
                pseudo_markers.append("formula_only_display_math")
            oversize_rows.append(
                {
                    "unit_id": unit["unit_id"],
                    "document_id": document_id,
                    "char_count": char_count,
                    "atom_count": len(unit["atom_ids"]),
                    "atom_id": unit["atom_ids"][0] if len(unit["atom_ids"]) == 1 else None,
                    "oversize_single_atom": unit.get("oversize_single_atom") is True,
                    "pseudo_narrative_markers": pseudo_markers,
                    "content_sha256": unit["content_sha256"],
                    "preview": content[:240],
                }
            )
    invalid_oversize = [
        row for row in oversize_rows
        if not row["oversize_single_atom"]
        or row["atom_count"] != 1
        or row["char_count"] > MAX_OVERSIZE_SINGLE_ATOM_CHARS
        or any(
            marker != "formula_only_display_math"
            for marker in row["pseudo_narrative_markers"]
        )
    ]

    checks.update(
        {
            "atom_count_match": atom_count == package_manifest["atom_count"],
            "unit_count_match": unit_count
            == package_manifest["unit_count"] == index_manifest["unit_count"],
            "unit_count_gate": unit_count <= 150_000,
            "atom_alignment_exact": atom_alignment_failures == 0,
            "unit_alignment_exact": unit_alignment_failures == 0,
            "no_table_html_atoms": table_html_atom_count == 0,
            "no_table_html_units": unit_table_html_count == 0,
            "all_unit_atoms_exist": missing_atom_references == 0,
            "no_cross_document_atom_references": cross_document_units == 0,
            "oversize_only_single_atom_exception": not invalid_oversize,
            "oversize_full_content_no_truncation": all(
                row["atom_id"] in atom_projection
                and row["char_count"] == len(atom_projection[row["atom_id"]][2])
                for row in oversize_rows
            ),
        }
    )

    manifest_documents = {str(row["document_id"]): row for row in index_manifest["documents"]}
    checks["index_documents_exact"] = set(manifest_documents) == set(documents)
    dense_bytes = 0
    index_unit_total = 0
    for document_id, document in manifest_documents.items():
        shard_dir = args.index_dir / str(document["shard_path"])
        for name, expected_hash in document["artifacts"].items():
            path = shard_dir / name
            checks[f"shard_hash:{document_id}:{name}"] = (
                path.is_file() and sha256_file(path) == expected_hash
            )
        shard_units = list(read_jsonl(shard_dir / "units.jsonl"))
        index_unit_total += len(shard_units)
        checks[f"shard_scope:{document_id}"] = all(
            row["document_id"] == document_id for row in shard_units
        )
        for name in ("dense_context.npy", "dense_content.npy"):
            path = shard_dir / name
            matrix = np.load(path, mmap_mode="r")
            checks[f"shape:{document_id}:{name}"] = matrix.shape == (
                int(document["unit_count"]), EMBEDDING_DIMENSION
            ) and matrix.dtype == np.float16
            dense_bytes += path.stat().st_size
    checks["index_shard_count_sum"] = index_unit_total == unit_count

    retriever_source = (ROOT / "src/finglmqa/type3_a2rag_retriever.py").read_text(
        encoding="utf-8"
    )
    builder_source = (ROOT / "scripts/build_type3_a2rag_index.py").read_text(
        encoding="utf-8"
    )
    checks["builder_has_no_question_input_cli"] = "--questions" not in builder_source
    checks["builder_has_no_reference_answer_field"] = "reference_answer" not in builder_source
    checks["retriever_has_no_reference_answer_field"] = "reference_answer" not in retriever_source
    checks["retriever_requires_document_id"] = "document_id: str" in retriever_source
    checks["retriever_exposes_batch_facets"] = "def retrieve_many(" in retriever_source
    checks["retriever_explicit_no_truncation"] = '"content_truncated": False' in retriever_source
    checks["retriever_filters_formula_only_ocr"] = "is_formula_only_block" in retriever_source

    artifact_bytes = sum(
        path.stat().st_size
        for root in (args.package_dir, args.index_dir)
        for path in root.rglob("*") if path.is_file()
    )
    checks["artifact_size_gate"] = artifact_bytes <= 3 * 1024**3
    failed = sorted(key for key, value in checks.items() if not value)
    report = {
        "schema_version": "finglmqa.type3.a2rag.validation_report.v1",
        "builder_version": BUILDER_VERSION,
        "corpus_id": profile["corpus_id"],
        "counts": {
            "documents": len(documents),
            "atoms": atom_count,
            "units": unit_count,
            "oversize_single_atom_units": len(oversize_rows),
            "max_unit_chars": max([1800] + [row["char_count"] for row in oversize_rows]),
            "oversize_pseudo_narrative_units": sum(
                bool(row["pseudo_narrative_markers"]) for row in oversize_rows
            ),
        },
        "oversize_policy": {
            "normal_max_chars": 1800,
            "exception": "one complete exact source atom only",
            "exception_max_chars": MAX_OVERSIZE_SINGLE_ATOM_CHARS,
            "retrieval_content_truncated": False,
            "online_answer_policy": "project exact atom_id; never cut a source atom mid-content",
            "oversize_samples": sorted(
                oversize_rows, key=lambda row: (-row["char_count"], row["unit_id"])
            )[:20],
        },
        "resources": {
            "artifact_bytes": artifact_bytes,
            "artifact_gib": round(artifact_bytes / 1024**3, 6),
            "dense_bytes": dense_bytes,
        },
        "safety": {
            "builder_question_inputs": [],
            "builder_reference_answer_fields": [],
            "retriever_reference_answer_fields": [],
            "source_markdown_read_only": True,
            "document_prefilter_before_scoring": True,
            "cross_document_units": cross_document_units,
            "table_html_in_text_units": unit_table_html_count,
        },
        "checks": checks,
        "failed_checks": failed,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "passed": not failed,
    }
    return report


def _load_v10_audit(path: Path) -> dict[str, dict[str, Any]]:
    audit: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        case_id = str(row["case_id"])
        selected_ids = [str(value) for value in row.get("selected_fragment_ids") or []]
        facets_by_fragment: dict[str, set[str]] = defaultdict(set)
        for run in row.get("selector_runs") or []:
            for selection in run.get("selections") or []:
                fragment_id = str(selection.get("fragment_id") or "")
                if fragment_id in selected_ids:
                    facets_by_fragment[fragment_id].update(
                        str(value) for value in selection.get("facet_ids") or []
                    )
        citations = {
            str(value.get("candidate_id")): value
            for value in row.get("citations") or []
            if value.get("candidate_id")
        }
        projections = {
            str(value["fragment_id"]): value
            for value in row.get("selected_fragment_projection") or []
        }
        gold = []
        for fragment_id in selected_ids:
            projection = projections.get(fragment_id)
            if not projection:
                continue
            citation = citations.get(fragment_id, {})
            gold.append(
                {
                    "fragment_id": fragment_id,
                    "source_evidence_id": citation.get("source_evidence_id"),
                    "text": projection.get("text") or "",
                    "line_range": projection.get("line_range") or [],
                    "facet_ids": sorted(facets_by_fragment.get(fragment_id, set())),
                }
            )
        trace = row.get("no_qwen_retrieval_trace") or []
        audit[case_id] = {
            "gold": gold,
            "legacy_trace": trace[0] if trace else {},
        }
    return audit


def _legacy_evidence(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        result[str(row["evidence_chunk_id"])] = {
            "candidate_id": str(row["evidence_chunk_id"]),
            "document_id": str(row["document_id"]),
            "line_range": row["line_range"],
            "content": str(row["content"]),
            "adjacent_table_ids": [],
        }
    return result


def _rrf_lists(channels: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    scores: dict[str, float] = defaultdict(float)
    values: dict[str, dict[str, Any]] = {}
    first: dict[str, tuple[int, int]] = {}
    for channel_index, channel in enumerate(channels):
        for rank, candidate in enumerate(channel, 1):
            candidate_id = str(candidate["candidate_id"])
            scores[candidate_id] += 1.0 / (60.0 + rank)
            values.setdefault(candidate_id, dict(candidate))
            first.setdefault(candidate_id, (channel_index, rank))
    return [
        values[candidate_id]
        for candidate_id in sorted(
            scores,
            key=lambda value: (-scores[value], first[value], value),
        )
    ]


def _legacy_candidates(
    audit: Mapping[str, Any], evidence: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    trace = audit.get("legacy_trace") or {}
    dense = []
    for candidate_id in trace.get("dense_retrieved_chunk_ids") or []:
        value = evidence.get(str(candidate_id))
        if value:
            dense.append(dict(value))
    sparse = [
        {
            "candidate_id": str(candidate_id),
            "document_id": "",
            "line_range": [],
            "content": "",
            "adjacent_table_ids": [],
        }
        for candidate_id in trace.get("sparse_fragment_ids") or []
    ]
    return _rrf_lists([dense, sparse])


def _new_candidates(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": str(row["unit_id"]),
            "document_id": str(row["document_id"]),
            "line_range": list(row["line_range"]),
            "content": str(row["content"]),
            "adjacent_table_ids": list(row["adjacent_table_ids"]),
        }
        for row in result["candidates"]
    ]


def _matches(candidate: Mapping[str, Any], gold: Mapping[str, Any]) -> bool:
    if candidate["candidate_id"] in {gold["fragment_id"], gold.get("source_evidence_id")}:
        return True
    text = _compact(str(gold.get("text") or ""))
    if text and text in _compact(str(candidate.get("content") or "")):
        return True
    candidate_lines = candidate.get("line_range") or []
    gold_lines = gold.get("line_range") or []
    return (
        len(candidate_lines) == 2
        and len(gold_lines) == 2
        and int(candidate_lines[0]) <= int(gold_lines[1])
        and int(candidate_lines[1]) >= int(gold_lines[0])
    )


def _variant_metrics(
    cases: Sequence[Mapping[str, Any]], variant: str
) -> dict[str, Any]:
    gold_total = 0
    hits = {5: 0, 15: 0}
    cases_with_gold = 0
    cases_hit = {5: 0, 15: 0}
    reciprocal_ranks: list[float] = []
    facet_total = 0
    facet_hits = {5: 0, 15: 0}
    for case in cases:
        gold = case["gold"]
        if not gold:
            continue
        cases_with_gold += 1
        candidates = case["variants"][variant]
        gold_total += len(gold)
        gold_facets = {value for row in gold for value in row.get("facet_ids") or []}
        facet_total += len(gold_facets)
        first_rank = 0
        for rank, candidate in enumerate(candidates, 1):
            if any(_matches(candidate, row) for row in gold):
                first_rank = rank
                break
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        for cutoff in (5, 15):
            selected = candidates[:cutoff]
            matched_gold = [
                row for row in gold if any(_matches(candidate, row) for candidate in selected)
            ]
            hits[cutoff] += len(matched_gold)
            cases_hit[cutoff] += bool(matched_gold)
            matched_facets = {
                value for row in matched_gold for value in row.get("facet_ids") or []
            }
            facet_hits[cutoff] += len(matched_facets)
    return {
        "audit_cases": cases_with_gold,
        "gold_evidence_count": gold_total,
        "recall_at_5": round(hits[5] / gold_total, 8) if gold_total else 0.0,
        "recall_at_15": round(hits[15] / gold_total, 8) if gold_total else 0.0,
        "mrr": round(statistics.mean(reciprocal_ranks), 8) if reciprocal_ranks else 0.0,
        "case_coverage_at_5": round(cases_hit[5] / cases_with_gold, 8) if cases_with_gold else 0.0,
        "case_coverage_at_15": round(cases_hit[15] / cases_with_gold, 8) if cases_with_gold else 0.0,
        "facet_coverage_at_5": round(facet_hits[5] / facet_total, 8) if facet_total else 0.0,
        "facet_coverage_at_15": round(facet_hits[15] / facet_total, 8) if facet_total else 0.0,
    }


def ablation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    profile = validate_corpus_profile(read_json(args.corpus_manifest))
    question_profile, questions = load_question_profile(
        args.question_profile, corpus_profile=profile
    )
    if any(set(row) != {"question_id", "question", "document_id"} for row in questions):
        raise Type3A2RAGError("sanitized question records contain unexpected fields")
    audit = _load_v10_audit(args.v10_results)
    legacy_evidence = _legacy_evidence(args.legacy_evidence)
    retriever = Type3A2RAGRetriever(args.index_dir, model_path=args.model_path)

    embed_started = time.perf_counter()
    query_vectors = retriever.encode_queries([row["question"] for row in questions], batch_size=8)
    embedding_ms = (time.perf_counter() - embed_started) * 1000.0
    variants = (
        "A0_legacy_v10_cached",
        "A1_dense",
        "A2_dense_sparse",
        "no_heading",
        "no_adjacency",
        "legacy_plus_adjacent_delta",
    )
    latencies: dict[str, list[float]] = defaultdict(list)
    cases: list[dict[str, Any]] = []
    for ordinal, question in enumerate(questions):
        case_id = str(question["question_id"])
        case_audit = audit.get(case_id, {"gold": [], "legacy_trace": {}})
        legacy = _legacy_candidates(case_audit, legacy_evidence)
        outputs: dict[str, list[dict[str, Any]]] = {"A0_legacy_v10_cached": legacy}
        shard_started = time.perf_counter()
        retriever._load_shard(str(question["document_id"]))
        latencies["document_shard_load_or_lru"].append(
            (time.perf_counter() - shard_started) * 1000.0
        )
        configs = {
            "A1_dense": ("dense", True, False),
            "A2_dense_sparse": ("hybrid", True, True),
            "no_heading": ("hybrid", False, True),
            "no_adjacency": ("hybrid", True, False),
        }
        for name, (mode, include_heading, include_adjacency) in configs.items():
            search_started = time.perf_counter()
            result = retriever.retrieve(
                str(question["question"]),
                document_id=str(question["document_id"]),
                top_k=15,
                mode=mode,
                include_heading=include_heading,
                include_adjacency=include_adjacency,
                query_vector=query_vectors[ordinal],
            )
            latencies[name].append((time.perf_counter() - search_started) * 1000.0)
            candidates = _new_candidates(result)
            if any(row["document_id"] != question["document_id"] for row in candidates):
                raise Type3A2RAGError(f"cross-document result: {case_id}:{name}")
            outputs[name] = candidates
        adjacent = [
            row for row in outputs["A2_dense_sparse"] if row["adjacent_table_ids"]
        ]
        outputs["legacy_plus_adjacent_delta"] = _rrf_lists([legacy, adjacent])
        cases.append(
            {
                "question_id": case_id,
                "document_id": question["document_id"],
                "gold": case_audit["gold"],
                "variants": outputs,
            }
        )
        if (ordinal + 1) % 50 == 0:
            print(f"ablation {ordinal + 1}/{len(questions)}", file=sys.stderr, flush=True)

    metrics = {name: _variant_metrics(cases, name) for name in variants}
    recovery_gold = 0
    recovery_legacy_hits = 0
    recovery_adjacent_hits = 0
    recovery_union_hits = 0
    recovered_evidence = 0
    recovered_cases = 0
    for case in cases:
        gold = case["gold"]
        if not gold:
            continue
        legacy_top = case["variants"]["A0_legacy_v10_cached"][:15]
        adjacent_top = [
            row for row in case["variants"]["A2_dense_sparse"][:15]
            if row["adjacent_table_ids"]
        ]
        legacy_hit_ids = {
            row["fragment_id"] for row in gold
            if any(_matches(candidate, row) for candidate in legacy_top)
        }
        adjacent_hit_ids = {
            row["fragment_id"] for row in gold
            if any(_matches(candidate, row) for candidate in adjacent_top)
        }
        recovered = adjacent_hit_ids - legacy_hit_ids
        recovery_gold += len(gold)
        recovery_legacy_hits += len(legacy_hit_ids)
        recovery_adjacent_hits += len(adjacent_hit_ids)
        recovery_union_hits += len(legacy_hit_ids | adjacent_hit_ids)
        recovered_evidence += len(recovered)
        recovered_cases += bool(recovered)
    latency_report = {
        name: {
            "calls": len(values),
            "p50_ms": round(_percentile(values, 0.5), 6),
            "p95_ms": round(_percentile(values, 0.95), 6),
            "mean_ms": round(statistics.mean(values), 6) if values else 0.0,
        }
        for name, values in latencies.items()
    }
    compact_cases = []
    for case in cases:
        compact_cases.append(
            {
                "schema_version": "finglmqa.type3.a2rag.ablation_case.v1",
                "question_id": case["question_id"],
                "document_id": case["document_id"],
                "gold": case["gold"],
                "variant_candidate_ids": {
                    name: [row["candidate_id"] for row in case["variants"][name]][:15]
                    for name in variants
                },
                "variant_hits_at_15": {
                    name: [
                        row["fragment_id"] for row in case["gold"]
                        if any(
                            _matches(candidate, row)
                            for candidate in case["variants"][name][:15]
                        )
                    ]
                    for name in variants
                },
            }
        )
    atomic_jsonl(args.run_dir / "ablation_cases.jsonl", compact_cases)
    a0 = metrics["A0_legacy_v10_cached"]
    a2 = metrics["A2_dense_sparse"]
    adjacent = metrics["legacy_plus_adjacent_delta"]
    report = {
        "schema_version": "finglmqa.type3.a2rag.ablation_report.v1",
        "builder_version": BUILDER_VERSION,
        "corpus_id": profile["corpus_id"],
        "question_profile_id": question_profile["question_profile_id"],
        "sanitized_question_count": len(questions),
        "generator_visible_question_fields": list(question_profile["allowed_fields"]),
        "annotations_available_to_generator": question_profile[
            "annotations_available_to_generator"
        ],
        "v10_selected_source_evidence_audit_cases": sum(bool(row["gold"]) for row in cases),
        "metrics": metrics,
        "deltas": {
            "A2_minus_A0_recall_at_5": round(a2["recall_at_5"] - a0["recall_at_5"], 8),
            "A2_minus_A0_recall_at_15": round(a2["recall_at_15"] - a0["recall_at_15"], 8),
            "A2_minus_A0_mrr": round(a2["mrr"] - a0["mrr"], 8),
            "adjacent_delta_minus_A0_recall_at_15": round(
                adjacent["recall_at_15"] - a0["recall_at_15"], 8
            ),
        },
        "legacy_adjacent_recovery_audit": {
            "policy": "non-displacing union of A0 top15 and adjacent A2 top15; not a ranked-15 metric",
            "gold_evidence_count": recovery_gold,
            "legacy_hit_count": recovery_legacy_hits,
            "adjacent_hit_count": recovery_adjacent_hits,
            "recovered_evidence_count": recovered_evidence,
            "recovered_case_count": recovered_cases,
            "union_hit_count": recovery_union_hits,
            "union_recall": round(recovery_union_hits / recovery_gold, 8),
            "union_minus_legacy_recall": round(
                (recovery_union_hits - recovery_legacy_hits) / recovery_gold, 8
            ),
        },
        "latency": {
            "query_batch_count": len(questions),
            "query_batch_embedding_total_ms": round(embedding_ms, 6),
            "retrieval_scoring": latency_report,
        },
        "safety": {
            "cross_document_results": 0,
            "benchmark_answers_loaded_by_builder_or_retriever": [],
            "benchmark_keywords_loaded_by_builder_or_retriever": [],
            "benchmark_scores_loaded_by_builder_or_retriever": [],
            "evaluation_gold_source": "frozen v10 selected source evidence only",
        },
        "artifacts": {
            "ablation_cases": "ablation_cases.jsonl",
            "ablation_cases_sha256": sha256_file(args.run_dir / "ablation_cases.jsonl"),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "passed": True,
    }
    return report


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-manifest", type=Path,
        default=ROOT / "data/corpus_package/type3/annual_reports_170_v1/corpus_manifest.json",
    )
    parser.add_argument(
        "--question-profile", type=Path,
        default=ROOT / (
            "data/corpus_package/type3/annual_reports_170_v1/questions/"
            "type3_260_dev_v1/question_profile.json"
        ),
    )
    parser.add_argument(
        "--package-dir", type=Path,
        default=ROOT / "data/corpus_package/type3/annual_reports_170_v1/a2rag_text_v1",
    )
    parser.add_argument(
        "--index-dir", type=Path,
        default=ROOT / "data/indexes/type3/annual_reports_170_v1/a2rag",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=ROOT / "runs/type3_a2rag_tabgr_experiment_v1/annual_reports_170_v1/phase_2",
    )
    parser.add_argument(
        "--model-path", type=Path,
        default=Path(
            "/home/coder/demo/models/models--BAAI--bge-m3/snapshots/"
            "5617a9f61b028005a4858fdac845db406aefb181"
        ),
    )
    parser.add_argument(
        "--v10-results", type=Path,
        default=ROOT / "runs/type3_qwen36_hybrid_coverage_v10/full/results.jsonl",
    )
    parser.add_argument(
        "--legacy-evidence", type=Path,
        default=ROOT / "data/corpus_package/evidence_chunks.jsonl",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--structural", action="store_true")
    mode.add_argument("--ablation", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.structural:
        report = structural_validation(args)
        atomic_json(args.run_dir / "validation_report.json", report)
        if not report["passed"]:
            raise Type3A2RAGError(f"validation failed: {report['failed_checks'][:5]}")
    else:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        report = ablation(args)
        atomic_json(args.run_dir / "ablation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
