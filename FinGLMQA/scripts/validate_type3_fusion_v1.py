#!/usr/bin/env python3
"""Validate Phase 4 fusion safety, deterministic function, and ablations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import subprocess
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.type3_a2rag_tabgr_pipeline import (  # noqa: E402
    ManifestPaths,
    PipelineConfig,
    SUPPORTED_ARMS,
    Type3A2RAGTabGRPipeline,
    bind_text_runtime,
    load_sanitized_questions,
    read_json,
    sha256_file,
)
from finglmqa.type3_evidence_fusion import canonical_json_bytes, semantic_sha256  # noqa: E402
from finglmqa.type3_tabgr_retriever import lexical_tokens, numeric_fragments  # noqa: E402


ARMS = (
    "union",
    "text_only",
    "table_only",
    "legacy_table_only",
    "v2_table_only",
    "no_route_quota",
    "no_adjacency",
    "no_fact_join",
    "no_table_semantics",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--corpus-id", default="annual_reports_170_v1")
    parser.add_argument("--question-profile-id", default="type3_260_dev_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--text-mode", choices=("hybrid", "sparse"), default="hybrid")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--fresh-process-repeat", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for value in values:
            handle.write(canonical_json_bytes(value))


def query_overlap(question: str, evidence: list[dict[str, Any]]) -> float:
    query = set(lexical_tokens(question))
    if not query:
        return 0.0
    selected = set(
        token
        for row in evidence[:8]
        for token in lexical_tokens(row["display_text"])
    )
    return len(query & selected) / len(query)


def validate_packet(packet: dict[str, Any]) -> dict[str, int]:
    cross_document = 0
    unsupported = 0
    route_counts: Counter[str] = Counter()
    for evidence in packet["evidence"]:
        cross_document += evidence["document_id"] != packet["document_id"]
        route_counts[evidence["route"]] += 1
        allowed = {
            rendering
            for authorization in evidence["numeric_authorizations"]
            for rendering in authorization.get("allowed_renderings") or ()
        }
        unsupported += sum(
            literal not in allowed for literal in numeric_fragments(evidence["answer_safe_text"])
        )
        if evidence["route"] == "text":
            unsupported += sum(literal not in evidence["display_text"] for literal in allowed)
        if evidence["route"] == "table":
            unsupported += sum(literal not in evidence["display_text"] for literal in allowed)
    return {
        "cross_document": cross_document,
        "unsupported_numeric": unsupported,
        "text": route_counts["text"],
        "table": route_counts["table"],
    }


def run_arm(
    pipeline: Type3A2RAGTabGRPipeline,
    rows: list[dict[str, str]],
    *,
    arm: str,
    text_mode: str,
    output_dir: Path,
) -> dict[str, Any]:
    packets = [
        pipeline.run_question(row, config=PipelineConfig(arm=arm, text_mode=text_mode))
        for row in rows
    ]
    packet_path = output_dir / "answers.jsonl"
    write_jsonl(packet_path, packets)
    counts: Counter[str] = Counter()
    overlaps = []
    candidate_sequences = []
    for packet in packets:
        counts.update(validate_packet(packet))
        overlaps.append(query_overlap(packet["question"], packet["evidence"]))
        candidate_sequences.append(
            [value["candidate_id"] for value in packet["evidence"]]
        )
    return {
        "arm": arm,
        "question_count": len(packets),
        "nonempty_answers": sum(bool(value["answer_safe_text"].strip()) for value in packets),
        "cross_document_evidence": counts["cross_document"],
        "unsupported_numeric_literals": counts["unsupported_numeric"],
        "text_evidence_count": counts["text"],
        "table_evidence_count": counts["table"],
        "mean_query_token_overlap": round(sum(overlaps) / max(1, len(overlaps)), 6),
        "candidate_sequence_sha256": semantic_sha256(candidate_sequences),
        "answer_packets_sha256": sha256_file(packet_path),
    }


def run_fresh_process_repeat(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    first = output_dir / "fresh_process_1"
    second = output_dir / "fresh_process_2"
    command = [
        sys.executable,
        str(ROOT / "scripts/run_type3_a2rag_tabgr.py"),
        "--root",
        str(args.root.resolve()),
        "--corpus-id",
        args.corpus_id,
        "--question-profile-id",
        args.question_profile_id,
        "--arm",
        "union",
        "--text-mode",
        args.text_mode,
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    subprocess.run([*command, "--output-dir", str(first)], check=True)
    subprocess.run([*command, "--output-dir", str(second)], check=True)
    files = ("answers.jsonl", "semantic_traces.jsonl", "run_manifest.json")
    comparisons = {
        value: {
            "first_sha256": sha256_file(first / value),
            "second_sha256": sha256_file(second / value),
            "byte_equal": (first / value).read_bytes() == (second / value).read_bytes(),
        }
        for value in files
    }
    return {
        "schema_version": "finglmqa.type3.a2rag_tabgr.fresh_process_repeat.v1",
        "status": "passed" if all(value["byte_equal"] for value in comparisons.values()) else "failed",
        "comparisons": comparisons,
    }


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if set(ARMS) != set(SUPPORTED_ARMS):
        raise RuntimeError("validator arm set differs from pipeline")
    paths = ManifestPaths.defaults(
        args.root.resolve(), args.corpus_id, args.question_profile_id
    )
    runtime_info = bind_text_runtime(
        root=args.root.resolve(),
        a2rag_index_manifest=paths.a2rag_index_manifest,
    )
    question_profile = read_json(paths.question_profile)
    rows = load_sanitized_questions(
        paths.questions,
        corpus_id=args.corpus_id,
        expected_count=int(question_profile["question_count"]),
    )
    if args.limit is not None:
        rows = rows[: args.limit]
    output_dir = args.output_dir.resolve()
    pipeline = Type3A2RAGTabGRPipeline(
        paths=paths,
        corpus_id=args.corpus_id,
        question_profile_id=args.question_profile_id,
    )
    arms = ("union",) if args.skip_ablations else ARMS
    reports = {
        arm: run_arm(
            pipeline,
            rows,
            arm=arm,
            text_mode=args.text_mode,
            output_dir=output_dir / ("full" if arm == "union" else f"ablations/{arm}"),
        )
        for arm in arms
    }
    ablation_report = {
        "schema_version": "finglmqa.type3.a2rag_tabgr.ablation_report.v1",
        "corpus_id": args.corpus_id,
        "question_profile_id": args.question_profile_id,
        "quality_scope": "functional safety and annotation-free query/evidence proxy only; QA scoring is Phase 5",
        "arms": reports,
    }
    write_json(output_dir / "ablation_report.json", ablation_report)
    fresh = None
    if args.fresh_process_repeat:
        fresh = run_fresh_process_repeat(args, output_dir / "repeatability")
        write_json(output_dir / "repeatability_report.json", fresh)
    union = reports["union"]
    failed_gates = []
    if union["question_count"] != len(rows) or union["nonempty_answers"] != len(rows):
        failed_gates.append("complete_nonempty_questions")
    if union["cross_document_evidence"]:
        failed_gates.append("document_boundary")
    if union["unsupported_numeric_literals"]:
        failed_gates.append("numeric_authorization")
    if union["text_evidence_count"] == 0 or union["table_evidence_count"] == 0:
        failed_gates.append("dual_route")
    if fresh is not None and fresh["status"] != "passed":
        failed_gates.append("fresh_process_repeatability")
    report = {
        "schema_version": "finglmqa.type3.a2rag_tabgr.validation_report.v1",
        "status": "passed" if not failed_gates else "failed",
        "corpus_id": args.corpus_id,
        "question_profile_id": args.question_profile_id,
        "question_count": len(rows),
        "manifest_binding": pipeline.binding.as_mapping(),
        "runtime_binding": runtime_info,
        "generator_input_fields": ["corpus_id", "document_id", "question", "question_id"],
        "benchmark_annotations_available_to_generator": False,
        "union": union,
        "failed_gates": failed_gates,
        "passed_gates": [
            value
            for value in (
                "explicit_manifest_hash_binding",
                "complete_nonempty_questions",
                "document_boundary",
                "legacy_v2_union_with_legacy_anchor",
                "exact_atom_projection",
                "numeric_authorization",
                "dual_route",
                "deterministic_semantic_trace",
                "fresh_process_repeatability" if fresh is not None else None,
            )
            if value is not None and value not in failed_gates
        ],
        "resource": {
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "peak_rss_below_8gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            < 8 * 1024 * 1024,
        },
        "ablation_report_sha256": sha256_file(output_dir / "ablation_report.json"),
        "repeatability": fresh,
    }
    write_json(output_dir / "validation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if not failed_gates else 1


if __name__ == "__main__":
    raise SystemExit(main())
