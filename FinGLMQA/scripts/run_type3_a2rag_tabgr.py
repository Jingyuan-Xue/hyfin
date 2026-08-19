#!/usr/bin/env python3
"""Run the deterministic Type 3 A2RAG + TabGR answer-safe pipeline."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.type3_a2rag_tabgr_pipeline import (  # noqa: E402
    ManifestPaths,
    PipelineConfig,
    Type3A2RAGTabGRPipeline,
    bind_text_runtime,
    load_sanitized_questions,
    read_json,
    sha256_file,
)
from finglmqa.type3_evidence_fusion import canonical_json_bytes, semantic_sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--corpus-id", default="annual_reports_170_v1")
    parser.add_argument("--question-profile-id", default="type3_260_dev_v1")
    parser.add_argument("--arm", default="union")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-facets", type=int, default=6)
    parser.add_argument("--text-top-k", type=int, default=15)
    parser.add_argument("--table-top-k", type=int, default=12)
    parser.add_argument("--max-fused-candidates", type=int, default=18)
    parser.add_argument("--max-composed-items", type=int, default=8)
    parser.add_argument("--text-mode", choices=("hybrid", "sparse"), default="hybrid")
    parser.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--question-profile", type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--a2rag-package-manifest", type=Path)
    parser.add_argument("--a2rag-index-manifest", type=Path)
    parser.add_argument("--text-atoms", type=Path)
    parser.add_argument("--tabgr-package-manifest", type=Path)
    parser.add_argument("--tabgr-index-manifest", type=Path)
    parser.add_argument("--fact-manifest", type=Path)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> ManifestPaths:
    defaults = ManifestPaths.defaults(args.root.resolve(), args.corpus_id, args.question_profile_id)
    return ManifestPaths(
        corpus_manifest=(args.corpus_manifest or defaults.corpus_manifest).resolve(),
        question_profile=(args.question_profile or defaults.question_profile).resolve(),
        questions=(args.questions or defaults.questions).resolve(),
        a2rag_package_manifest=(
            args.a2rag_package_manifest or defaults.a2rag_package_manifest
        ).resolve(),
        a2rag_index_manifest=(args.a2rag_index_manifest or defaults.a2rag_index_manifest).resolve(),
        text_atoms=(args.text_atoms or defaults.text_atoms).resolve(),
        tabgr_package_manifest=(
            args.tabgr_package_manifest or defaults.tabgr_package_manifest
        ).resolve(),
        tabgr_index_manifest=(args.tabgr_index_manifest or defaults.tabgr_index_manifest).resolve(),
        fact_manifest=(args.fact_manifest or defaults.fact_manifest).resolve(),
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for value in values:
            handle.write(canonical_json_bytes(value))


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    paths = resolve_paths(args)
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
    pipeline = Type3A2RAGTabGRPipeline(
        paths=paths,
        corpus_id=args.corpus_id,
        question_profile_id=args.question_profile_id,
    )
    config = PipelineConfig(
        arm=args.arm,
        max_facets=args.max_facets,
        text_top_k=args.text_top_k,
        table_top_k=args.table_top_k,
        max_fused_candidates=args.max_fused_candidates,
        max_composed_items=args.max_composed_items,
        text_mode=args.text_mode,
    ).validate()
    packets = [pipeline.run_question(row, config=config) for row in rows]

    output_dir = args.output_dir.resolve()
    answers_path = output_dir / "answers.jsonl"
    traces_path = output_dir / "semantic_traces.jsonl"
    write_jsonl(answers_path, packets)
    write_jsonl(
        traces_path,
        [
            {
                "question_id": value["question_id"],
                "semantic_trace": value["semantic_trace"],
            }
            for value in packets
        ],
    )
    route_counts: Counter[str] = Counter()
    conflict_count = 0
    for packet in packets:
        for evidence in packet["evidence"]:
            route_counts[evidence["route"]] += 1
            conflict_count += evidence["conflict_status"] != "clear"
    manifest_unsigned = {
        "schema_version": "finglmqa.type3.a2rag_tabgr.run_manifest.v1",
        "pipeline_version": "type3-a2rag-tabgr-pipeline-v1",
        "corpus_id": args.corpus_id,
        "question_profile_id": args.question_profile_id,
        "arm": args.arm,
        "configuration": {
            "max_facets": config.max_facets,
            "text_top_k": config.text_top_k,
            "table_top_k": config.table_top_k,
            "max_fused_candidates": config.max_fused_candidates,
            "max_composed_items": config.max_composed_items,
            "text_mode": config.text_mode,
        },
        "manifest_binding": pipeline.binding.as_mapping(),
        "runtime_binding": runtime_info,
        "question_count": len(packets),
        "question_ids_sha256": semantic_sha256([value["question_id"] for value in packets]),
        "semantic_trace_sequence_sha256": semantic_sha256(
            [value["semantic_trace"]["semantic_trace_sha256"] for value in packets]
        ),
        "safety": {
            "generator_input_fields": ["corpus_id", "document_id", "question", "question_id"],
            "cross_document_evidence": sum(
                evidence["document_id"] != packet["document_id"]
                for packet in packets
                for evidence in packet["evidence"]
            ),
            "unsupported_numeric_literals": sum(
                len(packet["semantic_trace"]["numeric_safety"]["unsupported_numeric_literals"])
                for packet in packets
            ),
            "conflict_redacted_evidence": conflict_count,
        },
        "route_counts": dict(sorted(route_counts.items())),
        "artifacts": {
            "answers.jsonl": sha256_file(answers_path),
            "semantic_traces.jsonl": sha256_file(traces_path),
        },
    }
    write_json(
        output_dir / "run_manifest.json",
        {
            **manifest_unsigned,
            "run_fingerprint": semantic_sha256(manifest_unsigned),
        },
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "arm": args.arm,
                "questions": len(packets),
                "output_dir": output_dir.as_posix(),
                "answers_sha256": sha256_file(answers_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
