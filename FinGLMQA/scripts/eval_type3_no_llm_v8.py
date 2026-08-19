#!/usr/bin/env python3
"""Run the opt-in Type 3-1 v8 deterministic repair experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.contracts import canonical_json_bytes  # noqa: E402
from finglmqa.table_evidence import TableEvidenceIndex  # noqa: E402
from finglmqa.type3_v8 import (  # noqa: E402
    FrozenEvidenceCorpus,
    TYPE3_V8_STAGES,
    TYPE3_V8_VERSION,
    Type3V8Enhancer,
)


DEFAULT_PHASE8_BASELINE = ROOT / "runs/type3_no_llm_experiment_v4"
DEFAULT_ANSWER_BASELINE = ROOT / "runs/type3_no_llm_experiment_v7"
DEFAULT_OUTPUT = ROOT / "runs/type3_no_llm_experiment_v8"
DEFAULT_TABLE_INDEX = ROOT / "runs/table_evidence_experiment/table_evidence_fragments.jsonl"
DEFAULT_EVIDENCE_CHUNKS = ROOT / "data/corpus_package/evidence_chunks.jsonl"
DEFAULT_DOCUMENT_MAP = ROOT / "data/indexes/a2rag_index/document_chunk_map.jsonl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"expected object rows: {path}")
    return rows


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(path, canonical_json_bytes(dict(value)))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_write(path, b"".join(canonical_json_bytes(dict(row)) for row in rows))


def _verified_rows(path: Path, *, expected_profile: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = _read_json(path / "run_report.json")
    answer_path = path / "http_evaluation.jsonl"
    rows = _read_jsonl(answer_path)
    if len(rows) != 260 or len({row["case_id"] for row in rows}) != 260:
        raise RuntimeError(f"baseline identities are invalid: {path.name}")
    if expected_profile is not None and report.get("profile_version") != expected_profile:
        raise RuntimeError(f"unexpected answer baseline profile: {report.get('profile_version')}")
    expected_hash = (
        report.get("stages", {}).get("full", {}).get("answers_sha256")
        or report.get("artifacts", {}).get("answers_sha256")
    )
    if expected_hash != _sha256(answer_path):
        raise RuntimeError(f"baseline answer hash mismatch: {path.name}")
    return report, rows


def _scope(row: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
    scope = trace.get("scope_plan")
    resolutions = scope.get("entity_resolutions") if isinstance(scope, Mapping) else None
    if not isinstance(resolutions, list) or len(resolutions) != 1:
        raise RuntimeError(f"v8 requires one resolved entity: {row['case_id']}")
    resolution = resolutions[0]
    documents = resolution.get("document_set")
    identity = resolution.get("identity")
    if (
        resolution.get("status") != "unique"
        or not isinstance(documents, list)
        or len(documents) != 1
        or not isinstance(identity, Mapping)
    ):
        raise RuntimeError(f"v8 requires one resolved document: {row['case_id']}")
    document = documents[0]
    return {
        "case_id": row["case_id"],
        "question": row["request"]["question"],
        "document_id": document["document_id"],
        "company": identity["company_full"],
        "stock_code": identity["stock_code"],
        "report_year": document["report_year"],
    }


def _base_answer(response: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "answer_text": response.get("answer", ""),
        "citations": response.get("citations", []),
        "status": response.get("status"),
        "errors": response.get("errors", []),
        "warnings": response.get("warnings", []),
    }


def _projection(
    baseline_row: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    return {
        "case_id": baseline_row["case_id"],
        "kind": "benchmark",
        "oracle_match": True,
        "request": baseline_row["request"],
        "response": {
            "answer": result["answer"],
            "citations": result["citations"],
            "status": result["status"],
            "errors": result["errors"],
            "warnings": result["warnings"],
            "trace_hash": result["trace"]["trace_hash"],
            "generator_modes": ["deterministic_question_aware_extractive", TYPE3_V8_VERSION],
        },
        "experimental_profile": TYPE3_V8_VERSION,
        "ablation_stage": stage,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase8-baseline", type=Path, default=DEFAULT_PHASE8_BASELINE)
    parser.add_argument("--answer-baseline", type=Path, default=DEFAULT_ANSWER_BASELINE)
    parser.add_argument("--table-index", type=Path, default=DEFAULT_TABLE_INDEX)
    parser.add_argument("--evidence-chunks", type=Path, default=DEFAULT_EVIDENCE_CHUNKS)
    parser.add_argument("--document-map", type=Path, default=DEFAULT_DOCUMENT_MAP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    phase8_report, phase8_rows = _verified_rows(args.phase8_baseline.resolve())
    answer_report, answer_rows = _verified_rows(
        args.answer_baseline.resolve(),
        expected_profile="type3-v7-deterministic-evidence-v1",
    )
    phase8_by_case = {row["case_id"]: row for row in phase8_rows}
    answer_by_case = {row["case_id"]: row for row in answer_rows}
    if set(phase8_by_case) != set(answer_by_case):
        raise RuntimeError("Phase 8 trace baseline and v7 answer baseline differ")

    corpus = FrozenEvidenceCorpus(
        chunk_path=args.evidence_chunks,
        document_map_path=args.document_map,
    )
    enhancer = Type3V8Enhancer(
        root=ROOT,
        table_index=TableEvidenceIndex(args.table_index),
        evidence_corpus=corpus,
    )
    stage_rows: dict[str, list[dict[str, Any]]] = {stage: [] for stage, _ in TYPE3_V8_STAGES}
    stage_traces: dict[str, list[dict[str, Any]]] = {stage: [] for stage, _ in TYPE3_V8_STAGES}
    source_counts: dict[str, Counter[str]] = {stage: Counter() for stage, _ in TYPE3_V8_STAGES}

    for ordinal, case_id in enumerate(
        (row["case_id"] for row in answer_rows), start=1
    ):
        phase8_row = phase8_by_case[case_id]
        answer_row = answer_by_case[case_id]
        trace_path = args.phase8_baseline.resolve() / "traces" / f"{phase8_row['response']['trace_hash']}.json"
        trace = _read_json(trace_path)
        if trace.get("trace_hash") != phase8_row["response"]["trace_hash"]:
            raise RuntimeError(f"Phase 8 trace mismatch: {case_id}")
        prepared = enhancer.prepare(
            scope=_scope(phase8_row, trace),
            base_answer=_base_answer(answer_row["response"]),
            base_trace=trace,
        )
        for stage, features in TYPE3_V8_STAGES:
            result = enhancer.materialize(prepared, features)
            stage_rows[stage].append(_projection(answer_row, result, stage=stage))
            stage_traces[stage].append(result["trace"])
            source_counts[stage][result["trace"]["selected_source"]] += 1
        if ordinal % 25 == 0 or ordinal == len(answer_rows):
            print(f"prepared={ordinal}/{len(answer_rows)}", flush=True)

    out_dir = args.out_dir.resolve()
    stage_reports: dict[str, Any] = {}
    for stage, _ in TYPE3_V8_STAGES:
        stage_dir = out_dir if stage == "full" else out_dir / "ablations" / stage
        answer_path = stage_dir / "http_evaluation.jsonl"
        trace_path = stage_dir / "deterministic_traces.jsonl"
        _write_jsonl(answer_path, stage_rows[stage])
        _write_jsonl(trace_path, stage_traces[stage])
        stage_reports[stage] = {
            "rows": len(stage_rows[stage]),
            "nonempty_answers": sum(bool(row["response"]["answer"].strip()) for row in stage_rows[stage]),
            "selected_source_counts": dict(sorted(source_counts[stage].items())),
            "answers": answer_path.relative_to(ROOT).as_posix(),
            "answers_sha256": _sha256(answer_path),
            "traces": trace_path.relative_to(ROOT).as_posix(),
            "traces_sha256": _sha256(trace_path),
        }

    report = {
        "schema_version": "finglmqa.experimental.type3_v8_run_report.v1",
        "profile_version": TYPE3_V8_VERSION,
        "generative_llm_used": False,
        "rows": len(answer_rows),
        "source_freeze": {
            "phase8_baseline_answers_sha256": phase8_report["artifacts"]["answers_sha256"],
            "v7_baseline_answers_sha256": answer_report["stages"]["full"]["answers_sha256"],
            "table_index_sha256": _sha256(args.table_index.resolve()),
            "evidence_chunks_sha256": _sha256(args.evidence_chunks.resolve()),
            "document_map_sha256": _sha256(args.document_map.resolve()),
            "type3_v8_source_sha256": _sha256(ROOT / "src/finglmqa/type3_v8.py"),
            "intent_policy_sha256": _sha256(ROOT / "src/finglmqa/type3_v7_intent.py"),
            "table_policy_sha256": _sha256(ROOT / "src/finglmqa/type3_v7_table_upgrade.py"),
            "negative_policy_sha256": _sha256(ROOT / "src/finglmqa/type3_v7_negative.py"),
            "lowinfo_policy_sha256": _sha256(ROOT / "src/finglmqa/type3_v7_lowinfo.py"),
            "runner_sha256": _sha256(Path(__file__).resolve()),
        },
        "benchmark_fields_loaded_by_answer_chain": ["case_id", "question"],
        "forbidden_benchmark_fields_loaded_by_answer_chain": [],
        "stages": stage_reports,
    }
    _write_json(out_dir / "run_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
