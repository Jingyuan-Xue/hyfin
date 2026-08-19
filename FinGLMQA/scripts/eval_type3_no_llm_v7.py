#!/usr/bin/env python3
"""Materialize the opt-in Type 3-1 v7 deterministic evidence experiment.

The runner consumes a hash-verified frozen v4 evaluation and its Phase 8
traces.  It never loads benchmark prompts, keywords, or reference answers.
Each ablation is projected from the same prepared document candidates, so
stage comparisons cannot drift because of repeated dense retrieval calls.
"""

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
from finglmqa.type3_v7 import (  # noqa: E402
    STAGE_FEATURES,
    TYPE3_V7_VERSION,
    Type3V7Enhancer,
)


DEFAULT_BASELINE = ROOT / "runs/type3_no_llm_experiment_v4"
DEFAULT_OUTPUT = ROOT / "runs/type3_no_llm_experiment_v7"
DEFAULT_TABLE_INDEX = ROOT / "runs/table_evidence_experiment/table_evidence_fragments.jsonl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"expected JSON objects in {path.name}")
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


def _verify_baseline(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = _read_json(path / "run_report.json")
    evaluation = path / "http_evaluation.jsonl"
    if report.get("rows") != 260 or report.get("generative_llm_used") is not False:
        raise RuntimeError("baseline is not the frozen 260-row no-LLM v4 run")
    if _sha256(evaluation) != report["artifacts"]["answers_sha256"]:
        raise RuntimeError("baseline answer hash differs from its run report")
    frozen_files = {
        "evidence_executor_sha256": ROOT / "src/finglmqa/evidence_executor.py",
        "retriever_sha256": ROOT / "scripts/query_type3_evidence.py",
        "pipeline_sha256": ROOT / "src/finglmqa/pipeline.py",
    }
    for key, source in frozen_files.items():
        if _sha256(source) != report["inputs"][key]:
            raise RuntimeError(f"frozen v4 source changed: {source.relative_to(ROOT)}")
    rows = _read_jsonl(evaluation)
    if len(rows) != 260 or len({row["case_id"] for row in rows}) != 260:
        raise RuntimeError("baseline evaluation identities are invalid")
    return report, rows


def _scope(row: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
    scope = trace.get("scope_plan")
    resolutions = scope.get("entity_resolutions") if isinstance(scope, Mapping) else None
    if not isinstance(resolutions, list) or len(resolutions) != 1:
        raise RuntimeError(f"v7 requires one resolved entity: {row['case_id']}")
    resolution = resolutions[0]
    documents = resolution.get("document_set")
    identity = resolution.get("identity")
    if (
        resolution.get("status") != "unique"
        or not isinstance(documents, list)
        or len(documents) != 1
        or not isinstance(identity, Mapping)
    ):
        raise RuntimeError(f"v7 requires one resolved document: {row['case_id']}")
    document = documents[0]
    request = row["request"]
    return {
        "case_id": row["case_id"],
        "question": request["question"],
        "document_id": document["document_id"],
        "company": identity["company_full"],
        "stock_code": identity["stock_code"],
        "report_year": document["report_year"],
    }


def _baseline_answer(response: Mapping[str, Any]) -> dict[str, Any]:
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
    response = {
        "answer": result["answer"],
        "citations": result["citations"],
        "status": result["status"],
        "errors": result["errors"],
        "warnings": result["warnings"],
        "trace_hash": result["trace"]["trace_hash"],
        "generator_modes": ["deterministic_question_aware_extractive", TYPE3_V7_VERSION],
    }
    return {
        "case_id": baseline_row["case_id"],
        "kind": "benchmark",
        "oracle_match": True,
        "request": baseline_row["request"],
        "response": response,
        "experimental_profile": TYPE3_V7_VERSION,
        "ablation_stage": stage,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--table-index", type=Path, default=DEFAULT_TABLE_INDEX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_dir = args.baseline_dir.resolve()
    table_path = args.table_index.resolve()
    out_dir = args.out_dir.resolve()
    baseline_report, baseline_rows = _verify_baseline(baseline_dir)
    table_index = TableEvidenceIndex(table_path)
    enhancer = Type3V7Enhancer(root=ROOT, table_index=table_index)

    stage_rows: dict[str, list[dict[str, Any]]] = {stage: [] for stage, _ in STAGE_FEATURES}
    stage_traces: dict[str, list[dict[str, Any]]] = {stage: [] for stage, _ in STAGE_FEATURES}
    recovery: Counter[str] = Counter()
    for ordinal, baseline_row in enumerate(baseline_rows, start=1):
        response = baseline_row["response"]
        trace_path = baseline_dir / "traces" / f"{response['trace_hash']}.json"
        trace = _read_json(trace_path)
        if trace.get("trace_hash") != response["trace_hash"]:
            raise RuntimeError(f"baseline trace identity mismatch: {baseline_row['case_id']}")
        scope = _scope(baseline_row, trace)
        prepared = enhancer.prepare(
            scope=scope,
            base_answer=_baseline_answer(response),
            base_trace=trace,
        )
        for stage, features in STAGE_FEATURES:
            result = enhancer.materialize(prepared, features)
            projected = _projection(baseline_row, result, stage=stage)
            stage_rows[stage].append(projected)
            stage_traces[stage].append(result["trace"])
            if not response.get("answer") and result["answer"]:
                selected = result["trace"]["selected_group_ids"]
                recovery[stage] += 1
                if not selected:
                    raise RuntimeError(f"recovered answer lacks an audited group: {baseline_row['case_id']}")
        if ordinal % 25 == 0 or ordinal == len(baseline_rows):
            print(f"prepared={ordinal}/{len(baseline_rows)}", flush=True)

    stage_reports: dict[str, Any] = {}
    for stage, _ in STAGE_FEATURES:
        stage_dir = out_dir if stage == "full" else out_dir / "ablations" / stage
        answer_path = stage_dir / "http_evaluation.jsonl"
        trace_path = stage_dir / "deterministic_traces.jsonl"
        _write_jsonl(answer_path, stage_rows[stage])
        _write_jsonl(trace_path, stage_traces[stage])
        stage_reports[stage] = {
            "rows": len(stage_rows[stage]),
            "nonempty_answers": sum(bool(row["response"]["answer"].strip()) for row in stage_rows[stage]),
            "recovered_from_empty_v4": recovery[stage],
            "answers": answer_path.relative_to(ROOT).as_posix(),
            "answers_sha256": _sha256(answer_path),
            "traces": trace_path.relative_to(ROOT).as_posix(),
            "traces_sha256": _sha256(trace_path),
        }

    report = {
        "schema_version": "finglmqa.experimental.type3_v7_run_report.v1",
        "profile_version": TYPE3_V7_VERSION,
        "generative_llm_used": False,
        "rows": len(baseline_rows),
        "source_freeze": {
            "baseline_answers_sha256": baseline_report["artifacts"]["answers_sha256"],
            "baseline_run_report_sha256": _sha256(baseline_dir / "run_report.json"),
            "table_index_sha256": _sha256(table_path),
            "type3_v7_source_sha256": _sha256(ROOT / "src/finglmqa/type3_v7.py"),
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
