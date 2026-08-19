#!/usr/bin/env python3
"""Validate the frozen Type 3 v8 deterministic experiment artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.contracts import canonical_json_bytes, semantic_sha256  # noqa: E402
from finglmqa.type3_v7 import _compact, _normalize  # noqa: E402
from finglmqa.type3_v7_table_upgrade import (  # noqa: E402
    TableNumericAuthorization,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _validate_source_authorization(
    value: Mapping[str, Any],
    *,
    document_id: str,
    answer: str,
) -> None:
    if value.get("schema_version") != "finglmqa.experimental.source_numeric_authorization.v1":
        raise RuntimeError("unexpected source numeric authorization schema")
    if value.get("document_id") != document_id:
        raise RuntimeError("source numeric authorization crosses documents")
    path = PurePosixPath(str(value.get("source_markdown") or ""))
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError("source numeric authorization path is not portable")
    source = ROOT / path
    line_range = value.get("source_line_range")
    if not isinstance(line_range, list) or len(line_range) != 2 or line_range[0] != line_range[1]:
        raise RuntimeError("source numeric authorization is not line scoped")
    lines = source.read_text(encoding="utf-8").splitlines()
    line_number = int(line_range[0])
    if line_number < 1 or line_number > len(lines):
        raise RuntimeError("source numeric authorization line is outside document")
    source_line = _normalize(lines[line_number - 1])
    if hashlib.sha256(source_line.encode("utf-8")).hexdigest() != value.get("source_line_sha256"):
        raise RuntimeError("source numeric authorization line hash mismatch")
    allowed = value.get("allowed_renderings")
    if not isinstance(allowed, list) or not allowed:
        raise RuntimeError("source numeric authorization has no rendering")
    for rendering in allowed:
        if _compact(rendering) not in _compact(source_line):
            raise RuntimeError("authorized rendering is absent from source line")
        if _compact(rendering) not in _compact(answer):
            raise RuntimeError("authorized rendering is unused by answer")


def validate(out_dir: Path) -> dict[str, Any]:
    report = json.loads((out_dir / "run_report.json").read_text(encoding="utf-8"))
    answer_path = out_dir / "http_evaluation.jsonl"
    trace_path = out_dir / "deterministic_traces.jsonl"
    answers = _rows(answer_path)
    traces = _rows(trace_path)
    if len(answers) != 260 or len(traces) != 260:
        raise RuntimeError("v8 artifact row count differs from 260")
    if len({row["case_id"] for row in answers}) != 260:
        raise RuntimeError("v8 answer case ids are not unique")
    trace_by_case = {row["case_id"]: row for row in traces}
    if set(trace_by_case) != {row["case_id"] for row in answers}:
        raise RuntimeError("answer and trace cases differ")
    if report["stages"]["full"]["answers_sha256"] != _sha256(answer_path):
        raise RuntimeError("answer hash differs from run report")
    if report["stages"]["full"]["traces_sha256"] != _sha256(trace_path):
        raise RuntimeError("trace hash differs from run report")
    if report.get("forbidden_benchmark_fields_loaded_by_answer_chain"):
        raise RuntimeError("answer chain loaded forbidden benchmark fields")

    source_authorization_count = 0
    table_authorization_count = 0
    for row in answers:
        response = row["response"]
        answer = str(response.get("answer") or "").strip()
        if not answer or response.get("status") != "ok":
            raise RuntimeError(f"non-ok or empty answer: {row['case_id']}")
        trace = trace_by_case[row["case_id"]]
        unsigned = dict(trace)
        trace_hash = unsigned.pop("trace_hash")
        if semantic_sha256(unsigned) != trace_hash or response.get("trace_hash") != trace_hash:
            raise RuntimeError(f"trace hash mismatch: {row['case_id']}")
        document_id = trace["document_id"]
        if any(citation.get("document_id") != document_id for citation in response.get("citations") or []):
            raise RuntimeError(f"citation crosses documents: {row['case_id']}")
        authorizations = trace.get("numeric_authorizations") or []
        if len({value.get("authorization_id") for value in authorizations}) != len(authorizations):
            raise RuntimeError(f"duplicate numeric authorization: {row['case_id']}")
        for authorization in authorizations:
            schema = authorization.get("schema_version")
            if schema == "finglmqa.experimental.table_numeric_authorization.v1":
                TableNumericAuthorization.from_mapping(
                    authorization, expected_document_id=document_id
                )
                table_authorization_count += 1
            else:
                _validate_source_authorization(
                    authorization,
                    document_id=document_id,
                    answer=answer,
                )
                source_authorization_count += 1
    return {
        "schema_version": "finglmqa.experimental.type3_v8_validation.v1",
        "status": "passed",
        "rows": len(answers),
        "nonempty_answers": len(answers),
        "table_numeric_authorizations": table_authorization_count,
        "source_numeric_authorizations": source_authorization_count,
        "answers_sha256": _sha256(answer_path),
        "traces_sha256": _sha256(trace_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/type3_no_llm_experiment_v8")
    parser.add_argument("--compare-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    result = validate(out_dir)
    if args.compare_dir is not None:
        comparison = validate(args.compare_dir.resolve())
        result["repeat_answers_byte_identical"] = (
            result["answers_sha256"] == comparison["answers_sha256"]
        )
        result["repeat_traces_byte_identical"] = (
            result["traces_sha256"] == comparison["traces_sha256"]
        )
        if not result["repeat_answers_byte_identical"] or not result["repeat_traces_byte_identical"]:
            raise RuntimeError("repeat artifacts are not byte-identical")
    report_path = (args.report or (out_dir / "validation_report.json")).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=report_path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_json_bytes(result))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, report_path)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
