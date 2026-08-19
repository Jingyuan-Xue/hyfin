#!/usr/bin/env python3
"""Query the frozen Phase 8 static-composition QA pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.contracts import canonical_json_bytes, semantic_sha256  # noqa: E402
from finglmqa.pipeline import build_default_pipeline  # noqa: E402


def atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def request_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.request_json is not None:
        value = json.loads(Path(args.request_json).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("--request-json must contain an object")
        return value
    question = args.question
    if question is None:
        question = sys.stdin.read().strip()
    request_id = args.request_id or "qa_" + semantic_sha256({"question": question})[:16]
    request: dict[str, Any] = {
        "schema_version": "finglmqa.phase8.qa_request.v1",
        "request_id": request_id,
        "question": question,
        "locale": "zh-CN",
    }
    if args.company is not None:
        request["company"] = args.company
    if args.report_year is not None:
        request["report_year"] = args.report_year
    if args.metric_year:
        request["metric_years"] = args.metric_year
    if args.canonical_metric:
        request["canonical_metrics"] = args.canonical_metric
    return request


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--request-json")
    parser.add_argument("--request-id")
    parser.add_argument("--company")
    parser.add_argument("--report-year", type=int)
    parser.add_argument("--metric-year", type=int, action="append", default=[])
    parser.add_argument("--canonical-metric", action="append", default=[])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-cache")
    parser.add_argument("--no-evidence", action="store_true")
    parser.add_argument("--answer-only", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    try:
        request = request_from_args(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    pipeline, transport = build_default_pipeline(
        evidence_enabled=not args.no_evidence,
        device=args.device,
        model_cache=args.model_cache,
    )
    try:
        run = pipeline.run(request)
    finally:
        if transport is not None:
            transport.close()
    result = run.answer if args.answer_only else run.as_dict()
    if args.output:
        atomic_write(Path(args.output), result)
    sys.stdout.buffer.write(canonical_json_bytes(result))
    return 0 if run.answer["status"] not in {"blocked", "error"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
