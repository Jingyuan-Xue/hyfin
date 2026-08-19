#!/usr/bin/env python3
"""Prepare and record the fixed Phase 10 manual shadow audit sample."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finglmqa.contracts import canonical_json_bytes  # noqa: E402

RUN = ROOT / "runs/phase_10"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def candidates() -> list[dict[str, Any]]:
    results = read_jsonl(RUN / "shadow_results.jsonl")
    benchmark = {
        row["case_id"]: row["source"]["question"]
        for row in read_jsonl(ROOT / "runs/phase_08/benchmark_decomposition_oracle.jsonl")
    }
    fixtures = {row["case_id"]: row["question"] for row in read_jsonl(RUN / "shadow_adversarial_fixtures.jsonl")}
    questions = {**benchmark, **fixtures}
    answerable = [row for row in results if row["eligibility"] == "answerable"][:20]
    unanswerable = [row for row in results if row["eligibility"] != "answerable"][:10]
    if len(answerable) != 20 or len(unanswerable) != 10:
        raise RuntimeError("fixed 20 answerable + 10 unanswerable audit sample is unavailable")
    return [{
        "schema_version": "finglmqa.phase10.manual_shadow_audit.v1",
        "case_id": row["case_id"],
        "eligibility": row["eligibility"],
        "question": questions[row["case_id"]],
        "generator_outcome": row["generator_outcome"],
        "accepted_claim_projection": row["accepted_claim_projection"],
        "review": None,
    } for row in [*answerable, *unanswerable]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--relevance-fail", action="append", default=[])
    parser.add_argument("--citation-fail", action="append", default=[])
    args = parser.parse_args()
    rows = candidates()
    if args.finalize:
        relevance_fail = set(args.relevance_fail)
        citation_fail = set(args.citation_fail)
        known = {row["case_id"] for row in rows}
        if not relevance_fail | citation_fail <= known:
            raise RuntimeError("manual failure list contains a case outside the fixed sample")
        for row in rows:
            answerable = row["eligibility"] == "answerable"
            has_claim = bool(row["accepted_claim_projection"])
            row["review"] = {
                "reviewer": "primary_agent_manual",
                "relevant": bool(answerable and has_claim and row["case_id"] not in relevance_fail),
                "citation_sufficient": bool(answerable and has_claim and row["case_id"] not in citation_fail),
                "unsupported_output": bool(not answerable and has_claim),
                "review_state": "completed",
            }
    path = RUN / "reports/manual_shadow_audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, rows)
    print(json.dumps({"rows": len(rows), "finalized": args.finalize}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
