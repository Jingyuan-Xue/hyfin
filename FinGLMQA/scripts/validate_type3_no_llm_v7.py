#!/usr/bin/env python3
"""Fail-closed validation for the opt-in deterministic Type 3-1 v7 run."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.contracts import canonical_json_bytes, semantic_sha256  # noqa: E402


DEFAULT_RUN = ROOT / "runs/type3_no_llm_experiment_v7"
DEFAULT_REPEAT = ROOT / "runs/type3_no_llm_experiment_v7_repeat"
_DECIMAL_TOKEN = re.compile(r"(?<![\w])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)\.\d+[%％]?")
_PERCENT_TOKEN = re.compile(r"(?<![\w])[-+]?\d+(?:\.\d+)?[%％]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"expected object rows: {path}")
    return rows


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_json_bytes(dict(value)))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _score(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    report = _json(path / "score_report.json")
    details = {row["case_id"]: row for row in _jsonl(path / "score_details.jsonl")}
    return report, details


def _gate(name: str, condition: bool, details: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"gate": name, "passed": bool(condition), "details": dict(details or {})}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--repeat-dir", type=Path, default=DEFAULT_REPEAT)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN / "validation_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    repeat_dir = args.repeat_dir.resolve()
    report = _json(run_dir / "run_report.json")
    rows = _jsonl(run_dir / "http_evaluation.jsonl")
    traces = _jsonl(run_dir / "deterministic_traces.jsonl")
    checks: list[dict[str, Any]] = []

    checks.append(_gate(
        "source_and_artifact_hashes",
        report["source_freeze"]["type3_v7_source_sha256"]
        == _sha256(ROOT / "src/finglmqa/type3_v7.py")
        and report["source_freeze"]["runner_sha256"]
        == _sha256(ROOT / "scripts/eval_type3_no_llm_v7.py")
        and report["stages"]["full"]["answers_sha256"]
        == _sha256(run_dir / "http_evaluation.jsonl")
        and report["stages"]["full"]["traces_sha256"]
        == _sha256(run_dir / "deterministic_traces.jsonl"),
    ))
    checks.append(_gate(
        "coverage_and_no_generative_llm",
        len(rows) == len(traces) == 260
        and all(row["response"]["answer"].strip() for row in rows)
        and all(row["response"]["status"] == "ok" for row in rows)
        and report["generative_llm_used"] is False,
        {"rows": len(rows), "nonempty": sum(bool(row["response"]["answer"].strip()) for row in rows)},
    ))

    trace_errors: list[str] = []
    source_counts: Counter[str] = Counter()
    for row, trace in zip(rows, traces):
        case_id = row["case_id"]
        expected_hash = trace.get("trace_hash")
        unhashed = dict(trace)
        unhashed.pop("trace_hash", None)
        if expected_hash != semantic_sha256(unhashed) or row["response"]["trace_hash"] != expected_hash:
            trace_errors.append(f"{case_id}:trace_hash")
        document_id = trace["document_id"]
        groups = trace["selected_groups"]
        if len(groups) > 5:
            trace_errors.append(f"{case_id}:claim_group_limit")
        citation_ids = {citation["citation_id"] for citation in row["response"]["citations"]}
        for citation in row["response"]["citations"]:
            if citation.get("document_id") not in (None, document_id):
                trace_errors.append(f"{case_id}:cross_document_citation")
        authorizations = {
            authorization["authorization_id"]: authorization
            for authorization in trace["numeric_authorizations"]
        }
        for group in groups:
            source_counts[group["source_kind"]] += 1
            if not set(group["citation_ids"]).issubset(citation_ids):
                trace_errors.append(f"{case_id}:citation_projection")
            if not set(group["numeric_authorization_ids"]).issubset(authorizations):
                trace_errors.append(f"{case_id}:authorization_projection")
            if group["source_kind"] == "table" and len(group["citation_ids"]) > 12:
                trace_errors.append(f"{case_id}:table_row_limit")
            if group["source_kind"] == "table":
                renderings = [
                    rendering
                    for authorization_id in group["numeric_authorization_ids"]
                    for rendering in authorizations[authorization_id]["allowed_renderings"]
                ]
                numeric_tokens = set(_DECIMAL_TOKEN.findall(group["text"]))
                numeric_tokens.update(_PERCENT_TOKEN.findall(group["text"]))
                for token in numeric_tokens:
                    if not any(token in rendering for rendering in renderings):
                        trace_errors.append(f"{case_id}:unauthorized_table_number:{token}")
        audit = trace.get("document_absence_audit")
        if audit is not None and (
            audit.get("complete_scan") is not True
            or any(audit.get("alias_hit_counts", {}).values())
            or "未检索到" not in row["response"]["answer"]
        ):
            trace_errors.append(f"{case_id}:absence_audit")
    checks.append(_gate(
        "scope_numeric_and_absence_safety",
        not trace_errors,
        {"errors": trace_errors[:20], "selected_source_counts": dict(sorted(source_counts.items()))},
    ))

    base_report, base_details = _score(run_dir / "ablations/baseline/scoring")
    full_report, full_details = _score(run_dir / "scoring")
    score = full_report["scores"]["bge_m3"]["overall"]
    regressions = []
    for case_id, base in base_details.items():
        delta = full_details[case_id]["bge_m3"]["score"] - base["bge_m3"]["score"]
        if delta < -0.10:
            regressions.append({"case_id": case_id, "delta": round(delta, 6)})
    checks.append(_gate(
        "score_keyword_and_regression_thresholds",
        score["average_score"] >= 0.62
        and score["count"] == 260
        and full_report["scores"]["bge_m3"]["answered_only"]["average_score"] >= 0.631549
        and full_report["gates"]["keyword_matched"] >= 18
        and not regressions,
        {
            "average_score": score["average_score"],
            "answered_average": full_report["scores"]["bge_m3"]["answered_only"]["average_score"],
            "keyword_matched": full_report["gates"]["keyword_matched"],
            "regressions_over_0_10": regressions,
        },
    ))

    baseline_rows = _jsonl(run_dir / "ablations/baseline/http_evaluation.jsonl")
    baseline_empty = {row["case_id"] for row in baseline_rows if not row["response"]["answer"]}
    recovered = [
        trace for row, trace in zip(rows, traces)
        if row["case_id"] in baseline_empty and row["response"]["answer"]
    ]
    recovered_sources = Counter(
        source_kind
        for trace in recovered
        for source_kind in {group["source_kind"] for group in trace["selected_groups"]}
    )
    checks.append(_gate(
        "fourteen_empty_answers_recovered",
        len(baseline_empty) == len(recovered) == 14
        and recovered_sources == Counter({
            "text_checkbox_negative": 8,
            "table": 4,
            "document_absence": 2,
        }),
        {"recovered": len(recovered), "source_counts": dict(sorted(recovered_sources.items()))},
    ))

    v6_details = {
        row["case_id"]: row
        for row in _jsonl(ROOT / "runs/type3_no_llm_experiment_v6/scoring/score_details.jsonl")
    }
    changed = [
        case_id for case_id in base_details
        if base_details[case_id]["prediction"] != v6_details[case_id]["prediction"]
    ]
    v6_regressions = [
        case_id for case_id in changed
        if v6_details[case_id]["bge_m3"]["score"] < base_details[case_id]["bge_m3"]["score"]
    ]
    repaired = [
        case_id for case_id in v6_regressions
        if full_details[case_id]["bge_m3"]["score"]
        >= base_details[case_id]["bge_m3"]["score"] - 0.000001
    ]
    checks.append(_gate(
        "v6_regressions_repaired",
        len(changed) == 119 and len(v6_regressions) == len(repaired) == 87,
        {"v6_changed": len(changed), "v6_regressed": len(v6_regressions), "repaired": len(repaired)},
    ))

    score_inputs_match = True
    ablations: dict[str, Any] = {}
    for stage in ("baseline", "checkbox", "table_text", "table_numeric", "faceted_frame"):
        stage_dir = run_dir / "ablations" / stage
        stage_score = _json(stage_dir / "scoring/score_report.json")
        actual_hash = _sha256(stage_dir / "http_evaluation.jsonl")
        score_inputs_match &= stage_score["inputs"]["http_results"]["sha256"] == actual_hash
        overall = stage_score["scores"]["bge_m3"]["overall"]
        ablations[stage] = {
            "nonempty": stage_score["coverage"]["nonempty_answers"],
            "total_score": overall["total_score"],
            "average_score": overall["average_score"],
            "keyword_matched": stage_score["gates"]["keyword_matched"],
        }
    score_inputs_match &= (
        full_report["inputs"]["http_results"]["sha256"]
        == _sha256(run_dir / "http_evaluation.jsonl")
    )
    ablations["full"] = {
        "nonempty": full_report["coverage"]["nonempty_answers"],
        "total_score": score["total_score"],
        "average_score": score["average_score"],
        "keyword_matched": full_report["gates"]["keyword_matched"],
    }
    checks.append(_gate("ablation_score_inputs_match", score_inputs_match, {"stages": ablations}))

    repeat_ready = (repeat_dir / "run_report.json").is_file()
    repeat_equal = repeat_ready and all(
        _sha256(run_dir / relative) == _sha256(repeat_dir / relative)
        for relative in ("http_evaluation.jsonl", "deterministic_traces.jsonl")
    )
    checks.append(_gate(
        "byte_determinism",
        repeat_equal,
        {"repeat_present": repeat_ready, "answers_and_traces_equal": repeat_equal},
    ))

    result = {
        "schema_version": "finglmqa.experimental.type3_v7_validation_report.v1",
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "ablations": ablations,
        "artifact_hashes": {
            "answers_sha256": _sha256(run_dir / "http_evaluation.jsonl"),
            "traces_sha256": _sha256(run_dir / "deterministic_traces.jsonl"),
            "generator_source_sha256": _sha256(ROOT / "src/finglmqa/type3_v7.py"),
            "scorer_source_sha256": full_report["inputs"]["scorer_script_sha256"],
        },
        "recommendation": "promotion_candidate_only" if all(check["passed"] for check in checks) else "do_not_promote",
        "default_phase8_phase10_modified": False,
    }
    _atomic_json(args.output.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
