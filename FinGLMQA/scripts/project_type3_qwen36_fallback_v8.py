#!/usr/bin/env python3
"""Build the frozen safe-fallback projection for the Type 3 Qwen experiment.

The projection never reads benchmark prompts, keywords, or reference answers.
An accepted Qwen segment selection is used verbatim.  Every other outcome falls
back to the complete deterministic Type 3 v8 response for the same case.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QWEN_DIR = ROOT / "runs/type3_qwen36_organization_v8/full"
DEFAULT_V8_DIR = ROOT / "runs/type3_no_llm_experiment_v8"
DEFAULT_OUTPUT_DIR = ROOT / "runs/type3_qwen36_organization_v8/fallback"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise RuntimeError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, canonical_json_bytes(value))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_write(path, b"".join(canonical_json_bytes(dict(row)) for row in rows))


def keyed(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in result:
            raise RuntimeError(f"invalid or duplicate {label} case_id")
        result[case_id] = row
    return result


def verify_inputs(qwen_dir: Path, v8_dir: Path) -> dict[str, str]:
    qwen_report = read_json(qwen_dir / "run_report.json")
    qwen_safety = read_json(qwen_dir / "safety_validation.json")
    v8_report = read_json(v8_dir / "run_report.json")
    expected = {
        "qwen_http_evaluation_sha256": qwen_report["artifacts"]["http_evaluation_sha256"],
        "qwen_results_sha256": qwen_report["artifacts"]["results_sha256"],
        "qwen_safety_validation_sha256": qwen_report["artifacts"]["safety_validation_sha256"],
        "v8_http_evaluation_sha256": v8_report["stages"]["full"]["answers_sha256"],
    }
    paths = {
        "qwen_http_evaluation_sha256": qwen_dir / "http_evaluation.jsonl",
        "qwen_results_sha256": qwen_dir / "results.jsonl",
        "qwen_safety_validation_sha256": qwen_dir / "safety_validation.json",
        "v8_http_evaluation_sha256": v8_dir / "http_evaluation.jsonl",
    }
    for key, expected_hash in expected.items():
        if sha256_file(paths[key]) != expected_hash:
            raise RuntimeError(f"frozen input hash mismatch: {key}")
    if not qwen_report.get("safety_validation_passed") or not qwen_safety.get("passed"):
        raise RuntimeError("Qwen source did not pass its frozen safety validation")
    if not qwen_report.get("repeat_results_exact"):
        raise RuntimeError("Qwen source is not repeat deterministic")
    return expected


def project(qwen_dir: Path, v8_dir: Path, output_dir: Path) -> None:
    source_hashes = verify_inputs(qwen_dir, v8_dir)
    strict_rows = read_jsonl(qwen_dir / "http_evaluation.jsonl")
    result_rows = read_jsonl(qwen_dir / "results.jsonl")
    baseline_rows = read_jsonl(v8_dir / "http_evaluation.jsonl")
    if not (len(strict_rows) == len(result_rows) == len(baseline_rows) == 260):
        raise RuntimeError("safe fallback requires exactly 260 aligned cases")

    strict_by_case = keyed(strict_rows, "strict")
    results_by_case = keyed(result_rows, "result")
    baseline_by_case = keyed(baseline_rows, "baseline")
    if set(strict_by_case) != set(results_by_case) or set(strict_by_case) != set(baseline_by_case):
        raise RuntimeError("strict, result, and baseline case sets differ")

    output_rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    for baseline in baseline_rows:
        case_id = baseline["case_id"]
        strict = strict_by_case[case_id]
        result = results_by_case[case_id]
        outcome = result.get("generator_outcome")
        outcome_counts[str(outcome)] += 1
        strict_response = strict.get("response")
        baseline_response = baseline.get("response")
        if not isinstance(strict_response, dict) or not isinstance(baseline_response, dict):
            raise RuntimeError(f"invalid response projection for {case_id}")

        use_qwen = (
            outcome == "organized"
            and result.get("status") == "ok"
            and strict_response.get("status") == "ok"
            and isinstance(strict_response.get("answer"), str)
            and bool(strict_response["answer"].strip())
        )
        projected = dict(baseline)
        projected["ablation_stage"] = "qwen36_safe_fallback"
        projected["experimental_profile"] = "type3-v8-qwen36-safe-fallback-v1"
        projected["fallback_source"] = "qwen_organized" if use_qwen else "deterministic_v8"
        projected["response"] = strict_response if use_qwen else baseline_response
        output_rows.append(projected)
        source_counts[projected["fallback_source"]] += 1

    if any(not row["response"].get("answer", "").strip() for row in output_rows):
        raise RuntimeError("safe fallback projection contains an empty answer")

    output_path = output_dir / "http_evaluation.jsonl"
    write_jsonl(output_path, output_rows)

    # Verify the exact response-level branch contract after serialization.
    reloaded = read_jsonl(output_path)
    for row in reloaded:
        case_id = row["case_id"]
        expected_response = (
            strict_by_case[case_id]["response"]
            if row["fallback_source"] == "qwen_organized"
            else baseline_by_case[case_id]["response"]
        )
        if row["response"] != expected_response:
            raise RuntimeError(f"fallback response branch mismatch: {case_id}")

    safety = {
        "schema_version": "finglmqa.experimental.type3_qwen36_v8.fallback_safety.v1",
        "passed": True,
        "rows": len(reloaded),
        "nonempty_answers": sum(bool(row["response"]["answer"].strip()) for row in reloaded),
        "qwen_organized_responses_are_exact_strict_projections": True,
        "fallback_responses_are_exact_deterministic_v8_projections": True,
        "qwen_source_safety_validation_passed": True,
        "qwen_source_repeat_results_exact": True,
        "model_authored_text_accepted": False,
        "benchmark_reference_fields_read": [],
        "source_counts": dict(sorted(source_counts.items())),
        "generator_outcome_counts": dict(sorted(outcome_counts.items())),
    }
    safety_path = output_dir / "safety_validation.json"
    write_json(safety_path, safety)
    report = {
        "schema_version": "finglmqa.experimental.type3_qwen36_v8.fallback_report.v1",
        "profile_version": "type3-v8-qwen36-safe-fallback-v1",
        "rows": len(reloaded),
        "nonempty_answers": safety["nonempty_answers"],
        "source_counts": safety["source_counts"],
        "generator_outcome_counts": safety["generator_outcome_counts"],
        "selection_policy": (
            "use only organized, nonempty, gate-passed Qwen segment selections; "
            "otherwise use the exact deterministic Type 3 v8 response"
        ),
        "projection_frozen_before_scoring": True,
        "benchmark_scoring_used_for_selection": False,
        "benchmark_reference_fields_read": [],
        "source_hashes": dict(sorted(source_hashes.items())),
        "artifacts": {
            "http_evaluation_sha256": sha256_file(output_path),
            "safety_validation_sha256": sha256_file(safety_path),
            "projection_script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_json(output_dir / "run_report.json", report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-dir", type=Path, default=DEFAULT_QWEN_DIR)
    parser.add_argument("--v8-dir", type=Path, default=DEFAULT_V8_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    project(args.qwen_dir.resolve(), args.v8_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
