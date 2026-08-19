#!/usr/bin/env python3
"""Derive a strict 3/3 selector-consensus projection from frozen v9 traces.

This safety projection is created before benchmark scoring and does not invoke
Qwen or BGE.  A fragment survives only when all three configured selector
runs are valid and all three selected the same ID.  The exact v8 baseline is
always retained as the answer suffix; a source fragment that merely duplicates
the baseline is omitted.
"""

from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unicodedata
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.contracts import canonical_json_bytes, semantic_sha256  # noqa: E402


SOURCE_DIR = ROOT / "runs/type3_qwen36_faceted_v9/full"
OUTPUT_DIR = ROOT / "runs/type3_qwen36_faceted_v9/unanimous"
PROFILE_VERSION = "type3-qwen36-faceted-evidence-v9-unanimous-3of3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RuntimeError(f"expected JSON object: {path}:{line_number}")
        rows.append(value)
    return rows


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


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, canonical_json_bytes(value))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_write(path, b"".join(canonical_json_bytes(dict(row)) for row in rows))


def compact(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).lower()
    return re.sub(r"[\s\W_]+", "", normalized, flags=re.UNICODE)


def unanimous_ids(row: Mapping[str, Any]) -> list[str]:
    runs = row.get("selector_runs")
    if (
        not isinstance(runs, list)
        or len(runs) != 3
        or any(not isinstance(run, Mapping) or run.get("status") != "ok" for run in runs)
    ):
        return []
    ordered: list[list[str]] = []
    for run in runs:
        selections = run.get("selections")
        if not isinstance(selections, list):
            return []
        ordered.append([
            str(value["fragment_id"])
            for value in selections
            if isinstance(value, Mapping) and isinstance(value.get("fragment_id"), str)
        ])
    common = set(ordered[0]).intersection(ordered[1], ordered[2])
    return [fragment_id for fragment_id in ordered[0] if fragment_id in common]


def project(row: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(row))
    parent_fingerprint = str(value.get("result_fingerprint") or "")
    baseline = str(value.get("baseline_answer") or "").strip()
    projection_by_id = {
        str(item["fragment_id"]): item
        for item in value.get("selected_fragment_projection") or []
        if isinstance(item, Mapping) and isinstance(item.get("fragment_id"), str)
    }
    selected_ids = unanimous_ids(value)
    # The parent majority projection also applies a 2,400-character/10-item
    # cap after voting.  Never resurrect a unanimous ID whose source text was
    # not persisted beyond that cap.
    selected_ids = [
        fragment_id for fragment_id in selected_ids if fragment_id in projection_by_id
    ]

    # A selected source sentence that only normalizes punctuation/spacing of
    # v8 adds no information.  Retain the exact v8 spelling instead.
    if baseline:
        baseline_key = compact(baseline)
        selected_ids = [
            fragment_id for fragment_id in selected_ids
            if compact(str(projection_by_id[fragment_id].get("text") or "")) != baseline_key
        ]
    selected_projection = [projection_by_id[fragment_id] for fragment_id in selected_ids]
    parts = [str(item["text"]).strip() for item in selected_projection if str(item["text"]).strip()]
    if baseline:
        parts.append(baseline)
    answer = "\n".join(parts)

    citations: list[dict[str, Any]] = []
    for citation in value.get("citations") or []:
        if not isinstance(citation, Mapping):
            continue
        candidate_id = citation.get("candidate_id")
        if isinstance(candidate_id, str) and candidate_id.startswith("v9frag_"):
            if candidate_id not in selected_ids:
                continue
        citations.append(dict(citation))

    value.update({
        "profile_version": PROFILE_VERSION,
        "parent_result_fingerprint": parent_fingerprint,
        "answer": answer,
        "citations": citations,
        "status": "ok" if answer else "not_found",
        "selector_outcome": "unanimous_selected" if selected_ids else "unanimous_empty",
        "selected_fragment_ids": selected_ids,
        "selected_fragment_projection": selected_projection,
        "unanimous_policy": {
            "required_valid_runs": 3,
            "required_selection_votes": 3,
            "baseline_retained_as_exact_suffix": True,
            "baseline_duplicate_source_fragments_removed": True,
        },
    })
    value["result_fingerprint"] = semantic_sha256({
        "profile_version": PROFILE_VERSION,
        "case_id": value["case_id"],
        "question": value["question"],
        "document_id": value["document_id"],
        "answer": answer,
        "citations": citations,
        "status": value["status"],
        "facets": value["facets"],
        "selected_fragment_ids": selected_ids,
        "gate_report": value["gate_report"],
    })
    return value


def http_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "kind": "benchmark",
        "oracle_match": True,
        "request": {"question": row["question"]},
        "response": {
            "answer": row["answer"],
            "citations": row["citations"],
            "status": row["status"],
            "errors": [],
            "warnings": [],
            "generator_modes": [PROFILE_VERSION, "document_scoped_extractive_ids_only"],
        },
        "experimental_profile": PROFILE_VERSION,
    }


def main() -> int:
    source_results = SOURCE_DIR / "results.jsonl"
    source_repeats = SOURCE_DIR / "repeat_results.jsonl"
    source_report = SOURCE_DIR / "run_report.json"
    source_freeze = SOURCE_DIR / "freeze_manifest.json"
    manifest = {
        "schema_version": "finglmqa.experimental.type3_qwen36_faceted_v9.unanimous_freeze.v1",
        "profile_version": PROFILE_VERSION,
        "derived_before_scoring": True,
        "generative_model_invoked": False,
        "source_hashes": {
            "results_sha256": sha256_file(source_results),
            "repeats_sha256": sha256_file(source_repeats),
            "run_report_sha256": sha256_file(source_report),
            "freeze_manifest_sha256": sha256_file(source_freeze),
        },
        "projector_sha256": sha256_file(Path(__file__).resolve()),
        "policy": "three_valid_seed_runs_and_three_selection_votes",
        "benchmark_annotations_loaded": [],
        "manifest_fingerprint": "",
    }
    manifest["manifest_fingerprint"] = semantic_sha256({
        key: child for key, child in manifest.items() if key != "manifest_fingerprint"
    })
    write_json(OUTPUT_DIR / "freeze_manifest.json", manifest)

    results = [project(row) for row in read_jsonl(source_results)]
    repeats = [project(row) for row in read_jsonl(source_repeats)]
    if len(results) != 260 or len(repeats) != 10:
        raise RuntimeError("frozen v9 source counts differ from 260/10")
    by_case = {str(row["case_id"]): row for row in results}
    repeat_exact = all(
        repeated["result_fingerprint"]
        == by_case[str(repeated["case_id"])]["result_fingerprint"]
        for repeated in repeats
    )
    cross_document = sum(
        citation.get("document_id") not in (None, row["document_id"])
        for row in results
        for citation in row["citations"]
    )
    baseline_suffix_failures = sum(
        bool(row["baseline_answer"].strip())
        and not row["answer"].endswith(row["baseline_answer"].strip())
        for row in results
    )
    failed_source_gates = sum(not row["gate_report"]["passed"] for row in results)
    model_text = sum(row["gate_report"]["model_text_accepted"] for row in results)
    safety = {
        "schema_version": "finglmqa.experimental.type3_qwen36_faceted_v9.unanimous_safety.v1",
        "rows": len(results),
        "nonempty_answers": sum(bool(row["answer"].strip()) for row in results),
        "selected_fragment_count": sum(len(row["selected_fragment_ids"]) for row in results),
        "selected_row_count": sum(bool(row["selected_fragment_ids"]) for row in results),
        "cross_document_citation_count": cross_document,
        "baseline_suffix_failure_count": baseline_suffix_failures,
        "failed_source_gate_count": failed_source_gates,
        "model_free_text_accepted_count": model_text,
        "repeat_count": len(repeats),
        "repeat_final_projection_exact": repeat_exact,
        "passed": all((
            len(results) == 260,
            all(row["answer"].strip() for row in results),
            cross_document == 0,
            baseline_suffix_failures == 0,
            failed_source_gates == 0,
            model_text == 0,
            repeat_exact,
        )),
    }
    write_jsonl(OUTPUT_DIR / "results.jsonl", results)
    write_jsonl(OUTPUT_DIR / "repeat_results.jsonl", repeats)
    write_jsonl(OUTPUT_DIR / "http_evaluation.jsonl", map(http_projection, results))
    write_json(OUTPUT_DIR / "safety_validation.json", safety)
    report = {
        "schema_version": "finglmqa.experimental.type3_qwen36_faceted_v9.unanimous_report.v1",
        "profile_version": PROFILE_VERSION,
        "rows": len(results),
        "nonempty_answers": safety["nonempty_answers"],
        "selected_row_count": safety["selected_row_count"],
        "selected_fragment_count_distribution": dict(sorted(Counter(
            len(row["selected_fragment_ids"]) for row in results
        ).items())),
        "repeat_final_projection_exact": repeat_exact,
        "safety_validation_passed": safety["passed"],
        "freeze_manifest_fingerprint": manifest["manifest_fingerprint"],
        "artifacts": {
            "results_sha256": sha256_file(OUTPUT_DIR / "results.jsonl"),
            "repeat_results_sha256": sha256_file(OUTPUT_DIR / "repeat_results.jsonl"),
            "http_evaluation_sha256": sha256_file(OUTPUT_DIR / "http_evaluation.jsonl"),
            "safety_validation_sha256": sha256_file(OUTPUT_DIR / "safety_validation.json"),
        },
        "benchmark_scoring_used_for_projection": False,
    }
    write_json(OUTPUT_DIR / "run_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if safety["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
