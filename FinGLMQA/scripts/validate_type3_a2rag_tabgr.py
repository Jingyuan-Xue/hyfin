#!/usr/bin/env python3
"""Validate Phase 1 corpus contracts and the frozen Type 3 v10 baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.type3_corpus_profile import (  # noqa: E402
    canonical_json_bytes,
    load_corpus_profile,
    load_question_profile,
    profile_sha256,
    semantic_sha256,
    sha256_file,
    source_snapshot,
)


EXPECTED_SCORES = {
    "v8_no_llm": 0.688603,
    "v10_no_qwen_same_index": 0.712122,
    "v10_qwen_plan_deterministic_selector": 0.715633,
    "v10_qwen_coverage_unanimous_3of3": 0.751349,
    "v10_qwen_coverage_majority_2of3": 0.76664,
}
EXPECTED_V10_FREEZE_FINGERPRINT = (
    "2a59738fef8b09cc9a154ea08ee12ef961dbc5ddbd59491241997b917059c7e6"
)
FORBIDDEN_FIELDS = frozenset({
    "prompt", "prompt_answer", "prom_answer", "key_word", "keyword",
    "reference", "references", "reference_answer", "gold", "answer_key",
})


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise RuntimeError(f"blank JSONL row: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"expected JSON object: {path}:{line_number}")
            rows.append(value)
    return rows


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_FIELDS:
                found.add(str(key))
            found.update(_forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_forbidden_keys(child))
    return found


def _hashes(paths: list[Path], *, root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"frozen artifact is missing: {path}")
        result[path.relative_to(root).as_posix()] = sha256_file(path)
    return result


def _score(path: Path) -> float:
    report = _read_json(path)
    return float(report["scores"]["bge_m3"]["overall"]["average_score"])


def _verify_v8(
    v8_dir: Path,
    *,
    questions: list[dict[str, str]],
    document_ids: set[str],
) -> dict[str, Any]:
    report_path = v8_dir / "run_report.json"
    answers_path = v8_dir / "http_evaluation.jsonl"
    traces_path = v8_dir / "deterministic_traces.jsonl"
    report = _read_json(report_path)
    expected = report.get("stages", {}).get("full", {})
    if sha256_file(answers_path) != expected.get("answers_sha256"):
        raise RuntimeError("v8 answers differ from frozen run_report")
    if sha256_file(traces_path) != expected.get("traces_sha256"):
        raise RuntimeError("v8 traces differ from frozen run_report")
    answers = _read_jsonl(answers_path)
    traces = _read_jsonl(traces_path)
    if len(answers) != 260 or len(traces) != 260:
        raise RuntimeError("v8 must contain exactly 260 answers and traces")
    if _forbidden_keys(answers) or _forbidden_keys(traces):
        raise RuntimeError("forbidden benchmark annotations entered v8 generation inputs")
    trace_by_case = {str(row.get("case_id")): row for row in traces}
    answer_by_case = {str(row.get("case_id")): row for row in answers}
    if len(trace_by_case) != 260 or len(answer_by_case) != 260:
        raise RuntimeError("v8 case_id values are not unique")
    if set(trace_by_case) != set(answer_by_case):
        raise RuntimeError("v8 answer and trace case sets differ")
    if [row["question_id"] for row in questions] != [str(row["case_id"]) for row in answers]:
        raise RuntimeError("sanitized question order/identity differs from frozen v8")
    for question in questions:
        case_id = question["question_id"]
        answer = answer_by_case[case_id]
        trace = trace_by_case[case_id]
        request = answer.get("request")
        if not isinstance(request, Mapping) or request.get("question") != question["question"]:
            raise RuntimeError(f"question text differs from frozen v8: {case_id}")
        if trace.get("document_id") != question["document_id"]:
            raise RuntimeError(f"document boundary differs from frozen v8: {case_id}")
        if question["document_id"] not in document_ids:
            raise RuntimeError(f"v8 question references document outside corpus: {case_id}")
    safety = _read_json(v8_dir / "validation_report.json")
    if safety.get("status") != "passed" or safety.get("rows") != 260:
        raise RuntimeError("v8 validation report is not passed")
    return {
        "rows": 260,
        "answers_sha256": sha256_file(answers_path),
        "traces_sha256": sha256_file(traces_path),
        "run_report_sha256": sha256_file(report_path),
        "validation_passed": True,
    }


def _verify_artifact_hashes(
    report: Mapping[str, Any], *, base_dir: Path, mapping: Mapping[str, str]
) -> None:
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError(f"artifact report is incomplete: {base_dir}")
    for key, relative in mapping.items():
        expected = artifacts.get(key)
        actual = sha256_file(base_dir / relative)
        if expected != actual:
            raise RuntimeError(f"v10 artifact hash mismatch: {key}")


def _verify_safety(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    zero_fields = (
        "baseline_suffix_failure_count", "cross_document_citation_count",
        "failed_gate_count", "model_free_text_accepted_count",
        "unsupported_selected_number_count", "unsupported_selected_text_count",
    )
    if (
        value.get("passed") is not True
        or value.get("all_rows_terminal") is not True
        or value.get("rows") != 260
        or value.get("nonempty_answers") != 260
        or any(value.get(field) != 0 for field in zero_fields)
    ):
        raise RuntimeError(f"v10 safety gate failed or drifted: {path}")
    return {field: value[field] for field in ("passed", "rows", "nonempty_answers", *zero_fields)}


def _verify_v10(v10_dir: Path, *, repo_root: Path) -> dict[str, Any]:
    full = v10_dir / "full"
    unanimous = v10_dir / "unanimous"
    freeze = _read_json(full / "freeze_manifest.json")
    if freeze.get("manifest_fingerprint") != EXPECTED_V10_FREEZE_FINGERPRINT:
        raise RuntimeError("v10 freeze manifest fingerprint drifted")
    computed_fingerprint = semantic_sha256({
        key: value for key, value in freeze.items() if key != "manifest_fingerprint"
    })
    if computed_fingerprint != EXPECTED_V10_FREEZE_FINGERPRINT:
        raise RuntimeError("v10 freeze manifest content does not match fingerprint")
    if semantic_sha256(freeze["generation_config"]) != freeze["generation_config_sha256"]:
        raise RuntimeError("v10 generation config hash differs")
    index_paths = {
        "evidence_chunks_sha256": repo_root / "data/corpus_package/evidence_chunks.jsonl",
        "document_map_sha256": repo_root / "data/indexes/a2rag_index/document_chunk_map.jsonl",
        "dense_index_manifest_sha256": repo_root / "data/indexes/a2rag_index/index_manifest.json",
    }
    for key, path in index_paths.items():
        if sha256_file(path) != freeze["index_hashes"].get(key):
            raise RuntimeError(f"v10 frozen index differs: {key}")
    code_paths = {
        "coverage_v10_sha256": repo_root / "src/finglmqa/type3_qwen36_coverage_v10.py",
        "runner_sha256": repo_root / "scripts/run_type3_qwen36_coverage_v10.py",
        "base_retriever_sha256": repo_root / "scripts/query_type3_evidence.py",
    }
    for key, path in code_paths.items():
        if sha256_file(path) != freeze["code_hashes"].get(key):
            raise RuntimeError(f"v10 frozen code differs: {key}")

    full_report = _read_json(full / "run_report.json")
    unanimous_report = _read_json(unanimous / "run_report.json")
    if full_report.get("freeze_manifest_fingerprint") != EXPECTED_V10_FREEZE_FINGERPRINT:
        raise RuntimeError("v10 run report freeze binding differs")
    if unanimous_report.get("source_freeze_manifest_fingerprint") != EXPECTED_V10_FREEZE_FINGERPRINT:
        raise RuntimeError("v10 unanimous report freeze binding differs")
    _verify_artifact_hashes(full_report, base_dir=full, mapping={
        "results_sha256": "results.jsonl",
        "repeat_results_sha256": "repeat_results.jsonl",
        "http_evaluation_sha256": "http_evaluation.jsonl",
        "deterministic_http_sha256": "ablations/qwen_plan_deterministic/http_evaluation.jsonl",
        "no_qwen_same_index_http_sha256": "ablations/no_qwen_same_index/http_evaluation.jsonl",
        "safety_validation_sha256": "safety_validation.json",
        "unanimous_http_evaluation_sha256": "../unanimous/http_evaluation.jsonl",
    })
    _verify_artifact_hashes(unanimous_report, base_dir=unanimous, mapping={
        "results_sha256": "results.jsonl",
        "repeat_results_sha256": "repeat_results.jsonl",
        "http_evaluation_sha256": "http_evaluation.jsonl",
        "safety_validation_sha256": "safety_validation.json",
    })
    row_paths = [
        full / "results.jsonl", full / "http_evaluation.jsonl",
        full / "ablations/no_qwen_same_index/http_evaluation.jsonl",
        full / "ablations/qwen_plan_deterministic/http_evaluation.jsonl",
        unanimous / "results.jsonl", unanimous / "http_evaluation.jsonl",
    ]
    if any(len(_read_jsonl(path)) != 260 for path in row_paths):
        raise RuntimeError("a v10 baseline or ablation does not contain 260 rows")
    safety = {
        "majority_2of3": _verify_safety(full / "safety_validation.json"),
        "unanimous_3of3": _verify_safety(unanimous / "safety_validation.json"),
    }
    comparison = _read_json(v10_dir / "comparison_report.json")
    comparison_scores = comparison.get("scores", {})
    score_paths = {
        "v8_no_llm": repo_root / "runs/type3_no_llm_experiment_v8/scoring/score_report.json",
        "v10_no_qwen_same_index": full / "ablations/no_qwen_same_index/scoring/score_report.json",
        "v10_qwen_plan_deterministic_selector": full / "ablations/qwen_plan_deterministic/scoring/score_report.json",
        "v10_qwen_coverage_unanimous_3of3": unanimous / "scoring/score_report.json",
        "v10_qwen_coverage_majority_2of3": full / "scoring/score_report.json",
    }
    actual_scores = {key: _score(path) for key, path in score_paths.items()}
    if actual_scores != EXPECTED_SCORES:
        raise RuntimeError(f"frozen v10 ablation scores drifted: {actual_scores!r}")
    if any(float(comparison_scores.get(key, -1)) != score for key, score in EXPECTED_SCORES.items()):
        raise RuntimeError("v10 comparison report differs from score artifacts")
    if comparison.get("safety", {}).get("passed") is not True:
        raise RuntimeError("v10 comparison safety is not passed")
    return {
        "freeze_manifest_fingerprint": EXPECTED_V10_FREEZE_FINGERPRINT,
        "scores": actual_scores,
        "safety": safety,
        "index_hashes": freeze["index_hashes"],
        "primary_rows": full_report.get("input_rows"),
        "unanimous_rows": unanimous_report.get("rows"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("1",), required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--question-profile", type=Path, required=True)
    parser.add_argument("--v8-dir", type=Path, required=True)
    parser.add_argument("--v10-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    allowed = (
        repo_root / "runs/type3_a2rag_tabgr_experiment_v1/annual_reports_170_v1/phase_1"
    ).resolve()
    if output_dir != allowed:
        raise RuntimeError(f"Phase 1 output-dir must equal {allowed}")
    corpus = load_corpus_profile(args.corpus_manifest.resolve())
    question_profile, questions = load_question_profile(
        args.question_profile.resolve(), corpus_profile=corpus
    )
    if corpus["corpus_id"] != "annual_reports_170_v1" or corpus["document_count"] != 170:
        raise RuntimeError("Phase 1 expects the frozen annual_reports_170_v1 corpus")
    if question_profile["question_profile_id"] != "type3_260_dev_v1" or len(questions) != 260:
        raise RuntimeError("Phase 1 expects the frozen type3_260_dev_v1 questions")

    v8_dir = args.v8_dir.resolve()
    v10_dir = args.v10_dir.resolve()
    frozen_paths = [
        v8_dir / "http_evaluation.jsonl",
        v8_dir / "deterministic_traces.jsonl",
        v8_dir / "run_report.json",
        v8_dir / "validation_report.json",
        v8_dir / "scoring/score_report.json",
        v10_dir / "comparison_report.json",
        v10_dir / "full/freeze_manifest.json",
        v10_dir / "full/results.jsonl",
        v10_dir / "full/repeat_results.jsonl",
        v10_dir / "full/http_evaluation.jsonl",
        v10_dir / "full/run_report.json",
        v10_dir / "full/safety_validation.json",
        v10_dir / "full/scoring/score_report.json",
        v10_dir / "full/ablations/no_qwen_same_index/http_evaluation.jsonl",
        v10_dir / "full/ablations/no_qwen_same_index/scoring/score_report.json",
        v10_dir / "full/ablations/qwen_plan_deterministic/http_evaluation.jsonl",
        v10_dir / "full/ablations/qwen_plan_deterministic/scoring/score_report.json",
        v10_dir / "unanimous/results.jsonl",
        v10_dir / "unanimous/repeat_results.jsonl",
        v10_dir / "unanimous/http_evaluation.jsonl",
        v10_dir / "unanimous/run_report.json",
        v10_dir / "unanimous/safety_validation.json",
        v10_dir / "unanimous/scoring/score_report.json",
    ]
    frozen_before = _hashes(frozen_paths, root=repo_root)
    sources_before = source_snapshot(corpus, workspace_root=repo_root)
    document_ids = {row["document_id"] for row in corpus["documents"]}
    v8 = _verify_v8(v8_dir, questions=questions, document_ids=document_ids)
    v10 = _verify_v10(v10_dir, repo_root=repo_root)
    sources_after = source_snapshot(corpus, workspace_root=repo_root)
    frozen_after = _hashes(frozen_paths, root=repo_root)
    if sources_before != sources_after:
        raise RuntimeError("source Markdown changed during Phase 1 validation")
    if frozen_before != frozen_after:
        raise RuntimeError("a frozen v8/v10 artifact changed during Phase 1 validation")
    report = {
        "schema_version": "finglmqa.type3_a2rag_tabgr.phase1_validation.v1",
        "phase": 1,
        "status": "passed",
        "corpus": {
            "corpus_id": corpus["corpus_id"],
            "document_count": corpus["document_count"],
            "profile_sha256": corpus["profile_sha256"],
            "source_hashes_sha256_before": semantic_sha256(sources_before),
            "source_hashes_sha256_after": semantic_sha256(sources_after),
            "source_unchanged": True,
        },
        "questions": {
            "question_profile_id": question_profile["question_profile_id"],
            "question_count": question_profile["question_count"],
            "profile_sha256": question_profile["profile_sha256"],
            "annotations_available_to_generator": False,
            "allowed_fields": question_profile["allowed_fields"],
            "document_boundary_passed": True,
        },
        "v8": v8,
        "v10": v10,
        "frozen_artifacts": {
            "count": len(frozen_before),
            "hashes": frozen_before,
            "unchanged": True,
        },
        "expensive_inference": {
            "qwen_invoked": False,
            "bge_full_scoring_invoked": False,
            "corpus_rebuilt": False,
        },
    }
    report["report_sha256"] = profile_sha256(report)
    _atomic_write(output_dir / "validation_report.json", canonical_json_bytes(report))
    print(json.dumps({
        "status": "passed",
        "documents": corpus["document_count"],
        "questions": len(questions),
        "frozen_artifacts": len(frozen_before),
        "scores": v10["scores"],
        "output": (output_dir / "validation_report.json").as_posix(),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
