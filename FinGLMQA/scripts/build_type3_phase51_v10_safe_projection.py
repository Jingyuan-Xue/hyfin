#!/usr/bin/env python3
"""Build the annotation-free frozen v10 input projection for Phase 5.1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = (
    ROOT
    / "data/corpus_package/type3/annual_reports_170_v1/questions/"
    "type3_260_dev_v1/questions.jsonl"
)
DEFAULT_QUESTION_PROFILE = DEFAULT_QUESTIONS.with_name("question_profile.json")
DEFAULT_V10_RESULTS = (
    ROOT / "runs/type3_qwen36_hybrid_coverage_v10/full/http_evaluation.jsonl"
)
DEFAULT_V10_FREEZE = (
    ROOT / "runs/type3_qwen36_hybrid_coverage_v10/full/freeze_manifest.json"
)
DEFAULT_V10_RUN_REPORT = (
    ROOT / "runs/type3_qwen36_hybrid_coverage_v10/full/run_report.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "runs/type3_a2rag_tabgr_experiment_v1/annual_reports_170_v1/"
    "phase_5_1/phase_b_inputs/v10_safe_projection_v1"
)
SCHEMA_PATH = (
    ROOT / "data/schemas/type3/phase51_v10_safe_projection_v1.schema.json"
)

PROFILE_VERSION = "type3-phase51-v10-safe-projection-v1"
MANIFEST_SCHEMA = "finglmqa.type3.phase51.v10_safe_projection_manifest.v1"
V10_PROFILE = "type3-qwen36-hybrid-coverage-v10"
V10_FREEZE_SCHEMA = "finglmqa.experimental.type3_qwen36_coverage_v10.freeze.v1"
V10_REPORT_SCHEMA = (
    "finglmqa.experimental.type3_qwen36_coverage_v10.run_report.v1"
)
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_ANNOTATION_FRAGMENTS = (
    "oracle",
    "gold",
    "score",
    "delta",
    "loss",
    "win",
    "reference",
)
_ANSWER_FIELDS = (
    "question_id",
    "question",
    "document_id",
    "answer_safe_text",
    "citations",
)
_CITATION_FIELDS = (
    "citation_kind",
    "corpus_id",
    "document_id",
    "citation_id",
    "candidate_id",
    "source_kind",
    "source_markdown",
    "line_range",
    "content_sha256",
    "heading_path",
)
_ACTIVE_STAGING: Path | None = None

_PACKAGE_SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
Draft202012Validator.check_schema(_PACKAGE_SCHEMA)
_PACKAGE_VALIDATOR = Draft202012Validator(_PACKAGE_SCHEMA)


class SafeProjectionError(ValueError):
    """Raised when a frozen source or safe projection violates its contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SafeProjectionError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise SafeProjectionError(
                    f"blank JSONL record: {path}:{line_number}"
                )
            value = json.loads(line)
            if not isinstance(value, dict):
                raise SafeProjectionError(
                    f"expected JSON object: {path}:{line_number}"
                )
            rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row))


def _annotation_tokens(value: object) -> tuple[str, ...]:
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value))
    return tuple(
        token.lower() for token in re.findall(r"[A-Za-z0-9]+", words)
    )


def _enforce_safe_projection_keys(value: object, *, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            tokens = _annotation_tokens(key)
            compact = "".join(tokens)
            if (
                any(
                    fragment in token
                    for token in tokens
                    for fragment in _FORBIDDEN_ANNOTATION_FRAGMENTS
                )
                or "answerkey" in compact
            ):
                raise SafeProjectionError(
                    f"forbidden annotation key in safe projection: "
                    f"{location}.{key}"
                )
            _enforce_safe_projection_keys(
                child,
                location=f"{location}.{key}",
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _enforce_safe_projection_keys(
                child,
                location=f"{location}[{index}]",
            )


def _validate_package_value(value: Mapping[str, Any], *, label: str) -> None:
    errors = sorted(
        _PACKAGE_VALIDATOR.iter_errors(value),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if errors:
        first = errors[0]
        location = "$" + "".join(
            f"[{part!r}]" for part in first.absolute_path
        )
        raise SafeProjectionError(
            f"{label} failed closed schema at {location}: {first.message}"
        )


def _verify_question_profile(
    *,
    questions_path: Path,
    profile_path: Path,
    corpus_id: str,
    question_profile_id: str,
    expected_count: int,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    profile = _load_json(profile_path)
    expected_profile_fields = {
        "allowed_fields",
        "annotations_available_to_generator",
        "corpus_id",
        "profile_sha256",
        "question_count",
        "question_profile_id",
        "question_records_sha256",
        "questions_path",
        "questions_sha256",
        "schema_version",
    }
    if set(profile) != expected_profile_fields:
        raise SafeProjectionError("question profile shape differs")
    if (
        profile.get("schema_version") != "type3-question-profile-v1"
        or profile.get("corpus_id") != corpus_id
        or profile.get("question_profile_id") != question_profile_id
        or profile.get("allowed_fields")
        != ["question_id", "question", "document_id"]
        or profile.get("annotations_available_to_generator") is not False
        or profile.get("question_count") != expected_count
        or Path(str(profile.get("questions_path") or "")).name
        != questions_path.name
        or profile.get("questions_sha256") != sha256_file(questions_path)
        or not _HEX64_RE.fullmatch(str(profile.get("profile_sha256") or ""))
        or profile.get("profile_sha256")
        != semantic_sha256(
            {
                key: value
                for key, value in profile.items()
                if key != "profile_sha256"
            }
        )
    ):
        raise SafeProjectionError("question profile identity/freeze differs")
    raw_questions = _load_jsonl(questions_path)
    questions: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(raw_questions):
        if set(row) != {"question_id", "question", "document_id"}:
            raise SafeProjectionError(
                f"question row shape differs at index {index}"
            )
        projected = {
            "question_id": str(row.get("question_id") or ""),
            "question": str(row.get("question") or ""),
            "document_id": str(row.get("document_id") or ""),
        }
        if (
            not all(projected.values())
            or projected["question_id"] in seen_ids
        ):
            raise SafeProjectionError(
                f"question identity differs at index {index}"
            )
        seen_ids.add(projected["question_id"])
        questions.append(projected)
    if (
        len(questions) != expected_count
        or profile.get("question_records_sha256")
        != semantic_sha256(questions)
    ):
        raise SafeProjectionError("question count/semantic hash differs")
    return profile, questions


def _verify_v10_freeze(
    *,
    results_path: Path,
    freeze_path: Path,
    run_report_path: Path,
    expected_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    freeze = _load_json(freeze_path)
    fingerprint = str(freeze.get("manifest_fingerprint") or "")
    computed = semantic_sha256(
        {
            key: value
            for key, value in freeze.items()
            if key != "manifest_fingerprint"
        }
    )
    if (
        freeze.get("schema_version") != V10_FREEZE_SCHEMA
        or freeze.get("profile_version") != V10_PROFILE
        or fingerprint != computed
        or freeze.get("frozen_before_model_invocation") is not True
        or freeze.get("frozen_before_scoring") is not True
        or freeze.get("forbidden_benchmark_fields_consumed") != []
    ):
        raise SafeProjectionError("v10 freeze manifest contract differs")
    report = _load_json(run_report_path)
    results_hash = sha256_file(results_path)
    if (
        report.get("schema_version") != V10_REPORT_SCHEMA
        or report.get("profile_version") != V10_PROFILE
        or report.get("freeze_manifest_fingerprint") != fingerprint
        or report.get("input_rows") != expected_count
        or report.get("terminal_rows") != expected_count
        or report.get("nonempty_answers") != expected_count
        or report.get("safety_validation_passed") is not True
        or report.get("benchmark_scoring_used_for_prompt_or_rule_selection")
        is not False
        or report.get("forbidden_benchmark_fields_loaded_by_answer_chain")
        != []
        or report.get("artifacts", {}).get("http_evaluation_sha256")
        != results_hash
    ):
        raise SafeProjectionError("v10 run report/hash contract differs")
    return freeze, report


def _clean_legacy_citation(
    value: Mapping[str, Any],
    *,
    corpus_id: str,
    document_id: str,
) -> dict[str, Any]:
    if value.get("document_id") != document_id:
        raise SafeProjectionError("legacy citation crosses document boundary")
    provenance = value.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    citation_id = str(value.get("citation_id") or "")
    candidate_id = str(
        value.get("candidate_id")
        or value.get("source_evidence_id")
        or provenance.get("evidence_chunk_id")
        or citation_id
    )
    source_markdown = str(
        value.get("source_markdown")
        or provenance.get("source_markdown")
        or ""
    )
    line_range = value.get("line_range") or provenance.get("line_range")
    content_sha256 = str(
        value.get("content_sha256")
        or provenance.get("content_sha256")
        or ""
    )
    heading_path = value.get("heading_path")
    if heading_path is None:
        heading_path = provenance.get("section_path")
    if heading_path is None:
        heading_path = []
    if (
        not citation_id
        or not candidate_id
        or not source_markdown
        or not isinstance(line_range, list)
        or len(line_range) != 2
        or not all(isinstance(item, int) and item >= 0 for item in line_range)
        or line_range[0] > line_range[1]
        or not _HEX64_RE.fullmatch(content_sha256)
        or not isinstance(heading_path, list)
    ):
        raise SafeProjectionError("legacy citation projection is incomplete")
    citation = {
        "citation_kind": "legacy_evidence",
        "corpus_id": corpus_id,
        "document_id": document_id,
        "citation_id": citation_id,
        "candidate_id": candidate_id,
        "source_kind": str(
            value.get("source_kind") or "legacy_v10_evidence"
        ),
        "source_markdown": source_markdown,
        "line_range": [line_range[0], line_range[1]],
        "content_sha256": content_sha256,
        "heading_path": [str(item) for item in heading_path],
    }
    if not citation["source_kind"]:
        raise SafeProjectionError("legacy citation source_kind is empty")
    return citation


def _project_envelope_row(
    row: Mapping[str, Any],
    *,
    corpus_id: str,
    expected_question: Mapping[str, str],
) -> dict[str, Any]:
    expected_top = {
        "case_id",
        "experimental_profile",
        "kind",
        "oracle_match",
        "request",
        "response",
    }
    expected_response = {
        "answer",
        "citations",
        "errors",
        "generator_modes",
        "status",
        "warnings",
    }
    if (
        set(row) != expected_top
        or row.get("experimental_profile") != V10_PROFILE
        or row.get("kind") != "benchmark"
        or row.get("case_id") != expected_question["question_id"]
    ):
        raise SafeProjectionError("v10 envelope identity/shape differs")
    request = row.get("request")
    response = row.get("response")
    if (
        not isinstance(request, Mapping)
        or set(request) != {"question"}
        or request.get("question") != expected_question["question"]
        or not isinstance(response, Mapping)
        or set(response) != expected_response
        or response.get("status") != "ok"
        or response.get("errors") != []
        or not isinstance(response.get("warnings"), list)
        or not isinstance(response.get("generator_modes"), list)
    ):
        raise SafeProjectionError("v10 request/response contract differs")
    answer = str(response.get("answer") or "")
    citations_raw = response.get("citations")
    if not answer.strip() or not isinstance(citations_raw, list) or not citations_raw:
        raise SafeProjectionError("v10 answer/citations projection is incomplete")
    document_id = expected_question["document_id"]
    if any(
        not isinstance(citation, Mapping)
        or citation.get("document_id") != document_id
        for citation in citations_raw
    ):
        raise SafeProjectionError("v10 citations differ from question document")
    citations = [
        _clean_legacy_citation(
            citation,
            corpus_id=corpus_id,
            document_id=document_id,
        )
        for citation in citations_raw
    ]
    projection = {
        "question_id": expected_question["question_id"],
        "question": expected_question["question"],
        "document_id": document_id,
        "answer_safe_text": answer,
        "citations": citations,
    }
    _enforce_safe_projection_keys(projection)
    _validate_package_value(projection, label="safe answer")
    return projection


def _safe_output_dir(
    output_dir: Path,
    *,
    protected_paths: Sequence[Path],
) -> tuple[Path, Path]:
    resolved = output_dir.resolve()
    for protected_path in protected_paths:
        protected = protected_path.resolve()
        if (
            resolved == protected
            or resolved.is_relative_to(protected)
            or protected.is_relative_to(resolved)
        ):
            raise SafeProjectionError(
                f"output overlaps frozen input/code: {resolved} vs {protected}"
            )
    if resolved.exists():
        raise SafeProjectionError(f"output namespace already exists: {resolved}")
    staging = resolved.parent / f".{resolved.name}.staging"
    if staging.exists():
        raise SafeProjectionError(f"stale staging namespace exists: {staging}")
    return resolved, staging


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--corpus-id", default="annual_reports_170_v1")
    parser.add_argument("--question-profile-id", default="type3_260_dev_v1")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument(
        "--question-profile",
        type=Path,
        default=DEFAULT_QUESTION_PROFILE,
    )
    parser.add_argument("--v10-results", type=Path, default=DEFAULT_V10_RESULTS)
    parser.add_argument("--v10-freeze", type=Path, default=DEFAULT_V10_FREEZE)
    parser.add_argument(
        "--v10-run-report",
        type=Path,
        default=DEFAULT_V10_RUN_REPORT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-question-count", type=int, default=260)
    return parser.parse_args(argv)


def _main(argv: Sequence[str] | None = None) -> int:
    global _ACTIVE_STAGING
    args = parse_args(argv)
    if args.expected_question_count < 1:
        raise SafeProjectionError("expected question count must be positive")
    root = args.root.resolve()
    source_paths = {
        "v10_results": args.v10_results.resolve(),
        "v10_freeze": args.v10_freeze.resolve(),
        "v10_run_report": args.v10_run_report.resolve(),
        "questions": args.questions.resolve(),
        "question_profile": args.question_profile.resolve(),
    }
    code_paths = {
        "builder": Path(__file__).resolve(),
        "schema": SCHEMA_PATH.resolve(),
    }
    for path in [*source_paths.values(), *code_paths.values()]:
        if not path.is_file():
            raise SafeProjectionError(f"required frozen artifact is missing: {path}")
    final_dir, staging = _safe_output_dir(
        args.output_dir,
        protected_paths=[*source_paths.values(), *code_paths.values()],
    )
    source_hashes_before = {
        key: sha256_file(path) for key, path in sorted(source_paths.items())
    }
    code_hashes_before = {
        key: sha256_file(path) for key, path in sorted(code_paths.items())
    }
    question_profile, questions = _verify_question_profile(
        questions_path=source_paths["questions"],
        profile_path=source_paths["question_profile"],
        corpus_id=args.corpus_id,
        question_profile_id=args.question_profile_id,
        expected_count=args.expected_question_count,
    )
    freeze, _ = _verify_v10_freeze(
        results_path=source_paths["v10_results"],
        freeze_path=source_paths["v10_freeze"],
        run_report_path=source_paths["v10_run_report"],
        expected_count=args.expected_question_count,
    )
    envelopes = _load_jsonl(source_paths["v10_results"])
    if len(envelopes) != len(questions):
        raise SafeProjectionError("v10 envelope/question count differs")
    envelope_ids = [str(row.get("case_id") or "") for row in envelopes]
    question_ids = [row["question_id"] for row in questions]
    if envelope_ids != question_ids:
        raise SafeProjectionError("v10 envelope order differs from question profile")
    answers = [
        _project_envelope_row(
            envelope,
            corpus_id=args.corpus_id,
            expected_question=question,
        )
        for envelope, question in zip(envelopes, questions)
    ]

    _ACTIVE_STAGING = staging
    staging.mkdir(parents=True)
    answers_path = staging / "answers.jsonl"
    _write_jsonl(answers_path, answers)
    answers_hash = sha256_file(answers_path)

    source_hashes_after = {
        key: sha256_file(path) for key, path in sorted(source_paths.items())
    }
    code_hashes_after = {
        key: sha256_file(path) for key, path in sorted(code_paths.items())
    }
    if source_hashes_after != source_hashes_before:
        raise SafeProjectionError("frozen source artifact changed during projection")
    if code_hashes_after != code_hashes_before:
        raise SafeProjectionError("builder/schema changed during projection")

    manifest_unsigned = {
        "schema_version": MANIFEST_SCHEMA,
        "profile_version": PROFILE_VERSION,
        "corpus_id": args.corpus_id,
        "question_profile_id": args.question_profile_id,
        "question_profile_sha256": question_profile["profile_sha256"],
        "questions": {
            "count": len(questions),
            "question_ids_sha256": semantic_sha256(question_ids),
            "question_records_sha256": semantic_sha256(questions),
            "question_order_sha256": semantic_sha256(
                [
                    {"ordinal": index, "question_id": question_id}
                    for index, question_id in enumerate(question_ids)
                ]
            ),
        },
        "source_v10": {
            "profile_version": V10_PROFILE,
            "freeze_manifest_fingerprint": freeze["manifest_fingerprint"],
            "results_sha256": source_hashes_before["v10_results"],
        },
        "source_artifacts": {
            key: {
                "path": _relative(source_paths[key], root),
                "sha256_before": source_hashes_before[key],
                "sha256_after": source_hashes_after[key],
            }
            for key in sorted(source_paths)
        },
        "code": {
            key: {
                "path": _relative(code_paths[key], root),
                "sha256_before": code_hashes_before[key],
                "sha256_after": code_hashes_after[key],
            }
            for key in sorted(code_paths)
        },
        "projection_contract": {
            "allowed_answer_fields": list(_ANSWER_FIELDS),
            "allowed_legacy_citation_fields": list(_CITATION_FIELDS),
            "forbidden_annotation_key_fragments": [
                *_FORBIDDEN_ANNOTATION_FRAGMENTS,
                "answer_key",
            ],
            "annotation_fields_projected": False,
            "raw_envelope_retained": False,
            "source_artifacts_read_only": True,
            "closed_schema_validated": True,
        },
        "artifacts": {
            "answers.jsonl": {
                "path": "answers.jsonl",
                "row_count": len(answers),
                "sha256": answers_hash,
            }
        },
    }
    manifest = {
        **manifest_unsigned,
        "run_fingerprint": semantic_sha256(manifest_unsigned),
    }
    _enforce_safe_projection_keys(
        {"answers": answers, "manifest": manifest}
    )
    _validate_package_value(manifest, label="safe projection manifest")
    _write_json(staging / "manifest.json", manifest)
    staging.rename(final_dir)
    _ACTIVE_STAGING = None
    print(
        json.dumps(
            {
                "status": "passed",
                "rows": len(answers),
                "answers_sha256": answers_hash,
                "run_fingerprint": manifest["run_fingerprint"],
                "output_dir": final_dir.as_posix(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    global _ACTIVE_STAGING
    try:
        return _main(argv)
    finally:
        if _ACTIVE_STAGING is not None and _ACTIVE_STAGING.exists():
            shutil.rmtree(_ACTIVE_STAGING)
        _ACTIVE_STAGING = None


if __name__ == "__main__":
    raise SystemExit(main())
