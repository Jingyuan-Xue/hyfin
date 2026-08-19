#!/usr/bin/env python3
"""Generate the contract-hardened Phase 5 compact Type 3 answer profile."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.type3_a2rag_tabgr_pipeline import (  # noqa: E402
    ManifestPaths,
    bind_manifests,
)
from finglmqa.type3_corpus_profile import (  # noqa: E402
    load_corpus_profile,
    load_question_profile,
    source_snapshot,
)
from finglmqa.type3_evidence_fusion import (  # noqa: E402
    canonical_json_bytes,
    semantic_sha256,
)
from finglmqa.type3_phase5_compact_composer import (  # noqa: E402
    COMPOSER_VERSION,
    MASK,
    OUTPUT_SCHEMA,
    PROFILE_VERSION,
    compose_compact_answer,
)


DEFAULT_QUESTIONS = (
    ROOT
    / "data/corpus_package/type3/annual_reports_170_v1/questions/"
    "type3_260_dev_v1/questions.jsonl"
)
DEFAULT_PHASE4_ANSWERS = (
    ROOT
    / "runs/type3_a2rag_tabgr_experiment_v1/annual_reports_170_v1/phase_4/"
    "evaluation/independent_validator/full/answers.jsonl"
)
DEFAULT_PHASE4_MANIFEST = (
    ROOT
    / "runs/type3_a2rag_tabgr_experiment_v1/annual_reports_170_v1/phase_4/"
    "evaluation/fresh_process_1/run_manifest.json"
)
DEFAULT_LEGACY_RESULTS = (
    ROOT / "runs/type3_qwen36_hybrid_coverage_v10/full/http_evaluation.jsonl"
)
DEFAULT_LEGACY_FREEZE = (
    ROOT / "runs/type3_qwen36_hybrid_coverage_v10/full/freeze_manifest.json"
)
DEFAULT_LEGACY_RUN_REPORT = (
    ROOT / "runs/type3_qwen36_hybrid_coverage_v10/full/run_report.json"
)

RUN_TRACE_SCHEMA = "finglmqa.type3.phase5.compact_run_trace.v2"
RUN_MANIFEST_SCHEMA = "finglmqa.type3.phase5.compact_run_manifest.v2"
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_FIELDS = frozenset(
    {
        "answer_key",
        "best_reference",
        "bge_m3",
        "delta",
        "eval_spec",
        "gold",
        "key_word",
        "keyword",
        "prom_answer",
        "prompt_answer",
        "reference_answer",
        "references",
        "score",
        "score_details",
        "score_report",
        "sequence_audit",
        "v10_score",
    }
)
_SCORER_FILENAMES = frozenset(
    {
        "ablation_summary.json",
        "lowest_50_nonempty.jsonl",
        "paired_deltas.jsonl",
        "predictions.jsonl",
        "score_details.jsonl",
        "score_report.json",
        "score_report.md",
    }
)
_ACTIVE_STAGING: Path | None = None
_ANSWER_SCHEMA_PATH = (
    ROOT / "data/schemas/type3/phase5_compact_answer_v2.schema.json"
)
_ANSWER_SCHEMA = json.loads(_ANSWER_SCHEMA_PATH.read_text(encoding="utf-8"))
Draft202012Validator.check_schema(_ANSWER_SCHEMA)
_ANSWER_VALIDATOR = Draft202012Validator(_ANSWER_SCHEMA)


class Phase5RunnerError(ValueError):
    """Raised when a generator-side input violates the frozen contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def guard_generator_path(path: Path) -> Path:
    """Reject known scorer/gold locations before any generator-side read."""

    resolved = path.resolve()
    lowered_parts = [part.lower() for part in resolved.parts]
    lowered = resolved.as_posix().lower()
    if "/big-finbenchmark/data/by_type/" in lowered:
        raise Phase5RunnerError("generator cannot read benchmark gold")
    if any(
        part.startswith("scoring")
        or part in {"gold", "golden", "evaluation_gold"}
        for part in lowered_parts
    ):
        raise Phase5RunnerError("generator cannot read a scoring/gold directory")
    name = resolved.name.lower()
    if name in _SCORER_FILENAMES or "gold" in name:
        raise Phase5RunnerError("generator cannot read scorer/gold artifacts")
    return resolved


def _reject_forbidden_fields(value: object, *, location: str = "$") -> None:
    if isinstance(value, Mapping):
        lowered = {str(key).lower() for key in value}
        overlap = _FORBIDDEN_FIELDS.intersection(lowered)
        if overlap:
            raise Phase5RunnerError(
                f"generator value contains forbidden annotation fields at {location}: "
                f"{sorted(overlap)!r}"
            )
        for key, item in value.items():
            _reject_forbidden_fields(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_fields(item, location=f"{location}[{index}]")


def load_json(path: Path) -> dict[str, Any]:
    guarded = guard_generator_path(path)
    value = json.loads(guarded.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase5RunnerError(f"expected JSON object: {guarded}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    guarded = guard_generator_path(path)
    rows: list[dict[str, Any]] = []
    with guarded.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise Phase5RunnerError(f"blank JSONL record: {guarded}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise Phase5RunnerError(f"expected object at {guarded}:{line_number}")
            rows.append(value)
    return rows


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row))


def _safe_output_dir(
    output_dir: Path,
    *,
    protected_dirs: Sequence[Path],
) -> tuple[Path, Path]:
    resolved = output_dir.resolve()
    if any(part.lower().startswith("scoring") for part in resolved.parts):
        raise Phase5RunnerError("generator output cannot be inside a scoring directory")
    for protected in protected_dirs:
        frozen = protected.resolve()
        if (
            resolved == frozen
            or resolved.is_relative_to(frozen)
            or frozen.is_relative_to(resolved)
        ):
            raise Phase5RunnerError(
                f"generator output overlaps frozen input area: {resolved} vs {frozen}"
            )
    if resolved.exists():
        raise Phase5RunnerError(f"output namespace must not exist: {resolved}")
    staging = resolved.parent / f".{resolved.name}.staging"
    if staging.exists():
        raise Phase5RunnerError(f"stale generator staging directory exists: {staging}")
    return resolved, staging


def _verify_phase4(
    *,
    answers_path: Path,
    manifest_path: Path,
    expected_binding: Mapping[str, Any],
    corpus_id: str,
    question_profile_id: str,
    questions: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != "finglmqa.type3.a2rag_tabgr.run_manifest.v1":
        raise Phase5RunnerError("Phase 4 manifest schema differs")
    if manifest.get("pipeline_version") != "type3-a2rag-tabgr-pipeline-v1":
        raise Phase5RunnerError("Phase 4 pipeline version differs")
    if manifest.get("arm") != "union":
        raise Phase5RunnerError("Phase 5 requires the frozen Phase 4 union arm")
    if (
        manifest.get("corpus_id") != corpus_id
        or manifest.get("question_profile_id") != question_profile_id
    ):
        raise Phase5RunnerError("Phase 4 corpus/question profile binding differs")
    expected_ids = [row["question_id"] for row in questions]
    if (
        manifest.get("question_count") != len(questions)
        or manifest.get("question_ids_sha256") != semantic_sha256(expected_ids)
    ):
        raise Phase5RunnerError("Phase 4 question identity binding differs")
    if manifest.get("manifest_binding") != expected_binding:
        raise Phase5RunnerError("Phase 4 manifest binding differs from current frozen inputs")
    safety = manifest.get("safety") or {}
    if (
        safety.get("cross_document_evidence") != 0
        or safety.get("unsupported_numeric_literals") != 0
    ):
        raise Phase5RunnerError("Phase 4 safety gate is not clean")
    unsigned = {key: value for key, value in manifest.items() if key != "run_fingerprint"}
    if manifest.get("run_fingerprint") != semantic_sha256(unsigned):
        raise Phase5RunnerError("Phase 4 run fingerprint differs")
    answers_hash = sha256_file(answers_path)
    if manifest.get("artifacts", {}).get("answers.jsonl") != answers_hash:
        raise Phase5RunnerError("Phase 4 answer hash differs from frozen manifest")

    rows = load_jsonl(answers_path)
    _reject_forbidden_fields(rows, location="phase4_answers")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("arm") != "union":
            raise Phase5RunnerError("Phase 4 packet arm differs")
        question_id = str(row.get("question_id") or "")
        if not question_id or question_id in by_id:
            raise Phase5RunnerError("invalid or duplicate Phase 4 question_id")
        trace = row.get("semantic_trace") or {}
        if trace.get("arm") != "union":
            raise Phase5RunnerError("Phase 4 packet/trace arm mismatch")
        if row.get("corpus_id") != corpus_id:
            raise Phase5RunnerError("Phase 4 packet corpus differs")
        document_id = row.get("document_id")
        for evidence in row.get("evidence") or ():
            if (
                evidence.get("corpus_id") != corpus_id
                or evidence.get("document_id") != document_id
            ):
                raise Phase5RunnerError("Phase 4 evidence crosses corpus/document boundary")
        by_id[question_id] = row
    if set(by_id) != set(expected_ids):
        raise Phase5RunnerError("question profile and Phase 4 case sets differ")
    return by_id


def _clean_legacy_citation(
    value: Mapping[str, Any],
    *,
    corpus_id: str,
    document_id: str,
) -> dict[str, Any]:
    if value.get("document_id") != document_id:
        raise Phase5RunnerError("legacy citation crosses document boundary")

    provenance = value.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    source_markdown = str(
        value.get("source_markdown") or provenance.get("source_markdown") or ""
    )
    line_range = value.get("line_range") or provenance.get("line_range")
    content_sha256 = str(
        value.get("content_sha256") or provenance.get("content_sha256") or ""
    )
    citation_id = str(value.get("citation_id") or "")
    candidate_id = str(
        value.get("candidate_id")
        or value.get("source_evidence_id")
        or provenance.get("evidence_chunk_id")
        or citation_id
    )
    if (
        not citation_id
        or not candidate_id
        or not source_markdown
        or not isinstance(line_range, list)
        or len(line_range) != 2
        or not _HEX64_RE.fullmatch(content_sha256)
    ):
        raise Phase5RunnerError("legacy citation projection is incomplete")
    citation = {
        "citation_kind": "legacy_evidence",
        "corpus_id": corpus_id,
        "document_id": document_id,
        "citation_id": citation_id,
        "candidate_id": candidate_id,
        "source_kind": str(value.get("source_kind") or "legacy_v10_evidence"),
        "source_markdown": source_markdown,
        "line_range": [int(line_range[0]), int(line_range[1])],
        "content_sha256": content_sha256,
        "heading_path": [
            str(item)
            for item in (
                value.get("heading_path")
                or provenance.get("section_path")
                or ()
            )
        ],
    }
    _reject_forbidden_fields(citation, location="legacy_citation")
    return citation


def _verify_legacy_freeze(
    *,
    results_path: Path,
    freeze_path: Path,
    run_report_path: Path,
) -> None:
    freeze = load_json(freeze_path)
    fingerprint = freeze.get("manifest_fingerprint")
    computed = semantic_sha256(
        {key: value for key, value in freeze.items() if key != "manifest_fingerprint"}
    )
    if not isinstance(fingerprint, str) or fingerprint != computed:
        raise Phase5RunnerError("legacy freeze manifest fingerprint differs")
    if (
        freeze.get("profile_version") != "type3-qwen36-hybrid-coverage-v10"
        or freeze.get("frozen_before_model_invocation") is not True
        or freeze.get("frozen_before_scoring") is not True
        or freeze.get("forbidden_benchmark_fields_consumed") != []
    ):
        raise Phase5RunnerError("legacy freeze contract is not annotation-free")
    report = load_json(run_report_path)
    if (
        report.get("profile_version") != freeze["profile_version"]
        or report.get("freeze_manifest_fingerprint") != fingerprint
        or report.get("input_rows") != 260
        or report.get("terminal_rows") != 260
        or report.get("nonempty_answers") != 260
        or report.get("safety_validation_passed") is not True
        or report.get("benchmark_scoring_used_for_prompt_or_rule_selection") is not False
        or report.get("forbidden_benchmark_fields_loaded_by_answer_chain") != []
    ):
        raise Phase5RunnerError("legacy run report does not satisfy the frozen contract")
    if (
        report.get("artifacts", {}).get("http_evaluation_sha256")
        != sha256_file(results_path)
    ):
        raise Phase5RunnerError("legacy HTTP result hash differs from run report")


def _load_legacy(
    path: Path,
    *,
    corpus_id: str,
) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
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
    for row in rows:
        if set(row) != expected_top or row.get("kind") != "benchmark":
            raise Phase5RunnerError("legacy HTTP row schema differs")
        if row.get("experimental_profile") != "type3-qwen36-hybrid-coverage-v10":
            raise Phase5RunnerError("legacy HTTP profile differs")
        request = row.get("request")
        response = row.get("response")
        if (
            not isinstance(request, Mapping)
            or set(request) != {"question"}
            or not isinstance(response, Mapping)
            or set(response) != expected_response
        ):
            raise Phase5RunnerError("legacy request/response schema differs")
        question_id = str(row.get("case_id") or "")
        answer = str(response.get("answer") or "")
        question = str(request.get("question") or "")
        if (
            not question_id
            or question_id in by_id
            or not question
            or not answer
            or response.get("status") != "ok"
        ):
            raise Phase5RunnerError("legacy projection is incomplete")
        citations_raw = response.get("citations")
        if not isinstance(citations_raw, list):
            raise Phase5RunnerError("legacy citations must be an array")
        document_ids = {
            str(value.get("document_id") or "")
            for value in citations_raw
            if isinstance(value, Mapping)
        }
        if len(document_ids) != 1 or "" in document_ids:
            raise Phase5RunnerError("legacy citations do not identify exactly one document")
        document_id = next(iter(document_ids))
        citations = [
            _clean_legacy_citation(
                value,
                corpus_id=corpus_id,
                document_id=document_id,
            )
            for value in citations_raw
            if isinstance(value, Mapping)
        ]
        if len(citations) != len(citations_raw):
            raise Phase5RunnerError("legacy citation is not an object")
        projection = {
            "question": question,
            "document_id": document_id,
            "answer": answer,
            "citations": citations,
        }
        _reject_forbidden_fields(projection, location="legacy_projection")
        by_id[question_id] = projection
    return by_id


def _validate_output(
    row: Mapping[str, Any],
    *,
    corpus_id: str,
    question: Mapping[str, str],
) -> None:
    schema_errors = sorted(
        _ANSWER_VALIDATOR.iter_errors(row),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if schema_errors:
        first = schema_errors[0]
        location = "$" + "".join(f"[{part!r}]" for part in first.absolute_path)
        raise Phase5RunnerError(
            f"Phase 5 output schema validation failed at {location}: {first.message}"
        )
    if (
        row.get("schema_version") != OUTPUT_SCHEMA
        or row.get("profile_version") != PROFILE_VERSION
        or row.get("question_id") != question["question_id"]
        or row.get("question") != question["question"]
        or row.get("document_id") != question["document_id"]
    ):
        raise Phase5RunnerError("Phase 5 output identity/schema differs")
    answer = row.get("answer_safe_text")
    if not isinstance(answer, str) or not answer.strip() or MASK in answer:
        raise Phase5RunnerError("Phase 5 final answer is empty or masked")
    selected = row.get("selected_candidate_ids")
    if not isinstance(selected, list) or len(selected) > 1 or len(set(selected)) != len(selected):
        raise Phase5RunnerError("Phase 5 selected candidate contract differs")
    citations = row.get("citations")
    if not isinstance(citations, list):
        raise Phase5RunnerError("Phase 5 citations must be an array")
    selected_citations = 0
    for citation in citations:
        if not isinstance(citation, Mapping):
            raise Phase5RunnerError("Phase 5 citation is not an object")
        if (
            citation.get("corpus_id") != corpus_id
            or citation.get("document_id") != question["document_id"]
        ):
            raise Phase5RunnerError("Phase 5 citation crosses corpus/document boundary")
        if citation.get("citation_kind") == "phase5_a2rag_text":
            selected_citations += 1
            if (
                citation.get("route") != "text"
                or citation.get("candidate_id") not in selected
            ):
                raise Phase5RunnerError("Phase 5 selected citation differs")
    if selected_citations != len(selected):
        raise Phase5RunnerError("Phase 5 selected citation count differs")
    trace = row.get("semantic_trace")
    if not isinstance(trace, Mapping):
        raise Phase5RunnerError("Phase 5 semantic trace is missing")
    if (
        trace.get("schema_version") != RUN_TRACE_SCHEMA
        or trace.get("profile_version") != PROFILE_VERSION
        or trace.get("selected_candidate_ids") != selected
        or trace.get("semantic_input")
        != {
            "corpus_id": corpus_id,
            "document_id": question["document_id"],
            "question": question["question"],
        }
    ):
        raise Phase5RunnerError("Phase 5 semantic trace contract differs")
    trace_unsigned = {
        key: value for key, value in trace.items() if key != "semantic_trace_sha256"
    }
    if trace.get("semantic_trace_sha256") != semantic_sha256(trace_unsigned):
        raise Phase5RunnerError("Phase 5 semantic trace hash differs")
    composer = trace.get("composer")
    if not isinstance(composer, Mapping):
        raise Phase5RunnerError("Phase 5 composer trace is missing")
    composer_unsigned = {
        key: value for key, value in composer.items() if key != "semantic_trace_sha256"
    }
    if composer.get("semantic_trace_sha256") != semantic_sha256(composer_unsigned):
        raise Phase5RunnerError("Phase 5 composer trace hash differs")
    _reject_forbidden_fields(row, location="phase5_output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--corpus-id", default="annual_reports_170_v1")
    parser.add_argument("--question-profile-id", default="type3_260_dev_v1")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--question-profile", type=Path)
    parser.add_argument("--phase4-answers", type=Path, default=DEFAULT_PHASE4_ANSWERS)
    parser.add_argument("--phase4-manifest", type=Path, default=DEFAULT_PHASE4_MANIFEST)
    parser.add_argument("--legacy-http-results", type=Path, default=DEFAULT_LEGACY_RESULTS)
    parser.add_argument("--legacy-freeze-manifest", type=Path, default=DEFAULT_LEGACY_FREEZE)
    parser.add_argument("--legacy-run-report", type=Path, default=DEFAULT_LEGACY_RUN_REPORT)
    parser.add_argument("--without-legacy", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _main() -> int:
    global _ACTIVE_STAGING
    args = parse_args()
    root = args.root.resolve()
    manifest_paths = ManifestPaths.defaults(
        root,
        args.corpus_id,
        args.question_profile_id,
    )
    corpus_manifest_path = guard_generator_path(
        args.corpus_manifest or manifest_paths.corpus_manifest
    )
    question_profile_path = guard_generator_path(
        args.question_profile or manifest_paths.question_profile
    )
    questions_path = guard_generator_path(args.questions)
    if corpus_manifest_path != manifest_paths.corpus_manifest.resolve():
        raise Phase5RunnerError("corpus manifest path must match corpus/profile identity")
    if question_profile_path != manifest_paths.question_profile.resolve():
        raise Phase5RunnerError("question profile path must match corpus/profile identity")
    if questions_path != manifest_paths.questions.resolve():
        raise Phase5RunnerError("questions path must be the file bound by question profile")

    corpus = load_corpus_profile(corpus_manifest_path)
    question_profile, questions = load_question_profile(
        question_profile_path,
        corpus_profile=corpus,
    )
    if (
        corpus["corpus_id"] != args.corpus_id
        or question_profile["question_profile_id"] != args.question_profile_id
    ):
        raise Phase5RunnerError("corpus/question profile CLI identity differs")
    _reject_forbidden_fields(questions, location="sanitized_questions")
    source_before = source_snapshot(corpus, workspace_root=root)
    binding = bind_manifests(
        manifest_paths,
        expected_corpus_id=args.corpus_id,
        expected_question_profile_id=args.question_profile_id,
    ).as_mapping()

    phase4_path = guard_generator_path(args.phase4_answers)
    phase4_manifest_path = guard_generator_path(args.phase4_manifest)
    legacy_paths: tuple[Path, ...] = ()
    if not args.without_legacy:
        legacy_paths = (
            guard_generator_path(args.legacy_http_results),
            guard_generator_path(args.legacy_freeze_manifest),
            guard_generator_path(args.legacy_run_report),
        )

    input_paths: dict[str, Path] = {
        "corpus_manifest": corpus_manifest_path,
        "question_profile": question_profile_path,
        "questions": questions_path,
        "phase4_answers": phase4_path,
        "phase4_manifest": phase4_manifest_path,
        **{
            f"binding_{name}": Path(value["path"]).resolve()
            for name, value in binding["manifests"].items()
        },
    }
    if legacy_paths:
        input_paths.update(
            {
                "legacy_http_results": legacy_paths[0],
                "legacy_freeze_manifest": legacy_paths[1],
                "legacy_run_report": legacy_paths[2],
            }
        )
    for path in input_paths.values():
        guard_generator_path(path)
        if not path.is_file():
            raise Phase5RunnerError(f"frozen input is missing: {path}")
    frozen_hashes_before = {
        key: sha256_file(path) for key, path in sorted(input_paths.items())
    }

    phase4 = _verify_phase4(
        answers_path=phase4_path,
        manifest_path=phase4_manifest_path,
        expected_binding=binding,
        corpus_id=args.corpus_id,
        question_profile_id=args.question_profile_id,
        questions=questions,
    )
    legacy: dict[str, dict[str, Any]] = {}
    if legacy_paths:
        _verify_legacy_freeze(
            results_path=legacy_paths[0],
            freeze_path=legacy_paths[1],
            run_report_path=legacy_paths[2],
        )
        legacy = _load_legacy(legacy_paths[0], corpus_id=args.corpus_id)

    expected_ids = [row["question_id"] for row in questions]
    if not args.without_legacy and set(legacy) != set(expected_ids):
        raise Phase5RunnerError("question profile and legacy case sets differ")

    source_root = (root / corpus["source_ref"]).resolve()
    protected_dirs = [
        source_root,
        root / "data/corpus_package/type3" / args.corpus_id,
        root / "data/indexes/type3" / args.corpus_id,
        root / "data/facts/type3" / args.corpus_id,
        phase4_path.parent,
        phase4_manifest_path.parent,
    ]
    for ancestor in phase4_path.parents:
        if ancestor.name == "phase_4":
            protected_dirs.append(ancestor)
            break
    if legacy_paths:
        protected_dirs.append(legacy_paths[0].parent)
    final_output_dir, output_dir = _safe_output_dir(
        args.output_dir,
        protected_dirs=protected_dirs,
    )
    _ACTIVE_STAGING = output_dir

    outputs: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    modes: Counter[str] = Counter()
    for question_row in questions:
        question_id = question_row["question_id"]
        packet = phase4[question_id]
        if (
            packet.get("question") != question_row["question"]
            or packet.get("document_id") != question_row["document_id"]
        ):
            raise Phase5RunnerError("Phase 4 packet differs from sanitized question")
        legacy_row = legacy.get(question_id) or {
            "question": question_row["question"],
            "document_id": question_row["document_id"],
            "answer": "",
            "citations": [],
        }
        if (
            legacy_row["question"] != question_row["question"]
            or legacy_row["document_id"] != question_row["document_id"]
        ):
            raise Phase5RunnerError("legacy identity differs from sanitized question")
        composed = compose_compact_answer(
            question=question_row["question"],
            evidence=packet.get("evidence") or (),
            legacy_answer=legacy_row["answer"],
            legacy_citations=legacy_row["citations"],
        )
        composer_trace = composed["semantic_trace"]
        modes[str(composer_trace["mode"])] += 1
        semantic_trace_unsigned = {
            "schema_version": RUN_TRACE_SCHEMA,
            "profile_version": PROFILE_VERSION,
            "semantic_input": {
                "corpus_id": args.corpus_id,
                "document_id": question_row["document_id"],
                "question": question_row["question"],
            },
            "source_phase4_packet_sha256": semantic_sha256(packet),
            "composer": composer_trace,
            "selected_candidate_ids": composed["selected_candidate_ids"],
        }
        semantic_trace = {
            **semantic_trace_unsigned,
            "semantic_trace_sha256": semantic_sha256(semantic_trace_unsigned),
        }
        output = {
            "schema_version": OUTPUT_SCHEMA,
            "profile_version": PROFILE_VERSION,
            "question_id": question_id,
            "question": question_row["question"],
            "document_id": question_row["document_id"],
            "answer_safe_text": composed["answer_safe_text"],
            "citations": composed["citations"],
            "selected_candidate_ids": composed["selected_candidate_ids"],
            "semantic_trace": semantic_trace,
        }
        _validate_output(
            output,
            corpus_id=args.corpus_id,
            question=question_row,
        )
        outputs.append(output)
        traces.append({"question_id": question_id, "semantic_trace": semantic_trace})

    answers_path = output_dir / "answers.jsonl"
    traces_path = output_dir / "semantic_traces.jsonl"
    _write_jsonl(answers_path, outputs)
    _write_jsonl(traces_path, traces)

    source_after = source_snapshot(corpus, workspace_root=root)
    frozen_hashes_after = {
        key: sha256_file(path) for key, path in sorted(input_paths.items())
    }
    if source_after != source_before:
        raise Phase5RunnerError("source Markdown changed during Phase 5 generation")
    if frozen_hashes_after != frozen_hashes_before:
        raise Phase5RunnerError("frozen generator input changed during generation")

    nonempty = sum(bool(row["answer_safe_text"].strip()) for row in outputs)
    masked = sum(row["answer_safe_text"].count(MASK) for row in outputs)
    cross_document = sum(
        citation.get("document_id") != row["document_id"]
        or citation.get("corpus_id") != args.corpus_id
        for row in outputs
        for citation in row["citations"]
    )
    if nonempty != len(outputs) or masked != 0 or cross_document != 0:
        raise Phase5RunnerError("Phase 5 hard safety gate failed")

    manifest_unsigned = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "profile_version": PROFILE_VERSION,
        "composer_version": COMPOSER_VERSION,
        "corpus_id": args.corpus_id,
        "corpus_profile_sha256": corpus["profile_sha256"],
        "question_profile_id": args.question_profile_id,
        "question_profile_sha256": question_profile["profile_sha256"],
        "question_count": len(outputs),
        "question_ids_sha256": semantic_sha256(expected_ids),
        "configuration": {
            "legacy_baseline_enabled": not args.without_legacy,
            "maximum_selected_a2rag_sentences": 1,
            "minimum_question_bigram_recall": "0.08",
            "table_route_rendered_in_final_answer": False,
        },
        "manifest_binding": binding,
        "inputs": {
            key: {
                "path": _relative(path, root),
                "sha256": frozen_hashes_before[key],
            }
            for key, path in sorted(input_paths.items())
        },
        "source_freeze": {
            "source_ref": corpus["source_ref"],
            "document_count": len(source_before),
            "source_hashes_sha256_before": semantic_sha256(source_before),
            "source_hashes_sha256_after": semantic_sha256(source_after),
            "source_unchanged": True,
        },
        "code": {
            "composer_sha256": sha256_file(
                root / "src/finglmqa/type3_phase5_compact_composer.py"
            ),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "answer_schema_sha256": sha256_file(
                _ANSWER_SCHEMA_PATH
            ),
        },
        "generator_boundary": {
            "semantic_fields": ["corpus_id", "document_id", "question"],
            "question_id_use": "join_only",
            "benchmark_annotations_available": False,
            "scorer_outputs_available": False,
            "input_output_disjointness_enforced": True,
            "inputs_rehashed_after_generation": True,
        },
        "safety": {
            "nonempty_answers": nonempty,
            "masked_placeholders": masked,
            "cross_document_citations": cross_document,
            "selected_a2rag_sentences": sum(
                bool(row["selected_candidate_ids"]) for row in outputs
            ),
            "mode_counts": dict(sorted(modes.items())),
        },
        "artifacts": {
            "answers.jsonl": sha256_file(answers_path),
            "semantic_traces.jsonl": sha256_file(traces_path),
        },
    }
    manifest = {
        **manifest_unsigned,
        "run_fingerprint": semantic_sha256(manifest_unsigned),
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    output_dir.rename(final_output_dir)
    _ACTIVE_STAGING = None
    print(
        json.dumps(
            {
                "status": "passed",
                "questions": len(outputs),
                "answers_sha256": manifest["artifacts"]["answers.jsonl"],
                "traces_sha256": manifest["artifacts"]["semantic_traces.jsonl"],
                "output_dir": final_output_dir.as_posix(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    global _ACTIVE_STAGING
    try:
        return _main()
    finally:
        if _ACTIVE_STAGING is not None and _ACTIVE_STAGING.exists():
            shutil.rmtree(_ACTIVE_STAGING)
        _ACTIVE_STAGING = None


if __name__ == "__main__":
    raise SystemExit(main())
