#!/usr/bin/env python3
"""Build a read-only Type 3 corpus profile and a sanitized question profile."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.type3_corpus_profile import (  # noqa: E402
    CORPUS_PROFILE_SCHEMA,
    QUESTION_PROFILE_SCHEMA,
    QUESTION_RECORD_FIELDS,
    canonical_json_bytes,
    question_jsonl_bytes,
    semantic_sha256,
    sha256_file,
    source_snapshot,
    validate_corpus_profile,
    validate_question_profile,
    with_profile_sha256,
)


FORBIDDEN_ANNOTATION_FIELDS = frozenset({
    "answer", "answers", "reference_answer", "reference_answers", "gold_answer",
    "key_word", "keyword", "keywords", "score", "scores", "scoring",
    "label_answer", "prom_answer",
})
UPSTREAM_QUESTION_FIELDS = frozenset({
    "mapped_report_uids", "question", "question_file", "source_id",
    "source_split", "type", "uid",
})


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise RuntimeError(f"blank JSONL row: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"expected JSON object: {path}:{line_number}")
            yield value


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


def _safe_relative(value: str, *, field: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"{field} must be a workspace-relative path")
    return path.as_posix()


def _source_documents(
    upstream: Mapping[str, Any], *, source_root: Path
) -> list[dict[str, Any]]:
    selected = upstream.get("selected_reports")
    if not isinstance(selected, list) or not selected:
        raise RuntimeError("upstream manifest has no selected_reports")
    if upstream.get("report_count") != len(selected):
        raise RuntimeError("upstream report_count differs from selected_reports")
    documents: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        if not isinstance(row, Mapping):
            raise RuntimeError(f"selected_reports[{index}] must be an object")
        required = {"doc_id", "a2rag_doc", "stock_code", "company_full", "report_year"}
        if not required.issubset(row):
            raise RuntimeError(f"selected_reports[{index}] is incomplete")
        declared = Path(str(row["a2rag_doc"]))
        source = source_root / declared.name
        if not source.is_file() or source.resolve() != declared.resolve():
            raise RuntimeError(f"a2rag source mismatch: {declared}")
        documents.append({
            "document_id": str(row["doc_id"]),
            "company": str(row["company_full"]),
            "stock_code": str(row["stock_code"]),
            "report_year": int(row["report_year"]),
            "source_markdown": source.name,
            "source_sha256": sha256_file(source),
        })
    return sorted(documents, key=lambda row: row["document_id"])


def _sanitized_questions(
    question_source: Path,
    *,
    type_label: str,
    document_by_upstream_uid: Mapping[str, str],
) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in _read_jsonl(question_source):
        keys = set(row)
        forbidden = {str(key) for key in keys if str(key).lower() in FORBIDDEN_ANNOTATION_FIELDS}
        if forbidden:
            raise RuntimeError(f"question source contains forbidden annotations: {sorted(forbidden)!r}")
        if keys != UPSTREAM_QUESTION_FIELDS:
            raise RuntimeError(
                f"question source schema changed: missing={sorted(UPSTREAM_QUESTION_FIELDS - keys)!r}, "
                f"unknown={sorted(keys - UPSTREAM_QUESTION_FIELDS)!r}"
            )
        if str(row["type"]) != type_label:
            continue
        mapped = row["mapped_report_uids"]
        if not isinstance(mapped, list) or len(mapped) != 1 or not isinstance(mapped[0], str):
            raise RuntimeError(f"Type 3 question must map to one document: {row['uid']!r}")
        document_id = document_by_upstream_uid.get(mapped[0])
        if document_id is None:
            raise RuntimeError(f"question maps outside corpus: {mapped[0]!r}")
        question_id = f"benchmark:{type_label}:{row['uid']}"
        if question_id in seen:
            raise RuntimeError(f"duplicate question_id: {question_id}")
        seen.add(question_id)
        questions.append({
            "question_id": question_id,
            "question": str(row["question"]),
            "document_id": document_id,
        })
    if not questions:
        raise RuntimeError(f"no questions matched type label {type_label!r}")
    return questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--upstream-manifest", type=Path, required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--question-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--question-profile-id", required=True)
    parser.add_argument("--type-label", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    source_ref = _safe_relative(args.source_ref, field="source-ref")
    source_root = (repo_root / source_ref).resolve()
    output_root = args.output_root.resolve()
    allowed_root = (repo_root / "data/corpus_package/type3").resolve()
    if not output_root.is_relative_to(allowed_root):
        raise RuntimeError(f"output-root must remain under {allowed_root}")
    if output_root.name != args.corpus_id:
        raise RuntimeError("output-root basename must equal corpus-id")
    if not source_root.is_dir():
        raise RuntimeError(f"source-ref is not a directory: {source_root}")

    upstream_path = args.upstream_manifest.resolve()
    question_source = args.question_source.resolve()
    upstream = _read_json(upstream_path)
    documents = _source_documents(upstream, source_root=source_root)
    corpus_profile = with_profile_sha256({
        "schema_version": CORPUS_PROFILE_SCHEMA,
        "corpus_id": args.corpus_id,
        "source_ref": source_ref,
        "document_count": len(documents),
        "documents_sha256": semantic_sha256(documents),
        "documents": documents,
    })
    corpus_profile = validate_corpus_profile(corpus_profile)
    source_before = source_snapshot(corpus_profile, workspace_root=repo_root)

    document_by_upstream_uid = {
        f"A{row['stock_code']}_{row['report_year']}": row["document_id"]
        for row in documents
    }
    if len(document_by_upstream_uid) != len(documents):
        raise RuntimeError("stock/year keys are not unique in corpus")
    questions = _sanitized_questions(
        question_source,
        type_label=args.type_label,
        document_by_upstream_uid=document_by_upstream_uid,
    )
    question_dir = output_root / "questions" / args.question_profile_id
    questions_path = question_dir / "questions.jsonl"
    question_profile_path = question_dir / "question_profile.json"
    corpus_profile_path = output_root / "corpus_manifest.json"
    _atomic_write(questions_path, question_jsonl_bytes(questions))
    question_profile = with_profile_sha256({
        "schema_version": QUESTION_PROFILE_SCHEMA,
        "question_profile_id": args.question_profile_id,
        "corpus_id": args.corpus_id,
        "question_count": len(questions),
        "questions_path": "questions.jsonl",
        "questions_sha256": sha256_file(questions_path),
        "question_records_sha256": semantic_sha256(questions),
        "allowed_fields": list(QUESTION_RECORD_FIELDS),
        "annotations_available_to_generator": False,
    })
    question_profile = validate_question_profile(question_profile)
    _write_json(corpus_profile_path, corpus_profile)
    _write_json(question_profile_path, question_profile)

    source_after = source_snapshot(corpus_profile, workspace_root=repo_root)
    if source_before != source_after:
        raise RuntimeError("source Markdown hashes changed during build")
    source_freeze = {
        "schema_version": "finglmqa.type3.phase1.source_freeze.v1",
        "corpus_id": args.corpus_id,
        "document_count": len(source_before),
        "source_hashes_sha256_before": semantic_sha256(source_before),
        "source_hashes_sha256_after": semantic_sha256(source_after),
        "source_unchanged": True,
    }
    source_freeze_path = output_root / "source_freeze.json"
    _write_json(source_freeze_path, source_freeze)
    artifact_manifest = {
        "schema_version": "finglmqa.type3.phase1.artifact_manifest.v1",
        "corpus_id": args.corpus_id,
        "question_profile_id": args.question_profile_id,
        "inputs": {
            "upstream_manifest": args.upstream_manifest.as_posix(),
            "upstream_manifest_sha256": sha256_file(upstream_path),
            "question_source": args.question_source.as_posix(),
            "question_source_sha256": sha256_file(question_source),
        },
        "artifacts": {
            "corpus_manifest": corpus_profile_path.relative_to(output_root).as_posix(),
            "corpus_manifest_sha256": sha256_file(corpus_profile_path),
            "question_profile": question_profile_path.relative_to(output_root).as_posix(),
            "question_profile_sha256": sha256_file(question_profile_path),
            "questions": questions_path.relative_to(output_root).as_posix(),
            "questions_sha256": sha256_file(questions_path),
            "source_freeze": source_freeze_path.relative_to(output_root).as_posix(),
            "source_freeze_sha256": sha256_file(source_freeze_path),
        },
        "forbidden_annotation_fields_loaded": [],
        "generator_question_fields": list(QUESTION_RECORD_FIELDS),
    }
    _write_json(output_root / "artifact_manifest.json", artifact_manifest)
    print(json.dumps({
        "corpus_id": args.corpus_id,
        "document_count": len(documents),
        "question_profile_id": args.question_profile_id,
        "question_count": len(questions),
        "output_root": output_root.as_posix(),
        "source_unchanged": True,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
