"""Strict, corpus-agnostic contracts for Type 3 corpus and question profiles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping


CORPUS_PROFILE_SCHEMA = "type3-corpus-profile-v1"
QUESTION_PROFILE_SCHEMA = "type3-question-profile-v1"
QUESTION_RECORD_FIELDS = ("question_id", "question", "document_id")

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CORPUS_FIELDS = {
    "schema_version", "corpus_id", "source_ref", "document_count",
    "documents_sha256", "documents", "profile_sha256",
}
_DOCUMENT_FIELDS = {
    "document_id", "company", "stock_code", "report_year",
    "source_markdown", "source_sha256",
}
_QUESTION_PROFILE_FIELDS = {
    "schema_version", "question_profile_id", "corpus_id", "question_count",
    "questions_path", "questions_sha256", "question_records_sha256",
    "allowed_fields", "annotations_available_to_generator", "profile_sha256",
}


class Type3ProfileError(ValueError):
    """Raised when a Type 3 profile violates its public contract."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON with exactly one trailing newline."""

    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def profile_sha256(value: Mapping[str, Any]) -> str:
    """Hash a profile without its self-referential ``profile_sha256`` field."""

    return semantic_sha256({key: child for key, child in value.items() if key != "profile_sha256"})


def with_profile_sha256(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["profile_sha256"] = profile_sha256(result)
    return result


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Type3ProfileError(f"{path} must be an object")
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise Type3ProfileError(
            f"{path} fields differ: missing={sorted(missing)!r}, unknown={sorted(unknown)!r}"
        )


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise Type3ProfileError(f"{path} must be a non-empty trimmed string")
    if any(ord(character) < 32 for character in value):
        raise Type3ProfileError(f"{path} contains control characters")
    return value


def _identifier(value: Any, path: str) -> str:
    result = _string(value, path)
    if not _ID_RE.fullmatch(result):
        raise Type3ProfileError(f"{path} is not a stable identifier")
    return result


def _sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise Type3ProfileError(f"{path} must be a lowercase SHA256")
    return value


def _relative_path(value: Any, path: str, *, suffix: str | None = None) -> str:
    result = _string(value, path)
    pure = PurePosixPath(result)
    if pure.is_absolute() or ".." in pure.parts or result != pure.as_posix():
        raise Type3ProfileError(f"{path} must be a normalized relative POSIX path")
    if suffix is not None and pure.suffix.lower() != suffix:
        raise Type3ProfileError(f"{path} must end with {suffix}")
    return result


def validate_question_record(value: Any, *, path: str = "QuestionRecord") -> dict[str, str]:
    row = _object(value, path)
    _exact_fields(row, set(QUESTION_RECORD_FIELDS), path)
    return {
        "question_id": _identifier(row["question_id"], f"{path}.question_id"),
        "question": _string(row["question"], f"{path}.question"),
        "document_id": _string(row["document_id"], f"{path}.document_id"),
    }


def validate_corpus_profile(value: Any) -> dict[str, Any]:
    profile = _object(value, "CorpusProfile")
    _exact_fields(profile, _CORPUS_FIELDS, "CorpusProfile")
    if profile["schema_version"] != CORPUS_PROFILE_SCHEMA:
        raise Type3ProfileError("CorpusProfile.schema_version is unsupported")
    corpus_id = _identifier(profile["corpus_id"], "CorpusProfile.corpus_id")
    source_ref = _relative_path(profile["source_ref"], "CorpusProfile.source_ref")
    documents_value = profile["documents"]
    if not isinstance(documents_value, list) or not documents_value:
        raise Type3ProfileError("CorpusProfile.documents must be a non-empty array")
    documents: list[dict[str, Any]] = []
    document_ids: set[str] = set()
    source_paths: set[str] = set()
    for index, raw in enumerate(documents_value):
        path = f"CorpusProfile.documents[{index}]"
        row = _object(raw, path)
        _exact_fields(row, _DOCUMENT_FIELDS, path)
        document_id = _string(row["document_id"], f"{path}.document_id")
        source_markdown = _relative_path(
            row["source_markdown"], f"{path}.source_markdown", suffix=".md"
        )
        if document_id in document_ids or source_markdown in source_paths:
            raise Type3ProfileError(f"{path} duplicates a document_id or source_markdown")
        document_ids.add(document_id)
        source_paths.add(source_markdown)
        year = row["report_year"]
        if isinstance(year, bool) or not isinstance(year, int) or not 1900 <= year <= 2100:
            raise Type3ProfileError(f"{path}.report_year is invalid")
        stock_code = _string(row["stock_code"], f"{path}.stock_code")
        if not re.fullmatch(r"[0-9]{6}", stock_code):
            raise Type3ProfileError(f"{path}.stock_code must contain six digits")
        documents.append({
            "document_id": document_id,
            "company": _string(row["company"], f"{path}.company"),
            "stock_code": stock_code,
            "report_year": year,
            "source_markdown": source_markdown,
            "source_sha256": _sha256(row["source_sha256"], f"{path}.source_sha256"),
        })
    if documents != sorted(documents, key=lambda row: row["document_id"]):
        raise Type3ProfileError("CorpusProfile.documents must be sorted by document_id")
    count = profile["document_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count != len(documents):
        raise Type3ProfileError("CorpusProfile.document_count does not match documents")
    expected_documents_hash = semantic_sha256(documents)
    if _sha256(profile["documents_sha256"], "CorpusProfile.documents_sha256") != expected_documents_hash:
        raise Type3ProfileError("CorpusProfile.documents_sha256 does not match documents")
    expected_profile_hash = profile_sha256(profile)
    if _sha256(profile["profile_sha256"], "CorpusProfile.profile_sha256") != expected_profile_hash:
        raise Type3ProfileError("CorpusProfile.profile_sha256 does not match profile")
    return {
        "schema_version": CORPUS_PROFILE_SCHEMA,
        "corpus_id": corpus_id,
        "source_ref": source_ref,
        "document_count": count,
        "documents_sha256": expected_documents_hash,
        "documents": documents,
        "profile_sha256": expected_profile_hash,
    }


def validate_question_profile(value: Any) -> dict[str, Any]:
    profile = _object(value, "QuestionProfile")
    _exact_fields(profile, _QUESTION_PROFILE_FIELDS, "QuestionProfile")
    if profile["schema_version"] != QUESTION_PROFILE_SCHEMA:
        raise Type3ProfileError("QuestionProfile.schema_version is unsupported")
    count = profile["question_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise Type3ProfileError("QuestionProfile.question_count must be positive")
    allowed = profile["allowed_fields"]
    if allowed != list(QUESTION_RECORD_FIELDS):
        raise Type3ProfileError("QuestionProfile.allowed_fields must be the frozen safe fields")
    if profile["annotations_available_to_generator"] is not False:
        raise Type3ProfileError("QuestionProfile annotations must be unavailable to the generator")
    result = {
        "schema_version": QUESTION_PROFILE_SCHEMA,
        "question_profile_id": _identifier(
            profile["question_profile_id"], "QuestionProfile.question_profile_id"
        ),
        "corpus_id": _identifier(profile["corpus_id"], "QuestionProfile.corpus_id"),
        "question_count": count,
        "questions_path": _relative_path(
            profile["questions_path"], "QuestionProfile.questions_path", suffix=".jsonl"
        ),
        "questions_sha256": _sha256(
            profile["questions_sha256"], "QuestionProfile.questions_sha256"
        ),
        "question_records_sha256": _sha256(
            profile["question_records_sha256"], "QuestionProfile.question_records_sha256"
        ),
        "allowed_fields": list(QUESTION_RECORD_FIELDS),
        "annotations_available_to_generator": False,
        "profile_sha256": _sha256(profile["profile_sha256"], "QuestionProfile.profile_sha256"),
    }
    if result["profile_sha256"] != profile_sha256(profile):
        raise Type3ProfileError("QuestionProfile.profile_sha256 does not match profile")
    return result


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise Type3ProfileError(f"{path}:{line_number} is blank")
            rows.append(json.loads(line))
    return rows


def load_corpus_profile(path: Path) -> dict[str, Any]:
    return validate_corpus_profile(read_json(path))


def load_question_profile(
    path: Path, *, corpus_profile: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    profile = validate_question_profile(read_json(path))
    questions_path = path.parent / profile["questions_path"]
    if not questions_path.is_file():
        raise Type3ProfileError(f"questions file does not exist: {questions_path}")
    if sha256_file(questions_path) != profile["questions_sha256"]:
        raise Type3ProfileError("QuestionProfile.questions_sha256 does not match file")
    questions = [
        validate_question_record(row, path=f"QuestionRecord[{index}]")
        for index, row in enumerate(read_jsonl(questions_path))
    ]
    if len(questions) != profile["question_count"]:
        raise Type3ProfileError("QuestionProfile.question_count does not match questions")
    if len({row["question_id"] for row in questions}) != len(questions):
        raise Type3ProfileError("QuestionRecord.question_id values must be unique")
    if semantic_sha256(questions) != profile["question_records_sha256"]:
        raise Type3ProfileError("QuestionProfile.question_records_sha256 does not match questions")
    if corpus_profile is not None:
        corpus = validate_corpus_profile(corpus_profile)
        if profile["corpus_id"] != corpus["corpus_id"]:
            raise Type3ProfileError("QuestionProfile.corpus_id does not match CorpusProfile")
        document_ids = {row["document_id"] for row in corpus["documents"]}
        unknown = sorted({row["document_id"] for row in questions} - document_ids)
        if unknown:
            raise Type3ProfileError(f"questions reference unknown documents: {unknown[:5]!r}")
    return profile, questions


def source_snapshot(
    corpus_profile: Mapping[str, Any], *, workspace_root: Path
) -> dict[str, str]:
    """Hash every source file through the declared read-only corpus reference."""

    profile = validate_corpus_profile(corpus_profile)
    source_root = (workspace_root / profile["source_ref"]).resolve()
    if not source_root.is_dir():
        raise Type3ProfileError(f"source_ref does not resolve to a directory: {source_root}")
    hashes: dict[str, str] = {}
    for document in profile["documents"]:
        source = source_root / document["source_markdown"]
        # The current read-only corpus directory intentionally contains one
        # symlink per Markdown file.  Path traversal is already rejected by the
        # profile contract; the symlink target itself may live outside the link
        # directory and is frozen by its content hash.
        if not source.is_file():
            raise Type3ProfileError(f"source file is missing: {source}")
        actual = sha256_file(source)
        if actual != document["source_sha256"]:
            raise Type3ProfileError(f"source hash differs: {document['document_id']}")
        hashes[document["document_id"]] = actual
    return hashes


def question_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(validate_question_record(row)) for row in rows)
