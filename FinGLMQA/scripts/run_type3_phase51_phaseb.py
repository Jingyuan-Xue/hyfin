#!/usr/bin/env python3
"""Generate the four frozen generator-side Phase 5.1 Phase B TabGR arms."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import copy
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_type3_phase5_compact as phase5_helpers  # noqa: E402

from finglmqa.type3_a2rag_tabgr_pipeline import (  # noqa: E402
    ManifestPaths,
    bind_manifests,
)
from finglmqa.type3_corpus_profile import (  # noqa: E402
    load_corpus_profile,
    load_question_profile,
    source_snapshot,
)
from finglmqa.type3_phase51_compact_tabgr import (  # noqa: E402
    COMPOSER_VERSION,
    MASK,
    MAX_CLAIMS,
    MAX_TOTAL_CHARACTERS,
    PROFILE_VERSION,
    compose_compact_claims,
)
from finglmqa.type3_tabgr_retriever import (  # noqa: E402
    TABGR_RUNTIME_SHA256,
    TABGR_V2_BUILDER_VERSION,
    TABGR_V2_ROW_SCHEMA,
    canonical_json_bytes,
    normalize_text,
    numeric_fragments,
    semantic_sha256,
    sha256_text,
)


OUTPUT_SCHEMA = "finglmqa.type3.phase51.compact_tabgr_answer.v1"
CASE_TRACE_SCHEMA = "finglmqa.type3.phase51.phase_b_case_trace.v1"
RUN_MANIFEST_SCHEMA = "finglmqa.type3.phase51.phase_b_run_manifest.v1"

B1 = "b1_compact_tabgr_only"
B2 = "b2_v10_plus_compact_tabgr"
B3 = "b3_compact_tabgr_no_complementarity"
B4 = "b4_compact_tabgr_no_route_gate"
ARMS = (B1, B2, B3, B4)
ARM_CONFIGURATION = {
    B1: {
        "base_kind": "none",
        "route_gate_enabled": True,
        "complementarity_enabled": True,
    },
    B2: {
        "base_kind": "frozen_v10",
        "route_gate_enabled": True,
        "complementarity_enabled": True,
    },
    B3: {
        "base_kind": "frozen_v10",
        "route_gate_enabled": True,
        "complementarity_enabled": False,
    },
    B4: {
        "base_kind": "frozen_v10",
        "route_gate_enabled": False,
        "complementarity_enabled": True,
    },
}

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
DEFAULT_V10_SAFE_DIR = (
    ROOT
    / "runs/type3_a2rag_tabgr_experiment_v1/annual_reports_170_v1/"
    "phase_5_1/phase_b_inputs/v10_safe_projection_v1"
)
DEFAULT_R2_DIR = (
    ROOT
    / "runs/type3_a2rag_tabgr_experiment_v1/annual_reports_170_v1/"
    "phase_5/r2_compact_baseline_v2_frozen/fresh_process_1"
)
ANSWER_SCHEMA_PATH = (
    ROOT / "data/schemas/type3/phase51_compact_tabgr_answer_v1.schema.json"
)
SAFE_V10_SCHEMA_PATH = (
    ROOT / "data/schemas/type3/phase51_v10_safe_projection_v1.schema.json"
)

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_ROW_PROJECTION_FIELDS = (
    "schema_version",
    "builder_version",
    "record_type",
    "evidence_id",
    "corpus_id",
    "document_id",
    "table_id",
    "table_index",
    "row_index",
    "heading_path",
    "active_header_rows",
    "flattened_column_headers",
    "row_path",
    "cells",
    "source_markdown",
    "table_line_range",
    "table_sha256",
    "numeric_authorizations",
    "semantic_states",
)
_FORBIDDEN_PATH_NAMES = {
    "paired_deltas.jsonl",
    "score_details.jsonl",
    "score_report.json",
    "score_report.md",
    "ablation_summary.json",
}
_ANNOTATION_KEY_FRAGMENTS = (
    "gold",
    "oracle",
    "reference",
    "score",
    "delta",
    "loss",
    "win",
    "answerkey",
)
_ACTIVE_STAGING: Path | None = None

Phase51RunnerError = phase5_helpers.Phase5RunnerError
sha256_file = phase5_helpers.sha256_file
_write_json = phase5_helpers._write_json
_write_jsonl = phase5_helpers._write_jsonl
_relative = phase5_helpers._relative

_ANSWER_SCHEMA = json.loads(ANSWER_SCHEMA_PATH.read_text(encoding="utf-8"))
Draft202012Validator.check_schema(_ANSWER_SCHEMA)
_ANSWER_VALIDATOR = Draft202012Validator(_ANSWER_SCHEMA)
_SAFE_V10_SCHEMA = json.loads(
    SAFE_V10_SCHEMA_PATH.read_text(encoding="utf-8")
)
Draft202012Validator.check_schema(_SAFE_V10_SCHEMA)
_SAFE_V10_VALIDATOR = Draft202012Validator(_SAFE_V10_SCHEMA)


def _annotation_key_tokens(value: object) -> tuple[str, ...]:
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value))
    return tuple(
        token.lower() for token in re.findall(r"[A-Za-z0-9]+", words)
    )


def _enforce_annotation_policy(value: object, *, location: str = "$") -> None:
    """Reject annotation/scorer-derived keys at every recursive object level."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            tokens = _annotation_key_tokens(key)
            compact = "".join(tokens)
            if (
                any(
                    fragment in token
                    for token in tokens
                    for fragment in _ANNOTATION_KEY_FRAGMENTS
                    if fragment != "answerkey"
                )
                or "answerkey" in compact
            ):
                raise Phase51RunnerError(
                    f"generator value contains forbidden annotation key at "
                    f"{location}.{key}"
                )
            _enforce_annotation_policy(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _enforce_annotation_policy(child, location=f"{location}[{index}]")


def _reject_forbidden_fields(value: object, *, location: str = "$") -> None:
    _enforce_annotation_policy(value, location=location)


def guard_generator_path(path: Path) -> Path:
    """Resolve symlinks and reject Phase 5/5.1 scorer and annotation paths."""

    resolved = phase5_helpers.guard_generator_path(path)
    parts = [part.lower() for part in resolved.parts]
    lowered = resolved.as_posix().lower()
    for phase in ("phase_5", "phase_5_1"):
        if phase in parts:
            index = parts.index(phase)
            if index + 1 < len(parts) and parts[index + 1] == "evaluation":
                raise Phase51RunnerError(
                    "Phase 5.1 generator cannot read Phase 5/5.1 evaluation"
                )
    if any(part in {"gold", "golden", "oracle"} for part in parts):
        raise Phase51RunnerError("Phase 5.1 generator cannot read gold/oracle")
    if resolved.name.lower() in _FORBIDDEN_PATH_NAMES:
        raise Phase51RunnerError("Phase 5.1 generator cannot read score/delta artifacts")
    if "/big-finbenchmark/data/by_type/" in lowered:
        raise Phase51RunnerError("Phase 5.1 generator cannot read benchmark gold")
    return resolved


def _guard_manifest_paths(paths: ManifestPaths) -> ManifestPaths:
    """Resolve and guard every bound path before any binder/loader can open it."""

    return ManifestPaths(
        corpus_manifest=guard_generator_path(paths.corpus_manifest),
        question_profile=guard_generator_path(paths.question_profile),
        questions=guard_generator_path(paths.questions),
        a2rag_package_manifest=guard_generator_path(
            paths.a2rag_package_manifest
        ),
        a2rag_index_manifest=guard_generator_path(paths.a2rag_index_manifest),
        text_atoms=guard_generator_path(paths.text_atoms),
        tabgr_package_manifest=guard_generator_path(
            paths.tabgr_package_manifest
        ),
        tabgr_index_manifest=guard_generator_path(paths.tabgr_index_manifest),
        fact_manifest=guard_generator_path(paths.fact_manifest),
    )


def load_json(path: Path) -> dict[str, Any]:
    guarded = guard_generator_path(path)
    value = json.loads(guarded.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase51RunnerError(f"expected JSON object: {guarded}")
    _reject_forbidden_fields(value, location=str(guarded))
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    guarded = guard_generator_path(path)
    rows: list[dict[str, Any]] = []
    with guarded.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise Phase51RunnerError(
                    f"blank JSONL record: {guarded}:{line_number}"
                )
            value = json.loads(line)
            if not isinstance(value, dict):
                raise Phase51RunnerError(
                    f"expected JSON object: {guarded}:{line_number}"
                )
            rows.append(value)
    _reject_forbidden_fields(rows, location=str(guarded))
    return rows


def _safe_output_dir(
    output_dir: Path,
    *,
    protected_paths: Sequence[Path],
) -> tuple[Path, Path]:
    resolved = output_dir.resolve()
    guard_parts = [part.lower() for part in resolved.parts]
    if any(part.startswith("scoring") for part in guard_parts):
        raise Phase51RunnerError("generator output cannot be a scoring namespace")
    for phase in ("phase_5", "phase_5_1"):
        if phase in guard_parts:
            index = guard_parts.index(phase)
            if index + 1 < len(guard_parts) and guard_parts[index + 1] == "evaluation":
                raise Phase51RunnerError("generator output cannot be an evaluation namespace")
    for path in protected_paths:
        protected = path.resolve()
        if (
            resolved == protected
            or resolved.is_relative_to(protected)
            or protected.is_relative_to(resolved)
        ):
            raise Phase51RunnerError(
                f"output overlaps frozen input: {resolved} vs {protected}"
            )
    if resolved.exists():
        raise Phase51RunnerError(f"output namespace must not exist: {resolved}")
    staging = resolved.parent / f".{resolved.name}.staging"
    if staging.exists():
        raise Phase51RunnerError(f"stale staging namespace exists: {staging}")
    return resolved, staging


def _safe_relative_path(value: object) -> str:
    text = str(value or "")
    pure = PurePosixPath(text)
    if (
        not text
        or pure.is_absolute()
        or ".." in pure.parts
        or pure.as_posix() != text
    ):
        raise Phase51RunnerError("TabGR shard path is not a safe relative path")
    return text


class BoundTabGRStore:
    """Manifest-bound TabGR document loader with a small verified LRU cache."""

    def __init__(
        self,
        index_dir: Path,
        *,
        corpus_id: str,
        maximum_cached_documents: int = 4,
    ) -> None:
        self.root = guard_generator_path(index_dir).resolve()
        self.corpus_id = corpus_id
        self.maximum_cached_documents = maximum_cached_documents
        self.manifest_path = guard_generator_path(self.root / "manifest.json")
        self.document_manifest_path = guard_generator_path(
            self.root / "document_manifest.jsonl"
        )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        _enforce_annotation_policy(manifest, location="tabgr_index_manifest")
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version")
            != "finglmqa.type3.tabgr.lexical_index_manifest.v2"
            or manifest.get("builder_version") != TABGR_V2_BUILDER_VERSION
            or manifest.get("tabgr_runtime_sha256") != TABGR_RUNTIME_SHA256
            or manifest.get("corpus_id") != corpus_id
            or manifest.get("document_prefilter_required") is not True
            or manifest.get("online_source_table_reparse_allowed") is not False
        ):
            raise Phase51RunnerError("TabGR index manifest contract differs")
        if manifest.get("document_manifest_sha256") != sha256_file(
            self.document_manifest_path
        ):
            raise Phase51RunnerError("TabGR document manifest hash differs")
        entries: dict[str, dict[str, Any]] = {}
        with self.document_manifest_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise Phase51RunnerError(
                        f"blank TabGR document manifest row: {line_number}"
                    )
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    raise Phase51RunnerError("TabGR document manifest row differs")
                _enforce_annotation_policy(
                    entry,
                    location=f"tabgr_document_manifest[{line_number}]",
                )
                document_id = str(entry.get("document_id") or "")
                if not document_id or document_id in entries:
                    raise Phase51RunnerError("TabGR document manifest ids differ")
                if (
                    entry.get("schema_version")
                    != "finglmqa.type3.tabgr.document_shard.v2"
                ):
                    raise Phase51RunnerError("TabGR shard manifest schema differs")
                relative = _safe_relative_path(entry.get("shard_path"))
                shard_path = guard_generator_path(self.root / relative)
                if not shard_path.is_relative_to(self.root):
                    raise Phase51RunnerError("TabGR shard escapes index directory")
                expected_hash = str(entry.get("shard_sha256") or "")
                if not _HEX64_RE.fullmatch(expected_hash):
                    raise Phase51RunnerError("TabGR shard hash is invalid")
                entries[document_id] = {
                    **entry,
                    "relative_path": relative,
                    "resolved_path": shard_path,
                }
        if len(entries) != int(manifest.get("document_count", -1)):
            raise Phase51RunnerError("TabGR document manifest count differs")
        self.entries = entries
        self._cache: OrderedDict[str, dict[str, dict[str, Any]]] = OrderedDict()
        self.used_shards: dict[str, dict[str, str]] = {}

    def _load_document(self, document_id: str) -> dict[str, dict[str, Any]]:
        cached = self._cache.get(document_id)
        if cached is not None:
            self._cache.move_to_end(document_id)
            return cached
        entry = self.entries.get(document_id)
        if entry is None:
            raise Phase51RunnerError("document_id is outside frozen TabGR index")
        digest = hashlib.sha256()
        record_count = 0
        rows: dict[str, dict[str, Any]] = {}
        with Path(entry["resolved_path"]).open("rb") as handle:
            for raw_line in handle:
                digest.update(raw_line)
                if not raw_line.strip():
                    raise Phase51RunnerError("blank record in TabGR shard")
                record = json.loads(raw_line)
                if not isinstance(record, dict):
                    raise Phase51RunnerError("TabGR shard row differs")
                _enforce_annotation_policy(
                    record,
                    location=f"tabgr_shard[{document_id}][{record_count}]",
                )
                record_count += 1
                if (
                    record.get("corpus_id") != self.corpus_id
                    or record.get("document_id") != document_id
                ):
                    raise Phase51RunnerError(
                        "cross-corpus/document row in TabGR shard"
                    )
                evidence_id = str(record.get("evidence_id") or "")
                if evidence_id in rows:
                    raise Phase51RunnerError("duplicate evidence_id in TabGR shard")
                if record.get("record_type") == "table_row":
                    rows[evidence_id] = {
                        key: record.get(key) for key in _ROW_PROJECTION_FIELDS
                    }
        actual_hash = digest.hexdigest()
        if (
            actual_hash != entry["shard_sha256"]
            or record_count != int(entry.get("record_count", -1))
        ):
            raise Phase51RunnerError("TabGR shard hash/count differs")
        self.used_shards[document_id] = {
            "path": Path(entry["resolved_path"]).as_posix(),
            "relative_path": str(entry["relative_path"]),
            "sha256": actual_hash,
        }
        self._cache[document_id] = rows
        self._cache.move_to_end(document_id)
        while len(self._cache) > self.maximum_cached_documents:
            self._cache.popitem(last=False)
        return rows

    def hydrate(
        self,
        *,
        candidate_id: str,
        document_id: str,
    ) -> dict[str, Any]:
        rows = self._load_document(document_id)
        row = rows.get(candidate_id)
        if row is None:
            raise Phase51RunnerError(
                "Phase 4 candidate_id did not hydrate in its bound document shard"
            )
        if (
            row.get("schema_version") != TABGR_V2_ROW_SCHEMA
            or row.get("builder_version") != TABGR_V2_BUILDER_VERSION
            or row.get("record_type") != "table_row"
            or row.get("evidence_id") != candidate_id
        ):
            raise Phase51RunnerError("candidate_id does not identify a rich row")
        projection_hash = semantic_sha256(row)
        entry = self.entries[document_id]
        return {
            "row": copy.deepcopy(row),
            "hydration": {
                "candidate_id": candidate_id,
                "document_shard_path": str(entry["relative_path"]),
                "document_shard_sha256": str(entry["shard_sha256"]),
                "rich_row_projection_sha256": projection_hash,
            },
        }

    def verify_used_shards_unchanged(self) -> None:
        for value in self.used_shards.values():
            if sha256_file(Path(value["path"])) != value["sha256"]:
                raise Phase51RunnerError(
                    "frozen TabGR shard changed during generation"
                )


def _verify_r2(
    directory: Path,
    *,
    corpus_id: str,
    question_profile_id: str,
    questions: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    root = guard_generator_path(directory)
    manifest_path = guard_generator_path(root / "run_manifest.json")
    answers_path = guard_generator_path(root / "answers.jsonl")
    traces_path = guard_generator_path(root / "semantic_traces.jsonl")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version")
        != "finglmqa.type3.phase5.compact_run_manifest.v2"
        or manifest.get("profile_version")
        != "type3-a2rag-tabgr-compact-baseline-v2"
        or manifest.get("corpus_id") != corpus_id
        or manifest.get("question_profile_id") != question_profile_id
        or manifest.get("question_count") != len(questions)
        or manifest.get("question_ids_sha256")
        != semantic_sha256([row["question_id"] for row in questions])
    ):
        raise Phase51RunnerError("frozen R2 manifest identity differs")
    artifacts = manifest.get("artifacts") or {}
    if (
        artifacts.get("answers.jsonl") != sha256_file(answers_path)
        or artifacts.get("semantic_traces.jsonl") != sha256_file(traces_path)
    ):
        raise Phase51RunnerError("frozen R2 artifact hash differs")
    unsigned = {
        key: value for key, value in manifest.items() if key != "run_fingerprint"
    }
    if manifest.get("run_fingerprint") != semantic_sha256(unsigned):
        raise Phase51RunnerError("frozen R2 run fingerprint differs")
    rows = load_jsonl(answers_path)
    by_id = {str(row.get("question_id") or ""): row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != {
        row["question_id"] for row in questions
    }:
        raise Phase51RunnerError("frozen R2 case identity differs")
    for question in questions:
        row = by_id[question["question_id"]]
        if (
            row.get("question") != question["question"]
            or row.get("document_id") != question["document_id"]
            or not str(row.get("answer_safe_text") or "").strip()
        ):
            raise Phase51RunnerError("frozen R2 answer identity differs")
    _enforce_annotation_policy(by_id, location="b0_r2_answers")
    return {
        "directory": root,
        "manifest_path": manifest_path,
        "answers_path": answers_path,
        "traces_path": traces_path,
        "manifest": manifest,
        "answers": by_id,
    }


def _verify_phase4(
    *,
    answers_path: Path,
    manifest_path: Path,
    binding: Mapping[str, Any],
    corpus_id: str,
    question_profile_id: str,
    questions: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    rows = phase5_helpers._verify_phase4(
        answers_path=guard_generator_path(answers_path),
        manifest_path=guard_generator_path(manifest_path),
        expected_binding=binding,
        corpus_id=corpus_id,
        question_profile_id=question_profile_id,
        questions=questions,
    )
    # `fusion_score` is deterministic Phase 4 retrieval metadata that this
    # generator does not consume. Drop it before applying the centralized
    # no-score policy to the exact Phase 5.1 semantic input projection.
    projected = copy.deepcopy(rows)
    for packet in projected.values():
        for evidence in packet.get("evidence") or ():
            if isinstance(evidence, dict):
                evidence.pop("fusion_score", None)
    _enforce_annotation_policy(projected, location="phase4_projection")
    return projected


def _validate_safe_v10_value(
    value: Mapping[str, Any],
    *,
    label: str,
) -> None:
    errors = sorted(
        _SAFE_V10_VALIDATOR.iter_errors(value),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if errors:
        first = errors[0]
        location = "$" + "".join(
            f"[{part!r}]" for part in first.absolute_path
        )
        raise Phase51RunnerError(
            f"{label} failed closed safe-v10 schema at "
            f"{location}: {first.message}"
        )


def _verify_and_load_safe_v10(
    *,
    directory: Path,
    corpus_id: str,
    question_profile_id: str,
    question_profile_sha256: str,
    questions: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    root = guard_generator_path(directory)
    manifest_path = guard_generator_path(root / "manifest.json")
    answers_path = guard_generator_path(root / "answers.jsonl")
    manifest = load_json(manifest_path)
    _validate_safe_v10_value(manifest, label="safe v10 manifest")
    unsigned = {
        key: value
        for key, value in manifest.items()
        if key != "run_fingerprint"
    }
    question_ids = [row["question_id"] for row in questions]
    question_records = [
        {
            "question_id": row["question_id"],
            "question": row["question"],
            "document_id": row["document_id"],
        }
        for row in questions
    ]
    expected_order_hash = semantic_sha256(
        [
            {"ordinal": index, "question_id": question_id}
            for index, question_id in enumerate(question_ids)
        ]
    )
    artifact = manifest.get("artifacts", {}).get("answers.jsonl", {})
    question_contract = manifest.get("questions", {})
    if (
        manifest.get("schema_version")
        != "finglmqa.type3.phase51.v10_safe_projection_manifest.v1"
        or manifest.get("profile_version")
        != "type3-phase51-v10-safe-projection-v1"
        or manifest.get("corpus_id") != corpus_id
        or manifest.get("question_profile_id") != question_profile_id
        or manifest.get("question_profile_sha256")
        != question_profile_sha256
        or manifest.get("run_fingerprint") != semantic_sha256(unsigned)
        or question_contract.get("count") != len(questions)
        or question_contract.get("question_ids_sha256")
        != semantic_sha256(question_ids)
        or question_contract.get("question_records_sha256")
        != semantic_sha256(question_records)
        or question_contract.get("question_order_sha256")
        != expected_order_hash
        or artifact.get("path") != "answers.jsonl"
        or artifact.get("row_count") != len(questions)
        or artifact.get("sha256") != sha256_file(answers_path)
    ):
        raise Phase51RunnerError("safe v10 projection manifest differs")

    raw_rows = load_jsonl(answers_path)
    if len(raw_rows) != len(questions):
        raise Phase51RunnerError("safe v10 projection count differs")
    rows: dict[str, dict[str, Any]] = {}
    for index, (row, question) in enumerate(zip(raw_rows, questions)):
        _validate_safe_v10_value(
            row,
            label=f"safe v10 answer[{index}]",
        )
        question_id = str(row.get("question_id") or "")
        citations = row.get("citations")
        if (
            question_id != question["question_id"]
            or question_id in rows
            or row.get("question") != question["question"]
            or row.get("document_id") != question["document_id"]
            or not str(row.get("answer_safe_text") or "").strip()
            or not isinstance(citations, list)
            or any(
                citation.get("corpus_id") != corpus_id
                or citation.get("document_id") != question["document_id"]
                for citation in citations
            )
        ):
            raise Phase51RunnerError(
                "safe v10 answer identity/provenance differs"
            )
        rows[question_id] = {
            "question": question["question"],
            "document_id": question["document_id"],
            "answer": str(row["answer_safe_text"]),
            "citations": copy.deepcopy(citations),
        }
    _enforce_annotation_policy(rows, location="safe_v10_projection")
    return rows


def _document_map(corpus: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    documents = corpus.get("documents")
    if not isinstance(documents, list):
        raise Phase51RunnerError("corpus manifest documents differ")
    return {str(row["document_id"]): row for row in documents}


def _selected_table_evidence(
    packet: Mapping[str, Any],
    *,
    corpus_id: str,
    document: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    evidence = packet.get("evidence")
    if not isinstance(evidence, list):
        raise Phase51RunnerError("Phase 4 packet evidence differs")
    selected: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for item in evidence:
        if not isinstance(item, Mapping):
            raise Phase51RunnerError("Phase 4 evidence row differs")
        if item.get("route") != "table":
            continue
        if item.get("evidence_type") != "table_row":
            raise Phase51RunnerError("selected Phase 4 table evidence is not a row")
        candidate_id = str(item.get("candidate_id") or "")
        if not candidate_id or candidate_id in seen:
            continue
        seen.add(candidate_id)
        if (
            item.get("corpus_id") != corpus_id
            or item.get("document_id") != document["document_id"]
            or item.get("source_sha256") != document["source_sha256"]
            or Path(str(item.get("source_markdown") or "")).name
            != document["source_markdown"]
        ):
            raise Phase51RunnerError(
                "Phase 4 table evidence source/corpus/document binding differs"
            )
        selected.append(item)
    return selected


def _strip_join_fields(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_join_fields(child)
            for key, child in value.items()
            if str(key).lower() not in {"question_id", "case_id"}
        }
    if isinstance(value, list):
        return [_strip_join_fields(child) for child in value]
    return value


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _compact_citation(claim: Mapping[str, Any]) -> dict[str, Any]:
    cells = claim["selected_cells"]
    return {
        "citation_kind": "compact_tabgr_cell_fact",
        "claim_kind": claim["claim_kind"],
        "candidate_id": claim["candidate_id"],
        "corpus_id": claim["corpus_id"],
        "document_id": claim["document_id"],
        "source_markdown": claim["source_markdown"],
        "source_sha256": claim["source_sha256"],
        "table_id": claim["table_id"],
        "table_index": claim["table_index"],
        "table_sha256": claim["table_sha256"],
        "table_line_range": claim["table_line_range"],
        "heading_path": claim["heading_path"],
        "row_index": claim["row_index"],
        "row_path": claim["row_path"],
        "row_label_cell": claim["row_label_cell"],
        "cell_coordinates": [cell["coordinate"] for cell in cells],
        "origin_coordinates": [cell["origin_coordinate"] for cell in cells],
        "origin_cell_hashes": [cell["origin_cell_hash"] for cell in cells],
        "authorization_ids": [
            cell["numeric_authorization"]["authorization_id"] for cell in cells
        ],
        "hydrated_projection_sha256": claim["hydrated_projection_sha256"],
        "claim_sha256": claim["claim_sha256"],
    }


def _compose_arm_case(
    *,
    arm: str,
    corpus_id: str,
    question: Mapping[str, str],
    document: Mapping[str, Any],
    phase4_packet: Mapping[str, Any],
    hydrated_candidates: Sequence[Mapping[str, Any]],
    v10: Mapping[str, Any],
) -> dict[str, Any]:
    configuration = ARM_CONFIGURATION[arm]
    base_kind = str(configuration["base_kind"])
    base_answer = "" if base_kind == "none" else str(v10["answer"])
    base_citations = [] if base_kind == "none" else list(v10["citations"])
    composed = compose_compact_claims(
        question=question["question"],
        corpus_id=corpus_id,
        document_id=question["document_id"],
        candidates=hydrated_candidates,
        base_answer=base_answer,
        enable_route_gate=bool(configuration["route_gate_enabled"]),
        enable_complementarity=bool(configuration["complementarity_enabled"]),
    )
    append_text = str(composed["append_text"])
    answer = (
        append_text
        if not base_answer
        else base_answer + (f"\n{append_text}" if append_text else "")
    )
    if arm != B1 and not answer.strip():
        raise Phase51RunnerError(f"{arm} produced an empty primary answer")
    claims = list(composed["claims"])
    selected_candidate_ids = _unique(
        [str(claim["candidate_id"]) for claim in claims]
    )
    selected_claim_sha256 = [str(claim["claim_sha256"]) for claim in claims]
    composer_trace = copy.deepcopy(composed["semantic_trace"])
    composer_trace["selected_candidate_ids"] = selected_candidate_ids
    composer_unsigned = {
        key: value
        for key, value in composer_trace.items()
        if key != "semantic_trace_sha256"
    }
    composer_trace["semantic_trace_sha256"] = semantic_sha256(composer_unsigned)
    hydration_trace = [
        copy.deepcopy(packet["hydration"]) for packet in hydrated_candidates
    ]
    trace_unsigned = {
        "schema_version": CASE_TRACE_SCHEMA,
        "profile_version": PROFILE_VERSION,
        "arm": arm,
        "semantic_input": {
            "corpus_id": corpus_id,
            "document_id": question["document_id"],
            "question": question["question"],
        },
        "source_phase4_packet_sha256": semantic_sha256(
            _strip_join_fields(phase4_packet)
        ),
        "base_kind": base_kind,
        "base_answer_sha256": sha256_text(base_answer),
        "hydrated_candidates": hydration_trace,
        "composer": composer_trace,
        "selected_candidate_ids": selected_candidate_ids,
        "selected_claim_sha256": selected_claim_sha256,
    }
    trace = {
        **trace_unsigned,
        "semantic_trace_sha256": semantic_sha256(trace_unsigned),
    }
    output = {
        "schema_version": OUTPUT_SCHEMA,
        "profile_version": PROFILE_VERSION,
        "arm": arm,
        "corpus_id": corpus_id,
        "question_id": question["question_id"],
        "question": question["question"],
        "document_id": question["document_id"],
        "base_kind": base_kind,
        "base_answer_sha256": sha256_text(base_answer),
        "answer_safe_text": answer,
        "claims": claims,
        "citations": [
            *copy.deepcopy(base_citations),
            *[_compact_citation(claim) for claim in claims],
        ],
        "selected_candidate_ids": selected_candidate_ids,
        "selected_claim_sha256": selected_claim_sha256,
        "semantic_trace": trace,
    }
    _validate_output(
        output,
        corpus_id=corpus_id,
        question=question,
        document=document,
        phase4_packet=phase4_packet,
        base_answer=base_answer,
        base_citations=base_citations,
        hydrated_candidates=hydrated_candidates,
        arm_configuration=configuration,
    )
    return output


def _validate_claim_contract(
    claim: Mapping[str, Any],
    *,
    corpus_id: str,
    question: Mapping[str, str],
    document: Mapping[str, Any],
    hydration_by_candidate: Mapping[str, Mapping[str, Any]],
    row_by_candidate: Mapping[str, Mapping[str, Any]],
) -> None:
    candidate_id = str(claim.get("candidate_id") or "")
    selected_cells = claim.get("selected_cells")
    row_label_cell = claim.get("row_label_cell")
    if (
        not candidate_id
        or not isinstance(selected_cells, list)
        or not 1 <= len(selected_cells) <= MAX_CLAIMS
        or not isinstance(row_label_cell, Mapping)
    ):
        raise Phase51RunnerError("Phase B claim structure differs")
    if (
        claim.get("corpus_id") != corpus_id
        or claim.get("document_id") != question["document_id"]
        or claim.get("source_sha256") != document["source_sha256"]
        or Path(str(claim.get("source_markdown") or "")).name
        != document["source_markdown"]
        or not str(claim.get("table_id") or "")
        or not _HEX64_RE.fullmatch(str(claim.get("table_sha256") or ""))
        or not isinstance(claim.get("row_index"), int)
        or not isinstance(claim.get("table_index"), int)
    ):
        raise Phase51RunnerError("Phase B claim source/table identity differs")
    row_index = int(claim["row_index"])
    row_path = claim.get("row_path")
    if (
        not isinstance(row_path, list)
        or not row_path
        or normalize_text(row_path[-1])
        != normalize_text(row_label_cell.get("raw_value"))
    ):
        raise Phase51RunnerError("Phase B row label binding differs")
    label_coordinate = row_label_cell.get("coordinate")
    label_origin = row_label_cell.get("origin_coordinate")
    label_raw = str(row_label_cell.get("raw_value") or "")
    if (
        not isinstance(label_coordinate, list)
        or len(label_coordinate) != 2
        or label_coordinate[0] != row_index
        or not isinstance(label_origin, list)
        or len(label_origin) != 2
        or label_origin[0] > label_coordinate[0]
        or label_origin[1] > label_coordinate[1]
        or row_label_cell.get("raw_value_sha256") != sha256_text(label_raw)
        or not _HEX64_RE.fullmatch(
            str(row_label_cell.get("origin_cell_hash") or "")
        )
        or numeric_fragments(label_raw)
        or MASK in label_raw
    ):
        raise Phase51RunnerError("Phase B row-label cell provenance differs")

    hydration = hydration_by_candidate.get(candidate_id)
    row = row_by_candidate.get(candidate_id)
    if (
        hydration is None
        or row is None
        or hydration.get("candidate_id") != candidate_id
        or hydration.get("rich_row_projection_sha256")
        != claim.get("hydrated_projection_sha256")
        or semantic_sha256(
            {key: row.get(key) for key in _ROW_PROJECTION_FIELDS}
        )
        != claim.get("hydrated_projection_sha256")
    ):
        raise Phase51RunnerError("Phase B claim hydration binding differs")
    exact_row_fields = {
        "corpus_id": corpus_id,
        "document_id": question["document_id"],
        "source_markdown": row.get("source_markdown"),
        "table_id": row.get("table_id"),
        "table_index": row.get("table_index"),
        "table_sha256": row.get("table_sha256"),
        "table_line_range": row.get("table_line_range"),
        "heading_path": row.get("heading_path"),
        "row_index": row.get("row_index"),
        "row_path": row.get("row_path"),
    }
    for key, expected in exact_row_fields.items():
        if claim.get(key) != expected:
            raise Phase51RunnerError(
                f"Phase B claim hydrated row binding differs at {key}"
            )
    row_cells = row.get("cells")
    row_authorizations = row.get("numeric_authorizations")
    if not isinstance(row_cells, list) or not isinstance(row_authorizations, list):
        raise Phase51RunnerError("Phase B hydrated row cell/auth shape differs")
    cells_by_coordinate = {
        tuple(cell.get("coordinate") or ()): cell
        for cell in row_cells
        if isinstance(cell, Mapping)
    }
    authorization_by_id = {
        str(value.get("authorization_id") or ""): value
        for value in row_authorizations
        if isinstance(value, Mapping)
    }
    hydrated_label = cells_by_coordinate.get(tuple(label_coordinate))
    expected_label = (
        {
            "coordinate": hydrated_label.get("coordinate"),
            "origin_coordinate": hydrated_label.get("origin_coordinate"),
            "origin_cell_hash": hydrated_label.get("origin_cell_hash"),
            "raw_value": hydrated_label.get("raw_value"),
            "raw_value_sha256": hydrated_label.get("raw_value_sha256"),
        }
        if isinstance(hydrated_label, Mapping)
        else None
    )
    if (
        expected_label is None
        or hydrated_label.get("numeric_status") != "not_numeric"
        or dict(row_label_cell) != expected_label
    ):
        raise Phase51RunnerError("Phase B hydrated row-label cell differs")

    allowed_renderings: set[str] = set()
    seen_coordinates: set[tuple[int, int]] = set()
    for cell in selected_cells:
        if not isinstance(cell, Mapping):
            raise Phase51RunnerError("Phase B selected cell differs")
        coordinate = cell.get("coordinate")
        origin_coordinate = cell.get("origin_coordinate")
        if (
            not isinstance(coordinate, list)
            or len(coordinate) != 2
            or coordinate[0] != row_index
            or not all(isinstance(value, int) and value >= 0 for value in coordinate)
            or not isinstance(origin_coordinate, list)
            or len(origin_coordinate) != 2
            or not all(
                isinstance(value, int) and value >= 0
                for value in origin_coordinate
            )
            or origin_coordinate[0] > coordinate[0]
            or origin_coordinate[1] > coordinate[1]
        ):
            raise Phase51RunnerError("Phase B selected cell coordinates differ")
        coordinate_key = (coordinate[0], coordinate[1])
        if coordinate_key in seen_coordinates:
            raise Phase51RunnerError("Phase B selected cell is duplicated")
        seen_coordinates.add(coordinate_key)
        raw_value = str(cell.get("raw_value") or "")
        authorization = cell.get("numeric_authorization")
        hydrated_cell = cells_by_coordinate.get(coordinate_key)
        if (
            not isinstance(authorization, Mapping)
            or not isinstance(hydrated_cell, Mapping)
            or cell.get("raw_value_sha256") != sha256_text(raw_value)
            or not _HEX64_RE.fullmatch(str(cell.get("origin_cell_hash") or ""))
        ):
            raise Phase51RunnerError("Phase B selected cell value/hash differs")
        authorization_id = str(authorization.get("authorization_id") or "")
        hydrated_authorization = authorization_by_id.get(authorization_id)
        column = coordinate[1]
        flattened_headers = row.get("flattened_column_headers")
        semantic_states = row.get("semantic_states")
        if (
            not isinstance(flattened_headers, list)
            or column >= len(flattened_headers)
            or not isinstance(semantic_states, Mapping)
        ):
            raise Phase51RunnerError(
                "Phase B hydrated header/semantic-state shape differs"
            )
        state_values: dict[str, str | None] = {}
        for state_field, output_field in (
            ("unit_by_column", "unit"),
            ("period_by_column", "period"),
            ("accounting_scope_by_column", "accounting_scope"),
        ):
            by_column = semantic_states.get(state_field)
            hydrated_state = hydrated_cell.get(
                {
                    "unit_by_column": "unit",
                    "period_by_column": "period",
                    "accounting_scope_by_column": "accounting_scope",
                }[state_field]
            )
            if (
                not isinstance(by_column, Mapping)
                or by_column.get(str(column)) != hydrated_state
                or not isinstance(hydrated_state, Mapping)
            ):
                raise Phase51RunnerError(
                    f"Phase B hydrated {state_field} binding differs"
                )
            status = hydrated_state.get("status")
            value = hydrated_state.get("value")
            if status == "resolved" and isinstance(value, str) and value:
                state_values[output_field] = value
            elif status == "unknown" and output_field == "accounting_scope":
                state_values[output_field] = None
            else:
                raise Phase51RunnerError(
                    f"Phase B hydrated {state_field} is not renderable"
                )
        expected_cell_provenance = {
            "coordinate": hydrated_cell.get("coordinate"),
            "origin_coordinate": hydrated_cell.get("origin_coordinate"),
            "origin_cell_hash": hydrated_cell.get("origin_cell_hash"),
            "raw_value": hydrated_cell.get("raw_value"),
            "raw_value_sha256": hydrated_cell.get("raw_value_sha256"),
            "column_header": normalize_text(flattened_headers[column]),
            **state_values,
        }
        if (
            hydrated_cell.get("numeric_status") != "authorized"
            or normalize_text(hydrated_cell.get("column_header"))
            != normalize_text(flattened_headers[column])
            or any(
                cell.get(key) != expected
                for key, expected in expected_cell_provenance.items()
            )
            or not isinstance(hydrated_authorization, Mapping)
            or dict(authorization) != dict(hydrated_authorization)
            or hydrated_cell.get("authorization_ids") != [authorization_id]
        ):
            raise Phase51RunnerError(
                "Phase B selected cell/hydrated authorization differs"
            )
        exact_authorization = {
            "schema_version": "finglmqa.type3.tabgr.numeric_authorization.v1",
            "corpus_id": corpus_id,
            "document_id": question["document_id"],
            "table_id": claim["table_id"],
            "table_sha256": claim["table_sha256"],
            "source_markdown": claim["source_markdown"],
            "table_line_range": claim["table_line_range"],
            "cell_coordinate": coordinate,
            "raw_value": raw_value,
            "raw_value_sha256": cell["raw_value_sha256"],
        }
        for key, expected in exact_authorization.items():
            if authorization.get(key) != expected:
                raise Phase51RunnerError(
                    f"Phase B authorization binding differs at {key}"
                )
        if (
            authorization.get("allowed_renderings") != [raw_value]
            or str(authorization.get("metric_year")) != str(cell.get("period"))
            or authorization.get("normalized_unit") != cell.get("unit")
            or not authorization_id
        ):
            raise Phase51RunnerError(
                "Phase B authorization semantic binding differs"
            )
        allowed_renderings.update(authorization["allowed_renderings"])

    claim_numeric_fragments = set(numeric_fragments(str(claim.get("text") or "")))
    if claim_numeric_fragments != allowed_renderings:
        raise Phase51RunnerError(
            "Phase B claim numeric fragments differ from allowed renderings"
        )
    expected_kind = (
        "compact_tabgr_comparison"
        if len(selected_cells) == 2
        else "compact_tabgr_single_value"
    )
    if claim.get("claim_kind") != expected_kind:
        raise Phase51RunnerError("Phase B claim kind/cell count differs")
    expected_claim_hash = semantic_sha256(
        {
            "text": claim["text"],
            "candidate_id": candidate_id,
            "row_label_cell": dict(row_label_cell),
            "selected_cells": selected_cells,
        }
    )
    if claim.get("claim_sha256") != expected_claim_hash:
        raise Phase51RunnerError("Phase B claim hash differs")


def _validate_output(
    row: Mapping[str, Any],
    *,
    corpus_id: str,
    question: Mapping[str, str],
    document: Mapping[str, Any],
    phase4_packet: Mapping[str, Any],
    base_answer: str,
    base_citations: Sequence[Mapping[str, Any]],
    hydrated_candidates: Sequence[Mapping[str, Any]],
    arm_configuration: Mapping[str, Any],
) -> None:
    errors = sorted(
        _ANSWER_VALIDATOR.iter_errors(row),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if errors:
        first = errors[0]
        location = "$" + "".join(
            f"[{part!r}]" for part in first.absolute_path
        )
        raise Phase51RunnerError(
            f"Phase B output schema failed at {location}: {first.message}"
        )
    arm = str(row["arm"])
    expected_configuration = ARM_CONFIGURATION.get(arm)
    if (
        expected_configuration is None
        or dict(arm_configuration) != expected_configuration
        or set(arm_configuration)
        != {
            "base_kind",
            "route_gate_enabled",
            "complementarity_enabled",
        }
    ):
        raise Phase51RunnerError("Phase B authoritative arm configuration differs")
    expected_base_kind = str(arm_configuration["base_kind"])
    if (
        (expected_base_kind == "none" and base_answer != "")
        or (expected_base_kind == "frozen_v10" and not base_answer)
    ):
        raise Phase51RunnerError("Phase B authoritative base answer/config differs")
    if (
        row.get("schema_version") != OUTPUT_SCHEMA
        or row.get("profile_version") != PROFILE_VERSION
        or row.get("corpus_id") != corpus_id
        or row.get("question_id") != question["question_id"]
        or row.get("question") != question["question"]
        or row.get("document_id") != question["document_id"]
        or row.get("base_kind") != expected_base_kind
        or row.get("base_answer_sha256") != sha256_text(base_answer)
    ):
        raise Phase51RunnerError("Phase B output identity differs")
    if (
        phase4_packet.get("corpus_id") != corpus_id
        or phase4_packet.get("document_id") != question["document_id"]
        or phase4_packet.get("question") != question["question"]
    ):
        raise Phase51RunnerError("authoritative Phase 4 packet identity differs")

    expected_composed = compose_compact_claims(
        question=question["question"],
        corpus_id=corpus_id,
        document_id=question["document_id"],
        candidates=hydrated_candidates,
        base_answer=base_answer,
        enable_route_gate=bool(arm_configuration["route_gate_enabled"]),
        enable_complementarity=bool(
            arm_configuration["complementarity_enabled"]
        ),
    )
    expected_claims = copy.deepcopy(list(expected_composed["claims"]))
    expected_selected_ids = _unique(
        [str(claim["candidate_id"]) for claim in expected_claims]
    )
    expected_claim_hashes = [
        str(claim["claim_sha256"]) for claim in expected_claims
    ]
    expected_composer = copy.deepcopy(expected_composed["semantic_trace"])
    expected_composer["selected_candidate_ids"] = expected_selected_ids
    expected_composer_unsigned = {
        key: value
        for key, value in expected_composer.items()
        if key != "semantic_trace_sha256"
    }
    expected_composer["semantic_trace_sha256"] = semantic_sha256(
        expected_composer_unsigned
    )
    expected_hydration = [
        copy.deepcopy(packet["hydration"]) for packet in hydrated_candidates
    ]
    expected_trace_unsigned = {
        "schema_version": CASE_TRACE_SCHEMA,
        "profile_version": PROFILE_VERSION,
        "arm": arm,
        "semantic_input": {
            "corpus_id": corpus_id,
            "document_id": question["document_id"],
            "question": question["question"],
        },
        "source_phase4_packet_sha256": semantic_sha256(
            _strip_join_fields(phase4_packet)
        ),
        "base_kind": expected_base_kind,
        "base_answer_sha256": sha256_text(base_answer),
        "hydrated_candidates": expected_hydration,
        "composer": expected_composer,
        "selected_candidate_ids": expected_selected_ids,
        "selected_claim_sha256": expected_claim_hashes,
    }
    expected_trace = {
        **expected_trace_unsigned,
        "semantic_trace_sha256": semantic_sha256(expected_trace_unsigned),
    }

    answer = str(row["answer_safe_text"])
    if MASK in answer:
        raise Phase51RunnerError("Phase B output contains a mask")
    if arm != B1 and (not answer or not answer.startswith(base_answer)):
        raise Phase51RunnerError("Phase B v10-base projection differs")
    claims = row["claims"]
    if claims != expected_claims:
        raise Phase51RunnerError(
            "Phase B claims differ from authoritative recomposition"
        )
    if len(claims) > MAX_CLAIMS:
        raise Phase51RunnerError("Phase B selected too many claims")
    selected_ids = _unique([claim["candidate_id"] for claim in claims])
    claim_hashes = [claim["claim_sha256"] for claim in claims]
    if (
        row["selected_candidate_ids"] != selected_ids
        or row["selected_claim_sha256"] != claim_hashes
    ):
        raise Phase51RunnerError("Phase B selected claim identity differs")
    expected_answer = "\n".join(str(claim["text"]) for claim in claims)
    if base_answer:
        expected_answer = base_answer + (
            f"\n{expected_answer}" if expected_answer else ""
        )
    if answer != expected_answer:
        raise Phase51RunnerError("Phase B answer/claim projection differs")
    expected_citations = [
        *copy.deepcopy(list(base_citations)),
        *[_compact_citation(claim) for claim in claims],
    ]
    if row["citations"] != expected_citations:
        raise Phase51RunnerError("Phase B citation projection differs")
    trace = row["semantic_trace"]
    if trace != expected_trace:
        raise Phase51RunnerError(
            "Phase B trace differs from authoritative recomposition"
        )
    if trace["hydrated_candidates"] != expected_hydration:
        raise Phase51RunnerError("Phase B hydration trace projection differs")
    hydration_by_candidate: dict[str, Mapping[str, Any]] = {}
    row_by_candidate: dict[str, Mapping[str, Any]] = {}
    for packet in hydrated_candidates:
        hydration = packet.get("hydration")
        hydrated_row = packet.get("row")
        if not isinstance(hydration, Mapping) or not isinstance(
            hydrated_row, Mapping
        ):
            raise Phase51RunnerError("Phase B hydrated candidate shape differs")
        candidate_id = str(hydration.get("candidate_id") or "")
        if (
            not candidate_id
            or candidate_id in hydration_by_candidate
            or candidate_id in row_by_candidate
        ):
            raise Phase51RunnerError("Phase B hydrated candidate ids differ")
        hydration_by_candidate[candidate_id] = hydration
        row_by_candidate[candidate_id] = hydrated_row
    for claim in claims:
        _validate_claim_contract(
            claim,
            corpus_id=corpus_id,
            question=question,
            document=document,
            hydration_by_candidate=hydration_by_candidate,
            row_by_candidate=row_by_candidate,
        )
    if (
        trace["selected_candidate_ids"] != selected_ids
        or trace["selected_claim_sha256"] != claim_hashes
        or trace["semantic_input"]
        != {
            "corpus_id": corpus_id,
            "document_id": question["document_id"],
            "question": question["question"],
        }
    ):
        raise Phase51RunnerError("Phase B semantic trace differs")
    trace_unsigned = {
        key: value for key, value in trace.items()
        if key != "semantic_trace_sha256"
    }
    if trace["semantic_trace_sha256"] != semantic_sha256(trace_unsigned):
        raise Phase51RunnerError("Phase B semantic trace hash differs")
    composer = trace["composer"]
    if composer != expected_composer:
        raise Phase51RunnerError(
            "Phase B composer differs from authoritative recomposition"
        )
    if (
        composer["selected_candidate_ids"] != selected_ids
        or composer["selected_claim_sha256"] != claim_hashes
        or composer["selected_row_label_cells"]
        != [claim["row_label_cell"] for claim in claims]
        or composer["selected_claim_count"] != len(claims)
    ):
        raise Phase51RunnerError("Phase B composer selection projection differs")
    composer_unsigned = {
        key: value for key, value in composer.items()
        if key != "semantic_trace_sha256"
    }
    if composer["semantic_trace_sha256"] != semantic_sha256(composer_unsigned):
        raise Phase51RunnerError("Phase B composer trace hash differs")
    _reject_forbidden_fields(row, location="phase_b_output")


def _membership_row(output: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "question_id": output["question_id"],
        "document_id": output["document_id"],
        "question": output["question"],
        "selected_candidate_ids": output["selected_candidate_ids"],
        "selected_claim_sha256": output["selected_claim_sha256"],
    }


def _semantic_membership_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    semantic_rows = [
        {
            "document_id": row["document_id"],
            "question": row["question"],
            "selected_candidate_ids": row["selected_candidate_ids"],
            "selected_claim_sha256": row["selected_claim_sha256"],
        }
        for row in rows
    ]
    semantic_rows.sort(
        key=lambda row: (
            row["document_id"],
            row["question"],
            tuple(row["selected_candidate_ids"]),
            tuple(row["selected_claim_sha256"]),
        )
    )
    return semantic_sha256(semantic_rows)


def _arm_coverage(outputs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [row for row in outputs if row["claims"]]
    clusters = {(row["document_id"], row["question"]) for row in selected}
    documents = {row["document_id"] for row in selected}
    routes = Counter(
        row["semantic_trace"]["composer"]["route"] for row in outputs
    )
    claims = Counter(str(len(row["claims"])) for row in outputs)
    return {
        "cases": len(outputs),
        "nonempty_answers": sum(bool(row["answer_safe_text"]) for row in outputs),
        "empty_answers": sum(not bool(row["answer_safe_text"]) for row in outputs),
        "selected_cases": len(selected),
        "selected_semantic_tasks": len(clusters),
        "selected_documents": len(documents),
        "rendered_claims": sum(len(row["claims"]) for row in outputs),
        "route_counts": dict(sorted(routes.items())),
        "claim_count_distribution": dict(sorted(claims.items())),
        "maximum_append_characters": max(
            (
                row["semantic_trace"]["composer"]["append_characters"]
                for row in outputs
            ),
            default=0,
        ),
    }


def _code_dependency_paths(root: Path) -> dict[str, Path]:
    """Return the currently loaded repo-local runtime and schema closure."""

    paths = {
        "core_module": root / "src/finglmqa/type3_phase51_compact_tabgr.py",
        "runner": Path(__file__).resolve(),
        "answer_schema": ANSWER_SCHEMA_PATH,
        "safe_v10_schema": SAFE_V10_SCHEMA_PATH,
        "phase5_helper_runner": root / "scripts/run_type3_phase5_compact.py",
        "phase5_helper_composer": (
            root / "src/finglmqa/type3_phase5_compact_composer.py"
        ),
        "phase5_helper_answer_schema": (
            root / "data/schemas/type3/phase5_compact_answer_v2.schema.json"
        ),
        "a2rag_tabgr_pipeline": (
            root / "src/finglmqa/type3_a2rag_tabgr_pipeline.py"
        ),
        "corpus_profile": root / "src/finglmqa/type3_corpus_profile.py",
        "tabgr_retriever": root / "src/finglmqa/type3_tabgr_retriever.py",
    }
    registered = {path.resolve() for path in paths.values()}
    loaded_paths: set[Path] = set()
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix in {".pyc", ".pyo"}:
            try:
                path = Path(importlib.util.source_from_cache(path.as_posix()))
            except (ValueError, NotImplementedError):
                continue
        if (
            path.suffix != ".py"
            or not path.is_relative_to(root)
            or path.resolve() in registered
        ):
            continue
        relative = path.relative_to(root)
        if relative.parts[0] not in {"src", "scripts"}:
            continue
        loaded_paths.add(path)
    for path in sorted(loaded_paths, key=lambda value: value.as_posix()):
        relative = path.relative_to(root).as_posix()
        key = "loaded_" + re.sub(r"[^a-z0-9]+", "_", relative.lower()).strip("_")
        if key in paths:
            raise Phase51RunnerError(
                f"runtime code dependency key collision: {relative}"
            )
        paths[key] = path
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--corpus-id", default="annual_reports_170_v1")
    parser.add_argument("--question-profile-id", default="type3_260_dev_v1")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--question-profile", type=Path)
    parser.add_argument("--phase4-answers", type=Path, default=DEFAULT_PHASE4_ANSWERS)
    parser.add_argument("--phase4-manifest", type=Path, default=DEFAULT_PHASE4_MANIFEST)
    parser.add_argument("--tabgr-index-dir", type=Path)
    parser.add_argument(
        "--v10-safe-dir",
        type=Path,
        default=DEFAULT_V10_SAFE_DIR,
    )
    parser.add_argument("--b0-r2-dir", type=Path, default=DEFAULT_R2_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _main(argv: Sequence[str] | None = None) -> int:
    global _ACTIVE_STAGING
    args = parse_args(argv)
    root = args.root.resolve()
    manifest_paths = _guard_manifest_paths(
        ManifestPaths.defaults(
            root, args.corpus_id, args.question_profile_id
        )
    )
    corpus_manifest_path = guard_generator_path(
        args.corpus_manifest or manifest_paths.corpus_manifest
    )
    question_profile_path = guard_generator_path(
        args.question_profile or manifest_paths.question_profile
    )
    questions_path = guard_generator_path(args.questions)
    if (
        corpus_manifest_path != manifest_paths.corpus_manifest.resolve()
        or question_profile_path != manifest_paths.question_profile.resolve()
        or questions_path != manifest_paths.questions.resolve()
    ):
        raise Phase51RunnerError(
            "corpus/question paths must match the declared corpus profile"
        )
    corpus = load_corpus_profile(corpus_manifest_path)
    question_profile, questions = load_question_profile(
        question_profile_path, corpus_profile=corpus
    )
    _enforce_annotation_policy(corpus, location="corpus_profile")
    _enforce_annotation_policy(
        question_profile, location="question_profile"
    )
    if (
        corpus["corpus_id"] != args.corpus_id
        or question_profile["question_profile_id"] != args.question_profile_id
    ):
        raise Phase51RunnerError("CLI corpus/question identity differs")
    _enforce_annotation_policy(questions, location="sanitized_questions")
    source_before = source_snapshot(corpus, workspace_root=root)
    binding = bind_manifests(
        manifest_paths,
        expected_corpus_id=args.corpus_id,
        expected_question_profile_id=args.question_profile_id,
    ).as_mapping()
    _enforce_annotation_policy(binding, location="manifest_binding")

    phase4_path = guard_generator_path(args.phase4_answers)
    phase4_manifest_path = guard_generator_path(args.phase4_manifest)
    tabgr_index_dir = guard_generator_path(
        args.tabgr_index_dir or manifest_paths.tabgr_index_manifest.parent
    )
    v10_safe_dir = guard_generator_path(args.v10_safe_dir)
    v10_safe_paths = (
        guard_generator_path(v10_safe_dir / "manifest.json"),
        guard_generator_path(v10_safe_dir / "answers.jsonl"),
    )
    r2_dir = guard_generator_path(args.b0_r2_dir)
    r2_paths = (
        guard_generator_path(r2_dir / "answers.jsonl"),
        guard_generator_path(r2_dir / "semantic_traces.jsonl"),
        guard_generator_path(r2_dir / "run_manifest.json"),
    )
    input_paths: dict[str, Path] = {
        "corpus_manifest": corpus_manifest_path,
        "question_profile": question_profile_path,
        "questions": questions_path,
        "phase4_answers": phase4_path,
        "phase4_manifest": phase4_manifest_path,
        "tabgr_index_manifest": tabgr_index_dir / "manifest.json",
        "tabgr_document_manifest": tabgr_index_dir / "document_manifest.jsonl",
        "tabgr_package_manifest": manifest_paths.tabgr_package_manifest,
        "v10_safe_manifest": v10_safe_paths[0],
        "v10_safe_answers": v10_safe_paths[1],
        "v10_safe_schema": SAFE_V10_SCHEMA_PATH,
        "b0_r2_answers": r2_paths[0],
        "b0_r2_traces": r2_paths[1],
        "b0_r2_manifest": r2_paths[2],
        "answer_schema": ANSWER_SCHEMA_PATH,
        **{
            f"binding_{name}": Path(value["path"]).resolve()
            for name, value in binding["manifests"].items()
        },
    }
    for path in input_paths.values():
        guarded = guard_generator_path(path)
        if not guarded.is_file():
            raise Phase51RunnerError(f"frozen input is missing: {guarded}")
    input_hashes_before = {
        key: sha256_file(path) for key, path in sorted(input_paths.items())
    }
    code_paths = _code_dependency_paths(root)
    for path in code_paths.values():
        if not path.is_file():
            raise Phase51RunnerError(f"executed code/schema dependency is missing: {path}")
    code_hashes_before = {
        key: sha256_file(path) for key, path in sorted(code_paths.items())
    }

    phase4 = _verify_phase4(
        answers_path=phase4_path,
        manifest_path=phase4_manifest_path,
        binding=binding,
        corpus_id=args.corpus_id,
        question_profile_id=args.question_profile_id,
        questions=questions,
    )
    v10 = _verify_and_load_safe_v10(
        directory=v10_safe_dir,
        corpus_id=args.corpus_id,
        question_profile_id=args.question_profile_id,
        question_profile_sha256=question_profile["profile_sha256"],
        questions=questions,
    )
    r2 = _verify_r2(
        r2_dir,
        corpus_id=args.corpus_id,
        question_profile_id=args.question_profile_id,
        questions=questions,
    )
    documents = _document_map(corpus)
    store = BoundTabGRStore(tabgr_index_dir, corpus_id=args.corpus_id)

    protected_paths = [
        *input_paths.values(),
        phase4_path.parent,
        tabgr_index_dir,
        r2_dir,
        v10_safe_dir,
        root / "data/corpus_package/type3" / args.corpus_id,
        root / "data/indexes/type3" / args.corpus_id,
        root / "data/facts/type3" / args.corpus_id,
    ]
    final_dir, staging = _safe_output_dir(
        args.output_dir, protected_paths=protected_paths
    )
    _ACTIVE_STAGING = staging
    staging.mkdir(parents=True)

    outputs_by_arm: dict[str, list[dict[str, Any]]] = {
        arm: [] for arm in ARMS
    }
    for question in questions:
        question_id = question["question_id"]
        packet = phase4[question_id]
        if (
            packet.get("question") != question["question"]
            or packet.get("document_id") != question["document_id"]
        ):
            raise Phase51RunnerError(
                "Phase 4 packet differs from sanitized question"
            )
        document = documents[question["document_id"]]
        selected_evidence = _selected_table_evidence(
            packet, corpus_id=args.corpus_id, document=document
        )
        hydrated = []
        for evidence in selected_evidence:
            value = store.hydrate(
                candidate_id=str(evidence["candidate_id"]),
                document_id=question["document_id"],
            )
            hydrated.append({"evidence": evidence, **value})
        for arm in ARMS:
            outputs_by_arm[arm].append(
                _compose_arm_case(
                    arm=arm,
                    corpus_id=args.corpus_id,
                    question=question,
                    document=document,
                    phase4_packet=packet,
                    hydrated_candidates=hydrated,
                    v10=v10[question_id],
                )
            )

    arm_manifests: dict[str, Any] = {}
    for arm in ARMS:
        arm_dir = staging / "arms" / arm
        outputs = outputs_by_arm[arm]
        traces = [
            {
                "question_id": row["question_id"],
                "semantic_trace": row["semantic_trace"],
            }
            for row in outputs
        ]
        membership = [_membership_row(row) for row in outputs if row["claims"]]
        answers_path = arm_dir / "answers.jsonl"
        traces_path = arm_dir / "traces.jsonl"
        membership_path = arm_dir / "selected_membership.jsonl"
        _write_jsonl(answers_path, outputs)
        _write_jsonl(traces_path, traces)
        _write_jsonl(membership_path, membership)
        coverage = _arm_coverage(outputs)
        if arm == B1:
            if coverage["nonempty_answers"] + coverage["empty_answers"] != len(
                questions
            ):
                raise Phase51RunnerError("b1 terminal case accounting differs")
        elif coverage["nonempty_answers"] != len(questions):
            raise Phase51RunnerError(f"{arm} contains an empty primary answer")
        arm_manifests[arm] = {
            "configuration": ARM_CONFIGURATION[arm],
            "coverage": coverage,
            "artifacts": {
                "answers.jsonl": sha256_file(answers_path),
                "traces.jsonl": sha256_file(traces_path),
                "selected_membership.jsonl": sha256_file(membership_path),
            },
            "selected_membership_count": len(membership),
            "selected_membership_semantic_sha256": _semantic_membership_hash(
                membership
            ),
        }

    store.verify_used_shards_unchanged()
    source_after = source_snapshot(corpus, workspace_root=root)
    if source_after != source_before:
        raise Phase51RunnerError(
            "frozen source Markdown changed during Phase B generation"
        )
    input_hashes_after = {
        key: sha256_file(path) for key, path in sorted(input_paths.items())
    }
    if input_hashes_after != input_hashes_before:
        raise Phase51RunnerError("frozen generator input changed during generation")
    code_paths_after = _code_dependency_paths(root)
    if {
        key: path.resolve() for key, path in code_paths_after.items()
    } != {
        key: path.resolve() for key, path in code_paths.items()
    }:
        raise Phase51RunnerError(
            "repo-local runtime code dependency closure changed during generation"
        )
    code_hashes_after = {
        key: sha256_file(path)
        for key, path in sorted(code_paths_after.items())
    }
    if code_hashes_after != code_hashes_before:
        raise Phase51RunnerError("Phase B code/schema changed during generation")

    manifest_unsigned = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "profile_version": PROFILE_VERSION,
        "composer_version": COMPOSER_VERSION,
        "corpus_id": args.corpus_id,
        "corpus_profile_sha256": corpus["profile_sha256"],
        "question_profile_id": args.question_profile_id,
        "question_profile_sha256": question_profile["profile_sha256"],
        "question_count": len(questions),
        "question_ids_sha256": semantic_sha256(
            [row["question_id"] for row in questions]
        ),
        "configuration": {
            "arms": list(ARMS),
            "maximum_claims_per_question": MAX_CLAIMS,
            "maximum_appended_characters": MAX_TOTAL_CHARACTERS,
            "phase4_selected_table_evidence_only": True,
            "document_shard_cache_size": store.maximum_cached_documents,
            "b0_regenerated": False,
        },
        "manifest_binding": binding,
        "inputs": {
            key: {
                "path": _relative(path, root),
                "sha256_before": input_hashes_before[key],
                "sha256_after": input_hashes_after[key],
            }
            for key, path in sorted(input_paths.items())
        },
        "used_tabgr_document_shards": {
            document_id: {
                "path": _relative(Path(value["path"]), root),
                "sha256_before": value["sha256"],
                "sha256_after": sha256_file(Path(value["path"])),
            }
            for document_id, value in sorted(store.used_shards.items())
        },
        "source_freeze": {
            "source_ref": corpus["source_ref"],
            "document_count": len(source_before),
            "source_hashes_sha256_before": semantic_sha256(source_before),
            "source_hashes_sha256_after": semantic_sha256(source_after),
            "source_unchanged": True,
        },
        "b0_phase5_r2_frozen": {
            "regenerated": False,
            "directory": _relative(r2["directory"], root),
            "run_manifest_sha256": sha256_file(r2["manifest_path"]),
            "answers_sha256": sha256_file(r2["answers_path"]),
            "traces_sha256": sha256_file(r2["traces_path"]),
        },
        "arms": arm_manifests,
        "code": {
            key: {
                "path": _relative(code_paths[key], root),
                "sha256_before": code_hashes_before[key],
                "sha256_after": code_hashes_after[key],
            }
            for key in sorted(code_paths)
        },
        "generator_boundary": {
            "semantic_fields": ["corpus_id", "document_id", "question"],
            "question_id_use": "join_and_output_order_only",
            "gold_or_scoring_available": False,
            "scorer_imported": False,
            "input_output_disjointness_enforced": True,
            "recursive_forbidden_field_scan": True,
            "safe_v10_projection_only": True,
            "raw_v10_envelope_opened": False,
            "inputs_rehashed_after_generation": True,
            "hidden_staging_atomic_publish": True,
        },
        "safety": {
            "schema_failures": 0,
            "masked_placeholders": 0,
            "cross_document_claims": 0,
            "source_sha_mismatches": 0,
            "b1_empty_answers": arm_manifests[B1]["coverage"]["empty_answers"],
            "b2_b3_b4_nonempty_answers": {
                arm: arm_manifests[arm]["coverage"]["nonempty_answers"]
                for arm in (B2, B3, B4)
            },
        },
    }
    manifest = {
        **manifest_unsigned,
        "run_fingerprint": semantic_sha256(manifest_unsigned),
    }
    _write_json(staging / "run_manifest.json", manifest)
    staging.rename(final_dir)
    _ACTIVE_STAGING = None
    print(
        json.dumps(
            {
                "status": "passed",
                "questions": len(questions),
                "output_dir": final_dir.as_posix(),
                "run_fingerprint": manifest["run_fingerprint"],
                "selected_cases": {
                    arm: arm_manifests[arm]["coverage"]["selected_cases"]
                    for arm in ARMS
                },
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
