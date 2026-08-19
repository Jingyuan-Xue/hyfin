#!/usr/bin/env python3
"""Generate the four scorer-free Type 3 Phase 5.1 Phase B.1 arms."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
from referencing import Registry, Resource


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_type3_phase51_phaseb as phaseb  # noqa: E402

from finglmqa.type3_a2rag_tabgr_pipeline import (  # noqa: E402
    ManifestPaths,
    bind_manifests,
)
from finglmqa.type3_corpus_profile import (  # noqa: E402
    load_corpus_profile,
    load_question_profile,
    source_snapshot,
)
from finglmqa.type3_phase51_b1_compact import (  # noqa: E402
    COMPOSER_VERSION,
    MASK,
    MAX_CLAIMS,
    MAX_TOTAL_CHARACTERS,
    PROFILE_VERSION,
    compose_phase51_b1_claims,
)
from finglmqa.type3_tabgr_retriever import (  # noqa: E402
    TABGR_RUNTIME_SHA256,
    TABGR_V2_RETRIEVER_VERSION,
    Type3TabGRCandidate,
    Type3TabGRRetriever,
    normalize_text,
    numeric_fragments,
    semantic_sha256,
    sha256_text,
)


OUTPUT_SCHEMA = "finglmqa.type3.phase51.phase_b1_answer.v1"
CASE_TRACE_SCHEMA = "finglmqa.type3.phase51.phase_b1_case_trace.v1"
RUN_MANIFEST_SCHEMA = "finglmqa.type3.phase51.phase_b1_run_manifest.v1"
LITERAL_SCHEMA = "finglmqa.type3.tabgr.cell_literal_authorization.v1"
LITERAL_MANIFEST_SCHEMA = (
    "finglmqa.type3.tabgr.cell_literal_authorization_manifest.v1"
)
LITERAL_DOCUMENT_MANIFEST_SCHEMA = (
    "finglmqa.type3.tabgr.cell_literal_authorization_document_manifest.v1"
)

B1A = "b1a_current_shortlist_existing_auth"
B1B = "b1b_current_shortlist_literal_auth"
B1C = "b1c_full_document_tabgr_literal_auth"
B1D = "b1d_full_retrieval_literal_render_suppressed"
ARMS = (B1A, B1B, B1C, B1D)
ARM_CONFIGURATION = {
    B1A: {
        "retrieval": "phase4_shortlist",
        "authorization_mode": "existing_only",
        "suppress_rendering": False,
        "phase_c_bridge": False,
    },
    B1B: {
        "retrieval": "phase4_shortlist",
        "authorization_mode": "existing_plus_literal",
        "suppress_rendering": False,
        "phase_c_bridge": False,
    },
    B1C: {
        "retrieval": "full_document_tabgr_lexical_top100",
        "authorization_mode": "existing_plus_literal",
        "suppress_rendering": False,
        "phase_c_bridge": True,
    },
    B1D: {
        "retrieval": "full_document_tabgr_lexical_top100",
        "authorization_mode": "existing_plus_literal",
        "suppress_rendering": True,
        "phase_c_bridge": False,
    },
}

DEFAULT_QUESTIONS = phaseb.DEFAULT_QUESTIONS
DEFAULT_PHASE4_ANSWERS = phaseb.DEFAULT_PHASE4_ANSWERS
DEFAULT_PHASE4_MANIFEST = phaseb.DEFAULT_PHASE4_MANIFEST
DEFAULT_V10_SAFE_DIR = phaseb.DEFAULT_V10_SAFE_DIR
DEFAULT_V3_DIR = (
    ROOT
    / "runs/type3_a2rag_tabgr_experiment_v1/annual_reports_170_v1/"
    "phase_5_1/phase_b_compact_tabgr_v3_frozen/fresh_process_1"
)
DEFAULT_LITERAL_DIR = (
    ROOT
    / "data/authorizations/type3/annual_reports_170_v1/"
    "tabgr_cell_literal_v1"
)

ANSWER_SCHEMA_PATH = (
    ROOT / "data/schemas/type3/phase51_phaseb1_answer_v1.schema.json"
)
LITERAL_SCHEMA_PATH = (
    ROOT / "data/schemas/type3/tabgr_cell_literal_authorization_v1.schema.json"
)
LITERAL_MANIFEST_SCHEMA_PATH = (
    ROOT
    / "data/schemas/type3/"
    "tabgr_cell_literal_authorization_manifest_v1.schema.json"
)
LITERAL_DOCUMENT_MANIFEST_SCHEMA_PATH = (
    ROOT
    / "data/schemas/type3/"
    "tabgr_cell_literal_authorization_document_manifest_v1.schema.json"
)
NUMERIC_SCHEMA_PATH = (
    ROOT / "data/schemas/type3/tabgr_numeric_authorization_v1.schema.json"
)

TOP_K = 100
EXPECTED_V3_B2_COVERAGE = {
    "selected_cases": 4,
    "selected_semantic_tasks": 2,
    "selected_documents": 2,
    "rendered_claims": 8,
}
EXPECTED_V3_B2_MEMBERSHIP_SHA256 = (
    "34e9d6399deec82b90f238f54c2a9d629d12fd407fb0053138b370b407287f90"
)
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_ACTIVE_STAGING: Path | None = None
_PERCENT_UNITS = frozenset({"%", "％", "百分比", "百分数"})

PhaseB1RunnerError = phaseb.Phase51RunnerError
guard_generator_path = phaseb.guard_generator_path
sha256_file = phaseb.sha256_file
_write_json = phaseb._write_json
_write_jsonl = phaseb._write_jsonl
_relative = phaseb._relative
_safe_output_dir = phaseb._safe_output_dir
_safe_relative_path = phaseb._safe_relative_path
_strip_join_fields = phaseb._strip_join_fields
_unique = phaseb._unique

_SAFE_FALSE_SCORE_CONTROL_KEYS = {
    "gold_or_scoring_inputs_read",
    "gold_or_scoring_available",
    "scorer_imported",
}


def _enforce_annotation_policy(
    value: object,
    *,
    location: str = "$",
) -> None:
    """Reject scorer/annotation keys while permitting frozen retrieval metadata."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text == "retrieval_score":
                pass
            elif key_text in _SAFE_FALSE_SCORE_CONTROL_KEYS:
                if child is not False:
                    raise PhaseB1RunnerError(
                        "trusted gold/scoring control must be exactly false at "
                        f"{location}.{key_text}"
                    )
            else:
                tokens = phaseb._annotation_key_tokens(key_text)
                compact = "".join(tokens)
                if (
                    any(
                        fragment in token
                        for token in tokens
                        for fragment in phaseb._ANNOTATION_KEY_FRAGMENTS
                        if fragment != "answerkey"
                    )
                    or "answerkey" in compact
                ):
                    raise PhaseB1RunnerError(
                        "generator value contains forbidden annotation key at "
                        f"{location}.{key_text}"
                    )
            _enforce_annotation_policy(
                child,
                location=f"{location}.{key_text}",
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _enforce_annotation_policy(
                child,
                location=f"{location}[{index}]",
            )


def _reject_forbidden_fields(
    value: object,
    *,
    location: str = "$",
) -> None:
    _enforce_annotation_policy(value, location=location)


def _load_validator(path: Path) -> Draft202012Validator:
    value = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(value)
    return Draft202012Validator(value)


def _load_answer_validator() -> Draft202012Validator:
    value = json.loads(ANSWER_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(value)
    registry = Registry()
    for path in (
        phaseb.ANSWER_SCHEMA_PATH,
        LITERAL_SCHEMA_PATH,
        NUMERIC_SCHEMA_PATH,
    ):
        resource_value = json.loads(path.read_text(encoding="utf-8"))
        resource = Resource.from_contents(resource_value)
        registry = registry.with_resource(resource_value["$id"], resource)
    return Draft202012Validator(value, registry=registry)


_ANSWER_VALIDATOR = _load_answer_validator()
_LITERAL_VALIDATOR = _load_validator(LITERAL_SCHEMA_PATH)
_LITERAL_MANIFEST_VALIDATOR = _load_validator(LITERAL_MANIFEST_SCHEMA_PATH)
_LITERAL_DOCUMENT_MANIFEST_VALIDATOR = _load_validator(
    LITERAL_DOCUMENT_MANIFEST_SCHEMA_PATH
)


def _validate_schema(
    validator: Draft202012Validator,
    value: object,
    *,
    label: str,
) -> None:
    errors = sorted(
        validator.iter_errors(value),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if not errors:
        return
    first = errors[0]
    location = "$" + "".join(
        f"[{part!r}]" for part in first.absolute_path
    )
    raise PhaseB1RunnerError(
        f"{label} failed closed schema at {location}: {first.message}"
    )


def _guard_manifest_paths(paths: ManifestPaths) -> ManifestPaths:
    return phaseb._guard_manifest_paths(paths)


def load_json(path: Path) -> dict[str, Any]:
    guarded = guard_generator_path(path)
    value = json.loads(guarded.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PhaseB1RunnerError(f"expected JSON object: {guarded}")
    _enforce_annotation_policy(value, location=str(guarded))
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    guarded = guard_generator_path(path)
    rows: list[dict[str, Any]] = []
    with guarded.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise PhaseB1RunnerError(
                    f"blank JSONL record: {guarded}:{line_number}"
                )
            value = json.loads(line)
            if not isinstance(value, dict):
                raise PhaseB1RunnerError(
                    f"expected JSON object: {guarded}:{line_number}"
                )
            rows.append(value)
    _enforce_annotation_policy(rows, location=str(guarded))
    return rows


def _literal_hash_chain(record: Mapping[str, Any]) -> tuple[str, str, str]:
    source_hash = semantic_sha256(
        {
            "corpus_id": record["corpus_id"],
            "document_id": record["document_id"],
            "row_evidence_id": record["row_evidence_id"],
            "source_binding": record["source_binding"],
            "row_index": record["row_binding"]["row_index"],
            "cell_binding": record["cell_binding"],
        }
    )
    semantic_hash = semantic_sha256(
        {
            "row_path": record["row_binding"]["row_path"],
            "row_label_cell": record["row_binding"]["row_label_cell"],
            "header_binding": record["header_binding"],
            "semantic_context": record["semantic_context"],
        }
    )
    payload = {
        key: value
        for key, value in record.items()
        if key not in {"authorization_id", "authorization_sha256"}
    }
    payload["source_cell_binding_sha256"] = source_hash
    payload["semantic_context_sha256"] = semantic_hash
    authorization_hash = semantic_sha256(payload)
    return source_hash, semantic_hash, authorization_hash


def validate_literal_hash_chain(record: Mapping[str, Any]) -> None:
    source_hash, semantic_hash, authorization_hash = _literal_hash_chain(record)
    if (
        record.get("source_cell_binding_sha256") != source_hash
        or record.get("semantic_context_sha256") != semantic_hash
        or record.get("authorization_sha256") != authorization_hash
        or record.get("authorization_id")
        != "t3tabgr-lit-" + authorization_hash[:24]
    ):
        raise PhaseB1RunnerError("literal authorization hash chain differs")


class BoundLiteralAuthorizationStore:
    """Manifest-bound exact-literal authorization document store."""

    def __init__(
        self,
        package_dir: Path,
        *,
        corpus_id: str,
        corpus_profile_sha256: str,
        maximum_cached_documents: int = 4,
    ) -> None:
        self.root = guard_generator_path(package_dir).resolve()
        self.corpus_id = corpus_id
        self.maximum_cached_documents = maximum_cached_documents
        self.manifest_path = guard_generator_path(self.root / "manifest.json")
        self.document_manifest_path = guard_generator_path(
            self.root / "document_manifest.jsonl"
        )
        self.rejection_audit_path = guard_generator_path(
            self.root / "rejection_audit.jsonl"
        )
        manifest = load_json(self.manifest_path)
        _validate_schema(
            _LITERAL_MANIFEST_VALIDATOR,
            manifest,
            label="literal authorization manifest",
        )
        unsigned = {
            key: value
            for key, value in manifest.items()
            if key != "manifest_fingerprint"
        }
        document_binding = manifest["document_manifest"]
        rejection_binding = manifest["rejection_audit"]
        if (
            manifest.get("schema_version") != LITERAL_MANIFEST_SCHEMA
            or manifest.get("corpus_id") != corpus_id
            or manifest.get("corpus_profile_sha256")
            != corpus_profile_sha256
            or manifest.get("manifest_fingerprint") != semantic_sha256(unsigned)
            or document_binding.get("path") != "document_manifest.jsonl"
            or document_binding.get("sha256")
            != sha256_file(self.document_manifest_path)
            or rejection_binding.get("path") != "rejection_audit.jsonl"
            or rejection_binding.get("sha256")
            != sha256_file(self.rejection_audit_path)
            or manifest.get("build_contract", {}).get(
                "gold_or_scoring_inputs_read"
            )
            is not False
            or manifest.get("build_contract", {}).get(
                "numeric_authorization_priority"
            )
            is not True
            or manifest.get("build_contract", {}).get("exact_literal_only")
            is not True
        ):
            raise PhaseB1RunnerError(
                "literal authorization manifest contract differs"
            )
        entries: dict[str, dict[str, Any]] = {}
        with self.document_manifest_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise PhaseB1RunnerError(
                        "blank literal document manifest row"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise PhaseB1RunnerError(
                        "literal document manifest row is not an object"
                    )
                _enforce_annotation_policy(
                    value,
                    location=f"literal_document_manifest[{line_number}]",
                )
                _validate_schema(
                    _LITERAL_DOCUMENT_MANIFEST_VALIDATOR,
                    value,
                    label=f"literal document manifest[{line_number}]",
                )
                document_id = str(value.get("document_id") or "")
                if (
                    value.get("schema_version")
                    != LITERAL_DOCUMENT_MANIFEST_SCHEMA
                    or value.get("corpus_id") != corpus_id
                    or not document_id
                    or document_id in entries
                ):
                    raise PhaseB1RunnerError(
                        "literal document manifest identity differs"
                    )
                relative = _safe_relative_path(value.get("shard_path"))
                shard = guard_generator_path(self.root / relative)
                if not shard.is_relative_to(self.root):
                    raise PhaseB1RunnerError(
                        "literal authorization shard escapes package"
                    )
                entries[document_id] = {
                    **value,
                    "relative_path": relative,
                    "resolved_path": shard,
                }
        if (
            len(entries) != int(manifest.get("document_count", -1))
            or len(entries)
            != int(document_binding.get("record_count", -1))
            or sum(
                int(entry["authorization_count"])
                for entry in entries.values()
            )
            != int(manifest.get("authorization_count", -1))
        ):
            raise PhaseB1RunnerError(
                "literal authorization manifest counts differ"
            )
        self.manifest = manifest
        self.entries = entries
        self._cache: OrderedDict[
            str, dict[str, tuple[dict[str, Any], ...]]
        ] = OrderedDict()
        self.used_shards: dict[str, dict[str, str]] = {}
        self.access_count = 0

    def _load_document(
        self,
        document_id: str,
    ) -> dict[str, tuple[dict[str, Any], ...]]:
        cached = self._cache.get(document_id)
        if cached is not None:
            self._cache.move_to_end(document_id)
            return cached
        entry = self.entries.get(document_id)
        if entry is None:
            raise PhaseB1RunnerError(
                "document_id is outside literal authorization package"
            )
        digest = hashlib.sha256()
        record_count = 0
        by_row: dict[str, list[dict[str, Any]]] = {}
        ids: set[str] = set()
        coordinates: set[tuple[str, int, int]] = set()
        with Path(entry["resolved_path"]).open("rb") as handle:
            for raw_line in handle:
                digest.update(raw_line)
                if not raw_line.strip():
                    raise PhaseB1RunnerError(
                        "blank literal authorization shard row"
                    )
                record = json.loads(raw_line)
                if not isinstance(record, dict):
                    raise PhaseB1RunnerError(
                        "literal authorization shard row differs"
                    )
                _enforce_annotation_policy(
                    record,
                    location=(
                        f"literal_authorization[{document_id}]"
                        f"[{record_count}]"
                    ),
                )
                _validate_schema(
                    _LITERAL_VALIDATOR,
                    record,
                    label=f"literal authorization[{document_id}]",
                )
                validate_literal_hash_chain(record)
                record_count += 1
                row_id = str(record.get("row_evidence_id") or "")
                authorization_id = str(record.get("authorization_id") or "")
                coordinate = record.get("cell_binding", {}).get("coordinate")
                coordinate_key = (
                    row_id,
                    int(coordinate[0]),
                    int(coordinate[1]),
                )
                if (
                    record.get("schema_version") != LITERAL_SCHEMA
                    or record.get("corpus_id") != self.corpus_id
                    or record.get("document_id") != document_id
                    or record.get("source_binding", {}).get("source_sha256")
                    != entry["source_sha256"]
                    or not row_id
                    or authorization_id in ids
                    or coordinate_key in coordinates
                ):
                    raise PhaseB1RunnerError(
                        "literal authorization identity/provenance differs"
                    )
                ids.add(authorization_id)
                coordinates.add(coordinate_key)
                by_row.setdefault(row_id, []).append(record)
        actual_hash = digest.hexdigest()
        if (
            actual_hash != entry["shard_sha256"]
            or record_count != int(entry["authorization_count"])
        ):
            raise PhaseB1RunnerError(
                "literal authorization shard hash/count differs"
            )
        frozen = {
            row_id: tuple(
                sorted(
                    rows,
                    key=lambda row: (
                        tuple(row["cell_binding"]["coordinate"]),
                        row["authorization_id"],
                    ),
                )
            )
            for row_id, rows in by_row.items()
        }
        self.used_shards[document_id] = {
            "path": Path(entry["resolved_path"]).as_posix(),
            "relative_path": str(entry["relative_path"]),
            "sha256": actual_hash,
        }
        self._cache[document_id] = frozen
        self._cache.move_to_end(document_id)
        while len(self._cache) > self.maximum_cached_documents:
            self._cache.popitem(last=False)
        return frozen

    def authorizations_for(
        self,
        *,
        document_id: str,
        row_evidence_id: str,
    ) -> list[dict[str, Any]]:
        self.access_count += 1
        rows = self._load_document(document_id).get(row_evidence_id, ())
        return copy.deepcopy(list(rows))

    def verify_used_shards_unchanged(self) -> None:
        for value in self.used_shards.values():
            if sha256_file(Path(value["path"])) != value["sha256"]:
                raise PhaseB1RunnerError(
                    "literal authorization shard changed during generation"
                )


class BoundFullTabGRStore(phaseb.BoundTabGRStore):
    """Phase B store plus complete-document row accounting."""

    def __init__(
        self,
        index_dir: Path,
        *,
        corpus_id: str,
        maximum_cached_documents: int = 4,
    ) -> None:
        super().__init__(
            index_dir,
            corpus_id=corpus_id,
            maximum_cached_documents=maximum_cached_documents,
        )
        self._authoritative_hashes: OrderedDict[
            str, dict[str, str]
        ] = OrderedDict()

    def _load_document(self, document_id: str) -> dict[str, dict[str, Any]]:
        cached = self._cache.get(document_id)
        if cached is not None:
            self._cache.move_to_end(document_id)
            self._authoritative_hashes.move_to_end(document_id)
            return cached
        entry = self.entries.get(document_id)
        if entry is None:
            raise PhaseB1RunnerError(
                "document_id is outside frozen TabGR index"
            )
        digest = hashlib.sha256()
        record_count = 0
        rows: dict[str, dict[str, Any]] = {}
        authoritative_hashes: dict[str, str] = {}
        with Path(entry["resolved_path"]).open("rb") as handle:
            for raw_line in handle:
                digest.update(raw_line)
                if not raw_line.strip():
                    raise PhaseB1RunnerError("blank record in TabGR shard")
                record = json.loads(raw_line)
                if not isinstance(record, dict):
                    raise PhaseB1RunnerError("TabGR shard row differs")
                _enforce_annotation_policy(
                    record,
                    location=(
                        f"tabgr_shard[{document_id}][{record_count}]"
                    ),
                )
                record_count += 1
                if (
                    record.get("corpus_id") != self.corpus_id
                    or record.get("document_id") != document_id
                ):
                    raise PhaseB1RunnerError(
                        "cross-corpus/document row in TabGR shard"
                    )
                evidence_id = str(record.get("evidence_id") or "")
                if evidence_id in rows:
                    raise PhaseB1RunnerError(
                        "duplicate evidence_id in TabGR shard"
                    )
                if record.get("record_type") == "table_row":
                    rows[evidence_id] = {
                        key: record.get(key)
                        for key in phaseb._ROW_PROJECTION_FIELDS
                    }
                    authoritative_hashes[evidence_id] = semantic_sha256(
                        record
                    )
        actual_hash = digest.hexdigest()
        if (
            actual_hash != entry["shard_sha256"]
            or record_count != int(entry.get("record_count", -1))
        ):
            raise PhaseB1RunnerError("TabGR shard hash/count differs")
        self.used_shards[document_id] = {
            "path": Path(entry["resolved_path"]).as_posix(),
            "relative_path": str(entry["relative_path"]),
            "sha256": actual_hash,
        }
        self._cache[document_id] = rows
        self._authoritative_hashes[document_id] = authoritative_hashes
        self._cache.move_to_end(document_id)
        self._authoritative_hashes.move_to_end(document_id)
        while len(self._cache) > self.maximum_cached_documents:
            evicted, _ = self._cache.popitem(last=False)
            self._authoritative_hashes.pop(evicted, None)
        return rows

    def hydrate(
        self,
        *,
        candidate_id: str,
        document_id: str,
    ) -> dict[str, Any]:
        value = super().hydrate(
            candidate_id=candidate_id,
            document_id=document_id,
        )
        authoritative_hash = self._authoritative_hashes.get(
            document_id,
            {},
        ).get(candidate_id)
        if not authoritative_hash:
            raise PhaseB1RunnerError(
                "authoritative TabGR row hash is unavailable"
            )
        value["hydration"]["authoritative_row_evidence_sha256"] = (
            authoritative_hash
        )
        return value

    def table_row_count(self, document_id: str) -> int:
        return len(self._load_document(document_id))


def _verify_v3_b2_reference(
    directory: Path,
    *,
    corpus_id: str,
    question_profile_id: str,
    questions: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    root = guard_generator_path(directory)
    manifest_path = guard_generator_path(root / "run_manifest.json")
    answers_path = guard_generator_path(
        root / "arms" / phaseb.B2 / "answers.jsonl"
    )
    traces_path = guard_generator_path(
        root / "arms" / phaseb.B2 / "traces.jsonl"
    )
    membership_path = guard_generator_path(
        root / "arms" / phaseb.B2 / "selected_membership.jsonl"
    )
    manifest = load_json(manifest_path)
    unsigned = {
        key: value for key, value in manifest.items()
        if key != "run_fingerprint"
    }
    arm = manifest.get("arms", {}).get(phaseb.B2, {})
    artifacts = arm.get("artifacts", {})
    coverage = arm.get("coverage", {})
    if (
        manifest.get("schema_version") != phaseb.RUN_MANIFEST_SCHEMA
        or manifest.get("corpus_id") != corpus_id
        or manifest.get("question_profile_id") != question_profile_id
        or manifest.get("question_count") != len(questions)
        or manifest.get("run_fingerprint") != semantic_sha256(unsigned)
        or artifacts.get("answers.jsonl") != sha256_file(answers_path)
        or artifacts.get("traces.jsonl") != sha256_file(traces_path)
        or artifacts.get("selected_membership.jsonl")
        != sha256_file(membership_path)
        or any(
            coverage.get(key) != expected
            for key, expected in EXPECTED_V3_B2_COVERAGE.items()
        )
        or arm.get("selected_membership_semantic_sha256")
        != EXPECTED_V3_B2_MEMBERSHIP_SHA256
    ):
        raise PhaseB1RunnerError("Phase B v3 b2 regression reference differs")
    rows = load_jsonl(answers_path)
    by_id = {str(row.get("question_id") or ""): row for row in rows}
    if (
        len(rows) != len(questions)
        or len(by_id) != len(rows)
        or set(by_id) != {row["question_id"] for row in questions}
    ):
        raise PhaseB1RunnerError("Phase B v3 b2 reference identity differs")
    return {
        "directory": root,
        "manifest_path": manifest_path,
        "answers_path": answers_path,
        "traces_path": traces_path,
        "membership_path": membership_path,
        "manifest": manifest,
        "answers": by_id,
    }


def _phase4_shortlist(
    *,
    packet: Mapping[str, Any],
    corpus_id: str,
    document: Mapping[str, Any],
    tabgr_store: BoundFullTabGRStore,
    literal_store: BoundLiteralAuthorizationStore | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evidence_rows = phaseb._selected_table_evidence(
        packet,
        corpus_id=corpus_id,
        document=document,
    )
    hydrated: list[dict[str, Any]] = []
    for rank, evidence in enumerate(evidence_rows, 1):
        packet_value = tabgr_store.hydrate(
            candidate_id=str(evidence["candidate_id"]),
            document_id=str(document["document_id"]),
        )
        candidate = {
            "evidence": copy.deepcopy(dict(evidence)),
            **packet_value,
        }
        if literal_store is not None:
            candidate["literal_authorizations"] = (
                literal_store.authorizations_for(
                    document_id=str(document["document_id"]),
                    row_evidence_id=str(evidence["candidate_id"]),
                )
            )
        else:
            candidate["literal_authorizations"] = []
        hydrated.append(candidate)
    ordered = [str(row["evidence"]["candidate_id"]) for row in hydrated]
    trace = {
        "retrieval_kind": "phase4_shortlist",
        "authorization_blind": True,
        "top_k": None,
        "ablations": [],
        "ppr_enabled": False,
        "eligible_row_count": len(hydrated),
        "retrieved_count": len(hydrated),
        "ordered_candidates": [
            {
                "rank": rank,
                "candidate_id": candidate_id,
                "retrieval_channel": "phase4_frozen_shortlist",
                "retrieval_score": None,
            }
            for rank, candidate_id in enumerate(ordered, 1)
        ],
    }
    trace["ordered_candidates_semantic_sha256"] = semantic_sha256(
        trace["ordered_candidates"]
    )
    return hydrated, trace


def _retrieved_evidence(
    candidate: Type3TabGRCandidate,
    *,
    rank: int,
    document: Mapping[str, Any],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        candidate.corpus_id != row.get("corpus_id")
        or candidate.document_id != row.get("document_id")
        or candidate.evidence_id != row.get("evidence_id")
        or candidate.table_id != row.get("table_id")
        or candidate.source_markdown != row.get("source_markdown")
        or list(candidate.line_range) != row.get("table_line_range")
        or Path(candidate.source_markdown).name
        != document["source_markdown"]
    ):
        raise PhaseB1RunnerError("full retrieval candidate hydration differs")
    cells = row.get("cells")
    if not isinstance(cells, list):
        raise PhaseB1RunnerError("full retrieval hydrated cells differ")
    return {
        "schema_version": "finglmqa.type3.phase51.full_tabgr_candidate.v1",
        "candidate_id": candidate.evidence_id,
        "corpus_id": candidate.corpus_id,
        "document_id": candidate.document_id,
        "route": "table",
        "evidence_type": "table_row",
        "source_markdown": candidate.source_markdown,
        "source_sha256": document["source_sha256"],
        "table_id": candidate.table_id,
        "table_sha256": row["table_sha256"],
        "line_range": list(candidate.line_range),
        "heading_path": list(candidate.heading_path),
        "row_path": list(candidate.row_path),
        "cell_coordinates": [
            copy.deepcopy(cell["coordinate"])
            for cell in cells
            if isinstance(cell, Mapping)
        ],
        "origin_cell_hashes": [
            str(cell["origin_cell_hash"])
            for cell in cells
            if isinstance(cell, Mapping)
        ],
        "numeric_authorizations": copy.deepcopy(
            list(row.get("numeric_authorizations") or ())
        ),
        "rank_signals": [
            {
                "rank": rank,
                "channel": "tabgr_lexical_v2",
            }
        ],
        "retrieval_score": format(candidate.retrieval_score, ".8f"),
    }


def _full_document_retrieval(
    *,
    question: Mapping[str, str],
    document: Mapping[str, Any],
    retriever: Type3TabGRRetriever,
    tabgr_store: BoundFullTabGRStore,
    literal_store: BoundLiteralAuthorizationStore,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Retrieval is deliberately completed before the authorization store is
    # touched.  Literal availability therefore cannot affect ranking/truncation.
    authorization_access_before = literal_store.access_count
    retrieved = retriever.retrieve(
        question["question"],
        document_id=question["document_id"],
        top_k=TOP_K,
        ablations=(),
        ppr_scores=None,
    )
    if literal_store.access_count != authorization_access_before:
        raise PhaseB1RunnerError(
            "literal authorization was accessed before retrieval completed"
        )
    candidates: list[dict[str, Any]] = []
    ordered_candidates: list[dict[str, Any]] = []
    previous_score: float | None = None
    previous_id = ""
    for rank, candidate in enumerate(retrieved, 1):
        if (
            candidate.retrieval_channel != "tabgr_lexical_v2"
            or candidate.retrieval_score <= 0
        ):
            raise PhaseB1RunnerError(
                "full retrieval channel/score contract differs"
            )
        if previous_score is not None:
            if candidate.retrieval_score > previous_score:
                raise PhaseB1RunnerError(
                    "full retrieval score order differs"
                )
            if (
                candidate.retrieval_score == previous_score
                and candidate.evidence_id < previous_id
            ):
                raise PhaseB1RunnerError(
                    "full retrieval evidence-id tie order differs"
                )
        previous_score = candidate.retrieval_score
        previous_id = candidate.evidence_id
        packet = tabgr_store.hydrate(
            candidate_id=candidate.evidence_id,
            document_id=question["document_id"],
        )
        evidence = _retrieved_evidence(
            candidate,
            rank=rank,
            document=document,
            row=packet["row"],
        )
        literal_authorizations = literal_store.authorizations_for(
            document_id=question["document_id"],
            row_evidence_id=candidate.evidence_id,
        )
        candidates.append(
            {
                "evidence": evidence,
                **packet,
                "literal_authorizations": literal_authorizations,
            }
        )
        ordered_candidates.append(
            {
                "rank": rank,
                "candidate_id": candidate.evidence_id,
                "retrieval_channel": candidate.retrieval_channel,
                "retrieval_score": format(
                    candidate.retrieval_score,
                    ".8f",
                ),
            }
        )
    trace = {
        "retrieval_kind": "full_document_tabgr_lexical_top100",
        "authorization_blind": True,
        "retriever_version": TABGR_V2_RETRIEVER_VERSION,
        "tabgr_runtime_sha256": TABGR_RUNTIME_SHA256,
        "top_k": TOP_K,
        "ablations": [],
        "ppr_enabled": False,
        "eligible_row_count": tabgr_store.table_row_count(
            question["document_id"]
        ),
        "retrieved_count": len(candidates),
        "ordered_candidates": ordered_candidates,
        "document_shard_sha256": (
            tabgr_store.entries[question["document_id"]]["shard_sha256"]
        ),
    }
    trace["ordered_candidates_semantic_sha256"] = semantic_sha256(
        ordered_candidates
    )
    return candidates, trace


def _compact_citation(claim: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        "citation_kind": "compact_tabgr_cell_claim",
        "claim_sha256": claim["claim_sha256"],
        "authorization_kind": claim["authorization_kind"],
        "candidate_id": claim["candidate_id"],
        "corpus_id": claim["corpus_id"],
        "document_id": claim["document_id"],
        "source_markdown": claim["source_markdown"],
        "source_sha256": claim["source_sha256"],
        "table_id": claim["table_id"],
        "table_index": claim["table_index"],
        "table_sha256": claim["table_sha256"],
        "table_line_range": claim["table_line_range"],
        "row_index": claim["row_index"],
        "row_path": claim["row_path"],
        "row_label_cell": claim["row_label_cell"],
        "selected_cells": claim["selected_cells"],
        "hydrated_projection_sha256": claim[
            "hydrated_projection_sha256"
        ],
    }
    return {**unsigned, "citation_sha256": semantic_sha256(unsigned)}


def _rendered_cell_value(raw_value: str, unit: str) -> str:
    """Mirror the frozen core's exact, non-normalizing unit suffix rule."""

    has_percent = "%" in raw_value or "％" in raw_value
    is_percent_unit = normalize_text(unit) in _PERCENT_UNITS
    if has_percent and not is_percent_unit:
        raise PhaseB1RunnerError(
            "percent literal conflicts with non-percent unit"
        )
    if has_percent or (unit and raw_value.endswith(unit)):
        return raw_value
    return raw_value + unit


def _assert_v3_b2_regression(
    generated: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> None:
    for key in (
        "corpus_id",
        "question_id",
        "question",
        "document_id",
        "base_kind",
        "base_answer_sha256",
        "answer_safe_text",
        "claims",
        "citations",
        "selected_candidate_ids",
        "selected_claim_sha256",
    ):
        if generated.get(key) != reference.get(key):
            raise PhaseB1RunnerError(
                f"b1a Phase B v3 b2 regression differs at {key}"
            )
    if (
        generated.get("semantic_trace", {}).get("composer")
        != reference.get("semantic_trace", {}).get("composer")
    ):
        raise PhaseB1RunnerError(
            "b1a Phase B v3 b2 composer regression differs"
        )


def _compose_b1a_case(
    *,
    corpus_id: str,
    question: Mapping[str, str],
    document: Mapping[str, Any],
    phase4_packet: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    retrieval_trace: Mapping[str, Any],
    v10: Mapping[str, Any],
    v3_reference: Mapping[str, Any],
) -> dict[str, Any]:
    v3_candidates = copy.deepcopy(list(candidates))
    for candidate in v3_candidates:
        hydration = candidate.get("hydration")
        if isinstance(hydration, dict):
            hydration.pop("authoritative_row_evidence_sha256", None)
    generated = phaseb._compose_arm_case(
        arm=phaseb.B2,
        corpus_id=corpus_id,
        question=question,
        document=document,
        phase4_packet=phase4_packet,
        hydrated_candidates=v3_candidates,
        v10=v10,
    )
    _assert_v3_b2_regression(generated, v3_reference)
    claims = copy.deepcopy(generated["claims"])
    base_answer = str(v10["answer"])
    trace_unsigned = {
        "schema_version": CASE_TRACE_SCHEMA,
        "profile_version": PROFILE_VERSION,
        "arm": B1A,
        "semantic_input": {
            "corpus_id": corpus_id,
            "document_id": question["document_id"],
            "question": question["question"],
        },
        "source_phase4_packet_sha256": semantic_sha256(
            _strip_join_fields(phase4_packet)
        ),
        "base_kind": "frozen_v10",
        "base_answer_sha256": sha256_text(base_answer),
        "retrieval": copy.deepcopy(dict(retrieval_trace)),
        "authorization_mode": "existing_only",
        "render_suppressed": False,
        "composer": copy.deepcopy(
            generated["semantic_trace"]["composer"]
        ),
        "selected_candidate_ids": copy.deepcopy(
            generated["selected_candidate_ids"]
        ),
        "selected_claim_sha256": copy.deepcopy(
            generated["selected_claim_sha256"]
        ),
        "would_render_claim_sha256": copy.deepcopy(
            generated["selected_claim_sha256"]
        ),
        "v3_b2_regression_case_sha256": semantic_sha256(
            _strip_join_fields(v3_reference)
        ),
    }
    trace = {
        **trace_unsigned,
        "semantic_trace_sha256": semantic_sha256(trace_unsigned),
    }
    return {
        "schema_version": OUTPUT_SCHEMA,
        "profile_version": PROFILE_VERSION,
        "arm": B1A,
        "corpus_id": corpus_id,
        "question_id": question["question_id"],
        "question": question["question"],
        "document_id": question["document_id"],
        "base_kind": "frozen_v10",
        "base_answer_sha256": sha256_text(base_answer),
        "answer_safe_text": generated["answer_safe_text"],
        "claims": claims,
        "would_render_claims": copy.deepcopy(claims),
        "citations": copy.deepcopy(generated["citations"]),
        "selected_candidate_ids": copy.deepcopy(
            generated["selected_candidate_ids"]
        ),
        "selected_claim_sha256": copy.deepcopy(
            generated["selected_claim_sha256"]
        ),
        "would_render_claim_sha256": copy.deepcopy(
            generated["selected_claim_sha256"]
        ),
        "semantic_trace": trace,
    }


def _compose_literal_case(
    *,
    arm: str,
    corpus_id: str,
    question: Mapping[str, str],
    phase4_packet: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    retrieval_trace: Mapping[str, Any],
    v10: Mapping[str, Any],
) -> dict[str, Any]:
    configuration = ARM_CONFIGURATION[arm]
    base_answer = str(v10["answer"])
    composed = compose_phase51_b1_claims(
        question=question["question"],
        corpus_id=corpus_id,
        document_id=question["document_id"],
        candidates=candidates,
        base_answer=base_answer,
        authorization_mode=str(configuration["authorization_mode"]),
        suppress_rendering=bool(configuration["suppress_rendering"]),
    )
    claims = copy.deepcopy(list(composed["claims"]))
    would_render_claims = copy.deepcopy(
        list(composed["would_render_claims"])
    )
    append_text = str(composed["append_text"])
    answer = base_answer + (f"\n{append_text}" if append_text else "")
    selected_candidate_ids = _unique(
        [str(claim["candidate_id"]) for claim in would_render_claims]
    )
    selected_claim_sha256 = [
        str(claim["claim_sha256"]) for claim in claims
    ]
    would_render_hashes = [
        str(claim["claim_sha256"]) for claim in would_render_claims
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
        "base_kind": "frozen_v10",
        "base_answer_sha256": sha256_text(base_answer),
        "retrieval": copy.deepcopy(dict(retrieval_trace)),
        "authorization_mode": configuration["authorization_mode"],
        "render_suppressed": configuration["suppress_rendering"],
        "composer": copy.deepcopy(composed["semantic_trace"]),
        "selected_candidate_ids": selected_candidate_ids,
        "selected_claim_sha256": selected_claim_sha256,
        "would_render_claim_sha256": would_render_hashes,
        "v3_b2_regression_case_sha256": None,
    }
    trace = {
        **trace_unsigned,
        "semantic_trace_sha256": semantic_sha256(trace_unsigned),
    }
    return {
        "schema_version": OUTPUT_SCHEMA,
        "profile_version": PROFILE_VERSION,
        "arm": arm,
        "corpus_id": corpus_id,
        "question_id": question["question_id"],
        "question": question["question"],
        "document_id": question["document_id"],
        "base_kind": "frozen_v10",
        "base_answer_sha256": sha256_text(base_answer),
        "answer_safe_text": answer,
        "claims": claims,
        "would_render_claims": would_render_claims,
        "citations": [
            *copy.deepcopy(list(v10["citations"])),
            *copy.deepcopy(list(composed["citations"])),
        ],
        "selected_candidate_ids": selected_candidate_ids,
        "selected_claim_sha256": selected_claim_sha256,
        "would_render_claim_sha256": would_render_hashes,
        "semantic_trace": trace,
    }


def _validate_output(
    row: Mapping[str, Any],
    *,
    question: Mapping[str, str],
    corpus_id: str,
    v10: Mapping[str, Any],
) -> None:
    _validate_schema(_ANSWER_VALIDATOR, row, label="Phase B.1 output")
    arm = str(row["arm"])
    configuration = ARM_CONFIGURATION.get(arm)
    base_answer = str(v10["answer"])
    claims = row["claims"]
    would_render_claims = row["would_render_claims"]
    if (
        configuration is None
        or row.get("schema_version") != OUTPUT_SCHEMA
        or row.get("profile_version") != PROFILE_VERSION
        or row.get("corpus_id") != corpus_id
        or row.get("question_id") != question["question_id"]
        or row.get("question") != question["question"]
        or row.get("document_id") != question["document_id"]
        or row.get("base_kind") != "frozen_v10"
        or row.get("base_answer_sha256") != sha256_text(base_answer)
        or not str(row.get("answer_safe_text") or "").strip()
        or not str(row["answer_safe_text"]).startswith(base_answer)
        or MASK in str(row["answer_safe_text"])
        or len(claims) > MAX_CLAIMS
        or len(would_render_claims) > MAX_CLAIMS
    ):
        raise PhaseB1RunnerError("Phase B.1 output identity/safety differs")
    suppressed = bool(configuration["suppress_rendering"])
    if suppressed:
        if claims or row["answer_safe_text"] != base_answer:
            raise PhaseB1RunnerError(
                "render-suppressed arm changed the base answer"
            )
    else:
        expected_answer = base_answer + (
            "\n" + "\n".join(claim["text"] for claim in claims)
            if claims
            else ""
        )
        if (
            claims != would_render_claims
            or row["answer_safe_text"] != expected_answer
        ):
            raise PhaseB1RunnerError(
                "rendered Phase B.1 answer/claim projection differs"
            )
    expected_citations = copy.deepcopy(list(v10["citations"]))
    if arm == B1A:
        expected_citations.extend(
            phaseb._compact_citation(claim) for claim in claims
        )
    else:
        expected_citations.extend(_compact_citation(claim) for claim in claims)
    if row["citations"] != expected_citations:
        raise PhaseB1RunnerError(
            "Phase B.1 citation projection differs"
        )
    for claim in would_render_claims:
        if "authorization_kind" in claim:
            unsigned_claim = {
                key: value
                for key, value in claim.items()
                if key != "claim_sha256"
            }
            if claim.get("claim_sha256") != semantic_sha256(unsigned_claim):
                raise PhaseB1RunnerError("Phase B.1 claim hash differs")
            expected_fragments: list[str] = []
            for cell in claim["selected_cells"]:
                raw_value = str(cell["raw_value"])
                unit = str(cell["unit"])
                if (
                    cell["raw_value_sha256"] != sha256_text(raw_value)
                    or claim["text"].count(raw_value) != 1
                    or cell["authorization_kind"]
                    != claim["authorization_kind"]
                ):
                    raise PhaseB1RunnerError(
                        "Phase B.1 selected-cell literal differs"
                    )
                expected_fragments.extend(
                    numeric_fragments(
                        _rendered_cell_value(raw_value, unit)
                    )
                )
            if Counter(numeric_fragments(claim["text"])) != Counter(
                expected_fragments
            ):
                raise PhaseB1RunnerError(
                    "Phase B.1 claim numeric fragments differ"
                )
        else:
            expected_hash = semantic_sha256(
                {
                    "text": claim["text"],
                    "candidate_id": claim["candidate_id"],
                    "row_label_cell": claim["row_label_cell"],
                    "selected_cells": claim["selected_cells"],
                }
            )
            if claim.get("claim_sha256") != expected_hash:
                raise PhaseB1RunnerError(
                    "b1a Phase B v3 claim hash differs"
                )
    expected_selected = _unique(
        [claim["candidate_id"] for claim in would_render_claims]
    )
    expected_actual_hashes = [
        claim["claim_sha256"] for claim in claims
    ]
    expected_would_hashes = [
        claim["claim_sha256"] for claim in would_render_claims
    ]
    if (
        row["selected_candidate_ids"] != expected_selected
        or row["selected_claim_sha256"] != expected_actual_hashes
        or row["would_render_claim_sha256"] != expected_would_hashes
    ):
        raise PhaseB1RunnerError(
            "Phase B.1 selected/would-render identity differs"
        )
    trace = row["semantic_trace"]
    if (
        trace.get("selected_candidate_ids") != expected_selected
        or trace.get("selected_claim_sha256") != expected_actual_hashes
        or trace.get("would_render_claim_sha256") != expected_would_hashes
        or trace.get("render_suppressed") is not suppressed
        or trace.get("authorization_mode")
        != configuration["authorization_mode"]
    ):
        raise PhaseB1RunnerError("Phase B.1 trace selection differs")
    trace_unsigned = {
        key: value for key, value in trace.items()
        if key != "semantic_trace_sha256"
    }
    if trace["semantic_trace_sha256"] != semantic_sha256(trace_unsigned):
        raise PhaseB1RunnerError("Phase B.1 trace hash differs")
    _reject_forbidden_fields(row, location="phase_b1_output")


def _membership_row(output: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "question_id": output["question_id"],
        "document_id": output["document_id"],
        "question": output["question"],
        "selected_candidate_ids": output["selected_candidate_ids"],
        "would_render_claim_sha256": output["would_render_claim_sha256"],
        "rendered_claim_sha256": output["selected_claim_sha256"],
    }


def _is_literal_claim(claim: Mapping[str, Any]) -> bool:
    return claim.get("authorization_kind") == "source_cell_exact_literal"


def _literal_render_membership_row(
    output: Mapping[str, Any],
) -> dict[str, Any] | None:
    claims = [
        claim
        for claim in output["claims"]
        if isinstance(claim, Mapping) and _is_literal_claim(claim)
    ]
    if not claims:
        return None
    return {
        "question_id": output["question_id"],
        "document_id": output["document_id"],
        "question": output["question"],
        "literal_candidate_ids": _unique(
            [str(claim["candidate_id"]) for claim in claims]
        ),
        "literal_claim_sha256": [
            str(claim["claim_sha256"]) for claim in claims
        ],
    }


def _literal_membership_hash(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    semantic_rows = [
        {
            "document_id": row["document_id"],
            "question": row["question"],
            "literal_candidate_ids": row["literal_candidate_ids"],
            "literal_claim_sha256": row["literal_claim_sha256"],
        }
        for row in rows
    ]
    semantic_rows.sort(
        key=lambda row: (
            row["document_id"],
            row["question"],
            tuple(row["literal_candidate_ids"]),
            tuple(row["literal_claim_sha256"]),
        )
    )
    return semantic_sha256(semantic_rows)


def _semantic_membership_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    semantic_rows = [
        {
            "document_id": row["document_id"],
            "question": row["question"],
            "selected_candidate_ids": row["selected_candidate_ids"],
            "would_render_claim_sha256": row["would_render_claim_sha256"],
        }
        for row in rows
    ]
    semantic_rows.sort(
        key=lambda row: (
            row["document_id"],
            row["question"],
            tuple(row["selected_candidate_ids"]),
            tuple(row["would_render_claim_sha256"]),
        )
    )
    return semantic_sha256(semantic_rows)


def _arm_coverage(outputs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    overall_rendered = [row for row in outputs if row["claims"]]
    would_render = [row for row in outputs if row["would_render_claims"]]
    literal_rendered = [
        row
        for row in outputs
        if any(
            isinstance(claim, Mapping) and _is_literal_claim(claim)
            for claim in row["claims"]
        )
    ]
    literal_would_render = [
        row
        for row in outputs
        if any(
            isinstance(claim, Mapping) and _is_literal_claim(claim)
            for claim in row["would_render_claims"]
        )
    ]
    clusters = {
        (row["document_id"], row["question"])
        for row in literal_rendered
    }
    documents = {row["document_id"] for row in literal_rendered}
    would_clusters = {
        (row["document_id"], row["question"])
        for row in literal_would_render
    }
    would_documents = {
        row["document_id"] for row in literal_would_render
    }
    overall_clusters = {
        (row["document_id"], row["question"])
        for row in overall_rendered
    }
    overall_documents = {
        row["document_id"] for row in overall_rendered
    }
    routes = Counter(
        str(row["semantic_trace"]["composer"]["route"])
        for row in outputs
    )
    return {
        "cases": len(outputs),
        "nonempty_answers": sum(
            bool(row["answer_safe_text"]) for row in outputs
        ),
        "rendered_cases": len(literal_rendered),
        "rendered_semantic_tasks": len(clusters),
        "rendered_documents": len(documents),
        "rendered_claims": sum(
            sum(
                isinstance(claim, Mapping) and _is_literal_claim(claim)
                for claim in row["claims"]
            )
            for row in outputs
        ),
        "would_render_cases": len(literal_would_render),
        "would_render_semantic_tasks": len(would_clusters),
        "would_render_documents": len(would_documents),
        "would_render_claims": sum(
            sum(
                isinstance(claim, Mapping) and _is_literal_claim(claim)
                for claim in row["would_render_claims"]
            )
            for row in outputs
        ),
        "overall_rendered_cases": len(overall_rendered),
        "overall_rendered_semantic_tasks": len(overall_clusters),
        "overall_rendered_documents": len(overall_documents),
        "overall_rendered_claims": sum(
            len(row["claims"]) for row in outputs
        ),
        "overall_would_render_cases": len(would_render),
        "overall_would_render_claims": sum(
            len(row["would_render_claims"]) for row in outputs
        ),
        "route_counts": dict(sorted(routes.items())),
    }


def _code_dependency_paths(root: Path) -> dict[str, Path]:
    paths = {
        "runner": Path(__file__).resolve(),
        "core_module": (
            root / "src/finglmqa/type3_phase51_b1_compact.py"
        ),
        "answer_schema": ANSWER_SCHEMA_PATH,
        "literal_schema": LITERAL_SCHEMA_PATH,
        "literal_manifest_schema": LITERAL_MANIFEST_SCHEMA_PATH,
        "literal_document_manifest_schema": (
            LITERAL_DOCUMENT_MANIFEST_SCHEMA_PATH
        ),
        "numeric_authorization_schema": NUMERIC_SCHEMA_PATH,
        "phase_b_v3_runner": (
            root / "scripts/run_type3_phase51_phaseb.py"
        ),
        "phase_b_v3_core": (
            root / "src/finglmqa/type3_phase51_compact_tabgr.py"
        ),
        "phase_b_v3_answer_schema": phaseb.ANSWER_SCHEMA_PATH,
        "safe_v10_schema": phaseb.SAFE_V10_SCHEMA_PATH,
    }
    registered = {path.resolve() for path in paths.values()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix in {".pyc", ".pyo"}:
            try:
                path = Path(
                    importlib.util.source_from_cache(path.as_posix())
                )
            except (ValueError, NotImplementedError):
                continue
        if (
            path.suffix != ".py"
            or not path.is_relative_to(root)
            or path in registered
            or path.relative_to(root).parts[0] not in {"src", "scripts"}
        ):
            continue
        relative = path.relative_to(root).as_posix()
        key = (
            "loaded_"
            + re.sub(r"[^a-z0-9]+", "_", relative.lower()).strip("_")
        )
        if key in paths:
            raise PhaseB1RunnerError(
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
    parser.add_argument(
        "--phase4-answers",
        type=Path,
        default=DEFAULT_PHASE4_ANSWERS,
    )
    parser.add_argument(
        "--phase4-manifest",
        type=Path,
        default=DEFAULT_PHASE4_MANIFEST,
    )
    parser.add_argument("--tabgr-index-dir", type=Path)
    parser.add_argument(
        "--literal-authorization-dir",
        type=Path,
        default=DEFAULT_LITERAL_DIR,
    )
    parser.add_argument(
        "--v10-safe-dir",
        type=Path,
        default=DEFAULT_V10_SAFE_DIR,
    )
    parser.add_argument(
        "--v3-reference-dir",
        type=Path,
        default=DEFAULT_V3_DIR,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _main(argv: Sequence[str] | None = None) -> int:
    global _ACTIVE_STAGING
    args = parse_args(argv)
    root = args.root.resolve()
    manifest_paths = _guard_manifest_paths(
        ManifestPaths.defaults(
            root,
            args.corpus_id,
            args.question_profile_id,
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
        raise PhaseB1RunnerError(
            "corpus/question paths must match declared profiles"
        )
    corpus = load_corpus_profile(corpus_manifest_path)
    question_profile, questions = load_question_profile(
        question_profile_path,
        corpus_profile=corpus,
    )
    _enforce_annotation_policy(corpus, location="corpus_profile")
    _enforce_annotation_policy(
        question_profile,
        location="question_profile",
    )
    _enforce_annotation_policy(questions, location="sanitized_questions")
    if (
        corpus["corpus_id"] != args.corpus_id
        or question_profile["question_profile_id"]
        != args.question_profile_id
    ):
        raise PhaseB1RunnerError("CLI corpus/question identity differs")
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
        args.tabgr_index_dir
        or manifest_paths.tabgr_index_manifest.parent
    )
    literal_dir = guard_generator_path(args.literal_authorization_dir)
    v10_dir = guard_generator_path(args.v10_safe_dir)
    v3_dir = guard_generator_path(args.v3_reference_dir)
    literal_manifest = guard_generator_path(literal_dir / "manifest.json")
    literal_document_manifest = guard_generator_path(
        literal_dir / "document_manifest.jsonl"
    )
    literal_rejections = guard_generator_path(
        literal_dir / "rejection_audit.jsonl"
    )
    v10_paths = (
        guard_generator_path(v10_dir / "manifest.json"),
        guard_generator_path(v10_dir / "answers.jsonl"),
    )
    v3_paths = (
        guard_generator_path(v3_dir / "run_manifest.json"),
        guard_generator_path(
            v3_dir / "arms" / phaseb.B2 / "answers.jsonl"
        ),
        guard_generator_path(
            v3_dir / "arms" / phaseb.B2 / "traces.jsonl"
        ),
        guard_generator_path(
            v3_dir
            / "arms"
            / phaseb.B2
            / "selected_membership.jsonl"
        ),
    )
    input_paths: dict[str, Path] = {
        "corpus_manifest": corpus_manifest_path,
        "question_profile": question_profile_path,
        "questions": questions_path,
        "phase4_answers": phase4_path,
        "phase4_manifest": phase4_manifest_path,
        "tabgr_index_manifest": tabgr_index_dir / "manifest.json",
        "tabgr_document_manifest": (
            tabgr_index_dir / "document_manifest.jsonl"
        ),
        "tabgr_package_manifest": manifest_paths.tabgr_package_manifest,
        "literal_manifest": literal_manifest,
        "literal_document_manifest": literal_document_manifest,
        "literal_rejection_audit": literal_rejections,
        "v10_safe_manifest": v10_paths[0],
        "v10_safe_answers": v10_paths[1],
        "v10_safe_schema": phaseb.SAFE_V10_SCHEMA_PATH,
        "v3_run_manifest": v3_paths[0],
        "v3_b2_answers": v3_paths[1],
        "v3_b2_traces": v3_paths[2],
        "v3_b2_membership": v3_paths[3],
        "answer_schema": ANSWER_SCHEMA_PATH,
        "literal_schema": LITERAL_SCHEMA_PATH,
        "literal_manifest_schema": LITERAL_MANIFEST_SCHEMA_PATH,
        "literal_document_manifest_schema": (
            LITERAL_DOCUMENT_MANIFEST_SCHEMA_PATH
        ),
        "numeric_authorization_schema": NUMERIC_SCHEMA_PATH,
        **{
            f"binding_{name}": Path(value["path"]).resolve()
            for name, value in binding["manifests"].items()
        },
    }
    for path in input_paths.values():
        guarded = guard_generator_path(path)
        if not guarded.is_file():
            raise PhaseB1RunnerError(f"frozen input is missing: {guarded}")
    input_hashes_before = {
        key: sha256_file(path)
        for key, path in sorted(input_paths.items())
    }
    code_paths = _code_dependency_paths(root)
    for path in code_paths.values():
        if not path.is_file():
            raise PhaseB1RunnerError(
                f"executed dependency is missing: {path}"
            )
    code_hashes_before = {
        key: sha256_file(path)
        for key, path in sorted(code_paths.items())
    }

    phase4 = phaseb._verify_phase4(
        answers_path=phase4_path,
        manifest_path=phase4_manifest_path,
        binding=binding,
        corpus_id=args.corpus_id,
        question_profile_id=args.question_profile_id,
        questions=questions,
    )
    v10 = phaseb._verify_and_load_safe_v10(
        directory=v10_dir,
        corpus_id=args.corpus_id,
        question_profile_id=args.question_profile_id,
        question_profile_sha256=question_profile["profile_sha256"],
        questions=questions,
    )
    v3 = _verify_v3_b2_reference(
        v3_dir,
        corpus_id=args.corpus_id,
        question_profile_id=args.question_profile_id,
        questions=questions,
    )
    documents = phaseb._document_map(corpus)
    tabgr_store = BoundFullTabGRStore(
        tabgr_index_dir,
        corpus_id=args.corpus_id,
    )
    literal_store = BoundLiteralAuthorizationStore(
        literal_dir,
        corpus_id=args.corpus_id,
        corpus_profile_sha256=corpus["profile_sha256"],
    )
    retriever = Type3TabGRRetriever(
        tabgr_index_dir,
        expected_corpus_id=args.corpus_id,
    )

    protected_paths = [
        *input_paths.values(),
        phase4_path.parent,
        tabgr_index_dir,
        literal_dir,
        v10_dir,
        v3_dir,
        root / "data/corpus_package/type3" / args.corpus_id,
        root / "data/indexes/type3" / args.corpus_id,
        root / "data/facts/type3" / args.corpus_id,
        root / "data/authorizations/type3" / args.corpus_id,
    ]
    final_dir, staging = _safe_output_dir(
        args.output_dir,
        protected_paths=protected_paths,
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
            raise PhaseB1RunnerError(
                "Phase 4 packet differs from sanitized question"
            )
        document = documents[question["document_id"]]
        shortlist_existing, shortlist_trace = _phase4_shortlist(
            packet=packet,
            corpus_id=args.corpus_id,
            document=document,
            tabgr_store=tabgr_store,
            literal_store=None,
        )
        b1a = _compose_b1a_case(
            corpus_id=args.corpus_id,
            question=question,
            document=document,
            phase4_packet=packet,
            candidates=shortlist_existing,
            retrieval_trace=shortlist_trace,
            v10=v10[question_id],
            v3_reference=v3["answers"][question_id],
        )
        _validate_output(
            b1a,
            question=question,
            corpus_id=args.corpus_id,
            v10=v10[question_id],
        )
        outputs_by_arm[B1A].append(b1a)

        shortlist_literal, shortlist_literal_trace = _phase4_shortlist(
            packet=packet,
            corpus_id=args.corpus_id,
            document=document,
            tabgr_store=tabgr_store,
            literal_store=literal_store,
        )
        b1b = _compose_literal_case(
            arm=B1B,
            corpus_id=args.corpus_id,
            question=question,
            phase4_packet=packet,
            candidates=shortlist_literal,
            retrieval_trace=shortlist_literal_trace,
            v10=v10[question_id],
        )
        _validate_output(
            b1b,
            question=question,
            corpus_id=args.corpus_id,
            v10=v10[question_id],
        )
        outputs_by_arm[B1B].append(b1b)

        full_candidates, full_trace = _full_document_retrieval(
            question=question,
            document=document,
            retriever=retriever,
            tabgr_store=tabgr_store,
            literal_store=literal_store,
        )
        b1c = _compose_literal_case(
            arm=B1C,
            corpus_id=args.corpus_id,
            question=question,
            phase4_packet=packet,
            candidates=full_candidates,
            retrieval_trace=full_trace,
            v10=v10[question_id],
        )
        b1d = _compose_literal_case(
            arm=B1D,
            corpus_id=args.corpus_id,
            question=question,
            phase4_packet=packet,
            candidates=full_candidates,
            retrieval_trace=full_trace,
            v10=v10[question_id],
        )
        for arm, output in ((B1C, b1c), (B1D, b1d)):
            _validate_output(
                output,
                question=question,
                corpus_id=args.corpus_id,
                v10=v10[question_id],
            )
            outputs_by_arm[arm].append(output)
        if (
            b1c["selected_candidate_ids"]
            != b1d["selected_candidate_ids"]
            or b1c["would_render_claim_sha256"]
            != b1d["would_render_claim_sha256"]
            or b1c["would_render_claims"] != b1d["would_render_claims"]
            or b1c["semantic_trace"]["retrieval"]
            != b1d["semantic_trace"]["retrieval"]
            or b1d["claims"]
            or b1d["answer_safe_text"] != v10[question_id]["answer"]
        ):
            raise PhaseB1RunnerError(
                "b1c/b1d exact same-selection counterfactual differs"
            )

    arm_manifests: dict[str, Any] = {}
    for arm in ARMS:
        outputs = outputs_by_arm[arm]
        arm_dir = staging / "arms" / arm
        answers_path = arm_dir / "answers.jsonl"
        traces_path = arm_dir / "traces.jsonl"
        membership_path = arm_dir / "selected_membership.jsonl"
        literal_membership_path = (
            arm_dir / "literal_render_membership.jsonl"
        )
        traces = [
            {
                "question_id": row["question_id"],
                "semantic_trace": row["semantic_trace"],
            }
            for row in outputs
        ]
        membership = [
            _membership_row(row)
            for row in outputs
            if row["would_render_claims"]
        ]
        literal_membership = [
            value
            for row in outputs
            if (value := _literal_render_membership_row(row)) is not None
        ]
        _write_jsonl(answers_path, outputs)
        _write_jsonl(traces_path, traces)
        _write_jsonl(membership_path, membership)
        _write_jsonl(literal_membership_path, literal_membership)
        coverage = _arm_coverage(outputs)
        if coverage["nonempty_answers"] != len(questions):
            raise PhaseB1RunnerError(f"{arm} contains an empty answer")
        arm_manifests[arm] = {
            "configuration": ARM_CONFIGURATION[arm],
            "coverage": coverage,
            "artifacts": {
                "answers.jsonl": sha256_file(answers_path),
                "traces.jsonl": sha256_file(traces_path),
                "selected_membership.jsonl": sha256_file(
                    membership_path
                ),
                "literal_render_membership.jsonl": sha256_file(
                    literal_membership_path
                ),
            },
            "selected_membership_count": len(membership),
            "selected_membership_semantic_sha256": (
                _semantic_membership_hash(membership)
            ),
            "literal_render_membership_count": len(
                literal_membership
            ),
            "literal_render_membership_semantic_sha256": (
                _literal_membership_hash(literal_membership)
            ),
        }
    b1a_coverage = arm_manifests[B1A]["coverage"]
    if (
        b1a_coverage["overall_rendered_cases"] != 4
        or b1a_coverage["overall_rendered_semantic_tasks"] != 2
        or b1a_coverage["overall_rendered_documents"] != 2
        or b1a_coverage["overall_rendered_claims"] != 8
        or b1a_coverage["rendered_cases"] != 0
    ):
        raise PhaseB1RunnerError("b1a v3 regression coverage differs")
    b1a_v3_membership = [
        {
            "question_id": row["question_id"],
            "document_id": row["document_id"],
            "question": row["question"],
            "selected_candidate_ids": row["selected_candidate_ids"],
            "selected_claim_sha256": row["selected_claim_sha256"],
        }
        for row in outputs_by_arm[B1A]
        if row["claims"]
    ]
    if (
        phaseb._semantic_membership_hash(b1a_v3_membership)
        != EXPECTED_V3_B2_MEMBERSHIP_SHA256
    ):
        raise PhaseB1RunnerError(
            "b1a v3 regression membership semantic hash differs"
        )
    if (
        arm_manifests[B1C]["selected_membership_semantic_sha256"]
        != arm_manifests[B1D]["selected_membership_semantic_sha256"]
        or arm_manifests[B1D]["coverage"]["rendered_cases"] != 0
    ):
        raise PhaseB1RunnerError(
            "b1c/b1d run-level same-selection differs"
        )

    tabgr_store.verify_used_shards_unchanged()
    literal_store.verify_used_shards_unchanged()
    source_after = source_snapshot(corpus, workspace_root=root)
    if source_after != source_before:
        raise PhaseB1RunnerError(
            "source Markdown changed during Phase B.1 generation"
        )
    input_hashes_after = {
        key: sha256_file(path)
        for key, path in sorted(input_paths.items())
    }
    if input_hashes_after != input_hashes_before:
        raise PhaseB1RunnerError(
            "frozen input changed during Phase B.1 generation"
        )
    code_paths_after = _code_dependency_paths(root)
    if {
        key: path.resolve()
        for key, path in code_paths_after.items()
    } != {
        key: path.resolve() for key, path in code_paths.items()
    }:
        raise PhaseB1RunnerError(
            "runtime code dependency closure changed during generation"
        )
    code_hashes_after = {
        key: sha256_file(path)
        for key, path in sorted(code_paths_after.items())
    }
    if code_hashes_after != code_hashes_before:
        raise PhaseB1RunnerError(
            "code/schema changed during Phase B.1 generation"
        )

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
            "full_document_retrieval_top_k": TOP_K,
            "full_document_retrieval_ablations": [],
            "full_document_retrieval_ppr_enabled": False,
            "retrieval_authorization_blind": True,
            "maximum_claims_per_question": MAX_CLAIMS,
            "maximum_appended_characters": MAX_TOTAL_CHARACTERS,
            "document_shard_cache_size": (
                tabgr_store.maximum_cached_documents
            ),
            "literal_shard_cache_size": (
                literal_store.maximum_cached_documents
            ),
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
            for document_id, value in sorted(
                tabgr_store.used_shards.items()
            )
        },
        "used_literal_authorization_shards": {
            document_id: {
                "path": _relative(Path(value["path"]), root),
                "sha256_before": value["sha256"],
                "sha256_after": sha256_file(Path(value["path"])),
            }
            for document_id, value in sorted(
                literal_store.used_shards.items()
            )
        },
        "source_freeze": {
            "source_ref": corpus["source_ref"],
            "document_count": len(source_before),
            "source_hashes_sha256_before": semantic_sha256(source_before),
            "source_hashes_sha256_after": semantic_sha256(source_after),
            "source_unchanged": True,
        },
        "phase_b_v3_b2_reference": {
            "directory": _relative(v3["directory"], root),
            "run_manifest_sha256": sha256_file(v3["manifest_path"]),
            "answers_sha256": sha256_file(v3["answers_path"]),
            "traces_sha256": sha256_file(v3["traces_path"]),
            "membership_sha256": sha256_file(v3["membership_path"]),
            "expected_membership_semantic_sha256": (
                EXPECTED_V3_B2_MEMBERSHIP_SHA256
            ),
            "regression_failures": 0,
        },
        "literal_authorization_package": {
            "directory": _relative(literal_store.root, root),
            "manifest_fingerprint": literal_store.manifest[
                "manifest_fingerprint"
            ],
            "authorization_count": literal_store.manifest[
                "authorization_count"
            ],
            "documents": literal_store.manifest["document_count"],
        },
        "arms": arm_manifests,
        "coverage_gate": {
            "primary_arm": B1C,
            "minimum_rendered_cases": 20,
            "minimum_rendered_semantic_tasks": 15,
            "minimum_rendered_documents": 15,
            "observed": {
                "rendered_cases": arm_manifests[B1C]["coverage"][
                    "rendered_cases"
                ],
                "rendered_semantic_tasks": arm_manifests[B1C][
                    "coverage"
                ]["rendered_semantic_tasks"],
                "rendered_documents": arm_manifests[B1C]["coverage"][
                    "rendered_documents"
                ],
            },
            "passed": (
                arm_manifests[B1C]["coverage"]["rendered_cases"] >= 20
                and arm_manifests[B1C]["coverage"][
                    "rendered_semantic_tasks"
                ]
                >= 15
                and arm_manifests[B1C]["coverage"][
                    "rendered_documents"
                ]
                >= 15
            ),
        },
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
            "unauthorized_numeric_fragments": 0,
            "b1a_regression_failures": 0,
            "b1c_b1d_same_selection_failures": 0,
            "literal_authorization_hash_failures": 0,
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
                "coverage_gate_passed": manifest["coverage_gate"]["passed"],
                "rendered_cases": {
                    arm: arm_manifests[arm]["coverage"]["rendered_cases"]
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
