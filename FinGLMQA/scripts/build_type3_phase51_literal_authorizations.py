#!/usr/bin/env python3
"""Build exact source-cell TabGR literal authorizations for Type 3 Phase 5.1.

The builder is deliberately offline and corpus-scoped.  It reads frozen TabGR
structured tables and document shards, verifies original Markdown only by
hash, gives valid NumericAuthorization records strict precedence, and publishes
through a hidden sibling staging directory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
from typing import Any, Iterable, Iterator, Mapping, Sequence

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from finglmqa.type3_corpus_profile import (  # noqa: E402
    load_corpus_profile,
    sha256_file,
    source_snapshot,
)
from finglmqa.type3_tabgr_retriever import (  # noqa: E402
    TABGR_V2_BUILDER_VERSION,
    TABGR_V2_ROW_SCHEMA,
    TABGR_V2_TABLE_SCHEMA,
    flatten_headers,
    infer_period_state,
    infer_scope_state,
    infer_unit_state,
    normalize_text,
    numeric_fragments,
    semantic_sha256,
    sha256_text,
)


BUILDER_VERSION = "type3-phase51-tabgr-cell-literal-builder-v1"
PROFILE_VERSION = "tabgr-cell-literal-exact-v1"
AUTH_SCHEMA_VERSION = "finglmqa.type3.tabgr.cell_literal_authorization.v1"
MANIFEST_SCHEMA_VERSION = (
    "finglmqa.type3.tabgr.cell_literal_authorization_manifest.v1"
)
DOCUMENT_MANIFEST_SCHEMA_VERSION = (
    "finglmqa.type3.tabgr.cell_literal_authorization_document_manifest.v1"
)
REJECTION_SCHEMA_VERSION = (
    "finglmqa.type3.tabgr.cell_literal_authorization_rejection_audit.v1"
)
NUMERIC_AUTH_SCHEMA_VERSION = "finglmqa.type3.tabgr.numeric_authorization.v1"

DEFAULT_CORPUS_ROOT = (
    ROOT / "data/corpus_package/type3/annual_reports_170_v1"
)
DEFAULT_TABGR_PACKAGE = DEFAULT_CORPUS_ROOT / "tabgr_table_v2"
DEFAULT_TABGR_INDEX = ROOT / "data/indexes/type3/annual_reports_170_v1/tabgr"
DEFAULT_NUMERIC_ROOT = ROOT / "data/facts/type3/annual_reports_170_v1"
DEFAULT_OUTPUT = (
    ROOT
    / "data/authorizations/type3/annual_reports_170_v1/"
    "tabgr_cell_literal_v1"
)
CONTRACT_PATH = (
    ROOT
    / "runs/type3_a2rag_tabgr_experiment_v1/annual_reports_170_v1/"
    "phase_5_1/phase_b1_contract/literal_authorization_contract.json"
)
AUTH_SCHEMA_PATH = (
    ROOT
    / "data/schemas/type3/tabgr_cell_literal_authorization_v1.schema.json"
)
MANIFEST_SCHEMA_PATH = (
    ROOT
    / "data/schemas/type3/"
    "tabgr_cell_literal_authorization_manifest_v1.schema.json"
)
DOCUMENT_MANIFEST_SCHEMA_PATH = (
    ROOT
    / "data/schemas/type3/"
    "tabgr_cell_literal_authorization_document_manifest_v1.schema.json"
)
REJECTION_SCHEMA_PATH = (
    ROOT
    / "data/schemas/type3/"
    "tabgr_cell_literal_authorization_rejection_audit_v1.schema.json"
)
NUMERIC_AUTH_SCHEMA_PATH = (
    ROOT / "data/schemas/type3/tabgr_numeric_authorization_v1.schema.json"
)
PROFILE_MODULE_PATH = ROOT / "src/finglmqa/type3_corpus_profile.py"
TABGR_MODULE_PATH = ROOT / "src/finglmqa/type3_tabgr_retriever.py"

_ACTIVE_STAGING: Path | None = None
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_FALLBACK_HEADER_RE = re.compile(r"^第\s*\d+\s*列$")
_EXACT_NUMERIC_LITERAL_RE = re.compile(
    r"[-+]?(?:"
    r"\d+(?:\.\d+)?"
    r"|"
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|"
    r"\d{1,3}(?:，\d{3})+(?:\.\d+)?"
    r"|"
    r"\d{1,3}(?: \d{3})+(?:\.\d+)?"
    r")(?:%|％)?"
)
_FORBIDDEN_LABEL_RE = re.compile(
    r"(?:\[未经授权数值\]|不适用|不\s*适\s*用|N/?A|--+|—+|…+|\*{2,})",
    re.IGNORECASE,
)
_ANNOTATION_TOKENS = {
    "gold",
    "oracle",
    "reference",
    "score",
    "scores",
    "scorer",
    "scoring",
    "delta",
    "loss",
    "win",
}
_ANNOTATION_COMPACT_PATTERNS = (
    "goldanswer",
    "oracleanswer",
    "referenceanswer",
    "benchmarkscore",
    "pairedelta",
    "losslabel",
    "winqids",
    "answerkey",
)
_REJECTION_REASONS = (
    "header_fallback_or_ambiguous",
    "literal_not_single_numeric_fragment",
    "merged_cell",
    "not_data_cell",
    "numeric_authorization_integrity_conflict",
    "numeric_authorization_precedence",
    "period_unresolved_or_conflicting",
    "row_evidence_binding_invalid",
    "row_label_missing_or_ambiguous",
    "row_label_numeric_or_forbidden",
    "scope_conflicting",
    "semantic_slot_conflict",
    "semantic_state_mismatch",
    "source_cell_binding_mismatch",
    "table_anomaly",
    "table_not_ready",
    "table_source_binding_invalid",
    "table_span_invalid",
    "unit_unresolved_or_conflicting",
)


class LiteralAuthorizationError(ValueError):
    """Raised when the builder cannot safely publish its authorization set."""


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


def _load_json(
    path: Path,
    *,
    enforce_annotations: bool = True,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LiteralAuthorizationError(f"expected JSON object: {path}")
    if enforce_annotations:
        _enforce_annotation_policy(value, location=path.as_posix())
    return value


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise LiteralAuthorizationError(
                    f"blank JSONL row: {path}:{line_number}"
                )
            value = json.loads(line)
            if not isinstance(value, dict):
                raise LiteralAuthorizationError(
                    f"expected JSON object: {path}:{line_number}"
                )
            _enforce_annotation_policy(
                value,
                location=f"{path.as_posix()}:{line_number}",
            )
            yield value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(dict(value)))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(dict(row)))
            count += 1
    return count


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _annotation_tokens(value: object) -> tuple[str, ...]:
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value))
    return tuple(
        token.lower() for token in re.findall(r"[A-Za-z0-9]+", words)
    )


def _is_forbidden_annotation_name(value: object) -> bool:
    tokens = _annotation_tokens(value)
    compact = "".join(tokens)
    return (
        any(token in _ANNOTATION_TOKENS for token in tokens)
        or any(pattern in compact for pattern in _ANNOTATION_COMPACT_PATTERNS)
    )


def _enforce_annotation_policy(value: object, *, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _is_forbidden_annotation_name(key):
                raise LiteralAuthorizationError(
                    f"forbidden annotation key at {location}.{key}"
                )
            _enforce_annotation_policy(
                child,
                location=f"{location}.{key}",
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _enforce_annotation_policy(
                child,
                location=f"{location}[{index}]",
            )


def _guard_input_path(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    for candidate in (path.as_posix(), resolved.as_posix()):
        if _is_forbidden_annotation_name(candidate):
            raise LiteralAuthorizationError(
                f"forbidden annotation/scoring input path for {label}: {path}"
            )
    if not resolved.is_file():
        raise LiteralAuthorizationError(f"required input is missing: {path}")
    return resolved


def _safe_output_dir(
    output: Path,
    *,
    protected_paths: Sequence[Path],
) -> tuple[Path, Path]:
    final = output.resolve()
    for path in protected_paths:
        protected = path.resolve()
        if (
            final == protected
            or final.is_relative_to(protected)
            or protected.is_relative_to(final)
        ):
            raise LiteralAuthorizationError(
                f"output overlaps frozen input/code: {final} vs {protected}"
            )
    if final.exists():
        raise LiteralAuthorizationError(
            f"output namespace already exists: {final}"
        )
    staging = final.parent / f".{final.name}.staging"
    if staging.exists():
        raise LiteralAuthorizationError(
            f"stale staging namespace exists: {staging}"
        )
    return final, staging


def _validator(path: Path) -> Draft202012Validator:
    schema = _load_json(path, enforce_annotations=False)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _validate(
    validator: Draft202012Validator,
    value: Mapping[str, Any],
    *,
    label: str,
) -> None:
    errors = sorted(
        validator.iter_errors(value),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if errors:
        first = errors[0]
        location = "$" + "".join(
            f"[{part!r}]" for part in first.absolute_path
        )
        raise LiteralAuthorizationError(
            f"{label} failed closed schema at {location}: {first.message}"
        )


def _coordinate(value: object, *, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            for item in value
        )
    ):
        raise LiteralAuthorizationError(f"{label} is not a coordinate")
    return int(value[0]), int(value[1])


def _line_range(value: object, *, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 1
            for item in value
        )
        or value[0] > value[1]
    ):
        raise LiteralAuthorizationError(f"{label} is not a line range")
    return [int(value[0]), int(value[1])]


def _resolved_state(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"status", "value", "source", "candidates"}
        and value.get("status") == "resolved"
        and isinstance(value.get("value"), str)
        and bool(value["value"])
        and isinstance(value.get("source"), str)
        and bool(value["source"])
        and isinstance(value.get("candidates"), list)
        and bool(value["candidates"])
        and len(value["candidates"]) == len(set(value["candidates"]))
        and all(
            isinstance(candidate, str) and bool(candidate)
            for candidate in value["candidates"]
        )
    )


def _scope_state(value: object) -> bool:
    if _resolved_state(value):
        return True
    return (
        isinstance(value, Mapping)
        and dict(value)
        == {
            "status": "unknown",
            "value": None,
            "source": None,
            "candidates": [],
        }
    )


def _combine_state(
    primary: Mapping[str, Any],
    fallback: Mapping[str, Any],
) -> dict[str, Any]:
    return dict(primary if primary.get("status") != "unknown" else fallback)


def _origin_hash(table_id: str, origin: Mapping[str, Any]) -> str:
    return semantic_sha256(
        {
            "table_id": table_id,
            "source_coordinate": origin["source_coordinate"],
            "origin_coordinate": origin["origin_coordinate"],
            "rowspan": origin["rowspan"],
            "colspan": origin["colspan"],
            "tag": origin["tag"],
            "text": origin["text"],
        }
    )


class TableContext:
    def __init__(
        self,
        table: Mapping[str, Any],
        *,
        document: Mapping[str, Any],
        anomaly_table_ids: set[str],
    ) -> None:
        self.table = dict(table)
        self.document = dict(document)
        self.table_id = str(table.get("table_id") or "")
        self.projection_sha256 = semantic_sha256(table)
        self.reason: str | None = None
        self.origin_by_coordinate: dict[
            tuple[int, int], Mapping[str, Any]
        ] = {}
        self.processed_origins: set[tuple[int, int]] = set()

        if self.table_id in anomaly_table_ids:
            self.reason = "table_anomaly"
            return
        if (
            table.get("schema_version") != TABGR_V2_TABLE_SCHEMA
            or table.get("builder_version") != TABGR_V2_BUILDER_VERSION
            or table.get("parse_status") != "ready"
            or table.get("failure_reason") is not None
        ):
            self.reason = "table_not_ready"
            return
        expected_source = (
            f"refs/type3_corpora/{table.get('corpus_id')}/"
            f"{document['source_markdown']}"
        )
        source_binding = table.get("source_binding")
        if (
            table.get("document_id") != document["document_id"]
            or table.get("source_markdown") != expected_source
            or not isinstance(source_binding, Mapping)
            or source_binding.get("status") != "exact"
            or source_binding.get("source_sha256")
            != document["source_sha256"]
            or source_binding.get("line_range")
            != table.get("table_line_range")
            or table.get("corpus_id") is None
        ):
            self.reason = "table_source_binding_invalid"
            return
        span = table.get("span_validation")
        matrix = table.get("matrix")
        if (
            not isinstance(span, Mapping)
            or span.get("status") != "matrix_exact"
            or span.get("parser_overwrite_count") != 0
            or not isinstance(matrix, list)
            or span.get("matrix_sha256") != semantic_sha256(matrix)
        ):
            self.reason = "table_span_invalid"
            return
        expected_table_hash = semantic_sha256(
            {
                "corpus_id": table["corpus_id"],
                "document_id": table["document_id"],
                "table_id": self.table_id,
                "raw_markdown_sha1": table.get("raw_markdown_sha1"),
                "matrix": matrix,
            }
        )
        if (
            table.get("table_sha256") != expected_table_hash
            or not isinstance(table.get("table_index"), int)
            or table["table_index"] < 1
            or not isinstance(table.get("data_start_column"), int)
            or table["data_start_column"] < 0
        ):
            self.reason = "table_source_binding_invalid"
            return

        width = max(
            (
                len(row)
                for row in matrix
                if isinstance(row, list)
            ),
            default=0,
        )
        origins = table.get("origin_cells")
        if not isinstance(origins, list):
            self.reason = "table_span_invalid"
            return
        for raw in origins:
            if not isinstance(raw, Mapping):
                self.reason = "table_span_invalid"
                return
            try:
                origin_coordinate = _coordinate(
                    raw.get("origin_coordinate"),
                    label="origin.origin_coordinate",
                )
                _coordinate(
                    raw.get("source_coordinate"),
                    label="origin.source_coordinate",
                )
                rowspan = int(raw.get("rowspan"))
                colspan = int(raw.get("colspan"))
            except (LiteralAuthorizationError, TypeError, ValueError):
                self.reason = "table_span_invalid"
                return
            if (
                rowspan < 1
                or colspan < 1
                or raw.get("cell_hash") != _origin_hash(self.table_id, raw)
                or not isinstance(raw.get("text"), str)
            ):
                self.reason = "table_span_invalid"
                return
            for row_index in range(
                origin_coordinate[0],
                origin_coordinate[0] + rowspan,
            ):
                for column_index in range(
                    origin_coordinate[1],
                    origin_coordinate[1] + colspan,
                ):
                    if row_index >= len(matrix) or column_index >= width:
                        self.reason = "table_span_invalid"
                        return
                    coordinate = (row_index, column_index)
                    if coordinate in self.origin_by_coordinate:
                        self.reason = "table_span_invalid"
                        return
                    self.origin_by_coordinate[coordinate] = raw

    @property
    def matrix(self) -> list[list[str]]:
        value = self.table["matrix"]
        assert isinstance(value, list)
        return value

    @property
    def data_start_column(self) -> int:
        return int(self.table["data_start_column"])

    def origin(self, coordinate: tuple[int, int]) -> Mapping[str, Any] | None:
        return self.origin_by_coordinate.get(coordinate)


def _source_cell(
    coordinate: tuple[int, int],
    origin: Mapping[str, Any],
) -> dict[str, Any]:
    literal = str(origin["text"])
    return {
        "coordinate": list(coordinate),
        "origin_coordinate": list(origin["origin_coordinate"]),
        "origin_cell_hash": str(origin["cell_hash"]),
        "source_cell_literal": literal,
        "source_cell_literal_sha256": sha256_text(literal),
    }


def _validate_row_binding(
    row: Mapping[str, Any],
    *,
    context: TableContext,
    corpus_id: str,
) -> bool:
    table = context.table
    if (
        row.get("schema_version") != TABGR_V2_ROW_SCHEMA
        or row.get("builder_version") != TABGR_V2_BUILDER_VERSION
        or row.get("record_type") != "table_row"
        or row.get("corpus_id") != corpus_id
        or row.get("document_id") != table.get("document_id")
        or row.get("table_id") != table.get("table_id")
        or row.get("table_index") != table.get("table_index")
        or row.get("table_sha256") != table.get("table_sha256")
        or row.get("source_markdown") != table.get("source_markdown")
        or row.get("table_line_range") != table.get("table_line_range")
        or row.get("evidence_id")
        not in set(table.get("row_evidence_ids") or ())
        or not isinstance(row.get("row_index"), int)
        or not isinstance(row.get("row_path"), list)
        or not isinstance(row.get("cells"), list)
    ):
        return False
    evidence_unsigned = {
        "corpus_id": corpus_id,
        "document_id": row["document_id"],
        "table_id": row["table_id"],
        "row_index": row["row_index"],
        "row_path": row["row_path"],
        "cells": row["cells"],
    }
    expected_id = (
        f"{corpus_id}:{table['table_id']}:"
        f"r{int(row['row_index']):04d}:"
        f"{semantic_sha256(evidence_unsigned)[:12]}"
    )
    return row.get("evidence_id") == expected_id


def _row_label_binding(
    row: Mapping[str, Any],
    *,
    context: TableContext,
) -> tuple[dict[str, Any] | None, str | None]:
    row_path = row.get("row_path")
    if (
        not isinstance(row_path, list)
        or not row_path
        or any(not isinstance(value, str) or not value for value in row_path)
    ):
        return None, "row_label_missing_or_ambiguous"
    label = row_path[-1]
    if (
        numeric_fragments(label)
        or _FORBIDDEN_LABEL_RE.search(label)
        or _FALLBACK_HEADER_RE.fullmatch(label)
    ):
        return None, "row_label_numeric_or_forbidden"
    row_index = int(row["row_index"])
    matches: dict[tuple[Any, ...], dict[str, Any]] = {}
    for cell in row["cells"]:
        if (
            not isinstance(cell, Mapping)
            or cell.get("raw_value") != label
            or cell.get("numeric_status") != "not_numeric"
            or numeric_fragments(str(cell.get("raw_value") or ""))
        ):
            continue
        try:
            coordinate = _coordinate(
                cell.get("coordinate"),
                label="row_label.coordinate",
            )
            origin_coordinate = _coordinate(
                cell.get("origin_coordinate"),
                label="row_label.origin_coordinate",
            )
        except LiteralAuthorizationError:
            continue
        origin = context.origin(coordinate)
        if (
            coordinate[0] != row_index
            or origin is None
            or tuple(origin["origin_coordinate"]) != origin_coordinate
            or origin.get("cell_hash") != cell.get("origin_cell_hash")
            or origin.get("text") != label
            or cell.get("raw_value_sha256") != sha256_text(label)
        ):
            continue
        binding = _source_cell(coordinate, origin)
        matches[
            (
                tuple(binding["origin_coordinate"]),
                binding["origin_cell_hash"],
                binding["source_cell_literal_sha256"],
            )
        ] = binding
    if len(matches) != 1:
        return None, "row_label_missing_or_ambiguous"
    return next(iter(matches.values())), None


def _header_binding(
    row: Mapping[str, Any],
    *,
    context: TableContext,
    column: int,
) -> tuple[dict[str, Any] | None, str | None]:
    active_rows = row.get("active_header_rows")
    headers = row.get("flattened_column_headers")
    if (
        not isinstance(active_rows, list)
        or not active_rows
        or len(active_rows) != len(set(active_rows))
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            for item in active_rows
        )
        or not isinstance(headers, list)
        or column >= len(headers)
    ):
        return None, "header_fallback_or_ambiguous"
    recomputed = flatten_headers(context.matrix, active_rows)
    if column >= len(recomputed):
        return None, "header_fallback_or_ambiguous"
    header = recomputed[column]
    if (
        header != headers[column]
        or _FALLBACK_HEADER_RE.fullmatch(header)
        or not header
    ):
        return None, "header_fallback_or_ambiguous"
    source_cells: list[dict[str, Any]] = []
    for row_index in active_rows:
        if (
            row_index >= len(context.matrix)
            or column >= len(context.matrix[row_index])
        ):
            continue
        value = context.matrix[row_index][column]
        if not normalize_text(value):
            continue
        origin = context.origin((row_index, column))
        if origin is None or origin.get("text") != value:
            return None, "header_fallback_or_ambiguous"
        source_cells.append(_source_cell((row_index, column), origin))
    if not source_cells:
        return None, "header_fallback_or_ambiguous"
    return {
        "column_index": column,
        "active_header_rows": list(active_rows),
        "flattened_column_header": header,
        "flattened_column_header_sha256": sha256_text(header),
        "header_source_cells": source_cells,
        "fallback_used": False,
    }, None


def _numeric_precedence_reason(
    *,
    row: Mapping[str, Any],
    cell: Mapping[str, Any],
    numeric_by_coordinate: Mapping[
        tuple[str, str, int, int], Sequence[Mapping[str, Any]]
    ],
) -> str | None:
    coordinate = _coordinate(
        cell.get("coordinate"),
        label="numeric_precedence.coordinate",
    )
    key = (
        str(row["document_id"]),
        str(row["table_id"]),
        coordinate[0],
        coordinate[1],
    )
    matches = list(numeric_by_coordinate.get(key) or ())
    row_ids = cell.get("authorization_ids")
    if not isinstance(row_ids, list):
        return "numeric_authorization_integrity_conflict"
    if not matches and not row_ids:
        return None
    if len(matches) != 1 or len(row_ids) != 1:
        return "numeric_authorization_integrity_conflict"
    authorization = matches[0]
    literal = str(cell.get("raw_value") or "")
    expected = {
        "authorization_id": row_ids[0],
        "corpus_id": row["corpus_id"],
        "document_id": row["document_id"],
        "table_id": row["table_id"],
        "table_sha256": row["table_sha256"],
        "source_markdown": row["source_markdown"],
        "table_line_range": row["table_line_range"],
        "cell_coordinate": list(coordinate),
        "raw_value": literal,
        "raw_value_sha256": sha256_text(literal),
        "allowed_renderings": [literal],
    }
    if any(authorization.get(key) != value for key, value in expected.items()):
        return "numeric_authorization_integrity_conflict"
    return "numeric_authorization_precedence"


def _hash_chain(record: Mapping[str, Any]) -> tuple[str, str, str]:
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


def validate_record_hash_chain(record: Mapping[str, Any]) -> None:
    source_hash, semantic_hash, authorization_hash = _hash_chain(record)
    if record.get("source_cell_binding_sha256") != source_hash:
        raise LiteralAuthorizationError(
            "source_cell_binding_sha256 differs"
        )
    if record.get("semantic_context_sha256") != semantic_hash:
        raise LiteralAuthorizationError("semantic_context_sha256 differs")
    if record.get("authorization_sha256") != authorization_hash:
        raise LiteralAuthorizationError("authorization_sha256 differs")
    if record.get("authorization_id") != (
        "t3tabgr-lit-" + authorization_hash[:24]
    ):
        raise LiteralAuthorizationError("authorization_id differs")


def _candidate_record(
    *,
    corpus_id: str,
    document: Mapping[str, Any],
    context: TableContext,
    row: Mapping[str, Any],
    cell: Mapping[str, Any],
    numeric_by_coordinate: Mapping[
        tuple[str, str, int, int], Sequence[Mapping[str, Any]]
    ],
    auth_validator: Draft202012Validator,
) -> tuple[dict[str, Any] | None, str | None]:
    if context.reason is not None:
        return None, context.reason
    if not _validate_row_binding(row, context=context, corpus_id=corpus_id):
        return None, "row_evidence_binding_invalid"
    try:
        coordinate = _coordinate(
            cell.get("coordinate"),
            label="cell.coordinate",
        )
        origin_coordinate = _coordinate(
            cell.get("origin_coordinate"),
            label="cell.origin_coordinate",
        )
    except LiteralAuthorizationError:
        return None, "source_cell_binding_mismatch"
    origin = context.origin(coordinate)
    if origin is None:
        return None, "source_cell_binding_mismatch"
    physical_origin = tuple(origin["origin_coordinate"])
    if physical_origin != origin_coordinate:
        return None, "source_cell_binding_mismatch"
    if coordinate[1] < context.data_start_column:
        return None, "not_data_cell"
    if (
        int(origin["rowspan"]) != 1
        or int(origin["colspan"]) != 1
        or coordinate != origin_coordinate
    ):
        return None, "merged_cell"
    literal = str(origin["text"])
    if (
        cell.get("raw_value") != literal
        or cell.get("raw_value_sha256") != sha256_text(literal)
        or cell.get("origin_cell_hash") != origin.get("cell_hash")
        or context.matrix[coordinate[0]][coordinate[1]] != literal
    ):
        return None, "source_cell_binding_mismatch"
    precedence_reason = _numeric_precedence_reason(
        row=row,
        cell=cell,
        numeric_by_coordinate=numeric_by_coordinate,
    )
    if precedence_reason is not None:
        return None, precedence_reason
    fragments = numeric_fragments(literal)
    if (
        not literal
        or len(literal) > 96
        or fragments != [literal]
        or _EXACT_NUMERIC_LITERAL_RE.fullmatch(literal) is None
        or _FORBIDDEN_LABEL_RE.search(literal)
    ):
        return None, "literal_not_single_numeric_fragment"
    if (
        cell.get("numeric_status") != "unauthorized"
        or cell.get("authorization_ids") != []
    ):
        return None, "numeric_authorization_integrity_conflict"

    row_label_cell, reason = _row_label_binding(row, context=context)
    if reason is not None or row_label_cell is None:
        return None, reason
    header_binding, reason = _header_binding(
        row,
        context=context,
        column=coordinate[1],
    )
    if reason is not None or header_binding is None:
        return None, reason
    if cell.get("column_header") != header_binding["flattened_column_header"]:
        return None, "header_fallback_or_ambiguous"

    period = cell.get("period")
    unit = cell.get("unit")
    scope = cell.get("accounting_scope")
    if not _resolved_state(period):
        return None, "period_unresolved_or_conflicting"
    if not _resolved_state(unit):
        return None, "unit_unresolved_or_conflicting"
    if literal.endswith(("%", "％")) and unit.get("value") != "%":
        return None, "unit_unresolved_or_conflicting"
    if not _scope_state(scope):
        return None, "scope_conflicting"
    semantic_states = row.get("semantic_states")
    if not isinstance(semantic_states, Mapping):
        return None, "semantic_state_mismatch"
    column_key = str(coordinate[1])
    for group, selected in (
        ("period_by_column", period),
        ("unit_by_column", unit),
        ("accounting_scope_by_column", scope),
    ):
        by_column = semantic_states.get(group)
        if (
            not isinstance(by_column, Mapping)
            or by_column.get(column_key) != selected
        ):
            return None, "semantic_state_mismatch"

    header = header_binding["flattened_column_header"]
    inferred_period = infer_period_state(
        header,
        int(document["report_year"]),
    )
    if inferred_period != period:
        return None, "period_unresolved_or_conflicting"
    table_unit = context.table.get("unit")
    if not isinstance(table_unit, Mapping):
        return None, "unit_unresolved_or_conflicting"
    inferred_unit = _combine_state(infer_unit_state(header), table_unit)
    if inferred_unit != unit:
        return None, "unit_unresolved_or_conflicting"
    table_scope = context.table.get("accounting_scope")
    if not isinstance(table_scope, Mapping):
        return None, "scope_conflicting"
    inferred_scope = _combine_state(infer_scope_state(header), table_scope)
    if inferred_scope != scope:
        return None, "scope_conflicting"

    row_path = list(row["row_path"])
    record: dict[str, Any] = {
        "schema_version": AUTH_SCHEMA_VERSION,
        "authorization_kind": "source_cell_exact_literal",
        "authorization_id": "",
        "corpus_id": corpus_id,
        "document_id": document["document_id"],
        "row_evidence_id": row["evidence_id"],
        "source_binding": {
            "source_markdown": context.table["source_markdown"],
            "source_sha256": document["source_sha256"],
            "table_id": context.table["table_id"],
            "table_index": context.table["table_index"],
            "table_sha256": context.table["table_sha256"],
            "table_line_range": _line_range(
                context.table["table_line_range"],
                label="table.table_line_range",
            ),
            "structured_table_projection_sha256": (
                context.projection_sha256
            ),
            "row_evidence_projection_sha256": semantic_sha256(row),
        },
        "row_binding": {
            "row_index": row["row_index"],
            "row_path": row_path,
            "row_path_sha256": semantic_sha256(row_path),
            "row_label_cell": row_label_cell,
        },
        "header_binding": header_binding,
        "cell_binding": {
            "coordinate": list(coordinate),
            "origin_coordinate": list(origin_coordinate),
            "origin_cell_hash": origin["cell_hash"],
            "origin_rowspan": 1,
            "origin_colspan": 1,
            "source_cell_literal": literal,
            "source_cell_literal_sha256": sha256_text(literal),
            "numeric_fragments": fragments,
        },
        "semantic_context": {
            "period": dict(period),
            "unit": dict(unit),
            "accounting_scope": dict(scope),
        },
        "grant": {
            "literal_role": "answer_surface_only",
            "canonical_fact_status": "not_a_canonical_fact",
            "allowed_renderings": [literal],
            "normalization_allowed": False,
            "rounding_allowed": False,
            "arithmetic_allowed": False,
            "unit_conversion_allowed": False,
            "percentage_conversion_allowed": False,
            "whole_row_rendering_allowed": False,
            "mask_rendering_allowed": False,
        },
        "source_cell_binding_sha256": "",
        "semantic_context_sha256": "",
        "authorization_sha256": "",
    }
    source_hash, semantic_hash, authorization_hash = _hash_chain(record)
    record["source_cell_binding_sha256"] = source_hash
    record["semantic_context_sha256"] = semantic_hash
    record["authorization_sha256"] = authorization_hash
    record["authorization_id"] = "t3tabgr-lit-" + authorization_hash[:24]
    _validate(auth_validator, record, label="literal authorization")
    validate_record_hash_chain(record)
    return record, None


def _semantic_slot(record: Mapping[str, Any]) -> tuple[Any, ...]:
    scope = record["semantic_context"]["accounting_scope"]
    return (
        record["corpus_id"],
        record["document_id"],
        normalize_text(
            record["row_binding"]["row_label_cell"]["source_cell_literal"]
        ),
        record["header_binding"]["flattened_column_header"],
        record["semantic_context"]["period"]["value"],
        record["semantic_context"]["unit"]["value"],
        scope["value"] if scope["status"] == "resolved" else None,
    )


class RejectionAudit:
    def __init__(self, corpus_id: str) -> None:
        self.corpus_id = corpus_id
        self.counts: Counter[str] = Counter()
        self.examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def add(
        self,
        reason: str,
        *,
        document_id: str,
        row_evidence_id: str,
        table_id: str,
        coordinate: Sequence[int],
        raw_value_sha256: str,
    ) -> None:
        if reason not in _REJECTION_REASONS:
            raise LiteralAuthorizationError(
                f"unknown rejection reason: {reason}"
            )
        self.counts[reason] += 1
        if len(self.examples[reason]) < 20:
            self.examples[reason].append(
                {
                    "document_id": document_id,
                    "row_evidence_id": row_evidence_id,
                    "table_id": table_id,
                    "coordinate": list(coordinate),
                    "raw_value_sha256": raw_value_sha256,
                }
            )

    def add_record(self, reason: str, record: Mapping[str, Any]) -> None:
        self.add(
            reason,
            document_id=record["document_id"],
            row_evidence_id=record["row_evidence_id"],
            table_id=record["source_binding"]["table_id"],
            coordinate=record["cell_binding"]["coordinate"],
            raw_value_sha256=record["cell_binding"][
                "source_cell_literal_sha256"
            ],
        )

    def rows(self) -> list[dict[str, Any]]:
        return [
            {
                "schema_version": REJECTION_SCHEMA_VERSION,
                "corpus_id": self.corpus_id,
                "reason": reason,
                "rejected_count": self.counts[reason],
                "examples": self.examples[reason],
            }
            for reason in sorted(self.counts)
        ]

    @property
    def total(self) -> int:
        return sum(self.counts.values())


class StructuredCursor:
    def __init__(
        self,
        path: Path,
        *,
        documents: Mapping[str, Mapping[str, Any]],
        anomaly_table_ids: set[str],
    ) -> None:
        self._iterator = _iter_jsonl(path)
        self._documents = documents
        self._anomaly_table_ids = anomaly_table_ids
        self.current: TableContext | None = None
        self.count = 0
        self._advance()

    def _advance(self) -> None:
        try:
            table = next(self._iterator)
        except StopIteration:
            self.current = None
            return
        document_id = str(table.get("document_id") or "")
        document = self._documents.get(document_id)
        if document is None:
            raise LiteralAuthorizationError(
                "structured table references unknown document"
            )
        self.current = TableContext(
            table,
            document=document,
            anomaly_table_ids=self._anomaly_table_ids,
        )
        self.count += 1

    def seek(self, table_id: str) -> TableContext:
        while self.current is not None and self.current.table_id != table_id:
            self._advance()
        if self.current is None:
            raise LiteralAuthorizationError(
                f"row shard table not found in structured tables: {table_id}"
            )
        return self.current

    def finish(self) -> None:
        while self.current is not None:
            self._advance()


def _load_document_manifest(
    path: Path,
    *,
    index_root: Path,
    corpus_id: str,
    documents: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    rows = list(_iter_jsonl(path))
    expected_ids = [document["document_id"] for document in documents]
    actual_ids = [row.get("document_id") for row in rows]
    if actual_ids != expected_ids:
        raise LiteralAuthorizationError(
            "TabGR document manifest order/binding differs from corpus"
        )
    shard_paths: dict[str, Path] = {}
    expected_fields = {
        "schema_version",
        "document_id",
        "record_count",
        "table_count",
        "shard_path",
        "shard_sha256",
    }
    for row in rows:
        if set(row) != expected_fields:
            raise LiteralAuthorizationError(
                "TabGR document manifest is not closed"
            )
        shard_rel = str(row["shard_path"])
        pure = PurePosixPath(shard_rel)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or pure.as_posix() != shard_rel
        ):
            raise LiteralAuthorizationError(
                "TabGR document shard path is unsafe"
            )
        shard = (index_root / shard_rel).resolve()
        if not shard.is_relative_to(index_root.resolve()):
            raise LiteralAuthorizationError(
                "TabGR document shard escapes index root"
            )
        _guard_input_path(shard, label="tabgr_document_shard")
        if sha256_file(shard) != row["shard_sha256"]:
            raise LiteralAuthorizationError(
                f"TabGR document shard hash differs: {row['document_id']}"
            )
        shard_paths[str(row["document_id"])] = shard
    return rows, shard_paths


def _load_numeric_authorizations(
    path: Path,
    *,
    corpus_id: str,
    validator: Draft202012Validator,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str, int, int], list[dict[str, Any]]],
]:
    rows: list[dict[str, Any]] = []
    by_coordinate: dict[
        tuple[str, str, int, int], list[dict[str, Any]]
    ] = defaultdict(list)
    ids: set[str] = set()
    for row in _iter_jsonl(path):
        _validate(validator, row, label="NumericAuthorization")
        if (
            row.get("schema_version") != NUMERIC_AUTH_SCHEMA_VERSION
            or row.get("corpus_id") != corpus_id
        ):
            raise LiteralAuthorizationError(
                "NumericAuthorization corpus/version differs"
            )
        authorization_id = str(row["authorization_id"])
        if authorization_id in ids:
            raise LiteralAuthorizationError(
                "duplicate NumericAuthorization id"
            )
        ids.add(authorization_id)
        coordinate = _coordinate(
            row["cell_coordinate"],
            label="NumericAuthorization.cell_coordinate",
        )
        key = (
            str(row["document_id"]),
            str(row["table_id"]),
            coordinate[0],
            coordinate[1],
        )
        by_coordinate[key].append(row)
        rows.append(row)
    return rows, by_coordinate


def _verify_manifests(
    *,
    profile: Mapping[str, Any],
    package_manifest: Mapping[str, Any],
    index_manifest: Mapping[str, Any],
    numeric_manifest: Mapping[str, Any],
    paths: Mapping[str, Path],
    document_manifest_rows: Sequence[Mapping[str, Any]],
    numeric_rows: Sequence[Mapping[str, Any]],
) -> None:
    corpus_id = profile["corpus_id"]
    if (
        package_manifest.get("schema_version")
        != "finglmqa.type3.tabgr.package_manifest.v2"
        or package_manifest.get("corpus_id") != corpus_id
        or package_manifest.get("corpus_profile_sha256")
        != profile["profile_sha256"]
        or index_manifest.get("schema_version")
        != "finglmqa.type3.tabgr.lexical_index_manifest.v2"
        or index_manifest.get("corpus_id") != corpus_id
        or index_manifest.get("corpus_profile_sha256")
        != profile["profile_sha256"]
        or numeric_manifest.get("schema_version")
        != "finglmqa.type3.tabgr.fact_authorization_manifest.v1"
        or numeric_manifest.get("corpus_id") != corpus_id
        or numeric_manifest.get("corpus_profile_sha256")
        != profile["profile_sha256"]
    ):
        raise LiteralAuthorizationError("frozen manifest corpus binding differs")
    artifacts = package_manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise LiteralAuthorizationError("package artifact manifest is invalid")
    expected_hashes = {
        "structured_tables_sha256": sha256_file(paths["structured_tables"]),
        "anomaly_audit_sha256": sha256_file(paths["anomaly_audit"]),
        "selected_fact_authorizations_sha256": sha256_file(
            paths["numeric_authorizations"]
        ),
        "document_manifest_sha256": sha256_file(paths["document_manifest"]),
    }
    if any(artifacts.get(key) != value for key, value in expected_hashes.items()):
        raise LiteralAuthorizationError("frozen package artifact hash differs")
    if (
        index_manifest.get("document_manifest_sha256")
        != expected_hashes["document_manifest_sha256"]
        or index_manifest.get("document_count")
        != len(document_manifest_rows)
        or numeric_manifest.get("authorizations_sha256")
        != expected_hashes["selected_fact_authorizations_sha256"]
        or numeric_manifest.get("selected_fact_count") != len(numeric_rows)
    ):
        raise LiteralAuthorizationError("frozen index/fact manifest differs")


def _artifact_binding(
    name: str,
    path: Path,
    *,
    root: Path,
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "name": name,
        "path": _relative(path, root),
        "sha256_before": before[name],
        "sha256_after": after[name],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--corpus-profile",
        type=Path,
        default=DEFAULT_CORPUS_ROOT / "corpus_manifest.json",
    )
    parser.add_argument(
        "--package-manifest",
        type=Path,
        default=DEFAULT_TABGR_PACKAGE / "manifest.json",
    )
    parser.add_argument(
        "--index-manifest",
        type=Path,
        default=DEFAULT_TABGR_INDEX / "manifest.json",
    )
    parser.add_argument(
        "--document-manifest",
        type=Path,
        default=DEFAULT_TABGR_INDEX / "document_manifest.jsonl",
    )
    parser.add_argument(
        "--structured-tables",
        type=Path,
        default=DEFAULT_TABGR_PACKAGE / "structured_tables.jsonl",
    )
    parser.add_argument(
        "--anomaly-audit",
        type=Path,
        default=DEFAULT_TABGR_PACKAGE / "anomaly_audit.jsonl",
    )
    parser.add_argument(
        "--numeric-manifest",
        type=Path,
        default=DEFAULT_NUMERIC_ROOT / "manifest.json",
    )
    parser.add_argument(
        "--numeric-authorizations",
        type=Path,
        default=DEFAULT_NUMERIC_ROOT / "selected_fact_authorizations.jsonl",
    )
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--auth-schema", type=Path, default=AUTH_SCHEMA_PATH)
    parser.add_argument(
        "--manifest-schema",
        type=Path,
        default=MANIFEST_SCHEMA_PATH,
    )
    parser.add_argument(
        "--document-manifest-schema",
        type=Path,
        default=DOCUMENT_MANIFEST_SCHEMA_PATH,
    )
    parser.add_argument(
        "--rejection-schema",
        type=Path,
        default=REJECTION_SCHEMA_PATH,
    )
    parser.add_argument(
        "--numeric-auth-schema",
        type=Path,
        default=NUMERIC_AUTH_SCHEMA_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _main(argv: Sequence[str] | None = None) -> int:
    global _ACTIVE_STAGING
    args = parse_args(argv)
    root = args.root.resolve()
    initial_paths = {
        "corpus_profile": args.corpus_profile,
        "package_manifest": args.package_manifest,
        "index_manifest": args.index_manifest,
        "document_manifest": args.document_manifest,
        "structured_tables": args.structured_tables,
        "anomaly_audit": args.anomaly_audit,
        "numeric_manifest": args.numeric_manifest,
        "numeric_authorizations": args.numeric_authorizations,
        "frozen_contract": args.contract,
    }
    code_paths = {
        "builder": Path(__file__),
        "corpus_profile_module": PROFILE_MODULE_PATH,
        "tabgr_retriever_module": TABGR_MODULE_PATH,
        "authorization_schema": args.auth_schema,
        "manifest_schema": args.manifest_schema,
        "document_manifest_schema": args.document_manifest_schema,
        "rejection_schema": args.rejection_schema,
        "numeric_authorization_schema": args.numeric_auth_schema,
    }
    resolved_inputs = {
        name: _guard_input_path(path, label=name)
        for name, path in initial_paths.items()
    }
    resolved_code = {
        name: _guard_input_path(path, label=name)
        for name, path in code_paths.items()
    }
    final_dir, staging = _safe_output_dir(
        args.output_dir,
        protected_paths=[
            *resolved_inputs.values(),
            *resolved_code.values(),
        ],
    )

    auth_validator = _validator(resolved_code["authorization_schema"])
    manifest_validator = _validator(resolved_code["manifest_schema"])
    document_manifest_validator = _validator(
        resolved_code["document_manifest_schema"]
    )
    rejection_validator = _validator(resolved_code["rejection_schema"])
    numeric_validator = _validator(
        resolved_code["numeric_authorization_schema"]
    )
    contract = _load_json(
        resolved_inputs["frozen_contract"],
        enforce_annotations=False,
    )
    contract_schema = contract.get("closed_record_schema")
    if contract_schema != _load_json(
        resolved_code["authorization_schema"],
        enforce_annotations=False,
    ):
        raise LiteralAuthorizationError(
            "authorization schema differs from frozen Phase B.1 contract"
        )

    profile = load_corpus_profile(resolved_inputs["corpus_profile"])
    corpus_id = profile["corpus_id"]
    documents = profile["documents"]
    document_by_id = {
        document["document_id"]: document for document in documents
    }
    package_manifest = _load_json(resolved_inputs["package_manifest"])
    index_manifest = _load_json(resolved_inputs["index_manifest"])
    numeric_manifest = _load_json(resolved_inputs["numeric_manifest"])
    document_manifest_rows, shard_paths = _load_document_manifest(
        resolved_inputs["document_manifest"],
        index_root=resolved_inputs["document_manifest"].parent,
        corpus_id=corpus_id,
        documents=documents,
    )
    for row in document_manifest_rows:
        if row["schema_version"] != "finglmqa.type3.tabgr.document_shard.v2":
            raise LiteralAuthorizationError(
                "TabGR document manifest schema differs"
            )
    numeric_rows, numeric_by_coordinate = _load_numeric_authorizations(
        resolved_inputs["numeric_authorizations"],
        corpus_id=corpus_id,
        validator=numeric_validator,
    )
    _verify_manifests(
        profile=profile,
        package_manifest=package_manifest,
        index_manifest=index_manifest,
        numeric_manifest=numeric_manifest,
        paths=resolved_inputs,
        document_manifest_rows=document_manifest_rows,
        numeric_rows=numeric_rows,
    )
    anomalies = list(_iter_jsonl(resolved_inputs["anomaly_audit"]))
    anomaly_table_ids = {str(row.get("table_id") or "") for row in anomalies}
    if len(anomaly_table_ids) != len(anomalies):
        raise LiteralAuthorizationError("anomaly audit contains duplicate table ids")

    all_input_paths = dict(resolved_inputs)
    for document_id, shard in shard_paths.items():
        all_input_paths[f"tabgr_shard_{sha256_text(document_id)[:20]}"] = shard
    input_hashes_before = {
        name: sha256_file(path)
        for name, path in sorted(all_input_paths.items())
    }
    code_hashes_before = {
        name: sha256_file(path)
        for name, path in sorted(resolved_code.items())
    }
    sources_before = source_snapshot(profile, workspace_root=root)

    _ACTIVE_STAGING = staging
    staging.mkdir(parents=True)
    (staging / "documents").mkdir()

    audit = RejectionAudit(corpus_id)
    counts: Counter[str] = Counter()
    cursor = StructuredCursor(
        resolved_inputs["structured_tables"],
        documents=document_by_id,
        anomaly_table_ids=anomaly_table_ids,
    )
    document_rows_out: list[dict[str, Any]] = []
    seen_authorization_ids: dict[str, str] = {}

    for manifest_row, document in zip(document_manifest_rows, documents):
        document_id = str(document["document_id"])
        shard = shard_paths[document_id]
        records: list[dict[str, Any]] = []
        shard_record_count = 0
        shard_table_count = 0
        row_count = 0
        current_table_id: str | None = None
        current_context: TableContext | None = None
        for shard_record in _iter_jsonl(shard):
            shard_record_count += 1
            if shard_record.get("document_id") != document_id:
                raise LiteralAuthorizationError(
                    "TabGR document shard crosses document boundary"
                )
            record_type = shard_record.get("record_type")
            if record_type == "table":
                shard_table_count += 1
                continue
            if record_type != "table_row":
                raise LiteralAuthorizationError(
                    "TabGR document shard has unknown record type"
                )
            row_count += 1
            counts["rows_scanned"] += 1
            table_id = str(shard_record.get("table_id") or "")
            if current_table_id != table_id:
                current_context = cursor.seek(table_id)
                current_table_id = table_id
            assert current_context is not None
            row = shard_record
            for cell in row.get("cells") or ():
                if not isinstance(cell, Mapping):
                    raise LiteralAuthorizationError(
                        "TabGR row cell is not an object"
                    )
                raw_value = str(cell.get("raw_value") or "")
                if not numeric_fragments(raw_value):
                    continue
                try:
                    coordinate = _coordinate(
                        cell.get("coordinate"),
                        label="row.cell.coordinate",
                    )
                except LiteralAuthorizationError:
                    raise LiteralAuthorizationError(
                        "numeric row cell has invalid coordinate"
                    )
                origin = current_context.origin(coordinate)
                origin_key = (
                    tuple(origin["origin_coordinate"])
                    if origin is not None
                    else coordinate
                )
                if origin_key in current_context.processed_origins:
                    continue
                current_context.processed_origins.add(origin_key)
                counts["physical_numeric_cells_considered"] += 1
                record, reason = _candidate_record(
                    corpus_id=corpus_id,
                    document=document,
                    context=current_context,
                    row=row,
                    cell=cell,
                    numeric_by_coordinate=numeric_by_coordinate,
                    auth_validator=auth_validator,
                )
                if reason is not None:
                    audit.add(
                        reason,
                        document_id=document_id,
                        row_evidence_id=str(row.get("evidence_id") or ""),
                        table_id=table_id,
                        coordinate=coordinate,
                        raw_value_sha256=sha256_text(raw_value),
                    )
                    continue
                assert record is not None
                existing = seen_authorization_ids.get(
                    record["authorization_id"]
                )
                record_hash = semantic_sha256(record)
                if existing is not None and existing != record_hash:
                    raise LiteralAuthorizationError(
                        "duplicate literal authorization id differs"
                    )
                seen_authorization_ids[record["authorization_id"]] = record_hash
                records.append(record)
        if (
            shard_record_count != manifest_row["record_count"]
            or shard_table_count != manifest_row["table_count"]
        ):
            raise LiteralAuthorizationError(
                f"TabGR document shard counts differ: {document_id}"
            )

        slots: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            slots[_semantic_slot(record)].append(record)
        conflict_ids: set[str] = set()
        for group in slots.values():
            literals = {
                record["cell_binding"]["source_cell_literal"]
                for record in group
            }
            if len(literals) > 1:
                counts["semantic_conflict_groups"] += 1
                for record in group:
                    conflict_ids.add(record["authorization_id"])
                    audit.add_record("semantic_slot_conflict", record)
        records = [
            record
            for record in records
            if record["authorization_id"] not in conflict_ids
        ]
        records.sort(
            key=lambda record: (
                record["source_binding"]["table_index"],
                record["row_binding"]["row_index"],
                record["cell_binding"]["coordinate"][1],
                record["authorization_id"],
            )
        )
        shard_name = f"{sha256_text(document_id)[:20]}.jsonl"
        shard_rel = f"documents/{shard_name}"
        output_shard = staging / shard_rel
        authorization_count = _write_jsonl(output_shard, records)
        counts["authorizations_emitted"] += authorization_count
        if authorization_count:
            counts["documents_with_authorizations"] += 1
        output_manifest_row = {
            "schema_version": DOCUMENT_MANIFEST_SCHEMA_VERSION,
            "corpus_id": corpus_id,
            "document_id": document_id,
            "source_sha256": document["source_sha256"],
            "shard_path": shard_rel,
            "shard_sha256": sha256_file(output_shard),
            "row_evidence_count": row_count,
            "authorization_count": authorization_count,
        }
        _validate(
            document_manifest_validator,
            output_manifest_row,
            label="literal authorization document manifest row",
        )
        document_rows_out.append(output_manifest_row)

    cursor.finish()
    counts["tables_scanned"] = cursor.count
    expected_ready_tables = package_manifest.get("counts", {}).get(
        "ready_tables"
    )
    if counts["tables_scanned"] != expected_ready_tables:
        raise LiteralAuthorizationError(
            "structured table count differs from package manifest"
        )
    if counts["rows_scanned"] != sum(
        int(row["record_count"]) - int(row["table_count"])
        for row in document_manifest_rows
    ):
        raise LiteralAuthorizationError(
            "row-evidence count differs from document manifests"
        )

    rejection_rows = audit.rows()
    for row in rejection_rows:
        _validate(rejection_validator, row, label="rejection audit row")
    rejection_path = staging / "rejection_audit.jsonl"
    rejection_record_count = _write_jsonl(rejection_path, rejection_rows)
    document_manifest_path = staging / "document_manifest.jsonl"
    _write_jsonl(document_manifest_path, document_rows_out)
    if (
        counts["physical_numeric_cells_considered"]
        != counts["authorizations_emitted"] + audit.total
    ):
        raise LiteralAuthorizationError(
            "authorization/rejection accounting does not close"
        )

    input_hashes_after = {
        name: sha256_file(path)
        for name, path in sorted(all_input_paths.items())
    }
    code_hashes_after = {
        name: sha256_file(path)
        for name, path in sorted(resolved_code.items())
    }
    sources_after = source_snapshot(profile, workspace_root=root)
    if input_hashes_after != input_hashes_before:
        raise LiteralAuthorizationError(
            "frozen input changed during authorization build"
        )
    if code_hashes_after != code_hashes_before:
        raise LiteralAuthorizationError(
            "builder/schema closure changed during authorization build"
        )
    if sources_after != sources_before:
        raise LiteralAuthorizationError(
            "original Markdown changed during authorization build"
        )

    manifest_unsigned = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "profile_version": PROFILE_VERSION,
        "builder_version": BUILDER_VERSION,
        "corpus_id": corpus_id,
        "corpus_profile_sha256": profile["profile_sha256"],
        "document_count": len(documents),
        "authorization_count": counts["authorizations_emitted"],
        "documents_with_authorizations": counts[
            "documents_with_authorizations"
        ],
        "counts": {
            "tables_scanned": counts["tables_scanned"],
            "rows_scanned": counts["rows_scanned"],
            "physical_numeric_cells_considered": counts[
                "physical_numeric_cells_considered"
            ],
            "authorizations_emitted": counts["authorizations_emitted"],
            "rejections": audit.total,
            "semantic_conflict_groups": counts[
                "semantic_conflict_groups"
            ],
        },
        "input_artifacts": [
            _artifact_binding(
                name,
                path,
                root=root,
                before=input_hashes_before,
                after=input_hashes_after,
            )
            for name, path in sorted(all_input_paths.items())
        ],
        "code_and_schema_closure": [
            _artifact_binding(
                name,
                path,
                root=root,
                before=code_hashes_before,
                after=code_hashes_after,
            )
            for name, path in sorted(resolved_code.items())
        ],
        "source_snapshot": {
            "document_count": len(sources_before),
            "semantic_sha256_before": semantic_sha256(sources_before),
            "semantic_sha256_after": semantic_sha256(sources_after),
            "unchanged": True,
        },
        "document_manifest": {
            "path": "document_manifest.jsonl",
            "sha256": sha256_file(document_manifest_path),
            "record_count": len(document_rows_out),
        },
        "rejection_audit": {
            "path": "rejection_audit.jsonl",
            "sha256": sha256_file(rejection_path),
            "record_count": rejection_record_count,
            "rejected_count": audit.total,
        },
        "build_contract": {
            "source_markdown_read_only_hash_verification": True,
            "gold_or_scoring_inputs_read": False,
            "numeric_authorization_priority": True,
            "exact_literal_only": True,
            "hidden_staging_atomic_publish": True,
            "all_records_closed_schema_validated": True,
        },
    }
    manifest = {
        **manifest_unsigned,
        "manifest_fingerprint": semantic_sha256(manifest_unsigned),
    }
    _validate(manifest_validator, manifest, label="authorization manifest")
    _write_json(staging / "manifest.json", manifest)

    staging.rename(final_dir)
    _ACTIVE_STAGING = None
    print(
        json.dumps(
            {
                "status": "passed",
                "corpus_id": corpus_id,
                "authorization_count": counts["authorizations_emitted"],
                "documents_with_authorizations": counts[
                    "documents_with_authorizations"
                ],
                "rejections": audit.total,
                "manifest_fingerprint": manifest["manifest_fingerprint"],
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
