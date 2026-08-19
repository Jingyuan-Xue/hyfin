"""Read-only Phase 8 repositories over immutable Phase 2/5/6 artifacts.

All public methods exchange JSON-compatible dictionaries.  SQL text and
ordering are owned by this module; callers can supply only the allow-listed
filters frozen in :mod:`finglmqa.ports`.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    ContractValidationError,
    semantic_sha256,
    validate_missing_fact_request,
)
from .ports import (
    validate_fact_lookup_request,
    validate_fact_lookup_result,
    validate_fact_record,
    validate_selected_fact_filters,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FACT_DATABASE = ROOT / "data/facts/financial_facts.duckdb"
DEFAULT_COMPANY_YEAR_INDEX = ROOT / "data/corpus_package/company_year_index.jsonl"
DEFAULT_CORPUS_MANIFEST = ROOT / "data/corpus_package/corpus_manifest.json"
DEFAULT_CANDIDATE_INDEX = ROOT / "data/indexes/canonical_metric_candidates.jsonl"

FACT_LOOKUP_RESULT_SCHEMA = "finglmqa.phase8.fact_lookup_result.v1"
METADATA_REQUEST_SCHEMA = "finglmqa.phase8.metadata_lookup_request.v1"
METADATA_RESULT_SCHEMA = "finglmqa.phase8.metadata_lookup_result.v1"

_FACT_COLUMNS = (
    "fact_id", "document_id", "stock_code", "stock_name", "company_full",
    "report_year", "metric_year", "canonical_metric", "normalized_value_text",
    "normalized_unit", "source_table_id", "source_line_start", "source_line_end",
    "provenance_json",
)
_FACT_SELECT = ", ".join(_FACT_COLUMNS)
_FILTER_COLUMNS = {
    "document_ids": "document_id",
    "stock_codes": "stock_code",
    "report_years": "report_year",
    "metric_years": "metric_year",
    "canonical_metrics": "canonical_metric",
    "normalized_units": "normalized_unit",
}
_METADATA_FIELDS = frozenset({"stock_code", "stock_name", "company_full"})

_DURATION_METRICS = frozenset({
    "营业收入", "归属于上市公司股东的净利润", "扣除非经常性损益后的净利润",
    "经营活动产生的现金流量净额", "基本每股收益", "稀释每股收益",
    "加权平均净资产收益率",
})
_INSTANT_METRICS = frozenset({"总资产", "净资产", "股本"})
_COMPARISON_RE = re.compile(r"同比|环比|增减|增长|下降|变动|比例|比上|较上|差异|占比")
_QUARTER_RE = re.compile(r"季度|半年度|半年|[1-4一二三四]季|\d+\s*[-至]\s*\d+\s*月")
_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_CURRENT_RE = re.compile(r"本报告期|报告期|本年度|本年|本期|期末|年末")
_PREVIOUS_RE = re.compile(r"上年度|上年|上期|期初|年初")
_OPENING_DATE_RE = re.compile(r"(?:1月1日|01月01日|01-01|年初|期初)")
_YEAR_END_DATE_RE = re.compile(r"(?:12月31日|12-31|年末|期末)")
_MONTH_OR_DAY_RE = re.compile(r"\d{1,2}\s*月|\d{1,2}\s*日|\d{1,2}[-/]\d{1,2}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL row {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} must be an object")
            rows.append(row)
    return rows


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        # Preserve the immutable source payload exactly when DuckDB exposes its
        # JSON column as text.
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fact_record(row: Sequence[Any]) -> dict[str, Any]:
    source = dict(zip(_FACT_COLUMNS, row, strict=True))
    record = {
        "fact_id": str(source["fact_id"]),
        "document_id": str(source["document_id"]),
        "stock_code": str(source["stock_code"]),
        "company": str(source["stock_name"] or source["company_full"]),
        "report_year": int(source["report_year"]),
        "metric_year": int(source["metric_year"]),
        "canonical_metric": str(source["canonical_metric"]),
        # normalized_value_text is the audited Phase 6 Decimal rendering.  Do
        # not round-trip it through float or DuckDB's Decimal object.
        "normalized_value": str(source["normalized_value_text"]),
        "normalized_unit": str(source["normalized_unit"]),
        "source_table_id": str(source["source_table_id"]),
        "source_line_start": (
            int(source["source_line_start"]) if source["source_line_start"] is not None else None
        ),
        "source_line_end": (
            int(source["source_line_end"]) if source["source_line_end"] is not None else None
        ),
        "provenance_json": _json_text(source["provenance_json"]),
    }
    return validate_fact_record(record)


class FactRepository:
    """Selected-fact repository with a fresh read-only connection per call."""

    def __init__(self, database_path: str | Path = DEFAULT_FACT_DATABASE) -> None:
        self._database_path = Path(database_path)
        if not self._database_path.is_file():
            raise FileNotFoundError("financial fact database does not exist")
        # A content hash is deterministic and does not leak a host path into a
        # public result.  It also changes if the immutable input changes.
        self.repository_fingerprint = _sha256_file(self._database_path)

    def _connect(self):
        try:
            import duckdb
        except ModuleNotFoundError as exc:  # pragma: no cover - deployment guard
            raise RuntimeError("DuckDB is required by FactRepository") from exc
        return duckdb.connect(str(self._database_path), read_only=True)

    def lookup_fact(self, request: Mapping[str, Any]) -> dict[str, Any]:
        checked = validate_fact_lookup_request(dict(request))
        sql = f"""
            SELECT {_FACT_SELECT}
            FROM selected_financial_facts
            WHERE document_id = ?
              AND stock_code = ?
              AND report_year = ?
              AND metric_year = ?
              AND canonical_metric = ?
              AND normalized_unit = ?
            ORDER BY fact_id
        """
        parameters = [
            checked["document_id"], checked["stock_code"], checked["report_year"],
            checked["metric_year"], checked["canonical_metric"], checked["normalized_unit"],
        ]
        connection = self._connect()
        try:
            records = [_fact_record(row) for row in connection.execute(sql, parameters).fetchall()]
        finally:
            connection.close()
        status = "not_found" if not records else "found" if len(records) == 1 else "ambiguous"
        result = {
            "schema_version": FACT_LOOKUP_RESULT_SCHEMA,
            "requirement_id": checked["requirement_id"],
            "status": status,
            "records": records,
            "repository_fingerprint": self.repository_fingerprint,
        }
        return validate_fact_lookup_result(result)

    def query_selected_facts(self, filters: Mapping[str, Any]) -> list[dict[str, Any]]:
        checked = validate_selected_fact_filters(dict(filters))
        clauses: list[str] = []
        parameters: list[Any] = []
        for request_field, column in _FILTER_COLUMNS.items():
            values = checked[request_field]
            if not values:
                continue
            placeholders = ",".join("?" for _ in values)
            clauses.append(f"{column} IN ({placeholders})")
            parameters.extend(values)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        sql = f"SELECT {_FACT_SELECT} FROM selected_financial_facts{where} ORDER BY fact_id"
        connection = self._connect()
        try:
            records = [_fact_record(row) for row in connection.execute(sql, parameters).fetchall()]
        finally:
            connection.close()
        # _fact_record validates each record.  The static ORDER BY is repeated
        # as an assertion at the boundary to fail closed if it is changed.
        if [row["fact_id"] for row in records] != sorted(row["fact_id"] for row in records):
            raise RuntimeError("selected fact scan lost deterministic fact_id ordering")
        return records


def validate_metadata_lookup_request(request: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ContractValidationError("MetadataLookupRequest must be an object")
    fields = {"schema_version", "requirement_id", "entity_key", "document_id", "stock_code", "metadata_field"}
    if set(request) != fields:
        raise ContractValidationError("MetadataLookupRequest fields do not match the frozen repository boundary")
    if request["schema_version"] != METADATA_REQUEST_SCHEMA:
        raise ContractValidationError("MetadataLookupRequest.schema_version is unsupported")
    for field in ("requirement_id", "stock_code", "metadata_field"):
        if not isinstance(request[field], str) or not request[field].strip():
            raise ContractValidationError(f"MetadataLookupRequest.{field} must be a non-empty string")
    for field in ("entity_key", "document_id"):
        if request[field] is not None and (not isinstance(request[field], str) or not request[field].strip()):
            raise ContractValidationError(f"MetadataLookupRequest.{field} must be a string or null")
    return request


def validate_metadata_lookup_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen metadata repository response and record lineage."""

    if not isinstance(result, dict):
        raise ContractValidationError("MetadataLookupResult must be an object")
    fields = {"schema_version", "requirement_id", "status", "records", "repository_fingerprint"}
    if set(result) != fields or result.get("schema_version") != METADATA_RESULT_SCHEMA:
        raise ContractValidationError("MetadataLookupResult fields/schema are invalid")
    for field in ("requirement_id", "repository_fingerprint"):
        if not isinstance(result.get(field), str) or not result[field].strip():
            raise ContractValidationError(f"MetadataLookupResult.{field} must be non-empty")
    if result.get("status") not in {"found", "not_found", "ambiguous"}:
        raise ContractValidationError("MetadataLookupResult.status is invalid")
    records = result.get("records")
    if not isinstance(records, list):
        raise ContractValidationError("MetadataLookupResult.records must be an array")
    expected_status = "not_found" if not records else "found" if len(records) == 1 else "ambiguous"
    if result["status"] != expected_status:
        raise ContractValidationError("MetadataLookupResult.status does not match record cardinality")
    record_fields = {
        "metadata_field", "value", "entity_key", "stock_code", "document_ids",
        "report_years", "provenance",
    }
    for record in records:
        if not isinstance(record, dict) or set(record) != record_fields:
            raise ContractValidationError("MetadataLookupResult record shape is invalid")
        for field in ("metadata_field", "value", "stock_code"):
            if not isinstance(record[field], str) or not record[field].strip():
                raise ContractValidationError(f"MetadataLookupResult record {field} is invalid")
        if record["entity_key"] is not None and (
            not isinstance(record["entity_key"], str) or not record["entity_key"].strip()
        ):
            raise ContractValidationError("MetadataLookupResult record entity_key is invalid")
        documents = record["document_ids"]
        years = record["report_years"]
        if (
            not isinstance(documents, list) or not documents
            or any(not isinstance(value, str) or not value for value in documents)
            or documents != sorted(set(documents))
        ):
            raise ContractValidationError("MetadataLookupResult document_ids are invalid")
        if (
            not isinstance(years, list) or not years
            or any(isinstance(value, bool) or not isinstance(value, int) for value in years)
            or years != sorted(set(years))
        ):
            raise ContractValidationError("MetadataLookupResult report_years are invalid")
        if not isinstance(record["provenance"], dict) or not record["provenance"]:
            raise ContractValidationError("MetadataLookupResult provenance must be a non-empty object")
    return result


class MetadataRepository:
    """Lookup only identity fields physically present in the Phase 2 index."""

    def __init__(
        self,
        index_path: str | Path = DEFAULT_COMPANY_YEAR_INDEX,
        manifest_path: str | Path = DEFAULT_CORPUS_MANIFEST,
    ) -> None:
        self._index_path = Path(index_path)
        self._manifest_path = Path(manifest_path)
        rows = _read_jsonl(self._index_path)
        manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        fingerprints = {
            str(row["document_id"]): str(row["content_sha256"])
            for row in manifest.get("documents", [])
            if row.get("document_id") and row.get("content_sha256")
        }
        canonical: list[dict[str, Any]] = []
        for row in rows:
            canonical.append({
                "document_id": str(row["document_id"]),
                "stock_code": str(row["stock_code"]),
                "stock_name": str(row["stock_name"]),
                "company_full": str(row["company_full"]),
                "report_year": int(row["report_year"]),
                "artifact_fingerprint": fingerprints.get(str(row["document_id"]))
                or semantic_sha256({
                    "document_id": row["document_id"],
                    "stock_code": row["stock_code"],
                    "report_year": int(row["report_year"]),
                }),
            })
        self._rows = tuple(sorted(canonical, key=lambda row: (row["stock_code"], row["report_year"], row["document_id"])))
        self.repository_fingerprint = semantic_sha256({
            "company_year_index_sha256": _sha256_file(self._index_path),
            "corpus_manifest_sha256": _sha256_file(self._manifest_path),
        })

    def lookup_metadata(self, request: Mapping[str, Any]) -> dict[str, Any]:
        checked = validate_metadata_lookup_request(dict(request))
        field = checked["metadata_field"]
        if field not in _METADATA_FIELDS:
            records: list[dict[str, Any]] = []
            status = "not_found"
        else:
            matches = [row for row in self._rows if row["stock_code"] == checked["stock_code"]]
            if checked["document_id"] is not None:
                matches = [row for row in matches if row["document_id"] == checked["document_id"]]
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in matches:
                value = row[field]
                grouped.setdefault(str(value), []).append(row)
            records = []
            for value, sources in sorted(grouped.items()):
                records.append({
                    "metadata_field": field,
                    "value": value,
                    "entity_key": checked["entity_key"],
                    "stock_code": checked["stock_code"],
                    "document_ids": sorted(row["document_id"] for row in sources),
                    "report_years": sorted({row["report_year"] for row in sources}),
                    "provenance": {
                        "source_kind": "company_year_index",
                        "artifact_fingerprints": sorted({row["artifact_fingerprint"] for row in sources}),
                    },
                })
            status = "not_found" if not records else "found" if len(records) == 1 else "ambiguous"
        result = {
            "schema_version": METADATA_RESULT_SCHEMA,
            "requirement_id": checked["requirement_id"],
            "status": status,
            "records": records,
            "repository_fingerprint": self.repository_fingerprint,
        }
        return validate_metadata_lookup_result(result)


def _candidate_metric_year(candidate: Mapping[str, Any]) -> int | None:
    """Conservatively reproduce Phase 6 period routing for candidate discovery."""

    metric = str(candidate.get("canonical_metric") or "")
    label_field = "column_label" if candidate.get("metric_source") == "row_label" else "row_label"
    label = re.sub(r"\s+", "", str(candidate.get(label_field) or ""))
    try:
        report_year = int(candidate.get("report_year"))
    except (TypeError, ValueError):
        return None
    if not label or _COMPARISON_RE.search(label) or _QUARTER_RE.search(label):
        return None
    if metric in _DURATION_METRICS and (_MONTH_OR_DAY_RE.search(label) or "至" in label):
        return None
    years = sorted({int(value) for value in _YEAR_RE.findall(label)})
    if len(years) > 1:
        return None
    if years:
        year = years[0]
        if metric in _INSTANT_METRICS and _MONTH_OR_DAY_RE.search(label):
            if _OPENING_DATE_RE.search(label):
                year -= 1
            elif not _YEAR_END_DATE_RE.search(label):
                return None
        elif metric in _INSTANT_METRICS and re.search(r"年初|期初", label):
            year -= 1
    elif _PREVIOUS_RE.search(label):
        year = report_year - 1
    elif _CURRENT_RE.search(label):
        year = report_year
    else:
        return None
    if year > report_year or year < report_year - 2:
        return None
    return year


class FallbackCandidateIndex:
    """Stable table discovery over Phase 5 candidates; never executes TabGR."""

    def __init__(self, candidates_path: str | Path = DEFAULT_CANDIDATE_INDEX) -> None:
        self._candidates_path = Path(candidates_path)
        self._rows = tuple(_read_jsonl(self._candidates_path))
        self.repository_fingerprint = _sha256_file(self._candidates_path)

    def candidate_table_ids(self, request: Mapping[str, Any]) -> list[str]:
        checked = validate_missing_fact_request(dict(request))
        table_ids: set[str] = set()
        for row in self._rows:
            if str(row.get("document_id")) != checked["document_id"]:
                continue
            if str(row.get("stock_code")) != checked["stock_code"]:
                continue
            try:
                report_year = int(row.get("report_year"))
            except (TypeError, ValueError):
                continue
            if report_year != checked["report_year"]:
                continue
            if row.get("canonical_metric") != checked["canonical_metric"]:
                continue
            if row.get("normalized_unit") != checked["normalized_unit"]:
                continue
            if _candidate_metric_year(row) != checked["metric_year"]:
                continue
            table_id = row.get("table_id")
            if isinstance(table_id, str) and table_id:
                table_ids.add(table_id)
        return sorted(table_ids)


__all__ = [
    "FactRepository",
    "MetadataRepository",
    "FallbackCandidateIndex",
    "validate_metadata_lookup_request",
    "validate_metadata_lookup_result",
]
