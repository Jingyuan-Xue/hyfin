"""Separate Phase 9 store and v1-conformant selected/supplemental composition."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .ports import (
    validate_fact_lookup_request,
    validate_fact_lookup_result,
    validate_fact_record,
    validate_selected_fact_filters,
)
from .repositories import FACT_LOOKUP_RESULT_SCHEMA, FactRepository
from .supplement_contracts import (
    FACT_SOURCE,
    SCHEMA_SUPPLEMENT_LOOKUP,
    canonical_provenance_text,
    validate_supplement_decision,
    validate_supplement_lookup_result,
    validate_supplemental_fact,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE = ROOT / "data/facts/supplemental_facts.duckdb"
DEFAULT_JSONL = ROOT / "data/facts/supplemental_facts.jsonl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fact_to_record(fact: Mapping[str, Any]) -> dict[str, Any]:
    provenance = {
        "fact_source": FACT_SOURCE,
        "supplemental_fact_id": fact["supplemental_fact_id"],
        "sources": fact["provenance_json"],
        "validation_versions": fact["validation_versions"],
        "tabgr_trace_fingerprint": fact["tabgr_trace_fingerprint"],
    }
    return validate_fact_record({
        "fact_id": fact["supplemental_fact_id"],
        "document_id": fact["document_id"],
        "stock_code": fact["stock_code"],
        "company": fact["company"],
        "report_year": fact["report_year"],
        "metric_year": fact["metric_year"],
        "canonical_metric": fact["canonical_metric"],
        "normalized_value": fact["normalized_value"],
        "normalized_unit": fact["normalized_unit"],
        "source_table_id": fact["source_table_id"],
        "source_line_start": fact["source_line_start"],
        "source_line_end": fact["source_line_end"],
        "provenance_json": canonical_provenance_text(provenance),
    })


class SupplementalFactRepository:
    def __init__(self, database_path: str | Path = DEFAULT_DATABASE) -> None:
        self._database_path = Path(database_path)
        if not self._database_path.is_file():
            raise FileNotFoundError("supplemental fact database does not exist")
        self.repository_fingerprint = sha256_file(self._database_path)

    def _connect(self):
        import duckdb
        return duckdb.connect(str(self._database_path), read_only=True)

    @staticmethod
    def _decode(row: tuple[Any, ...]) -> dict[str, Any]:
        value = json.loads(str(row[0]))
        return validate_supplemental_fact(value)

    def lookup_supplement(self, request: Mapping[str, Any]) -> dict[str, Any]:
        checked = validate_fact_lookup_request(dict(request))
        connection = self._connect()
        try:
            rows = connection.execute("""
                SELECT fact_json FROM supplemental_facts
                WHERE document_id=? AND stock_code=? AND report_year=? AND metric_year=?
                  AND canonical_metric=? AND normalized_unit=?
                ORDER BY supplemental_fact_id
            """, [
                checked["document_id"], checked["stock_code"], checked["report_year"],
                checked["metric_year"], checked["canonical_metric"], checked["normalized_unit"],
            ]).fetchall()
        finally:
            connection.close()
        records = [self._decode(row) for row in rows]
        status = "not_found" if not records else "found" if len(records) == 1 else "ambiguous"
        return validate_supplement_lookup_result({
            "schema_version": SCHEMA_SUPPLEMENT_LOOKUP,
            "requirement_id": checked["requirement_id"],
            "status": status,
            "records": records,
            "repository_fingerprint": self.repository_fingerprint,
            "fact_source": FACT_SOURCE,
        })

    def query_supplements(self, filters: Mapping[str, Any]) -> list[dict[str, Any]]:
        checked = validate_selected_fact_filters(dict(filters))
        columns = {
            "document_ids": "document_id", "stock_codes": "stock_code",
            "report_years": "report_year", "metric_years": "metric_year",
            "canonical_metrics": "canonical_metric", "normalized_units": "normalized_unit",
        }
        clauses: list[str] = []
        parameters: list[Any] = []
        for field, column in columns.items():
            values = checked[field]
            if values:
                clauses.append(f"{column} IN ({','.join('?' for _ in values)})")
                parameters.extend(values)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        connection = self._connect()
        try:
            rows = connection.execute(
                f"SELECT fact_json FROM supplemental_facts{where} ORDER BY supplemental_fact_id",
                parameters,
            ).fetchall()
        finally:
            connection.close()
        return [_fact_to_record(self._decode(row)) for row in rows]


class SupplementAwareFactRepository:
    """Default-off injectable composition preserving Phase 8's exact v1 port."""

    def __init__(
        self,
        selected: FactRepository | None = None,
        supplemental: SupplementalFactRepository | None = None,
    ) -> None:
        self.selected = selected or FactRepository()
        self.supplemental = supplemental or SupplementalFactRepository()
        self.repository_fingerprint = hashlib.sha256(
            (self.selected.repository_fingerprint + ":" + self.supplemental.repository_fingerprint).encode("ascii")
        ).hexdigest()

    def lookup_fact(self, request: Mapping[str, Any]) -> dict[str, Any]:
        checked = validate_fact_lookup_request(dict(request))
        selected = validate_fact_lookup_result(self.selected.lookup_fact(checked))
        if selected["status"] != "not_found":
            return selected
        supplement = self.supplemental.lookup_supplement(checked)
        records = sorted((_fact_to_record(row) for row in supplement["records"]), key=lambda row: row["fact_id"])
        result = {
            "schema_version": FACT_LOOKUP_RESULT_SCHEMA,
            "requirement_id": checked["requirement_id"],
            "status": supplement["status"],
            "records": records,
            "repository_fingerprint": self.repository_fingerprint,
        }
        return validate_fact_lookup_result(result)

    def query_selected_facts(self, filters: Mapping[str, Any]) -> list[dict[str, Any]]:
        checked = validate_selected_fact_filters(dict(filters))
        selected = self.selected.query_selected_facts(checked)
        selected_keys = {
            (row["document_id"], row["stock_code"], row["report_year"], row["metric_year"], row["canonical_metric"], row["normalized_unit"])
            for row in selected
        }
        supplemental = [
            row for row in self.supplemental.query_supplements(checked)
            if (row["document_id"], row["stock_code"], row["report_year"], row["metric_year"], row["canonical_metric"], row["normalized_unit"])
            not in selected_keys
        ]
        return sorted([*selected, *supplemental], key=lambda row: row["fact_id"])


def materialize_store(
    facts: Iterable[Mapping[str, Any]],
    decisions: Iterable[Mapping[str, Any]],
    database_path: str | Path = DEFAULT_DATABASE,
) -> dict[str, Any]:
    fact_rows = [validate_supplemental_fact(dict(row)) for row in facts]
    decision_rows = [validate_supplement_decision(dict(row)) for row in decisions]
    fact_rows.sort(key=lambda row: row["supplemental_fact_id"])
    decision_rows.sort(key=lambda row: row["slot_key"])
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    import duckdb
    connection = duckdb.connect(str(temporary))
    try:
        connection.execute("""
            CREATE TABLE supplemental_facts (
                supplemental_fact_id VARCHAR PRIMARY KEY,
                document_id VARCHAR NOT NULL, stock_code VARCHAR NOT NULL,
                report_year INTEGER NOT NULL, metric_year INTEGER NOT NULL,
                canonical_metric VARCHAR NOT NULL, normalized_unit VARCHAR NOT NULL,
                normalized_value VARCHAR NOT NULL, fact_source VARCHAR NOT NULL,
                fact_json JSON NOT NULL
            )
        """)
        for row in fact_rows:
            connection.execute(
                "INSERT INTO supplemental_facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [row["supplemental_fact_id"], row["document_id"], row["stock_code"], row["report_year"],
                 row["metric_year"], row["canonical_metric"], row["normalized_unit"], row["normalized_value"],
                 row["fact_source"], canonical_provenance_text(row)],
            )
        connection.execute("""
            CREATE TABLE supplement_decisions (
                slot_fingerprint VARCHAR PRIMARY KEY, requirement_id VARCHAR NOT NULL,
                decision_status VARCHAR NOT NULL, failure_code VARCHAR,
                decision_json JSON NOT NULL
            )
        """)
        for row in decision_rows:
            connection.execute(
                "INSERT INTO supplement_decisions VALUES (?, ?, ?, ?, ?)",
                [row["slot_fingerprint"], row["requirement_id"], row["decision_status"], row["failure_code"], canonical_provenance_text(row)],
            )
        connection.execute("CREATE TABLE build_metadata (key VARCHAR PRIMARY KEY, value VARCHAR NOT NULL)")
        metadata = {
            "builder_version": "phase9-supplemental-fact-builder-v1",
            "fact_count": str(len(fact_rows)), "decision_count": str(len(decision_rows)),
        }
        for key, value in sorted(metadata.items()):
            connection.execute("INSERT INTO build_metadata VALUES (?, ?)", [key, value])
        connection.execute("CHECKPOINT")
    finally:
        connection.close()
    os.replace(temporary, path)
    return {"fact_count": len(fact_rows), "decision_count": len(decision_rows), "duckdb_sha256": sha256_file(path)}


__all__ = [
    "DEFAULT_DATABASE", "DEFAULT_JSONL", "SupplementAwareFactRepository",
    "SupplementalFactRepository", "materialize_store", "sha256_file",
]
