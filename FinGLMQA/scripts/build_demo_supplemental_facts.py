#!/usr/bin/env python3
"""Materialize audited demo supplements for four unresolved revenue slots.

The Phase 6 store correctly fails closed when one document yields several
values for the same metric/year.  For these four demo documents, the primary
company total is nevertheless explicit in both the annual-report summary and
the consolidated income statement.  This builder publishes only the reviewed
primary-total facts; it never applies a general "pick the largest" fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.contracts import canonical_json_bytes, semantic_sha256  # noqa: E402
from finglmqa.supplement_contracts import (  # noqa: E402
    FACT_SOURCE,
    SCHEMA_SUPPLEMENTAL_FACT,
    validate_supplemental_fact,
)
from finglmqa.supplement_store import materialize_store, sha256_file  # noqa: E402


SOURCE_DATABASE = ROOT / "data/facts/financial_facts.duckdb"
OUTPUT_JSONL = ROOT / "data/facts/supplemental_facts.jsonl"
OUTPUT_DATABASE = ROOT / "data/facts/supplemental_facts.duckdb"
OUTPUT_SUMMARY = ROOT / "runs/phase_10/reports/demo_supplemental_facts_summary.json"

# Each fact is the primary-company total from the annual-report summary table.
# The expected values and provenance identities make accidental source drift a
# hard build failure.
RESOLUTIONS = (
    {
        "source_fact_id": "fact_71d9d9348a89ad86ce78e02b",
        "requirement_id": "req_b849893adbdbb95f9d69",
        "document_id": "A002134_天津普林_2019年年度报告",
        "expected_value": "418242590.19",
        "expected_table": "A002134_天津普林_2019年年度报告_table_0007_f6cb70e09e",
    },
    {
        "source_fact_id": "fact_753c975c0c1ed2efadc359e9",
        "requirement_id": "req_2cf0c2a6cc63fe8a9825",
        "document_id": "A002165_红 宝 丽_2019年年度报告",
        "expected_value": "2382800057.56",
        "expected_table": "A002165_红 宝 丽_2019年年度报告_table_0007_fcb328c09f",
    },
    {
        "source_fact_id": "fact_46fef02e99ead874b88117b5",
        "requirement_id": "req_29751e441ca9731f9ae6",
        "document_id": "A300286_安科瑞_2019年年度报告",
        "expected_value": "600208305.71",
        "expected_table": "A300286_安科瑞_2019年年度报告_table_0006_6f15745780",
    },
    {
        "source_fact_id": "fact_95033d257c4f8d5d0df5b818",
        "requirement_id": "req_7e250b62b3d31e46d003",
        "document_id": "A300310_宜通世纪_2019年年度报告",
        "expected_value": "2485724600.26",
        "expected_table": "A300310_宜通世纪_2019年年度报告_table_0006_5f2b73b959",
    },
)


def _portable_source(value: str) -> str:
    marker = "/refs/source_markdown/"
    if marker in value:
        return "refs/source_markdown/" + value.split(marker, 1)[1]
    return Path(value).name


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row))
    temporary.replace(path)


def build(
    source_database: Path = SOURCE_DATABASE,
    output_jsonl: Path = OUTPUT_JSONL,
    output_database: Path = OUTPUT_DATABASE,
    output_summary: Path = OUTPUT_SUMMARY,
) -> dict[str, Any]:
    import duckdb

    connection = duckdb.connect(str(source_database), read_only=True)
    facts: list[dict[str, Any]] = []
    try:
        for resolution in RESOLUTIONS:
            row = connection.execute(
                """
                SELECT fact_id, document_id, stock_code, stock_name, report_year,
                       metric_year, statement, canonical_metric,
                       normalized_value_text, normalized_unit, confidence_score,
                       selection_status, conflict_group_id, source_table_id,
                       source_table_index, source_row_index, source_col_index,
                       source_markdown, source_line_start, source_line_end,
                       provenance_json
                FROM financial_facts WHERE fact_id = ?
                """,
                [resolution["source_fact_id"]],
            ).fetchone()
            if row is None:
                raise RuntimeError(f"reviewed source fact missing: {resolution['source_fact_id']}")
            (
                fact_id, document_id, stock_code, company, report_year,
                metric_year, statement, metric, value, unit, confidence,
                selection_status, conflict_group_id, table_id, table_index,
                row_index, col_index, source_markdown, line_start, line_end,
                provenance_json,
            ) = row
            if (
                document_id != resolution["document_id"]
                or value != resolution["expected_value"]
                or table_id != resolution["expected_table"]
                or metric != "营业收入"
                or unit != "元"
                or report_year != 2019
                or metric_year != 2019
                or selection_status != "unresolved_conflict"
                or str(confidence) != "1.0000"
            ):
                raise RuntimeError(f"reviewed source fact drifted: {fact_id}")

            source_provenance = json.loads(str(provenance_json))
            if not isinstance(source_provenance, list) or not source_provenance:
                raise RuntimeError(f"reviewed source provenance missing: {fact_id}")
            audit_provenance = [{
                "resolution_method": "reviewed_primary_company_total",
                "source_fact_id": fact_id,
                "source_selection_status": selection_status,
                "source_conflict_group_id": conflict_group_id,
                "source_candidate": candidate,
            } for candidate in source_provenance]
            supplement_id = "sf_demo_" + hashlib.sha256(
                canonical_json_bytes([fact_id, document_id, value, table_id])
            ).hexdigest()[:20]
            fact = {
                "supplemental_fact_id": supplement_id,
                "schema_version": SCHEMA_SUPPLEMENTAL_FACT,
                "document_id": document_id,
                "stock_code": stock_code,
                "company": company,
                "report_year": report_year,
                "metric_year": metric_year,
                "canonical_metric": metric,
                "normalized_unit": unit,
                "statement": statement,
                "normalized_value": value,
                "fact_source": FACT_SOURCE,
                "validation_versions": {
                    "review": "demo-primary-total-review-v1",
                    "source_store": "phase6-financial-facts-v1",
                },
                "tabgr_trace_fingerprint": semantic_sha256(audit_provenance),
                "source_table_id": table_id,
                "source_table_index": table_index,
                "source_row_index": row_index,
                "source_col_index": col_index,
                "source_line_start": line_start,
                "source_line_end": line_end,
                "source_markdown": _portable_source(str(source_markdown)),
                "provenance_json": audit_provenance,
                "created_from_requirement_ids": [resolution["requirement_id"]],
            }
            facts.append(validate_supplemental_fact(fact))
    finally:
        connection.close()

    facts.sort(key=lambda item: item["supplemental_fact_id"])
    _write_jsonl(output_jsonl, facts)
    store = materialize_store(facts, [], output_database)
    summary = {
        "schema_version": "finglmqa.demo_supplemental_summary.v1",
        "fact_count": len(facts),
        "document_ids": sorted(row["document_id"] for row in facts),
        "artifact_hashes": {
            "jsonl": sha256_file(output_jsonl),
            "duckdb": store["duckdb_sha256"],
        },
    }
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_bytes(canonical_json_bytes(summary))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-database", type=Path, default=SOURCE_DATABASE)
    parser.add_argument("--output-jsonl", type=Path, default=OUTPUT_JSONL)
    parser.add_argument("--output-database", type=Path, default=OUTPUT_DATABASE)
    parser.add_argument("--output-summary", type=Path, default=OUTPUT_SUMMARY)
    args = parser.parse_args()
    print(json.dumps(build(
        source_database=args.source_database,
        output_jsonl=args.output_jsonl,
        output_database=args.output_database,
        output_summary=args.output_summary,
    ), ensure_ascii=False, sort_keys=True))
    return 0



if __name__ == "__main__":
    raise SystemExit(main())

