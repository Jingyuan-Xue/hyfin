#!/usr/bin/env python3
"""Build the exact-unit Phase 9 MissingFactRequest audit universe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.contracts import (  # noqa: E402
    SCHEMA_MISSING_FACT_REQUEST,
    canonical_json_bytes,
    make_requirement_id,
    validate_missing_fact_request,
)
from finglmqa.repositories import FallbackCandidateIndex  # noqa: E402
from finglmqa.supplement_contracts import canonical_slot_key  # noqa: E402


DEFAULT_COMPANY_YEAR = ROOT / "data/corpus_package/company_year_index.jsonl"
DEFAULT_FACT_DB = ROOT / "data/facts/financial_facts.duckdb"
DEFAULT_OUTPUT = ROOT / "runs/phase_09/reports/supplement_requests.jsonl"
DEFAULT_SUMMARY = ROOT / "runs/phase_09/reports/request_universe_summary.json"

METRIC_UNITS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("营业收入", ("元",)),
    ("归属于上市公司股东的净利润", ("元",)),
    ("扣除非经常性损益后的净利润", ("元",)),
    ("经营活动产生的现金流量净额", ("元",)),
    ("总资产", ("元",)),
    ("净资产", ("元",)),
    ("基本每股收益", ("元/股",)),
    ("稀释每股收益", ("元/股",)),
    ("加权平均净资产收益率", ("ratio",)),
    ("股本", ("元", "股")),
)

STATEMENT_BY_METRIC = {
    "营业收入": "income_statement",
    "归属于上市公司股东的净利润": "income_statement",
    "扣除非经常性损益后的净利润": "income_statement",
    "经营活动产生的现金流量净额": "cash_flow_statement",
    "总资产": "balance_sheet",
    "净资产": "balance_sheet",
    "股本": "balance_sheet",
    "基本每股收益": "financial_indicator",
    "稀释每股收益": "financial_indicator",
    "加权平均净资产收益率": "financial_indicator",
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            yield value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row))
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    os.replace(temporary, path)


def _selected_and_fact_rows(database: Path) -> tuple[set[tuple[Any, ...]], dict[tuple[Any, ...], list[dict[str, Any]]]]:
    import duckdb

    connection = duckdb.connect(str(database), read_only=True)
    try:
        selected = {
            tuple(row)
            for row in connection.execute("""
                SELECT document_id, stock_code, report_year, metric_year,
                       canonical_metric, normalized_unit
                FROM selected_financial_facts
            """).fetchall()
        }
        rows = connection.execute("""
            SELECT document_id, stock_code, report_year, metric_year,
                   canonical_metric, normalized_unit, selection_status,
                   conflict_group_id, confidence_score
            FROM financial_facts
            ORDER BY fact_id
        """).fetchall()
    finally:
        connection.close()
    by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[:6])
        by_key[key].append({
            "selection_status": row[6],
            "conflict_group_id": row[7],
            "confidence_score": str(row[8]),
        })
    return selected, by_key


def build_universe(
    company_year_path: Path = DEFAULT_COMPANY_YEAR,
    database_path: Path = DEFAULT_FACT_DB,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    documents = sorted(
        read_jsonl(company_year_path),
        key=lambda row: (str(row["document_id"]), str(row["stock_code"])),
    )
    if len(documents) != 170 or any(row.get("status") != "unique" for row in documents):
        raise RuntimeError("Phase 9 requires exactly 170 uniquely resolved documents")
    selected, fact_rows = _selected_and_fact_rows(database_path)
    fallback = FallbackCandidateIndex()
    rows: list[dict[str, Any]] = []
    classes: Counter[str] = Counter()
    by_metric: Counter[str] = Counter()
    by_offset: Counter[str] = Counter()
    grid = 0
    covered = 0
    for document in documents:
        report_year = int(document["report_year"])
        for metric, units in METRIC_UNITS:
            for unit in units:
                for offset in (0, 1, 2):
                    grid += 1
                    slot = {
                        "document_id": str(document["document_id"]),
                        "stock_code": str(document["stock_code"]),
                        "report_year": report_year,
                        "metric_year": report_year - offset,
                        "canonical_metric": metric,
                        "normalized_unit": unit,
                    }
                    key = tuple(canonical_slot_key(slot))
                    if key in selected:
                        covered += 1
                        continue
                    exact_rows = fact_rows.get(key, [])
                    if any(row["selection_status"] == "unresolved_conflict" for row in exact_rows):
                        request_class = "conflict_group_open"
                    elif exact_rows:
                        request_class = "fact_withheld"
                    else:
                        request_class = "no_exact_unit_fact"
                    classes[request_class] += 1
                    by_metric[metric] += 1
                    by_offset[f"Y-{offset}" if offset else "Y"] += 1
                    slot_key = canonical_slot_key(slot)
                    subplan_id = "sp_" + hashlib.sha256(canonical_json_bytes(slot_key)).hexdigest()[:16]
                    request = {
                        "schema_version": SCHEMA_MISSING_FACT_REQUEST,
                        "requirement_id": "pending",
                        "origin_operation": "fact_lookup",
                        "formula_id": None,
                        "operand_role": None,
                        "subplan_id": subplan_id,
                        **slot,
                        "candidate_table_ids": [],
                    }
                    request["requirement_id"] = make_requirement_id(request)
                    # This call is deliberately made against the real Phase 8
                    # index and is discovery-only.
                    request["candidate_table_ids"] = fallback.candidate_table_ids(request)
                    validate_missing_fact_request(request)
                    rows.append(request)
    rows.sort(key=lambda row: tuple(canonical_slot_key(row)))
    expected = {
        "grid_slots": 5610,
        "covered_selected_slots": 4189,
        "missing_request_slots": 1421,
        "conflict_group_open": 501,
        "fact_withheld": 30,
        "no_exact_unit_fact": 890,
    }
    actual = {
        "grid_slots": grid,
        "covered_selected_slots": covered,
        "missing_request_slots": len(rows),
        **dict(classes),
    }
    if actual != expected:
        raise RuntimeError(f"exact-unit arithmetic mismatch: expected={expected}, actual={actual}")
    summary = {
        "schema_version": "finglmqa.phase9.request_universe_summary.v1",
        "arithmetic": actual,
        "counts_by_metric": dict(sorted(by_metric.items())),
        "counts_by_year_offset": dict(sorted(by_offset.items())),
        "request_file_sha256": hashlib.sha256(b"".join(canonical_json_bytes(row) for row in rows)).hexdigest(),
        "statement_by_metric": STATEMENT_BY_METRIC,
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    rows, summary = build_universe()
    write_jsonl(args.output, rows)
    write_json(args.summary, summary)
    print(json.dumps(summary["arithmetic"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
