#!/usr/bin/env python3
"""Run the deterministic Phase 9 offline supplementation batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finglmqa.contracts import canonical_json_bytes, semantic_sha256, validate_missing_fact_request  # noqa: E402
from finglmqa.supplement_contracts import (  # noqa: E402
    FACT_SOURCE,
    SCHEMA_SUPPLEMENTAL_FACT,
    SCHEMA_SUPPLEMENT_DECISION,
    canonical_slot_key,
    slot_fingerprint,
    validate_supplement_decision,
    validate_supplemental_fact,
)
from finglmqa.supplement_store import materialize_store, sha256_file  # noqa: E402
from finglmqa.supplement_validation import STATEMENT_BY_METRIC, SupplementValidator  # noqa: E402
from finglmqa.tabgr_adapter import TabGRAdapter, TabGRRuntimeUnavailable  # noqa: E402


REQUESTS = ROOT / "runs/phase_09/reports/supplement_requests.jsonl"
CANDIDATES = ROOT / "data/indexes/canonical_metric_candidates.jsonl"
TABLE_CORPUS = ROOT / "data/corpus_package/tabgr_table_corpus.jsonl"
TABLE_INDEX = ROOT / "data/indexes/tabgr_table_index.jsonl"
TABLE_CELLS = ROOT / "data/corpus_package/table_cells.jsonl"
FACT_DB = ROOT / "data/facts/financial_facts.duckdb"
FACT_JSONL = ROOT / "data/facts/supplemental_facts.jsonl"
DECISIONS = ROOT / "runs/phase_09/reports/supplement_decisions.jsonl"
TRACE = ROOT / "runs/phase_09/reports/supplement_trace.jsonl"
SUMMARY = ROOT / "runs/phase_09/reports/supplement_summary.json"
SUPPLEMENT_DB = ROOT / "data/facts/supplemental_facts.duckdb"
MAX_LEXICAL_TABLES = 40
MAX_RECORDED_RANKED_CELLS = 25


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            yield value


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row))
    os.replace(temporary, path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    os.replace(temporary, path)


def portable_source(path: object) -> str:
    value = str(path or "")
    marker = "/refs/source_markdown/"
    if marker in value:
        return "refs/source_markdown/" + value.split(marker, 1)[1]
    candidate = Path(value)
    try:
        return candidate.resolve(strict=False).relative_to(ROOT).as_posix()
    except (ValueError, OSError):
        return candidate.name


def _phase6_inventory() -> tuple[
    set[tuple[Any, ...]],
    dict[tuple[Any, ...], list[dict[str, Any]]],
]:
    import duckdb

    connection = duckdb.connect(str(FACT_DB), read_only=True)
    try:
        selected = {tuple(row) for row in connection.execute("""
            SELECT document_id, stock_code, report_year, metric_year,
                   canonical_metric, normalized_unit
            FROM selected_financial_facts
        """).fetchall()}
        values = connection.execute("""
            SELECT document_id, stock_code, report_year, metric_year,
                   canonical_metric, normalized_unit, statement,
                   normalized_value_text, selection_status, conflict_group_id,
                   confidence_score
            FROM financial_facts ORDER BY fact_id
        """).fetchall()
    finally:
        connection.close()
    inventory: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in values:
        inventory[tuple(row[:6])].append({
            "statement": row[6], "normalized_value": row[7],
            "selection_status": row[8], "conflict_group_id": row[9],
            "confidence_score": str(row[10]),
        })
    return selected, inventory


def _classify(
    request: Mapping[str, Any],
    selected: set[tuple[Any, ...]],
    inventory: Mapping[tuple[Any, ...], list[dict[str, Any]]],
) -> tuple[str, list[dict[str, Any]]]:
    key = tuple(canonical_slot_key(request))
    rows = list(inventory.get(key, []))
    if key in selected:
        return "already_selected", rows
    if any(row["selection_status"] == "unresolved_conflict" for row in rows):
        return "conflict_group_open", rows
    if rows:
        return "fact_withheld", rows
    return "no_exact_unit_fact", rows


def _metric_text(table: Mapping[str, Any]) -> str:
    return " ".join([
        *(str(row) for row in table.get("section_path") or []),
        str(table.get("caption") or table.get("table_caption") or ""),
        *(str(row) for row in table.get("header") or []),
        str(table.get("matrix_text") or ""),
    ]).replace(" ", "")


def _source_provenance(
    request: Mapping[str, Any],
    table: Mapping[str, Any],
    cell: Mapping[str, Any],
    ranked: Mapping[str, Any],
    detail: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "document_id": request["document_id"],
        "table_id": table["table_id"],
        "table_index": int(table["table_index"]),
        "row_index": int(cell["row_index"]),
        "col_index": int(cell["col_index"]),
        "row_label": str(cell.get("row_label") or ""),
        "column_label": str(cell.get("column_label") or ""),
        "raw_value": str(cell["raw_value"]),
        "source_markdown": portable_source(cell.get("source_markdown")),
        "line_range": list(cell.get("line_range") or [None, None]),
        "section_path": list(cell.get("section_path") or []),
        "raw_markdown_sha1": table["raw_markdown_sha1"],
        "tabgr_score": ranked["score"],
        "ranked_cell_fingerprint": ranked["cell_fingerprint"],
        "metric_source": detail["metric_source"],
        "metric_label": detail["metric_label"],
        "period": detail["period"],
        "unit_source": detail["unit_source"],
        "confidence_score": detail["confidence_score"],
        "score_reasons": detail["score_reasons"],
        "validation_fingerprint": detail["validation_fingerprint"],
    }


def build(
    *,
    requests_path: Path = REQUESTS,
    fact_jsonl: Path = FACT_JSONL,
    decisions_path: Path = DECISIONS,
    trace_path: Path = TRACE,
    summary_path: Path = SUMMARY,
    database_path: Path = SUPPLEMENT_DB,
) -> dict[str, Any]:
    requests = [validate_missing_fact_request(row) for row in iter_jsonl(requests_path)]
    if len(requests) != 1421:
        raise RuntimeError("Phase 9 full batch requires exactly 1,421 requests")
    selected, inventory = _phase6_inventory()
    validator = SupplementValidator()
    try:
        adapter = TabGRAdapter()
    except TabGRRuntimeUnavailable:
        # The batch has no lexical-only fallback. No artifact is replaced.
        raise

    request_by_id = {row["requirement_id"]: row for row in requests}
    classifications: dict[str, tuple[str, list[dict[str, Any]]]] = {
        request_id: _classify(request, selected, inventory)
        for request_id, request in request_by_id.items()
    }
    executable = {
        request_id: request for request_id, request in request_by_id.items()
        if classifications[request_id][0] == "no_exact_unit_fact"
    }
    if len(executable) != 890:
        raise RuntimeError(f"expected 890 executable exact-unit gaps, found {len(executable)}")

    tier_b: dict[tuple[str, str], set[str]] = defaultdict(set)
    executable_doc_metrics = {(row["document_id"], row["canonical_metric"]) for row in executable.values()}
    for candidate in iter_jsonl(CANDIDATES):
        key = (str(candidate.get("document_id")), str(candidate.get("canonical_metric")))
        if key in executable_doc_metrics and isinstance(candidate.get("table_id"), str):
            tier_b[key].add(candidate["table_id"])

    lexical: dict[tuple[str, str], list[str]] = defaultdict(list)
    table_records: dict[str, dict[str, Any]] = {}
    validator_aliases = {metric: tuple(alias.replace(" ", "") for alias in validator.aliases(metric)) for _, metric in executable_doc_metrics}
    needed_metrics_by_doc: dict[str, set[str]] = defaultdict(set)
    for document_id, metric in executable_doc_metrics:
        needed_metrics_by_doc[document_id].add(metric)
    last_table_order: dict[str, tuple[int, str]] = {}
    for table in iter_jsonl(TABLE_CORPUS):
        document_id = str(table.get("document_id"))
        metrics = needed_metrics_by_doc.get(document_id)
        if not metrics or table.get("tabgr_ready") is not True:
            continue
        table_id = str(table["table_id"])
        table_order = (int(table["table_index"]), table_id)
        if document_id in last_table_order and table_order <= last_table_order[document_id]:
            raise RuntimeError("TabGR corpus is not ordered by (table_index, table_id) within document")
        last_table_order[document_id] = table_order
        chosen = any(table_id in tier_b[(document_id, metric)] for metric in metrics)
        text = _metric_text(table)
        for metric in sorted(metrics):
            key = (document_id, metric)
            if len(lexical[key]) >= MAX_LEXICAL_TABLES:
                continue
            aliases = sorted(validator_aliases[metric], key=lambda item: (-len(item), item))
            if any(alias and alias in text for alias in aliases):
                lexical[key].append(table_id)
                chosen = True
        if chosen:
            table_records[table_id] = table

    shortlist_by_request: dict[str, list[dict[str, Any]]] = {}
    table_to_requests: dict[str, list[str]] = defaultdict(list)
    for request_id, request in executable.items():
        seen: set[str] = set()
        groups: list[dict[str, Any]] = []
        tiers = (
            ("a", list(request["candidate_table_ids"])),
            ("b", sorted(tier_b[(request["document_id"], request["canonical_metric"])], key=lambda table_id: (int(table_records.get(table_id, {}).get("table_index", 10**9)), table_id))),
            ("c", lexical[(request["document_id"], request["canonical_metric"])]),
        )
        for tier, values in tiers:
            ids = [table_id for table_id in values if table_id not in seen and table_id in table_records]
            seen.update(ids)
            groups.append({"tier": tier, "table_ids": ids})
            for table_id in ids:
                table_to_requests[table_id].append(request_id)
        shortlist_by_request[request_id] = groups
    for table_id in table_to_requests:
        table_to_requests[table_id].sort()

    wanted_tables = set(table_to_requests)
    index_records = {
        row["table_id"]: row
        for row in iter_jsonl(TABLE_INDEX)
        if row.get("table_id") in wanted_tables
    }
    if set(index_records) != wanted_tables:
        raise RuntimeError("shortlisted table lacks Phase 4 index provenance")

    states: dict[str, dict[str, Any]] = {
        request_id: {
            "ranked": [], "accepted": [], "stage_counts": Counter(),
            "adapter_counts": Counter(), "scored_tables": 0,
        }
        for request_id in executable
    }

    def process_table(table_id: str, cells: list[dict[str, Any]]) -> None:
        if table_id not in wanted_tables:
            return
        table = table_records[table_id]
        index = index_records[table_id]
        cell_by_coord = {(int(row["row_index"]), int(row["col_index"])): row for row in cells}
        for request_id in table_to_requests[table_id]:
            request = executable[request_id]
            ranked_result = adapter.rank_table(
                table, cells, aliases=validator.aliases(request["canonical_metric"]),
                metric_year=request["metric_year"], normalized_unit=request["normalized_unit"],
            )
            state = states[request_id]
            state["scored_tables"] += 1
            state["adapter_counts"].update(ranked_result["audit_counts"])
            for ranked in ranked_result["ranked_cells"]:
                state["ranked"].append({
                    "score": ranked["score"], "table_index": ranked["table_index"],
                    "row_index": ranked["row_index"], "col_index": ranked["col_index"],
                    "cell_fingerprint": ranked["cell_fingerprint"],
                })
                cell = cell_by_coord[(ranked["row_index"], ranked["col_index"])]
                outcome = validator.validate_cell(request, cell, table, index)
                state["stage_counts"][(outcome.stage, outcome.failure_code)] += 1
                if outcome.accepted:
                    state["accepted"].append({
                        "value": outcome.detail["normalized_value"],
                        "source": _source_provenance(request, table, cell, ranked, outcome.detail),
                    })

    current_table_id: str | None = None
    current_cells: list[dict[str, Any]] = []
    for cell in iter_jsonl(TABLE_CELLS):
        table_id = str(cell["table_id"])
        if table_id != current_table_id:
            if current_table_id is not None:
                process_table(current_table_id, current_cells)
            current_table_id = table_id
            current_cells = []
        if table_id in wanted_tables:
            current_cells.append(cell)
    if current_table_id is not None:
        process_table(current_table_id, current_cells)

    facts: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for request in requests:
        request_id = request["requirement_id"]
        request_class, phase6_rows = classifications[request_id]
        accepted_fact: dict[str, Any] | None = None
        failure_code: str | None
        shortlist = shortlist_by_request.get(request_id, [])
        ranked_summary: list[dict[str, Any]] = []
        rejected_values: list[str] = []
        audit_counts: dict[str, Any]
        if request_class == "already_selected":
            failure_code = "SUPPLEMENT_ALREADY_SELECTED"
            audit_counts = {"guard_rows": len(phase6_rows)}
        elif request_class == "conflict_group_open":
            failure_code = "SUPPLEMENT_CONFLICT_GROUP_OPEN"
            rejected_values = sorted({str(row["normalized_value"]) for row in phase6_rows})
            audit_counts = {"guard_rows": len(phase6_rows)}
        elif request_class == "fact_withheld":
            failure_code = "SUPPLEMENT_FACT_WITHHELD"
            rejected_values = sorted({str(row["normalized_value"]) for row in phase6_rows})
            audit_counts = {"guard_rows": len(phase6_rows)}
        else:
            state = states[request_id]
            ranked_summary = sorted(
                state["ranked"],
                key=lambda row: (-float(row["score"]), row["table_index"], row["row_index"], row["col_index"], row["cell_fingerprint"]),
            )[:MAX_RECORDED_RANKED_CELLS]
            audit_counts = {
                "shortlisted_tables": sum(len(row["table_ids"]) for row in shortlist),
                "scored_tables": state["scored_tables"],
                "ranked_cells_total": len(state["ranked"]),
                "accepted_source_cells": len(state["accepted"]),
                **{key: int(value) for key, value in sorted(state["adapter_counts"].items())},
            }
            if not table_to_requests or audit_counts["shortlisted_tables"] == 0:
                failure_code = "SUPPLEMENT_NO_CANDIDATE_TABLE"
            else:
                by_value: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in state["accepted"]:
                    by_value[row["value"]].append(row["source"])
                has_provenance_failure = any(
                    stage == 6 and code == "SUPPLEMENT_PROVENANCE_FAILED"
                    for stage, code in state["stage_counts"]
                )
                if has_provenance_failure:
                    failure_code = "SUPPLEMENT_PROVENANCE_FAILED"
                    rejected_values = sorted(by_value)
                elif len(by_value) > 1:
                    failure_code = "SUPPLEMENT_VALUE_CONFLICT"
                    rejected_values = sorted(by_value)
                elif len(by_value) == 1:
                    value = next(iter(by_value))
                    # Any retained Phase 6 value with this exact unit/group is
                    # a hard conflict, even if it was not selected.
                    retained = {
                        str(row["normalized_value"]) for row in phase6_rows
                        if row["statement"] == STATEMENT_BY_METRIC[request["canonical_metric"]]
                    }
                    if retained and retained != {value}:
                        failure_code = "SUPPLEMENT_VALUE_CONFLICT"
                        rejected_values = sorted({value, *retained})
                    else:
                        sources = sorted(
                            by_value[value],
                            key=lambda row: (row["table_index"], row["row_index"], row["col_index"], row["table_id"]),
                        )
                        coordinates = [[row["table_id"], row["row_index"], row["col_index"]] for row in sources]
                        fact_id = "sf_" + hashlib.sha256(canonical_json_bytes([canonical_slot_key(request), value, coordinates])).hexdigest()[:24]
                        primary = sources[0]
                        line_range = primary["line_range"]
                        accepted_fact = {
                            "supplemental_fact_id": fact_id,
                            "schema_version": SCHEMA_SUPPLEMENTAL_FACT,
                            **{field: request[field] for field in ("document_id", "stock_code", "report_year", "metric_year", "canonical_metric", "normalized_unit")},
                            "company": str(table_records[primary["table_id"]].get("stock_name") or table_records[primary["table_id"]].get("company_full")),
                            "statement": STATEMENT_BY_METRIC[request["canonical_metric"]],
                            "normalized_value": value,
                            "fact_source": FACT_SOURCE,
                            "validation_versions": validator.validation_versions,
                            "tabgr_trace_fingerprint": adapter.trace_fingerprint,
                            "source_table_id": primary["table_id"],
                            "source_table_index": primary["table_index"],
                            "source_row_index": primary["row_index"],
                            "source_col_index": primary["col_index"],
                            "source_line_start": line_range[0],
                            "source_line_end": line_range[1],
                            "source_markdown": primary["source_markdown"],
                            "provenance_json": sources,
                            "created_from_requirement_ids": [request_id],
                        }
                        validate_supplemental_fact(accepted_fact)
                        facts.append(accepted_fact)
                        failure_code = None
                else:
                    stage_counts: Counter[tuple[int, str | None]] = state["stage_counts"]
                    if not stage_counts:
                        failure_code = "SUPPLEMENT_CELL_NOT_FOUND"
                    else:
                        # Deepest stage reached; provenance is stage 6 and
                        # therefore overrides every ordinary cell rejection.
                        stage, code = max(stage_counts, key=lambda item: (item[0], str(item[1])))
                        failure_code = str(code or "SUPPLEMENT_CELL_NOT_FOUND")
        decision_status = "accepted" if accepted_fact is not None else "rejected"
        conflict_ids = sorted({str(row["conflict_group_id"]) for row in phase6_rows if row.get("conflict_group_id")})
        trace_semantics = {
            "slot_key": canonical_slot_key(request),
            "request_class": request_class,
            "failure_code": failure_code,
            "shortlist": shortlist,
            "ranked_cells": ranked_summary,
            "audit_counts": audit_counts,
            "accepted_fact_id": accepted_fact["supplemental_fact_id"] if accepted_fact else None,
            "rejected_values": rejected_values,
        }
        trace_fingerprint = semantic_sha256(trace_semantics)
        decision = {
            "schema_version": SCHEMA_SUPPLEMENT_DECISION,
            "slot_key": canonical_slot_key(request),
            "slot_fingerprint": slot_fingerprint(request),
            "requirement_id": request_id,
            "decision_status": decision_status,
            "failure_code": failure_code,
            "request_class": request_class,
            "conflict_group_ids": conflict_ids,
            "shortlist": shortlist,
            "ranked_cell_fingerprints": [row["cell_fingerprint"] for row in ranked_summary],
            "rejected_values": rejected_values,
            "accepted_fact_id": accepted_fact["supplemental_fact_id"] if accepted_fact else None,
            "audit_counts": audit_counts,
            "trace_fingerprint": trace_fingerprint,
        }
        validate_supplement_decision(decision)
        decisions.append(decision)
        traces.append({
            "schema_version": "finglmqa.phase9.supplement_trace.v1",
            "requirement_id": request_id,
            "trace_fingerprint": trace_fingerprint,
            **trace_semantics,
        })

    facts.sort(key=lambda row: row["supplemental_fact_id"])
    decisions.sort(key=lambda row: row["slot_key"])
    traces.sort(key=lambda row: row["slot_key"])
    write_jsonl(fact_jsonl, facts)
    write_jsonl(decisions_path, decisions)
    write_jsonl(trace_path, traces)
    store = materialize_store(facts, decisions, database_path)
    failure_counts = Counter(row["failure_code"] for row in decisions if row["failure_code"])
    metric_counts = Counter(request_by_id[row["requirement_id"]]["canonical_metric"] for row in decisions if row["decision_status"] == "accepted")
    offset_counts = Counter(
        f"Y-{request_by_id[row['requirement_id']]['report_year'] - request_by_id[row['requirement_id']]['metric_year']}"
        if request_by_id[row["requirement_id"]]["report_year"] != request_by_id[row["requirement_id"]]["metric_year"] else "Y"
        for row in decisions if row["decision_status"] == "accepted"
    )
    summary = {
        "schema_version": "finglmqa.phase9.supplement_summary.v1",
        "request_count": len(requests),
        "decision_count": len(decisions),
        "accepted_fact_count": len(facts),
        "rejected_request_count": len(decisions) - len(facts),
        "counts_by_failure_code": dict(sorted(failure_counts.items())),
        "accepted_counts_by_metric": dict(sorted(metric_counts.items())),
        "accepted_counts_by_year_offset": dict(sorted(offset_counts.items())),
        "artifact_hashes": {
            "supplemental_facts_jsonl": sha256_file(fact_jsonl),
            "supplement_decisions_jsonl": sha256_file(decisions_path),
            "supplement_trace_jsonl": sha256_file(trace_path),
            "supplemental_facts_duckdb": store["duckdb_sha256"],
        },
        "tabgr_trace_fingerprint": adapter.trace_fingerprint,
        "runtime_telemetry": None,
    }
    write_json(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, default=REQUESTS)
    parser.add_argument("--facts", type=Path, default=FACT_JSONL)
    parser.add_argument("--decisions", type=Path, default=DECISIONS)
    parser.add_argument("--trace", type=Path, default=TRACE)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--database", type=Path, default=SUPPLEMENT_DB)
    args = parser.parse_args()
    summary = build(
        requests_path=args.requests, fact_jsonl=args.facts, decisions_path=args.decisions,
        trace_path=args.trace, summary_path=args.summary, database_path=args.database,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
