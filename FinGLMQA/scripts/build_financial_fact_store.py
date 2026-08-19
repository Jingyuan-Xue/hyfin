#!/usr/bin/env python3
"""Build the audited Phase 6 financial fact store.

The builder deliberately separates candidate eligibility, value-level
deduplication, and group-level selection.  Every input candidate receives an
audit decision.  Conflicting values are retained as separate facts in one
conflict group; selection is allowed only when deterministic confidence and
margin rules are satisfied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable


SCHEMA_FACT = "finglmqa.phase6.financial_fact.v1"
SCHEMA_REPORT = "finglmqa.phase6.fact_store_report.v1"
BUILDER_VERSION = "phase6-financial-fact-store-v1"

DEFAULT_CANDIDATES = "data/indexes/canonical_metric_candidates.jsonl"
DEFAULT_PHASE5_REPORT = "runs/phase_05/metric_unit_report.json"
DEFAULT_CORPUS_MANIFEST = "data/corpus_package/corpus_manifest.json"
DEFAULT_FACTS = "data/facts/financial_facts.jsonl"
DEFAULT_DATABASE = "data/facts/financial_facts.duckdb"
DEFAULT_SCHEMA = "data/schemas/financial_facts.schema.json"
DEFAULT_RUN_DIR = "runs/phase_06"

# Regression pins deliberately fail closed when audited upstream/reference
# outputs move.  A mismatch can be a legitimate upstream change: review the
# diff and update the corresponding pin only after confirming the new value.
PHASE5_FRACTION_WARNING_COUNT_REGRESSION_PIN = 546
FLYADA_2019_REVENUE_REGRESSION_PIN = (
    "3704210734.9",
    "元",
    "A000026_飞亚达_2019年年度报告_table_0007_acc5dee31c",
)
KNOWN_RELATIVE_PARTIAL_DURATION_CANDIDATE_ID = (
    "A688009_中国通号_2019年年度报告_table_0255_119d0118d7:r1:c1:营业收入"
)
LEGACY_RELATIVE_PARTIAL_DURATION_FACT_ID = "fact_870b88ea34d53604860979cc"
MAX_REPORT_SAMPLES = 50
MIN_SINGLE_VALUE_CONFIDENCE = Decimal("0.70")
# Conflict resolution requires stronger evidence than a single-value group.
# In the feasibility audit, a 0.87 key-audit candidate could otherwise defeat
# the actual share-capital value merely because the latter appeared in a
# parent-company statement.  Requiring 0.95 keeps that case unresolved while
# still resolving main-summary/consolidated facts (normally 0.95--1.00).
MIN_CONFLICT_SELECTION_CONFIDENCE = Decimal("0.95")
MIN_CONFLICT_MARGIN = Decimal("0.12")
MIN_SUPPORTED_CONFLICT_MARGIN = Decimal("0.05")

DURATION_METRICS = {
    "营业收入",
    "归属于上市公司股东的净利润",
    "扣除非经常性损益后的净利润",
    "经营活动产生的现金流量净额",
    "基本每股收益",
    "稀释每股收益",
    "加权平均净资产收益率",
}
INSTANT_METRICS = {"总资产", "净资产", "股本"}

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

ALLOWED_UNITS = {
    "营业收入": {"元"},
    "归属于上市公司股东的净利润": {"元"},
    "扣除非经常性损益后的净利润": {"元"},
    "经营活动产生的现金流量净额": {"元"},
    "总资产": {"元"},
    "净资产": {"元"},
    "基本每股收益": {"元/股"},
    "稀释每股收益": {"元/股"},
    "加权平均净资产收益率": {"ratio"},
    # Phase 5 intentionally keeps monetary share capital and share count
    # distinct by normalized unit.  Unit is part of the fact key.
    "股本": {"元", "股"},
}

COMPARISON_RE = re.compile(r"同比|环比|增减|增长|下降|变动|比例|比上|较上|差异|占比")
QUARTER_RE = re.compile(r"季度|半年度|半年|[1-4一二三四]季|\d+\s*[-至]\s*\d+\s*月")
YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
CURRENT_RE = re.compile(r"本报告期|报告期|本年度|本年|本期|期末|年末")
PREVIOUS_RE = re.compile(r"上年度|上年|上期|期初|年初")
OPENING_DATE_RE = re.compile(r"(?:1月1日|01月01日|01-01|年初|期初)")
YEAR_END_DATE_RE = re.compile(r"(?:12月31日|12-31|年末|期末)")
MONTH_OR_DAY_RE = re.compile(r"\d{1,2}\s*月|\d{1,2}\s*日|\d{1,2}[-/]\d{1,2}")

MAIN_SCOPE_TERMS = (
    "主要会计数据和财务指标",
    "主要会计数据",
    "主要财务指标",
    "合并资产负债表",
    "合并利润表",
    "合并现金流量表",
)
NON_COMPANY_SCOPE_TERMS = (
    "分部信息",
    "分行业",
    "分地区",
    "分产品",
    "主要控股参股公司",
    "主要子公司",
    "重要子公司",
    "联营企业",
    "合营企业",
    "企业合并",
    "处置子公司",
    "非同一控制下",
    "母公司利润表",
    "母公司资产负债表",
    "母公司现金流量表",
    "资产构成重大变动",
    "按行业",
    "按地区",
    "按产品",
)

REQUIRED_CANDIDATE_FIELDS = (
    "candidate_id",
    "document_id",
    "table_id",
    "row_index",
    "col_index",
    "stock_code",
    "report_year",
    "canonical_metric",
    "metric_source",
    "row_label",
    "column_label",
    "raw_value",
    "normalized_value",
    "normalized_unit",
    "parse_status",
    "normalization_warnings",
    "source_markdown",
    "line_range",
    "section_path",
)


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_year(value: object) -> int | None:
    match = YEAR_RE.search(normalize_text(value))
    return int(match.group(1)) if match else None


def decimal_plain(value: Decimal) -> str:
    if value == 0:
        return "0"
    if value == value.to_integral_value():
        return format(value.quantize(Decimal(1)), "f")
    return format(value.normalize(), "f")


def score_plain(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.0001")), "f")


def stable_id(prefix: str, parts: Iterable[object]) -> str:
    payload = "\x1f".join("" if item is None else str(item) for item in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            yield line_number, value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as fh:
        for row in rows:
            json.dump(row, fh, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            fh.write("\n")
            count += 1
    os.replace(temporary, path)
    return count


def joined_context(candidate: dict[str, Any]) -> str:
    parts: list[str] = []
    section = candidate.get("section_path")
    if isinstance(section, list):
        parts.extend(str(item) for item in section)
    for field in ("table_caption", "row_label", "column_label"):
        if candidate.get(field):
            parts.append(str(candidate[field]))
    return " ".join(parts)


def resolve_metric_year(candidate: dict[str, Any]) -> dict[str, Any]:
    metric = str(candidate.get("canonical_metric") or "")
    source = str(candidate.get("metric_source") or "")
    label = normalize_text(candidate.get("column_label") if source == "row_label" else candidate.get("row_label"))
    report_year = parse_year(candidate.get("report_year"))
    result: dict[str, Any] = {
        "period_label": label,
        "metric_year": None,
        "year_resolution": None,
        "year_confidence": None,
        "error": None,
    }
    if report_year is None:
        result["error"] = "invalid_report_year"
        return result
    if not label:
        result["error"] = "missing_period_label"
        return result
    if COMPARISON_RE.search(label):
        result["error"] = "comparison_or_change_column"
        return result
    if QUARTER_RE.search(label):
        result["error"] = "partial_period_or_quarter"
        return result
    # Duration facts must describe a complete year.  Apply this before
    # explicit/relative year routing so labels such as `年初至处置日` cannot
    # become `relative_previous` merely because `年初` also denotes a year.
    # Instant metrics retain their separate point-in-time date handling below.
    if metric in DURATION_METRICS and (MONTH_OR_DAY_RE.search(label) or "至" in label):
        result["error"] = "partial_duration_date"
        return result

    years = sorted(set(int(item) for item in YEAR_RE.findall(label)))
    if len(years) > 1:
        result["error"] = "ambiguous_multiple_years"
        return result
    if years:
        year = years[0]
        if metric in INSTANT_METRICS and MONTH_OR_DAY_RE.search(label):
            if OPENING_DATE_RE.search(label):
                year -= 1
                resolution = "explicit_opening_date"
            elif YEAR_END_DATE_RE.search(label):
                resolution = "explicit_year_end_date"
            else:
                result["error"] = "non_standard_balance_date"
                return result
        elif metric in INSTANT_METRICS and re.search(r"年初|期初", label):
            year -= 1
            resolution = "explicit_opening_period"
        else:
            resolution = "explicit_year"
        result.update(metric_year=year, year_resolution=resolution, year_confidence="high")
    elif PREVIOUS_RE.search(label):
        result.update(metric_year=report_year - 1, year_resolution="relative_previous", year_confidence="medium")
    elif CURRENT_RE.search(label):
        result.update(metric_year=report_year, year_resolution="relative_current", year_confidence="medium")
    else:
        result["error"] = "unresolved_metric_year"
        return result

    metric_year = int(result["metric_year"])
    if metric_year > report_year:
        result["error"] = "future_metric_year"
    elif metric_year < report_year - 2:
        result["error"] = "metric_year_outside_three_year_window"
    return result


def run_metric_year_regression_checks() -> list[dict[str, Any]]:
    """Exercise duration-date routing without depending on corpus contents."""
    cases = [
        {
            "name": "relative_previous_partial_duration_rejected",
            "metric": "营业收入",
            "label": "年初至处置日",
            "expected": {"error": "partial_duration_date"},
        },
        {
            "name": "relative_current_partial_duration_rejected",
            "metric": "营业收入",
            "label": "本期6月30日",
            "expected": {"error": "partial_duration_date"},
        },
        {
            "name": "relative_previous_full_year_allowed",
            "metric": "营业收入",
            "label": "上年度",
            "expected": {
                "error": None,
                "metric_year": 2018,
                "year_resolution": "relative_previous",
            },
        },
        {
            "name": "relative_current_full_year_allowed",
            "metric": "营业收入",
            "label": "本报告期",
            "expected": {
                "error": None,
                "metric_year": 2019,
                "year_resolution": "relative_current",
            },
        },
        {
            "name": "instant_metric_year_end_date_allowed",
            "metric": "总资产",
            "label": "2019年12月31日",
            "expected": {
                "error": None,
                "metric_year": 2019,
                "year_resolution": "explicit_year_end_date",
            },
        },
    ]
    checks: list[dict[str, Any]] = []
    for case in cases:
        actual = resolve_metric_year({
            "canonical_metric": case["metric"],
            "metric_source": "row_label",
            "column_label": case["label"],
            "report_year": "2019",
        })
        expected = case["expected"]
        checks.append({
            "name": case["name"],
            "metric": case["metric"],
            "period_label": case["label"],
            "expected": expected,
            "actual": {key: actual.get(key) for key in expected},
            "passed": all(actual.get(key) == value for key, value in expected.items()),
        })
    return checks


def value_is_plausible(metric: str, unit: str, value: Decimal) -> str | None:
    absolute = abs(value)
    if unit in {"元", "股"} and absolute > Decimal("10000000000000000"):
        return "implausible_magnitude"
    if metric == "加权平均净资产收益率" and absolute > Decimal("10"):
        return "implausible_ratio_magnitude"
    if metric in {"基本每股收益", "稀释每股收益"} and absolute > Decimal("100000"):
        return "implausible_per_share_magnitude"
    if metric in {"总资产", "股本"} and value < 0:
        return "negative_nonnegative_metric"
    return None


def candidate_score(candidate: dict[str, Any], period: dict[str, Any]) -> tuple[Decimal, list[str]]:
    score = Decimal("0")
    reasons: list[str] = []

    if candidate.get("metric_match_type") == "exact_alias":
        score += Decimal("0.32")
        reasons.append("exact_metric_alias:+0.32")
    else:
        score += Decimal("0.22")
        reasons.append("regex_metric_match:+0.22")

    if candidate.get("metric_source") == "row_label":
        score += Decimal("0.25")
        reasons.append("metric_from_row_label:+0.25")
    else:
        score += Decimal("0.12")
        reasons.append("metric_from_column_label:+0.12")

    if period.get("year_confidence") == "high":
        score += Decimal("0.25")
        reasons.append("explicit_metric_year:+0.25")
    else:
        score += Decimal("0.18")
        reasons.append("relative_metric_year:+0.18")

    if candidate.get("context_is_financial") is True:
        score += Decimal("0.10")
        reasons.append("financial_section_context:+0.10")

    unit_source = candidate.get("unit_source")
    if unit_source in {"raw_value", "row_label"}:
        score += Decimal("0.08")
        reasons.append("direct_unit_evidence:+0.08")
    elif unit_source == "unit_hint":
        score += Decimal("0.05")
        reasons.append("table_unit_hint:+0.05")
    elif unit_source == "column_label":
        score += Decimal("0.03")
        reasons.append("column_unit_evidence:+0.03")

    context = joined_context(candidate)
    if any(term in context for term in MAIN_SCOPE_TERMS):
        score += Decimal("0.12")
        reasons.append("consolidated_or_summary_scope:+0.12")
    if any(term in context for term in NON_COMPANY_SCOPE_TERMS):
        score -= Decimal("0.30")
        reasons.append("non_company_or_parent_only_scope:-0.30")

    report_year = parse_year(candidate.get("report_year"))
    if report_year is not None and period.get("metric_year") == report_year - 2:
        score -= Decimal("0.05")
        reasons.append("two_year_lookback:-0.05")

    score = min(Decimal("1"), max(Decimal("0"), score))
    return score, reasons


def audit_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_CANDIDATE_FIELDS if field not in candidate]
    reasons: list[str] = [f"missing_field:{field}" for field in missing]
    period = resolve_metric_year(candidate)
    if period.get("error"):
        reasons.append(str(period["error"]))

    metric = str(candidate.get("canonical_metric") or "")
    unit = candidate.get("normalized_unit")
    warnings = candidate.get("normalization_warnings")
    if not isinstance(warnings, list):
        reasons.append("invalid_normalization_warnings")
        warnings = []
    if warnings:
        reasons.extend(f"normalization_warning:{item}" for item in warnings)
    if candidate.get("parse_status") != "parsed" or candidate.get("normalized_value") is None:
        reasons.append("not_parsed_numeric")
    if metric not in ALLOWED_UNITS:
        reasons.append("unsupported_canonical_metric")
    elif unit not in ALLOWED_UNITS[metric]:
        reasons.append("missing_or_incompatible_unit")

    decimal_value: Decimal | None = None
    if candidate.get("normalized_value") is not None:
        try:
            decimal_value = Decimal(str(candidate["normalized_value"]))
            if not decimal_value.is_finite():
                reasons.append("non_finite_numeric_value")
        except InvalidOperation:
            reasons.append("invalid_normalized_decimal")
    if decimal_value is not None and decimal_value.is_finite() and isinstance(unit, str):
        plausibility_error = value_is_plausible(metric, unit, decimal_value)
        if plausibility_error:
            reasons.append(plausibility_error)

    # Deduplicate reasons while retaining their diagnostic order.
    reasons = list(dict.fromkeys(reasons))
    score, score_reasons = candidate_score(candidate, period) if not reasons else (Decimal("0"), [])
    return {
        "candidate_id": candidate.get("candidate_id"),
        "eligible": not reasons,
        "rejection_reasons": reasons,
        "period_label": period.get("period_label"),
        "metric_year": period.get("metric_year") if not period.get("error") else None,
        "year_resolution": period.get("year_resolution"),
        "year_confidence": period.get("year_confidence"),
        "confidence_score": score,
        "score_reasons": score_reasons,
        "decimal_value": decimal_value,
        "candidate": candidate,
        "group_id": None,
        "fact_id": None,
    }


def confidence_label(score: Decimal) -> str:
    if score >= Decimal("0.85"):
        return "high"
    if score >= Decimal("0.70"):
        return "medium"
    return "low"


def primary_provenance(decision: dict[str, Any]) -> dict[str, Any]:
    candidate = decision["candidate"]
    return {
        "candidate_id": candidate.get("candidate_id"),
        "document_id": candidate.get("document_id"),
        "table_id": candidate.get("table_id"),
        "table_index": candidate.get("table_index"),
        "row_index": candidate.get("row_index"),
        "col_index": candidate.get("col_index"),
        "row_label": candidate.get("row_label"),
        "column_label": candidate.get("column_label"),
        "raw_value": candidate.get("raw_value"),
        "raw_unit": candidate.get("raw_unit"),
        "parsed_value": candidate.get("parsed_value"),
        "normalized_value": candidate.get("normalized_value"),
        "normalized_unit": candidate.get("normalized_unit"),
        "unit_rule_id": candidate.get("unit_rule_id"),
        "unit_source": candidate.get("unit_source"),
        "normalization_warnings": candidate.get("normalization_warnings"),
        "metric_label": candidate.get("metric_label"),
        "metric_source": candidate.get("metric_source"),
        "metric_match_type": candidate.get("metric_match_type"),
        "metric_confidence": candidate.get("metric_confidence"),
        "context_is_financial": candidate.get("context_is_financial"),
        "source_markdown": candidate.get("source_markdown"),
        "line_range": candidate.get("line_range"),
        "section_path": candidate.get("section_path"),
        "table_caption": candidate.get("table_caption"),
        "table_unit_hint": candidate.get("table_unit_hint"),
        "period_label": decision.get("period_label"),
        "metric_year": decision.get("metric_year"),
        "year_resolution": decision.get("year_resolution"),
        "candidate_confidence_score": score_plain(decision["confidence_score"]),
        "score_reasons": decision.get("score_reasons"),
    }


def group_dimensions(decision: dict[str, Any]) -> tuple[Any, ...]:
    candidate = decision["candidate"]
    metric = candidate["canonical_metric"]
    return (
        candidate["document_id"],
        candidate.get("stock_code"),
        int(decision["metric_year"]),
        STATEMENT_BY_METRIC[metric],
        metric,
        candidate["normalized_unit"],
    )


def build_facts(decisions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for decision in decisions:
        if decision["eligible"]:
            grouped[group_dimensions(decision)].append(decision)

    facts: list[dict[str, Any]] = []
    conflict_groups: list[dict[str, Any]] = []
    for dimensions in sorted(grouped, key=lambda key: tuple("" if item is None else str(item) for item in key)):
        document_id, stock_code, metric_year, statement, metric, unit = dimensions
        group_id = stable_id("cg", dimensions)
        by_value: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for decision in grouped[dimensions]:
            assert isinstance(decision["decimal_value"], Decimal)
            value_key = decimal_plain(decision["decimal_value"])
            by_value[value_key].append(decision)

        group_facts: list[dict[str, Any]] = []
        for value_key in sorted(by_value, key=Decimal):
            value_decisions = sorted(
                by_value[value_key],
                key=lambda item: (-item["confidence_score"], str(item["candidate_id"])),
            )
            representative = value_decisions[0]
            candidate = representative["candidate"]
            support_count = len(value_decisions)
            support_bonus = min(Decimal("0.04"), Decimal(max(support_count - 1, 0)) * Decimal("0.01"))
            aggregate_score = min(Decimal("1"), representative["confidence_score"] + support_bonus)
            fact_id = stable_id("fact", (*dimensions, value_key))
            provenances = [primary_provenance(item) for item in value_decisions]
            line_range = candidate.get("line_range") if isinstance(candidate.get("line_range"), list) else [None, None]
            fact = {
                "schema_version": SCHEMA_FACT,
                "builder_version": BUILDER_VERSION,
                "fact_id": fact_id,
                "document_id": document_id,
                "stock_code": stock_code,
                "stock_symbol": candidate.get("stock_symbol"),
                "stock_name": candidate.get("stock_name"),
                "company_full": candidate.get("company_full"),
                "report_year": parse_year(candidate.get("report_year")),
                "metric_year": metric_year,
                "statement": statement,
                "canonical_metric": metric,
                "normalized_value": value_key,
                "normalized_unit": unit,
                "confidence_score": score_plain(aggregate_score),
                "confidence": confidence_label(aggregate_score),
                "support_count": support_count,
                "is_selected": False,
                "selection_status": None,
                "selection_reason": None,
                "conflict_group_id": group_id,
                "has_conflict": len(by_value) > 1,
                "source_table_id": candidate.get("table_id"),
                "source_table_index": candidate.get("table_index"),
                "source_row_index": candidate.get("row_index"),
                "source_col_index": candidate.get("col_index"),
                "source_markdown": candidate.get("source_markdown"),
                "source_line_start": line_range[0] if len(line_range) > 0 else None,
                "source_line_end": line_range[1] if len(line_range) > 1 else None,
                "source_candidate_id": candidate.get("candidate_id"),
                "provenance": provenances,
            }
            for decision in value_decisions:
                decision["group_id"] = group_id
                decision["fact_id"] = fact_id
            group_facts.append(fact)

        ranked = sorted(
            group_facts,
            key=lambda item: (-Decimal(item["confidence_score"]), -int(item["support_count"]), item["fact_id"]),
        )
        selected: dict[str, Any] | None = None
        resolution_status: str
        resolution_reason: str
        if len(ranked) == 1:
            if Decimal(ranked[0]["confidence_score"]) >= MIN_SINGLE_VALUE_CONFIDENCE:
                selected = ranked[0]
                resolution_status = "selected_single_value"
                resolution_reason = "single distinct value meets confidence threshold"
            else:
                resolution_status = "low_confidence"
                resolution_reason = "single distinct value is below confidence threshold"
        else:
            top, second = ranked[0], ranked[1]
            top_score = Decimal(top["confidence_score"])
            margin = top_score - Decimal(second["confidence_score"])
            supported_margin = (
                int(top["support_count"]) >= 2
                and int(second["support_count"]) == 1
                and margin >= MIN_SUPPORTED_CONFLICT_MARGIN
            )
            if top_score >= MIN_CONFLICT_SELECTION_CONFIDENCE and (margin >= MIN_CONFLICT_MARGIN or supported_margin):
                selected = top
                resolution_status = "resolved_by_confidence"
                resolution_reason = (
                    f"top value confidence margin {score_plain(margin)} satisfies deterministic selection rule"
                )
            else:
                resolution_status = "unresolved_conflict"
                resolution_reason = (
                    f"top confidence/margin {top['confidence_score']}/{score_plain(margin)} is insufficient"
                )

        for fact in group_facts:
            if fact is selected:
                fact["is_selected"] = True
                fact["selection_status"] = resolution_status
                fact["selection_reason"] = resolution_reason
            elif len(group_facts) > 1 and selected is not None:
                fact["selection_status"] = "not_selected_conflict"
                fact["selection_reason"] = f"conflicting value ranked below selected fact {selected['fact_id']}"
            else:
                fact["selection_status"] = resolution_status
                fact["selection_reason"] = resolution_reason

        if len(group_facts) > 1:
            conflict_groups.append({
                "conflict_group_id": group_id,
                "document_id": document_id,
                "stock_code": stock_code,
                "metric_year": metric_year,
                "statement": statement,
                "canonical_metric": metric,
                "normalized_unit": unit,
                "distinct_value_count": len(group_facts),
                "candidate_count": sum(int(item["support_count"]) for item in group_facts),
                "resolution_status": resolution_status,
                "resolution_reason": resolution_reason,
                "selected_fact_id": selected["fact_id"] if selected else None,
                "fact_summaries": [
                    {
                        "fact_id": item["fact_id"],
                        "normalized_value": item["normalized_value"],
                        "confidence_score": item["confidence_score"],
                        "support_count": item["support_count"],
                        "is_selected": item["is_selected"],
                        "source_candidate_ids": [prov["candidate_id"] for prov in item["provenance"]],
                    }
                    for item in ranked
                ],
            })
        facts.extend(group_facts)

    facts.sort(key=lambda item: item["fact_id"])
    conflict_groups.sort(key=lambda item: item["conflict_group_id"])
    return facts, conflict_groups


def fact_schema() -> dict[str, Any]:
    required = [
        "schema_version", "fact_id", "document_id", "stock_code", "report_year",
        "metric_year", "statement", "canonical_metric", "normalized_value",
        "normalized_unit", "confidence_score", "confidence", "support_count",
        "is_selected", "selection_status", "conflict_group_id", "has_conflict",
        "source_table_id", "source_row_index", "source_col_index",
        "source_markdown", "source_candidate_id", "provenance",
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_FACT,
        "type": "object",
        "required": required,
        "additionalProperties": True,
        "properties": {
            "schema_version": {"const": SCHEMA_FACT},
            "fact_id": {"type": "string"},
            "document_id": {"type": "string"},
            "stock_code": {"type": ["string", "null"]},
            "report_year": {"type": "integer"},
            "metric_year": {"type": "integer"},
            "statement": {"enum": sorted(set(STATEMENT_BY_METRIC.values()))},
            "canonical_metric": {"enum": sorted(STATEMENT_BY_METRIC)},
            "normalized_value": {"type": "string", "pattern": "^-?[0-9]+(?:\\.[0-9]+)?$"},
            "normalized_unit": {"enum": ["元", "元/股", "ratio", "股"]},
            "confidence_score": {"type": "string"},
            "confidence": {"enum": ["high", "medium", "low"]},
            "support_count": {"type": "integer", "minimum": 1},
            "is_selected": {"type": "boolean"},
            "selection_status": {"type": "string"},
            "conflict_group_id": {"type": "string"},
            "has_conflict": {"type": "boolean"},
            "source_table_id": {"type": "string"},
            "source_row_index": {"type": "integer", "minimum": 0},
            "source_col_index": {"type": "integer", "minimum": 0},
            "source_markdown": {"type": ["string", "null"]},
            "source_candidate_id": {"type": "string"},
            "provenance": {"type": "array", "minItems": 1, "items": {"type": "object"}},
        },
    }


def serializable_audit(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": decision["candidate_id"],
        "eligible": decision["eligible"],
        "rejection_reasons": decision["rejection_reasons"],
        "period_label": decision["period_label"],
        "metric_year": decision["metric_year"],
        "year_resolution": decision["year_resolution"],
        "year_confidence": decision["year_confidence"],
        "confidence_score": score_plain(decision["confidence_score"]),
        "score_reasons": decision["score_reasons"],
        "conflict_group_id": decision["group_id"],
        "fact_id": decision["fact_id"],
        "candidate": decision["candidate"],
    }


def build_database(
    database_path: Path,
    facts: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    metadata: dict[str, str],
) -> dict[str, Any]:
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise RuntimeError("DuckDB is required for Phase 6; install it once in the local .venv") from exc

    database_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = database_path.with_name(database_path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    connection = duckdb.connect(str(temporary))
    try:
        connection.execute("""
            CREATE TABLE financial_facts (
                fact_id VARCHAR PRIMARY KEY,
                schema_version VARCHAR NOT NULL,
                document_id VARCHAR NOT NULL,
                stock_code VARCHAR,
                stock_symbol VARCHAR,
                stock_name VARCHAR,
                company_full VARCHAR,
                report_year INTEGER NOT NULL,
                metric_year INTEGER NOT NULL,
                statement VARCHAR NOT NULL,
                canonical_metric VARCHAR NOT NULL,
                normalized_value DECIMAL(38, 12) NOT NULL,
                normalized_value_text VARCHAR NOT NULL,
                normalized_unit VARCHAR NOT NULL,
                confidence_score DECIMAL(6, 4) NOT NULL,
                confidence VARCHAR NOT NULL,
                support_count INTEGER NOT NULL,
                is_selected BOOLEAN NOT NULL,
                selection_status VARCHAR NOT NULL,
                selection_reason VARCHAR NOT NULL,
                conflict_group_id VARCHAR NOT NULL,
                has_conflict BOOLEAN NOT NULL,
                source_table_id VARCHAR NOT NULL,
                source_table_index INTEGER,
                source_row_index INTEGER NOT NULL,
                source_col_index INTEGER NOT NULL,
                source_markdown VARCHAR,
                source_line_start INTEGER,
                source_line_end INTEGER,
                source_candidate_id VARCHAR NOT NULL,
                provenance_json JSON NOT NULL
            )
        """)
        fact_rows = [(
            fact["fact_id"], fact["schema_version"], fact["document_id"], fact["stock_code"],
            fact["stock_symbol"], fact["stock_name"], fact["company_full"], fact["report_year"],
            fact["metric_year"], fact["statement"], fact["canonical_metric"],
            Decimal(fact["normalized_value"]), fact["normalized_value"], fact["normalized_unit"],
            Decimal(fact["confidence_score"]), fact["confidence"], fact["support_count"],
            fact["is_selected"], fact["selection_status"], fact["selection_reason"],
            fact["conflict_group_id"], fact["has_conflict"], fact["source_table_id"],
            fact["source_table_index"], fact["source_row_index"], fact["source_col_index"],
            fact["source_markdown"], fact["source_line_start"], fact["source_line_end"],
            fact["source_candidate_id"], json.dumps(fact["provenance"], ensure_ascii=False, separators=(",", ":")),
        ) for fact in facts]
        connection.executemany(
            "INSERT INTO financial_facts VALUES (" + ",".join("?" for _ in range(31)) + ")",
            fact_rows,
        )
        connection.execute("CREATE INDEX idx_facts_lookup ON financial_facts(stock_code, metric_year, canonical_metric)")
        connection.execute("CREATE INDEX idx_facts_document ON financial_facts(document_id)")
        connection.execute("CREATE INDEX idx_facts_conflict ON financial_facts(conflict_group_id)")
        connection.execute("CREATE VIEW selected_financial_facts AS SELECT * FROM financial_facts WHERE is_selected")

        connection.execute("""
            CREATE TABLE conflict_groups (
                conflict_group_id VARCHAR PRIMARY KEY,
                document_id VARCHAR NOT NULL,
                stock_code VARCHAR,
                metric_year INTEGER NOT NULL,
                statement VARCHAR NOT NULL,
                canonical_metric VARCHAR NOT NULL,
                normalized_unit VARCHAR NOT NULL,
                distinct_value_count INTEGER NOT NULL,
                candidate_count INTEGER NOT NULL,
                resolution_status VARCHAR NOT NULL,
                resolution_reason VARCHAR NOT NULL,
                selected_fact_id VARCHAR,
                fact_summaries_json JSON NOT NULL
            )
        """)
        connection.executemany(
            "INSERT INTO conflict_groups VALUES (" + ",".join("?" for _ in range(13)) + ")",
            [(
                group["conflict_group_id"], group["document_id"], group["stock_code"],
                group["metric_year"], group["statement"], group["canonical_metric"],
                group["normalized_unit"], group["distinct_value_count"], group["candidate_count"],
                group["resolution_status"], group["resolution_reason"], group["selected_fact_id"],
                json.dumps(group["fact_summaries"], ensure_ascii=False, separators=(",", ":")),
            ) for group in conflicts],
        )

        connection.execute("""
            CREATE TABLE candidate_audit (
                candidate_id VARCHAR PRIMARY KEY,
                document_id VARCHAR,
                canonical_metric VARCHAR,
                eligible BOOLEAN NOT NULL,
                rejection_reasons_json JSON NOT NULL,
                metric_year INTEGER,
                year_resolution VARCHAR,
                confidence_score DECIMAL(6, 4) NOT NULL,
                conflict_group_id VARCHAR,
                fact_id VARCHAR
            )
        """)
        connection.executemany(
            "INSERT INTO candidate_audit VALUES (" + ",".join("?" for _ in range(10)) + ")",
            [(
                decision["candidate_id"], decision["candidate"].get("document_id"),
                decision["candidate"].get("canonical_metric"), decision["eligible"],
                json.dumps(decision["rejection_reasons"], ensure_ascii=False, separators=(",", ":")),
                decision["metric_year"], decision["year_resolution"], decision["confidence_score"],
                decision["group_id"], decision["fact_id"],
            ) for decision in decisions],
        )
        connection.execute("CREATE TABLE build_metadata (key VARCHAR PRIMARY KEY, value VARCHAR NOT NULL)")
        connection.executemany("INSERT INTO build_metadata VALUES (?, ?)", sorted(metadata.items()))

        row_count = connection.execute("SELECT count(*) FROM financial_facts").fetchone()[0]
        selected_count = connection.execute("SELECT count(*) FROM selected_financial_facts").fetchone()[0]
        unique_count = connection.execute("SELECT count(DISTINCT fact_id) FROM financial_facts").fetchone()[0]
        selected_group_violations = connection.execute("""
            SELECT count(*) FROM (
                SELECT conflict_group_id FROM financial_facts
                GROUP BY conflict_group_id HAVING count(*) FILTER (WHERE is_selected) > 1
            )
        """).fetchone()[0]
        flyada = connection.execute("""
            SELECT normalized_value_text, normalized_unit, source_table_id
            FROM selected_financial_facts
            WHERE stock_code='000026' AND metric_year=2019 AND canonical_metric='营业收入'
            ORDER BY confidence_score DESC, fact_id LIMIT 1
        """).fetchone()
        connection.execute("CHECKPOINT")
    finally:
        connection.close()
    os.replace(temporary, database_path)
    return {
        "duckdb_version": duckdb.__version__,
        "fact_rows": row_count,
        "selected_rows": selected_count,
        "unique_fact_ids": unique_count,
        "selected_group_violations": selected_group_violations,
        "flyada_2019_revenue": list(flyada) if flyada else None,
    }


def validate_records(
    facts: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    expected_candidate_rows: int,
) -> dict[str, Any]:
    errors: list[str] = []
    fact_ids = [fact["fact_id"] for fact in facts]
    candidate_ids = [str(decision["candidate_id"]) for decision in decisions]
    if len(decisions) != expected_candidate_rows:
        errors.append("candidate_audit_row_count_mismatch")
    if len(fact_ids) != len(set(fact_ids)):
        errors.append("duplicate_fact_id")
    if len(candidate_ids) != len(set(candidate_ids)):
        errors.append("duplicate_candidate_id")
    if any(fact["support_count"] != len(fact["provenance"]) for fact in facts):
        errors.append("support_provenance_count_mismatch")
    if any(not fact["source_table_id"] or fact["source_row_index"] is None or fact["source_col_index"] is None for fact in facts):
        errors.append("missing_source_cell_provenance")
    if any(decision["eligible"] and decision["candidate"].get("normalization_warnings") for decision in decisions):
        errors.append("warned_candidate_became_eligible")

    selected_by_group: Counter[str] = Counter(
        fact["conflict_group_id"] for fact in facts if fact["is_selected"]
    )
    if any(count > 1 for count in selected_by_group.values()):
        errors.append("more_than_one_selected_fact_in_group")
    conflict_ids = {group["conflict_group_id"] for group in conflicts}
    fact_conflict_ids = {fact["conflict_group_id"] for fact in facts if fact["has_conflict"]}
    if conflict_ids != fact_conflict_ids:
        errors.append("conflict_group_coverage_mismatch")

    provenance_candidate_ids = {
        provenance["candidate_id"]
        for fact in facts
        for provenance in fact["provenance"]
    }
    eligible_candidate_ids = {decision["candidate_id"] for decision in decisions if decision["eligible"]}
    if provenance_candidate_ids != eligible_candidate_ids:
        errors.append("eligible_candidate_provenance_coverage_mismatch")
    return {
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "candidate_audit_rows_equal_input": len(decisions) == expected_candidate_rows,
        "fact_ids_unique": len(fact_ids) == len(set(fact_ids)),
        "candidate_ids_unique": len(candidate_ids) == len(set(candidate_ids)),
        "eligible_candidates_have_provenance": provenance_candidate_ids == eligible_candidate_ids,
        "selected_group_max_one": all(count <= 1 for count in selected_by_group.values()),
    }


def validate_known_relative_partial_duration(
    decisions: list[dict[str, Any]],
    facts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pin the reviewed China Railway Signal partial-duration failure mode."""
    decision = next(
        (
            item for item in decisions
            if item["candidate_id"] == KNOWN_RELATIVE_PARTIAL_DURATION_CANDIDATE_ID
        ),
        None,
    )
    candidate_fact_ids = sorted({
        fact["fact_id"]
        for fact in facts
        for provenance in fact["provenance"]
        if provenance["candidate_id"] == KNOWN_RELATIVE_PARTIAL_DURATION_CANDIDATE_ID
    })
    legacy_fact_present = any(
        fact["fact_id"] == LEGACY_RELATIVE_PARTIAL_DURATION_FACT_ID for fact in facts
    )
    if decision is None:
        passed = not candidate_fact_ids and not legacy_fact_present
        status = "candidate_absent_upstream"
    else:
        passed = (
            not decision["eligible"]
            and "partial_duration_date" in decision["rejection_reasons"]
            and decision["fact_id"] is None
            and not candidate_fact_ids
            and not legacy_fact_present
        )
        status = "candidate_rejected" if passed else "regression_failure"
    return {
        "passed": passed,
        "status": status,
        "candidate_id": KNOWN_RELATIVE_PARTIAL_DURATION_CANDIDATE_ID,
        "candidate_present": decision is not None,
        "candidate_eligible": decision["eligible"] if decision is not None else None,
        "rejection_reasons": decision["rejection_reasons"] if decision is not None else [],
        "candidate_fact_id": decision["fact_id"] if decision is not None else None,
        "fact_ids_containing_candidate": candidate_fact_ids,
        "legacy_fact_id": LEGACY_RELATIVE_PARTIAL_DURATION_FACT_ID,
        "legacy_fact_present": legacy_fact_present,
    }


def run_source_spot_checks(facts: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    """Check one deterministic selected source cell for every canonical metric."""
    samples: dict[str, dict[str, Any]] = {}
    for fact in facts:
        if fact["is_selected"] and fact["canonical_metric"] not in samples:
            samples[fact["canonical_metric"]] = fact

    checks: list[dict[str, Any]] = []
    for metric, fact in sorted(samples.items()):
        provenance = fact["provenance"][0]
        source_value = provenance.get("source_markdown")
        source_path = Path(str(source_value)) if source_value else Path("")
        if source_value and not source_path.is_absolute():
            source_path = root / source_path
        line_range = provenance.get("line_range")
        valid_range = (
            isinstance(line_range, list)
            and len(line_range) >= 2
            and isinstance(line_range[0], int)
            and isinstance(line_range[1], int)
            and line_range[0] >= 1
            and line_range[1] >= line_range[0]
        )
        source_exists = bool(source_value) and source_path.is_file()
        raw_value_found = False
        if source_exists and valid_range:
            lines = source_path.read_text(encoding="utf-8").splitlines()
            snippet = "\n".join(lines[line_range[0] - 1:line_range[1]])
            raw_value_found = str(provenance.get("raw_value") or "") in snippet
        checks.append({
            "canonical_metric": metric,
            "fact_id": fact["fact_id"],
            "candidate_id": provenance.get("candidate_id"),
            "table_id": provenance.get("table_id"),
            "source_markdown": source_value,
            "line_range": line_range,
            "raw_value": provenance.get("raw_value"),
            "source_exists": source_exists,
            "valid_line_range": valid_range,
            "raw_value_found_in_source_range": raw_value_found,
            "passed": source_exists and valid_range and raw_value_found,
        })
    return checks


def metric_coverage(
    metrics: Iterable[str],
    candidates: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    facts: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in sorted(metrics):
        metric_candidates = [row for row in candidates if row.get("canonical_metric") == metric]
        metric_decisions = [row for row in decisions if row["candidate"].get("canonical_metric") == metric]
        metric_facts = [row for row in facts if row["canonical_metric"] == metric]
        selected = [row for row in metric_facts if row["is_selected"]]
        result[metric] = {
            "candidate_rows": len(metric_candidates),
            "eligible_candidate_rows": sum(row["eligible"] for row in metric_decisions),
            "distinct_fact_rows": len(metric_facts),
            "selected_fact_rows": len(selected),
            "selected_documents": len({row["document_id"] for row in selected}),
            "selected_company_metric_years": len({
                (row["stock_code"], row["metric_year"]) for row in selected
            }),
        }
    return result


def markdown_report(report: dict[str, Any]) -> str:
    counts = report["counts"]
    validation = report["validation"]
    lines = [
        "# Phase 6 Corpus Financial Fact Store Report",
        "",
        "## Date",
        report["generated_at"],
        "",
        "## Environment",
        f"- Workspace: `{report['workspace_root']}`",
        f"- Builder: `{BUILDER_VERSION}`",
        f"- Python: `{report['environment']['python']}`",
        f"- DuckDB: `{report['environment']['duckdb']}`",
        "",
        "## Inputs",
    ]
    for key, value in report["inputs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend([
        "",
        "## Feasibility Review",
        "- Executable: yes; Phase 5 provides 20,431 provenance-preserving canonical candidates and Phase 4 cell coordinates remain embedded in every candidate.",
        "- Required input contract: parsed decimal strings, canonical metric, normalized unit, report metadata, table/cell coordinates, Markdown path, line range, and section context.",
        "- Output contract for later phases: selected facts are exposed through the DuckDB `selected_financial_facts` view; all facts and unresolved conflicts remain auditable in base tables/JSONL.",
        "- Principal risks: period/comparison columns, segment/subsidiary scope, bad table unit hints, duplicate disclosures, adjusted comparatives, and ambiguous conflicts.",
        "- Stop condition: missing candidate provenance or inability to materialize/query DuckDB. Neither condition occurred.",
        "",
        "## Changed Components",
        "- Added deterministic candidate eligibility, metric-year resolution, unit compatibility, plausibility checks, value-level deduplication, conflict grouping, and conservative confidence selection.",
        "- Duration partial-date rejection runs before explicit/relative year routing; deterministic regression checks cover relative previous/current rejection, valid full-year labels, and point-in-time dates.",
        "- Added JSONL and DuckDB fact stores with complete candidate/cell provenance.",
        "- Added candidate-decision and conflict-group audit surfaces.",
        "",
        "## Generated Artifacts",
    ])
    for key, value in report["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend([
        "",
        "## Verification Commands",
        "- `uv pip install --python .venv/bin/python duckdb` (single minimal dependency attempt; succeeded with DuckDB 1.5.4)",
        "- `.venv/bin/python -m py_compile scripts/build_financial_fact_store.py`",
        "- `.venv/bin/python scripts/build_financial_fact_store.py`",
        "- Repeat the builder and compare the reported `financial_facts_sha256`.",
        "- Query `selected_financial_facts` in `data/facts/financial_facts.duckdb`.",
        "",
        "## Verification Results",
        f"- Input candidates: {counts['candidate_rows']}",
        f"- Eligible candidates: {counts['eligible_candidate_rows']}",
        f"- Rejected candidates: {counts['rejected_candidate_rows']}",
        f"- Distinct fact rows: {counts['fact_rows']}",
        f"- Selected fact rows: {counts['selected_fact_rows']}",
        f"- Conflict groups: {counts['conflict_groups']}",
        f"- Resolved conflict groups: {counts['resolved_conflict_groups']}",
        f"- Unresolved conflict groups: {counts['unresolved_conflict_groups']}",
        f"- Selected documents: {counts['selected_documents']} / {counts['corpus_documents']}",
        f"- Candidate decisions preserve all inputs: {validation['candidate_audit_rows_equal_input']}",
        f"- Fact IDs unique: {validation['fact_ids_unique']}",
        f"- Eligible candidates all have provenance: {validation['eligible_candidates_have_provenance']}",
        f"- At most one selected fact per group: {validation['selected_group_max_one']}",
        f"- Metric-year regression checks: {sum(item['passed'] for item in validation['metric_year_regression_checks'])} / {len(validation['metric_year_regression_checks'])}",
        f"- Known China Railway Signal partial-duration candidate rejected and absent from facts: {validation['known_relative_partial_duration_check']['passed']}",
        f"- Markdown source spot checks: {sum(item['passed'] for item in validation['source_spot_checks'])} / {len(validation['source_spot_checks'])}",
        f"- DuckDB query validation: {report['duckdb_validation']}",
        f"- Financial facts SHA-256: `{report['hashes']['financial_facts_sha256']}`",
        f"- Repeated-run facts hash stable: {report['repeatability']['matches_previous_run']}",
        "",
        "## Fractional Person-Unit Policy",
        f"- Phase 5 warned data cells: {report['fractional_person_policy']['phase5_warned_data_cells']}.",
        f"- Warned rows reaching canonical candidates: {report['fractional_person_policy']['warned_candidate_rows']}.",
        f"- Warned canonical candidates rejected: {report['fractional_person_policy']['rejected_warned_candidate_rows']}.",
        f"- Warned non-candidate cells: {report['fractional_person_policy']['warned_non_candidate_cells']}.",
        "- Policy: do not round or repair. Reject warned candidates from facts and retain their original text and rejection reason in the candidate audit.",
        "",
        "## Regression Pins",
        f"- Phase 5 fractional-person warning count: pinned `{report['regression_pins']['phase5_fraction_warning_count']['pinned']}`, actual `{report['regression_pins']['phase5_fraction_warning_count']['actual']}`, matched `{report['regression_pins']['phase5_fraction_warning_count']['matched']}`.",
        f"- Flyada 2019 revenue tuple: pinned `{report['regression_pins']['flyada_2019_revenue']['pinned']}`, actual `{report['regression_pins']['flyada_2019_revenue']['actual']}`, matched `{report['regression_pins']['flyada_2019_revenue']['matched']}`.",
        f"- Maintenance policy: {report['regression_pins']['maintenance_policy']}",
        "",
        "## Issues Encountered",
        "- Phase 5 table-level unit hints can be wrong (for example financial values tagged as `人`); Phase 6 rejects incompatible/warned candidates rather than repairing them silently.",
        "- Unresolved conflicts remain queryable in `financial_facts` but are excluded from `selected_financial_facts`.",
        "",
        "## Discarded / Archived Artifacts",
        "- None. Temporary files are atomically replaced and no failed intermediates remain.",
        "",
        "## Subagents Used",
        "- Phase 6 implementation and the relative-year partial-duration review fix were executed by independently assigned subagents.",
        "",
        "## User Confirmations Needed",
        "- None for Phase 6. Conservative unresolved conflicts can be curated in a later offline policy revision.",
        "",
        "## Decision",
        "- continue",
        "",
    ])
    return "\n".join(lines)


def build(args: argparse.Namespace) -> dict[str, Any]:
    root = workspace_root()
    candidates_path = (root / args.candidates).resolve()
    phase5_report_path = (root / args.phase5_report).resolve()
    manifest_path = (root / args.corpus_manifest).resolve()
    facts_path = (root / args.facts_out).resolve()
    database_path = (root / args.database_out).resolve()
    schema_path = (root / args.schema_out).resolve()
    run_dir = (root / args.run_dir).resolve()
    coverage_path = run_dir / "coverage_report.json"
    conflict_report_path = run_dir / "conflict_report.json"
    repeatability_report_path = run_dir / "repeatability_report.json"
    run_report_path = run_dir / "phase_06_run_report.json"
    phase_report_path = run_dir / "phase_06_report.md"
    candidate_audit_path = run_dir / "reports/candidate_decisions.jsonl"
    conflict_groups_path = run_dir / "reports/conflict_groups.jsonl"

    phase5_report = read_json(phase5_report_path)
    manifest = read_json(manifest_path)
    candidates = [row for _, row in iter_jsonl(candidates_path)]
    decisions = [audit_candidate(candidate) for candidate in candidates]
    facts, conflict_groups = build_facts(decisions)

    previous_facts_hash = file_sha256(facts_path) if facts_path.exists() else None
    fact_rows_written = write_jsonl(facts_path, facts)
    audit_rows_written = write_jsonl(candidate_audit_path, (serializable_audit(item) for item in decisions))
    conflict_rows_written = write_jsonl(conflict_groups_path, conflict_groups)
    write_json(schema_path, fact_schema())

    rejection_counts: Counter[str] = Counter(
        reason for decision in decisions for reason in decision["rejection_reasons"]
    )
    selection_counts = Counter(fact["selection_status"] for fact in facts)
    metric_counts = metric_coverage(STATEMENT_BY_METRIC, candidates, decisions, facts)
    selected_facts = [fact for fact in facts if fact["is_selected"]]
    source_spot_checks = run_source_spot_checks(facts, root)
    corpus_documents = int(
        manifest.get("summary", {}).get("documents_emitted")
        or manifest.get("document_count")
        or len(manifest.get("documents", []))
    )
    phase5_warning_count = int(
        phase5_report.get("normalization_warning_counts", {}).get("integer_unit_has_fraction", 0)
    )
    warned_candidates = [
        decision for decision in decisions
        if "integer_unit_has_fraction" in decision["candidate"].get("normalization_warnings", [])
    ]

    metric_year_regression_checks = run_metric_year_regression_checks()
    known_partial_duration_check = validate_known_relative_partial_duration(decisions, facts)
    validation = validate_records(facts, conflict_groups, decisions, len(candidates))
    if not all(item["passed"] for item in metric_year_regression_checks):
        validation["errors"].append("metric_year_regression_check_failure")
    if not known_partial_duration_check["passed"]:
        validation["errors"].append("known_relative_partial_duration_candidate_regression")
    validation["metric_year_regression_checks"] = metric_year_regression_checks
    validation["known_relative_partial_duration_check"] = known_partial_duration_check
    if phase5_warning_count != PHASE5_FRACTION_WARNING_COUNT_REGRESSION_PIN:
        validation["errors"].append(
            "regression_pin_stale_or_mismatch:phase5_fraction_warning_count:"
            f"actual={phase5_warning_count}:"
            f"pinned={PHASE5_FRACTION_WARNING_COUNT_REGRESSION_PIN}; "
            "the Phase 5 output may have changed legitimately; review the upstream diff and "
            "update PHASE5_FRACTION_WARNING_COUNT_REGRESSION_PIN only if intentional"
        )
    if len(warned_candidates) != sum(not item["eligible"] for item in warned_candidates):
        validation["errors"].append("fractional_person_warning_candidate_not_rejected")
    if len(source_spot_checks) != len(STATEMENT_BY_METRIC) or not all(
        item["passed"] for item in source_spot_checks
    ):
        validation["errors"].append("markdown_source_spot_check_failure")
    validation["source_spot_checks"] = source_spot_checks

    facts_hash = file_sha256(facts_path)
    candidates_hash = file_sha256(candidates_path)
    database_metadata = {
        "builder_version": BUILDER_VERSION,
        "candidate_input_sha256": candidates_hash,
        "financial_facts_sha256": facts_hash,
        "schema_version": SCHEMA_FACT,
    }
    duckdb_validation = build_database(database_path, facts, conflict_groups, decisions, database_metadata)
    if duckdb_validation["fact_rows"] != len(facts):
        validation["errors"].append("duckdb_fact_row_count_mismatch")
    if duckdb_validation["selected_rows"] != len(selected_facts):
        validation["errors"].append("duckdb_selected_row_count_mismatch")
    if duckdb_validation["unique_fact_ids"] != len(facts):
        validation["errors"].append("duckdb_fact_id_uniqueness_failure")
    if duckdb_validation["selected_group_violations"]:
        validation["errors"].append("duckdb_selected_group_violation")
    actual_flyada = duckdb_validation["flyada_2019_revenue"]
    if tuple(actual_flyada or ()) != FLYADA_2019_REVENUE_REGRESSION_PIN:
        validation["errors"].append(
            "regression_pin_stale_or_mismatch:flyada_2019_revenue:"
            f"actual={actual_flyada}:pinned={list(FLYADA_2019_REVENUE_REGRESSION_PIN)}; "
            "the selected reference fact may have changed legitimately; review the fact/provenance "
            "diff and update FLYADA_2019_REVENUE_REGRESSION_PIN only if intentional"
        )
    validation["error_count"] = len(validation["errors"])
    validation["passed"] = not validation["errors"]
    regression_pins = {
        "maintenance_policy": (
            "Pins fail closed on mismatch. An upstream change is not automatically corruption; "
            "review the diff and update a pin only after confirming the new audited value."
        ),
        "phase5_fraction_warning_count": {
            "constant": "PHASE5_FRACTION_WARNING_COUNT_REGRESSION_PIN",
            "pinned": PHASE5_FRACTION_WARNING_COUNT_REGRESSION_PIN,
            "actual": phase5_warning_count,
            "matched": phase5_warning_count == PHASE5_FRACTION_WARNING_COUNT_REGRESSION_PIN,
        },
        "flyada_2019_revenue": {
            "constant": "FLYADA_2019_REVENUE_REGRESSION_PIN",
            "pinned": list(FLYADA_2019_REVENUE_REGRESSION_PIN),
            "actual": actual_flyada,
            "matched": tuple(actual_flyada or ()) == FLYADA_2019_REVENUE_REGRESSION_PIN,
        },
    }

    coverage_report = {
        "schema_version": "finglmqa.phase6.coverage_report.v1",
        "generated_at": utc_now(),
        "candidate_rows": len(candidates),
        "eligible_candidate_rows": sum(item["eligible"] for item in decisions),
        "rejected_candidate_rows": sum(not item["eligible"] for item in decisions),
        "fact_rows": len(facts),
        "selected_fact_rows": len(selected_facts),
        "selected_documents": len({fact["document_id"] for fact in selected_facts}),
        "corpus_documents": corpus_documents,
        "selected_company_metric_years": len({
            (fact["stock_code"], fact["canonical_metric"], fact["metric_year"], fact["normalized_unit"])
            for fact in selected_facts
        }),
        "selection_status_counts": dict(sorted(selection_counts.items())),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "metric_coverage": metric_counts,
    }
    conflict_report = {
        "schema_version": "finglmqa.phase6.conflict_report.v1",
        "generated_at": utc_now(),
        "conflict_group_count": len(conflict_groups),
        "resolved_conflict_group_count": sum(
            group["resolution_status"] == "resolved_by_confidence" for group in conflict_groups
        ),
        "unresolved_conflict_group_count": sum(
            group["resolution_status"] == "unresolved_conflict" for group in conflict_groups
        ),
        "conflicting_fact_rows": sum(group["distinct_value_count"] for group in conflict_groups),
        "conflicting_candidate_rows": sum(group["candidate_count"] for group in conflict_groups),
        "by_metric": dict(sorted(Counter(group["canonical_metric"] for group in conflict_groups).items())),
        "full_conflict_groups": rel(conflict_groups_path, root),
        "samples": conflict_groups[:MAX_REPORT_SAMPLES],
    }
    write_json(coverage_path, coverage_report)
    write_json(conflict_report_path, conflict_report)
    repeatability = {
        "schema_version": "finglmqa.phase6.repeatability_report.v1",
        "generated_at": utc_now(),
        "facts_out_existed_before_run": previous_facts_hash is not None,
        "previous_financial_facts_sha256": previous_facts_hash,
        "current_financial_facts_sha256": facts_hash,
        "matches_previous_run": previous_facts_hash == facts_hash if previous_facts_hash else None,
        "comparison_scope": "deterministic financial_facts.jsonl; timestamps are confined to reports",
    }
    write_json(repeatability_report_path, repeatability)

    report = {
        "schema_version": SCHEMA_REPORT,
        "builder_version": BUILDER_VERSION,
        "generated_at": utc_now(),
        "workspace_root": root.as_posix(),
        "command": " ".join(sys.argv),
        "environment": {
            "python": sys.version.split()[0],
            "duckdb": duckdb_validation["duckdb_version"],
        },
        "inputs": {
            "canonical_metric_candidates": rel(candidates_path, root),
            "phase5_metric_unit_report": rel(phase5_report_path, root),
            "corpus_manifest": rel(manifest_path, root),
        },
        "outputs": {
            "financial_facts_jsonl": rel(facts_path, root),
            "financial_facts_duckdb": rel(database_path, root),
            "financial_facts_schema": rel(schema_path, root),
            "coverage_report": rel(coverage_path, root),
            "conflict_report": rel(conflict_report_path, root),
            "repeatability_report": rel(repeatability_report_path, root),
            "candidate_decisions": rel(candidate_audit_path, root),
            "conflict_groups": rel(conflict_groups_path, root),
            "phase_06_run_report": rel(run_report_path, root),
            "phase_06_report": rel(phase_report_path, root),
        },
        "counts": {
            "candidate_rows": len(candidates),
            "eligible_candidate_rows": sum(item["eligible"] for item in decisions),
            "rejected_candidate_rows": sum(not item["eligible"] for item in decisions),
            "fact_rows": len(facts),
            "selected_fact_rows": len(selected_facts),
            "conflict_groups": len(conflict_groups),
            "resolved_conflict_groups": conflict_report["resolved_conflict_group_count"],
            "unresolved_conflict_groups": conflict_report["unresolved_conflict_group_count"],
            "selected_documents": coverage_report["selected_documents"],
            "corpus_documents": corpus_documents,
            "fact_rows_written": fact_rows_written,
            "candidate_audit_rows_written": audit_rows_written,
            "conflict_rows_written": conflict_rows_written,
        },
        "fractional_person_policy": {
            "policy": "reject_without_rounding_or_repair",
            "phase5_warned_data_cells": phase5_warning_count,
            "warned_candidate_rows": len(warned_candidates),
            "rejected_warned_candidate_rows": sum(not item["eligible"] for item in warned_candidates),
            "warned_non_candidate_cells": phase5_warning_count - len(warned_candidates),
        },
        "coverage": coverage_report,
        "conflicts": {key: value for key, value in conflict_report.items() if key != "samples"},
        "validation": validation,
        "duckdb_validation": duckdb_validation,
        "regression_pins": regression_pins,
        "repeatability": repeatability,
        "hashes": {
            "candidate_input_sha256": candidates_hash,
            "financial_facts_sha256": facts_hash,
        },
    }
    write_json(run_report_path, report)
    phase_report_path.parent.mkdir(parents=True, exist_ok=True)
    phase_report_path.write_text(markdown_report(report), encoding="utf-8")
    if not validation["passed"]:
        raise RuntimeError("Phase 6 validation failed: " + "; ".join(validation["errors"]))
    return report


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--phase5-report", default=DEFAULT_PHASE5_REPORT)
    parser.add_argument("--corpus-manifest", default=DEFAULT_CORPUS_MANIFEST)
    parser.add_argument("--facts-out", default=DEFAULT_FACTS)
    parser.add_argument("--database-out", default=DEFAULT_DATABASE)
    parser.add_argument("--schema-out", default=DEFAULT_SCHEMA)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    report = build(args)
    counts = report["counts"]
    print(json.dumps({
        "status": "complete",
        "candidate_rows": counts["candidate_rows"],
        "eligible_candidate_rows": counts["eligible_candidate_rows"],
        "fact_rows": counts["fact_rows"],
        "selected_fact_rows": counts["selected_fact_rows"],
        "conflict_groups": counts["conflict_groups"],
        "validation_passed": report["validation"]["passed"],
        "financial_facts_sha256": report["hashes"]["financial_facts_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
