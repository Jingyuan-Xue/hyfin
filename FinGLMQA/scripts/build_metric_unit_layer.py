#!/usr/bin/env python3
"""Build Phase 5 canonical metric and unit-normalization artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, TextIO


SCHEMA_CANDIDATE = "finglmqa.phase5.canonical_metric_candidate.v1"
SCHEMA_REPORT = "finglmqa.phase5.metric_unit_report.v1"
ADAPTER_VERSION = "phase5-metric-unit-layer-v1"

DEFAULT_TABLE_CELLS = "data/corpus_package/table_cells.jsonl"
DEFAULT_TABLE_INDEX = "data/indexes/tabgr_table_index.jsonl"
DEFAULT_METRIC_ALIASES = "src/config/metric_aliases.json"
DEFAULT_UNIT_RULES = "src/config/unit_rules.json"
DEFAULT_CANDIDATES = "data/indexes/canonical_metric_candidates.jsonl"
DEFAULT_UNIT_SAMPLES = "data/indexes/unit_normalization_samples.jsonl"
DEFAULT_SCHEMA = "data/schemas/canonical_metric_candidates.schema.json"
DEFAULT_REPORT_JSON = "runs/phase_05/metric_unit_report.json"
DEFAULT_REPORT_MD = "runs/phase_05/phase_05_report.md"

MAX_REPORT_SAMPLES = 50
MAX_UNIT_SAMPLES_PER_RULE = 40

FLYADA_TABLE_ID = "A000026_飞亚达_2019年年度报告_table_0007_acc5dee31c"

FINANCIAL_CONTEXT_TERMS = (
    "财务",
    "会计",
    "主要会计数据",
    "财务指标",
    "资产负债",
    "利润",
    "现金流",
    "所有者权益",
    "股东权益",
    "营业收入",
    "股本",
    "每股收益",
)

FULLWIDTH_TRANSLATION = str.maketrans({
    "（": "(",
    "）": ")",
    "，": ",",
    "．": ".",
    "。": ".",
    "＋": "+",
    "－": "-",
    "—": "-",
    "–": "-",
    "／": "/",
    "％": "%",
    "　": " ",
    "０": "0",
    "１": "1",
    "２": "2",
    "３": "3",
    "４": "4",
    "５": "5",
    "６": "6",
    "７": "7",
    "８": "8",
    "９": "9",
})

NUMBER_PATTERN = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+)|(?:\d+))(?:\.\d+)?|[+-]?\.\d+"
NUMBER_WITH_UNIT_RE = re.compile(
    rf"^(?:人民币)?\s*(?P<number>{NUMBER_PATTERN})\s*(?P<unit>[\u4e00-\u9fffA-Za-z%/]+)?$"
)
BRACKET_RE = re.compile(r"[\(（]\s*([^()（）]{1,40})\s*[\)）]")
UNIT_HINT_RE = re.compile(r"(?:单位|金额单位|币种)\s*[:：]\s*([^,，;；。.\s]+)")


@dataclass(frozen=True)
class UnitRule:
    rule_id: str
    category: str
    aliases: tuple[str, ...]
    normalized_unit: str
    multiplier: Decimal
    integer: bool = False


@dataclass(frozen=True)
class ParsedNumber:
    status: str
    value: Decimal | None = None
    raw_unit: str | None = None
    unit_source: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class NormalizedNumber:
    raw_value: str
    raw_unit: str | None
    parsed_value: str | None
    normalized_value: str | None
    normalized_unit: str | None
    parse_status: str
    parse_error: str | None
    unit_rule_id: str | None
    unit_source: str | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class MetricMatch:
    canonical_metric: str
    source: str
    label: str
    normalized_label: str
    match_type: str
    confidence: str


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def open_jsonl(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def write_jsonl_row(fh: TextIO, row: dict[str, Any]) -> None:
    json.dump(row, fh, ensure_ascii=False, separators=(",", ":"))
    fh.write("\n")


def iter_jsonl(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            yield line_number, row


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def compact_text(value: object) -> str:
    return re.sub(r"\s+", "", normalize_text(value).translate(FULLWIDTH_TRANSLATION))


def normalize_unit_phrase(value: object) -> str:
    text = compact_text(value)
    text = re.sub(r"^(?:单位|金额单位|币种)[:：]?", "", text)
    text = text.strip("：:()[]【】,，;；。.")
    return text


def decimal_to_plain(value: Decimal, *, keep_trailing: bool = False) -> str:
    if keep_trailing:
        return format(value, "f")
    if value == value.to_integral_value():
        return format(value.quantize(Decimal(1)), "f")
    return format(value.normalize(), "f")


def line_range(value: object) -> list[int | None]:
    if not isinstance(value, list) or len(value) < 2:
        return [None, None]

    def to_int(item: object) -> int | None:
        if isinstance(item, bool):
            return None
        if isinstance(item, int):
            return item
        try:
            return int(str(item))
        except (TypeError, ValueError):
            return None

    return [to_int(value[0]), to_int(value[1])]


class UnitCatalog:
    def __init__(self, config: dict[str, Any]) -> None:
        self.schema_version = str(config.get("schema_version") or "")
        self.null_markers = {compact_text(item) for item in config.get("null_markers", [])}
        self.rules: list[UnitRule] = []
        self.alias_to_rule: dict[str, UnitRule] = {}
        for raw_rule in config.get("rules", []):
            if not isinstance(raw_rule, dict):
                continue
            try:
                rule = UnitRule(
                    rule_id=str(raw_rule["rule_id"]),
                    category=str(raw_rule["category"]),
                    aliases=tuple(str(item) for item in raw_rule.get("aliases", [])),
                    normalized_unit=str(raw_rule["normalized_unit"]),
                    multiplier=Decimal(str(raw_rule["multiplier"])),
                    integer=bool(raw_rule.get("integer", False)),
                )
            except (KeyError, InvalidOperation) as exc:
                raise ValueError(f"Invalid unit rule: {raw_rule}") from exc
            self.rules.append(rule)
            for alias in rule.aliases:
                normalized = normalize_unit_phrase(alias)
                self.alias_to_rule[normalized] = rule

    def resolve(self, unit: object) -> UnitRule | None:
        normalized = normalize_unit_phrase(unit)
        if not normalized:
            return None
        return self.alias_to_rule.get(normalized)

    def is_null_marker(self, raw_value: object) -> bool:
        return compact_text(raw_value) in self.null_markers

    def extract_from_label(self, label: object) -> tuple[str | None, UnitRule | None]:
        text = normalize_text(label)
        if not text:
            return None, None
        for match in BRACKET_RE.finditer(text):
            unit = match.group(1)
            rule = self.resolve(unit)
            if rule:
                return normalize_unit_phrase(unit), rule
        hint_match = UNIT_HINT_RE.search(text)
        if hint_match:
            unit = hint_match.group(1)
            rule = self.resolve(unit)
            if rule:
                return normalize_unit_phrase(unit), rule
        rule = self.resolve(text)
        if rule:
            return normalize_unit_phrase(text), rule
        return None, None


class MetricCatalog:
    def __init__(self, config: dict[str, Any], units: UnitCatalog) -> None:
        self.schema_version = str(config.get("schema_version") or "")
        self.units = units
        self.metrics: list[dict[str, Any]] = []
        self.alias_to_metric: dict[str, str] = {}
        self.patterns: list[tuple[re.Pattern[str], str, list[re.Pattern[str]]]] = []
        self.ambiguous_terms: list[tuple[str, str]] = []

        for raw_metric in config.get("metrics", []):
            if not isinstance(raw_metric, dict):
                continue
            canonical = str(raw_metric.get("canonical_metric") or "")
            if not canonical:
                continue
            metric = {
                "canonical_metric": canonical,
                "aliases": list(raw_metric.get("aliases", [])),
                "patterns": list(raw_metric.get("patterns", [])),
                "exclude_patterns": list(raw_metric.get("exclude_patterns", [])),
            }
            self.metrics.append(metric)
            for alias in metric["aliases"]:
                self.alias_to_metric[self.normalize_label(alias)] = canonical
            excludes = [re.compile(pattern) for pattern in metric["exclude_patterns"]]
            for pattern in metric["patterns"]:
                self.patterns.append((re.compile(pattern), canonical, excludes))

        for item in config.get("ambiguous_terms", []):
            if isinstance(item, dict):
                term = self.normalize_label(item.get("term"))
                reason = str(item.get("reason") or "")
                if term:
                    self.ambiguous_terms.append((term, reason))

    def canonical_metrics(self) -> list[str]:
        return [metric["canonical_metric"] for metric in self.metrics]

    def normalize_label(self, label: object) -> str:
        text = compact_text(label)
        text = re.sub(r"^[\(（]?[一二三四五六七八九十百\d]+[、.)）]", "", text)
        text = text.strip(":：,，;；.。")

        def strip_unit(match: re.Match[str]) -> str:
            candidate = match.group(1)
            return "" if self.units.resolve(candidate) else match.group(0)

        previous = None
        while previous != text:
            previous = text
            text = BRACKET_RE.sub(strip_unit, text)
        text = text.replace("(", "").replace(")", "")
        text = text.replace("（", "").replace("）", "")
        return text.strip(":：,，;；.。")

    def match_single_label(self, label: object, source: str) -> tuple[MetricMatch | None, list[dict[str, str]]]:
        raw_label = normalize_text(label)
        normalized = self.normalize_label(label)
        ambiguities: list[dict[str, str]] = []
        if not normalized:
            return None, ambiguities

        for term, reason in self.ambiguous_terms:
            if normalized == term:
                ambiguities.append({
                    "source": source,
                    "label": raw_label,
                    "normalized_label": normalized,
                    "reason": reason,
                })

        exact = self.alias_to_metric.get(normalized)
        if exact:
            return MetricMatch(exact, source, raw_label, normalized, "exact_alias", "high"), ambiguities

        pattern_matches: list[str] = []
        for pattern, canonical, excludes in self.patterns:
            if any(exclude.search(normalized) for exclude in excludes):
                continue
            if pattern.search(normalized):
                pattern_matches.append(canonical)

        unique_matches = sorted(set(pattern_matches))
        if len(unique_matches) == 1:
            return MetricMatch(unique_matches[0], source, raw_label, normalized, "regex_pattern", "medium"), ambiguities
        if len(unique_matches) > 1:
            ambiguities.append({
                "source": source,
                "label": raw_label,
                "normalized_label": normalized,
                "reason": "matched_multiple_canonical_metrics:" + ",".join(unique_matches),
            })
        return None, ambiguities

    def select_metric(self, row_label: object, column_label: object) -> tuple[MetricMatch | None, list[dict[str, str]]]:
        row_match, row_ambiguities = self.match_single_label(row_label, "row_label")
        column_match, column_ambiguities = self.match_single_label(column_label, "column_label")
        ambiguities = row_ambiguities + column_ambiguities
        if row_match and column_match and row_match.canonical_metric != column_match.canonical_metric:
            ambiguities.append({
                "source": "row_label,column_label",
                "label": f"{row_match.label} / {column_match.label}",
                "normalized_label": f"{row_match.normalized_label}/{column_match.normalized_label}",
                "reason": f"row_column_metric_conflict:{row_match.canonical_metric}!={column_match.canonical_metric}",
            })
            return row_match, ambiguities
        if row_match:
            return row_match, ambiguities
        if column_match:
            return column_match, ambiguities
        return None, ambiguities


def parse_number(raw_value: object, units: UnitCatalog) -> ParsedNumber:
    raw_text = normalize_text(raw_value)
    if units.is_null_marker(raw_text):
        return ParsedNumber(status="missing", error="null_marker")

    text = raw_text.translate(FULLWIDTH_TRANSLATION).strip()
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", "", text)
    if not text:
        return ParsedNumber(status="missing", error="empty")

    negative_parentheses = False
    if (text.startswith("(") and text.endswith(")")) or (text.startswith("（") and text.endswith("）")):
        negative_parentheses = True
        text = text[1:-1].strip()

    match = NUMBER_WITH_UNIT_RE.fullmatch(text)
    if not match:
        return ParsedNumber(status="unparseable", error="not_numeric")

    number_text = match.group("number").replace(",", "")
    unit_text = match.group("unit")
    if unit_text and not units.resolve(unit_text):
        return ParsedNumber(status="unparseable", raw_unit=unit_text, unit_source="raw_value", error="unknown_unit_suffix")
    try:
        value = Decimal(number_text)
    except InvalidOperation:
        return ParsedNumber(status="unparseable", raw_unit=unit_text, unit_source="raw_value", error="invalid_decimal")
    if negative_parentheses and value > 0:
        value = -value
    raw_unit = normalize_unit_phrase(unit_text) if unit_text else None
    return ParsedNumber(status="parsed", value=value, raw_unit=raw_unit, unit_source="raw_value" if raw_unit else None)


def infer_unit(
    parsed: ParsedNumber,
    row_label: object,
    column_label: object,
    unit_hint: object,
    units: UnitCatalog,
) -> tuple[str | None, UnitRule | None, str | None]:
    if parsed.raw_unit:
        rule = units.resolve(parsed.raw_unit)
        if rule:
            return parsed.raw_unit, rule, "raw_value"
    for source, label in (("row_label", row_label), ("column_label", column_label), ("unit_hint", unit_hint)):
        raw_unit, rule = units.extract_from_label(label)
        if rule:
            return raw_unit, rule, source
    return None, None, None


def normalize_number(
    raw_value: object,
    row_label: object,
    column_label: object,
    unit_hint: object,
    units: UnitCatalog,
) -> NormalizedNumber:
    raw_text = normalize_text(raw_value)
    parsed = parse_number(raw_text, units)
    if parsed.status != "parsed" or parsed.value is None:
        return NormalizedNumber(
            raw_value=raw_text,
            raw_unit=parsed.raw_unit,
            parsed_value=None,
            normalized_value=None,
            normalized_unit=None,
            parse_status=parsed.status,
            parse_error=parsed.error,
            unit_rule_id=None,
            unit_source=parsed.unit_source,
        )

    raw_unit, rule, unit_source = infer_unit(parsed, row_label, column_label, unit_hint, units)
    parsed_value = decimal_to_plain(parsed.value, keep_trailing=True)
    if not rule:
        return NormalizedNumber(
            raw_value=raw_text,
            raw_unit=raw_unit,
            parsed_value=parsed_value,
            normalized_value=parsed_value,
            normalized_unit=None,
            parse_status="parsed",
            parse_error=None,
            unit_rule_id=None,
            unit_source=unit_source,
        )

    normalized = parsed.value * rule.multiplier
    warnings: list[str] = []
    keep_trailing = rule.multiplier == Decimal(1) and rule.category in {"money", "money_per_share"}
    if rule.integer and normalized != normalized.to_integral_value():
        warnings.append("integer_unit_has_fraction")
    normalized_value = decimal_to_plain(normalized, keep_trailing=keep_trailing)
    return NormalizedNumber(
        raw_value=raw_text,
        raw_unit=raw_unit,
        parsed_value=parsed_value,
        normalized_value=normalized_value,
        normalized_unit=rule.normalized_unit,
        parse_status="parsed",
        parse_error=None,
        unit_rule_id=rule.rule_id,
        unit_source=unit_source,
        warnings=tuple(warnings),
    )


def load_table_index(path: Path) -> dict[str, dict[str, Any]]:
    table_index: dict[str, dict[str, Any]] = {}
    for _, row in iter_jsonl(path):
        table_id = row.get("table_id")
        if isinstance(table_id, str) and table_id:
            table_index[table_id] = row
    return table_index


def financial_context(section_path: object) -> bool:
    if not isinstance(section_path, list):
        return False
    joined = " ".join(str(item) for item in section_path)
    return any(term in joined for term in FINANCIAL_CONTEXT_TERMS)


def metadata_value(cell: dict[str, Any], table_meta: dict[str, Any], field: str) -> Any:
    if cell.get(field) is not None:
        return cell.get(field)
    metadata = cell.get("metadata") if isinstance(cell.get("metadata"), dict) else {}
    if metadata.get(field) is not None:
        return metadata.get(field)
    return table_meta.get(field)


def candidate_record(
    cell: dict[str, Any],
    table_meta: dict[str, Any],
    metric: MetricMatch,
    normalized: NormalizedNumber,
    context_is_financial: bool,
) -> dict[str, Any]:
    metadata = cell.get("metadata") if isinstance(cell.get("metadata"), dict) else {}
    table_id = str(cell.get("table_id") or "")
    row_index = cell.get("row_index")
    col_index = cell.get("col_index")
    return {
        "schema_version": SCHEMA_CANDIDATE,
        "adapter_version": ADAPTER_VERSION,
        "candidate_id": f"{table_id}:r{row_index}:c{col_index}:{metric.canonical_metric}",
        "document_id": cell.get("document_id"),
        "table_id": table_id,
        "table_index": cell.get("table_index") if cell.get("table_index") is not None else table_meta.get("table_index"),
        "row_index": row_index,
        "col_index": col_index,
        "stock_code": metadata_value(cell, table_meta, "stock_code"),
        "stock_symbol": metadata_value(cell, table_meta, "stock_symbol"),
        "stock_name": metadata_value(cell, table_meta, "stock_name"),
        "company_full": metadata_value(cell, table_meta, "company_full"),
        "report_year": metadata_value(cell, table_meta, "report_year"),
        "fiscal_year": metadata.get("fiscal_year") or table_meta.get("fiscal_year"),
        "canonical_metric": metric.canonical_metric,
        "metric_label": metric.label,
        "metric_normalized_label": metric.normalized_label,
        "metric_source": metric.source,
        "metric_match_type": metric.match_type,
        "metric_confidence": metric.confidence,
        "row_label": normalize_text(cell.get("row_label")),
        "column_label": normalize_text(cell.get("column_label")),
        "raw_value": normalized.raw_value,
        "raw_unit": normalized.raw_unit,
        "parsed_value": normalized.parsed_value,
        "normalized_value": normalized.normalized_value,
        "normalized_unit": normalized.normalized_unit,
        "parse_status": normalized.parse_status,
        "parse_error": normalized.parse_error,
        "unit_rule_id": normalized.unit_rule_id,
        "unit_source": normalized.unit_source,
        "normalization_warnings": list(normalized.warnings),
        "context_is_financial": context_is_financial,
        "source_markdown": cell.get("source_markdown") or table_meta.get("source_markdown"),
        "line_range": line_range(cell.get("line_range") or table_meta.get("line_range")),
        "section_path": cell.get("section_path") if isinstance(cell.get("section_path"), list) else table_meta.get("section_path", []),
        "table_caption": table_meta.get("caption"),
        "table_unit_hint": cell.get("unit_hint") if cell.get("unit_hint") is not None else table_meta.get("unit_hint"),
        "metadata": metadata,
    }


def sample_from_cell(
    cell: dict[str, Any],
    table_meta: dict[str, Any],
    normalized: NormalizedNumber | None = None,
    metric: MetricMatch | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    row = {
        "document_id": cell.get("document_id"),
        "table_id": cell.get("table_id"),
        "table_index": cell.get("table_index") if cell.get("table_index") is not None else table_meta.get("table_index"),
        "row_index": cell.get("row_index"),
        "col_index": cell.get("col_index"),
        "row_label": normalize_text(cell.get("row_label")),
        "column_label": normalize_text(cell.get("column_label")),
        "raw_value": normalize_text(cell.get("raw_value")),
        "unit_hint": cell.get("unit_hint") if cell.get("unit_hint") is not None else table_meta.get("unit_hint"),
        "section_path": cell.get("section_path") if isinstance(cell.get("section_path"), list) else table_meta.get("section_path", []),
        "reason": reason,
    }
    if normalized:
        row.update({
            "raw_unit": normalized.raw_unit,
            "parsed_value": normalized.parsed_value,
            "normalized_value": normalized.normalized_value,
            "normalized_unit": normalized.normalized_unit,
            "parse_status": normalized.parse_status,
            "parse_error": normalized.parse_error,
            "unit_rule_id": normalized.unit_rule_id,
            "unit_source": normalized.unit_source,
            "normalization_warnings": list(normalized.warnings),
        })
    if metric:
        row.update({
            "canonical_metric": metric.canonical_metric,
            "metric_source": metric.source,
            "metric_label": metric.label,
        })
    return row


def validation_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_CANDIDATE,
        "type": "object",
        "required": [
            "schema_version",
            "candidate_id",
            "document_id",
            "table_id",
            "row_index",
            "col_index",
            "canonical_metric",
            "metric_label",
            "metric_source",
            "row_label",
            "column_label",
            "raw_value",
            "raw_unit",
            "parsed_value",
            "normalized_value",
            "normalized_unit",
            "parse_status",
            "unit_rule_id",
            "unit_source",
            "source_markdown",
            "line_range",
            "section_path",
            "metadata"
        ],
        "additionalProperties": True,
        "properties": {
            "schema_version": {"const": SCHEMA_CANDIDATE},
            "candidate_id": {"type": "string"},
            "document_id": {"type": "string"},
            "table_id": {"type": "string"},
            "table_index": {"type": ["integer", "null"], "minimum": 1},
            "row_index": {"type": "integer", "minimum": 0},
            "col_index": {"type": "integer", "minimum": 0},
            "canonical_metric": {"type": "string"},
            "metric_label": {"type": "string"},
            "metric_source": {"enum": ["row_label", "column_label"]},
            "row_label": {"type": "string"},
            "column_label": {"type": "string"},
            "raw_value": {"type": "string"},
            "raw_unit": {"type": ["string", "null"]},
            "parsed_value": {"type": ["string", "null"]},
            "normalized_value": {"type": ["string", "null"]},
            "normalized_unit": {"type": ["string", "null"]},
            "parse_status": {"enum": ["parsed", "missing", "unparseable"]},
            "parse_error": {"type": ["string", "null"]},
            "unit_rule_id": {"type": ["string", "null"]},
            "unit_source": {"type": ["string", "null"]},
            "source_markdown": {"type": ["string", "null"]},
            "line_range": {
                "type": "array",
                "prefixItems": [{"type": ["integer", "null"]}, {"type": ["integer", "null"]}]
            },
            "section_path": {"type": "array", "items": {"type": "string"}},
            "metadata": {"type": "object"}
        },
    }


def run_unit_self_tests(units: UnitCatalog) -> list[dict[str, Any]]:
    cases = [
        {
            "name": "yuan_with_commas",
            "raw_value": "3,704,210,734.90",
            "row_label": "营业收入（元)",
            "column_label": "2019年",
            "unit_hint": None,
            "expected_value": "3704210734.90",
            "expected_unit": "元",
        },
        {
            "name": "wanyuan_to_yuan",
            "raw_value": "12.34",
            "row_label": "营业收入",
            "column_label": "金额",
            "unit_hint": "万元",
            "expected_value": "123400",
            "expected_unit": "元",
        },
        {
            "name": "yiyuan_to_yuan",
            "raw_value": "1.2",
            "row_label": "总资产",
            "column_label": "金额",
            "unit_hint": "亿元",
            "expected_value": "120000000",
            "expected_unit": "元",
        },
        {
            "name": "percent_to_ratio",
            "raw_value": "8.93%",
            "row_label": "营业收入（元）",
            "column_label": "本年比上年增减",
            "unit_hint": None,
            "expected_value": "0.0893",
            "expected_unit": "ratio",
        },
        {
            "name": "money_per_share",
            "raw_value": "0.6243",
            "row_label": "基本每股收益(元/股)",
            "column_label": "2019年",
            "unit_hint": None,
            "expected_value": "0.6243",
            "expected_unit": "元/股",
        },
    ]
    results: list[dict[str, Any]] = []
    for case in cases:
        normalized = normalize_number(
            case["raw_value"],
            case["row_label"],
            case["column_label"],
            case["unit_hint"],
            units,
        )
        passed = (
            normalized.normalized_value == case["expected_value"]
            and normalized.normalized_unit == case["expected_unit"]
        )
        results.append({
            "name": case["name"],
            "passed": passed,
            "raw_value": case["raw_value"],
            "expected_value": case["expected_value"],
            "expected_unit": case["expected_unit"],
            "actual_value": normalized.normalized_value,
            "actual_unit": normalized.normalized_unit,
            "parse_status": normalized.parse_status,
            "parse_error": normalized.parse_error,
        })
    return results


def make_report_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# Phase 5 Metric And Unit Layer Report",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Adapter: `{ADAPTER_VERSION}`",
        f"- Table cells input: `{report['inputs']['table_cells']}`",
        f"- Table index input: `{report['inputs']['tabgr_table_index']}`",
        "",
        "## Summary",
        "",
        f"- Scanned data cells: {counts['scanned_data_cells']}",
        f"- Parsed numeric cells: {counts['parsed_numeric_cells']}",
        f"- Canonical metric candidate rows: {counts['canonical_metric_hits']}",
        f"- Unit conversion hits: {counts['unit_conversion_hits']}",
        f"- Ambiguity/conflict samples retained: {len(report['samples']['ambiguity_or_conflict'])}",
        f"- Unable-parse samples retained: {len(report['samples']['unable_parse'])}",
        "",
        "## Canonical Metric Hits",
        "",
        "| Canonical metric | Candidate rows | Parsed rows | Unit-normalized rows |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in sorted(report["metric_counts"]):
        metric_counts = report["metric_counts"][metric]
        lines.append(
            f"| {metric} | {metric_counts['candidate_rows']} | {metric_counts['parsed_rows']} | "
            f"{metric_counts['unit_normalized_rows']} |"
        )
    lines.extend([
        "",
        "## Unit Rule Hits",
        "",
        "| Unit rule | Hits |",
        "| --- | ---: |",
    ])
    for rule_id, count in sorted(report["unit_rule_counts"].items()):
        lines.append(f"| {rule_id} | {count} |")
    lines.extend([
        "",
        "## Normalization Warnings",
        "",
    ])
    if report["normalization_warning_counts"]:
        for warning, count in sorted(report["normalization_warning_counts"].items()):
            lines.append(f"- {warning}: {count}")
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Validation",
        "",
    ])
    for item in report["validation"]["unit_self_tests"]:
        status = "passed" if item["passed"] else "failed"
        lines.append(f"- {item['name']}: {status}")
    for name, passed in report["validation"]["spot_checks"].items():
        status = "passed" if passed else "failed"
        lines.append(f"- {name}: {status}")
    missing = report["validation"]["missing_core_metric_samples"]
    lines.append(f"- Missing canonical metric samples: {', '.join(missing) if missing else 'none'}")
    lines.extend([
        "",
        "## Outputs",
        "",
    ])
    for key, value in report["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def build_layer(args: argparse.Namespace) -> dict[str, Any]:
    root = workspace_root()
    table_cells_path = (root / args.table_cells).resolve()
    table_index_path = (root / args.table_index).resolve()
    metric_config_path = (root / args.metric_aliases).resolve()
    unit_config_path = (root / args.unit_rules).resolve()
    candidates_path = (root / args.candidates_out).resolve()
    unit_samples_path = (root / args.unit_samples_out).resolve()
    schema_path = (root / args.schema_out).resolve()
    report_json_path = (root / args.report_json).resolve()
    report_md_path = (root / args.report_md).resolve()

    units = UnitCatalog(read_json(unit_config_path))
    metrics = MetricCatalog(read_json(metric_config_path), units)
    unit_self_tests = run_unit_self_tests(units)
    if not all(item["passed"] for item in unit_self_tests):
        failures = [item for item in unit_self_tests if not item["passed"]]
        raise RuntimeError(f"Unit self-tests failed: {failures}")

    table_index = load_table_index(table_index_path)

    counts: Counter[str] = Counter()
    parse_error_counts: Counter[str] = Counter()
    parse_status_counts: Counter[str] = Counter()
    unit_rule_counts: Counter[str] = Counter()
    unit_source_counts: Counter[str] = Counter()
    normalization_warning_counts: Counter[str] = Counter()
    metric_counts: dict[str, Counter[str]] = defaultdict(Counter)
    context_counts: Counter[str] = Counter()
    unable_parse_samples: list[dict[str, Any]] = []
    ambiguity_samples: list[dict[str, Any]] = []
    unit_samples_by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    canonical_sample_seen: dict[str, bool] = {metric: False for metric in metrics.canonical_metrics()}
    spot_checks = {
        "flyada_revenue_canonicalized": False,
        "flyada_parent_net_profit_canonicalized": False,
        "flyada_yoy_percent_is_ratio": False,
    }

    with open_jsonl(candidates_path) as candidate_fh:
        for _, cell in iter_jsonl(table_cells_path):
            if cell.get("cell_role") != "data":
                continue
            counts["scanned_data_cells"] += 1
            table_id = str(cell.get("table_id") or "")
            table_meta = table_index.get(table_id, {})
            section_path = cell.get("section_path") if isinstance(cell.get("section_path"), list) else table_meta.get("section_path", [])
            context_is_financial = financial_context(section_path)
            context_counts["financial" if context_is_financial else "other"] += 1

            normalized = normalize_number(
                cell.get("raw_value"),
                cell.get("row_label"),
                cell.get("column_label"),
                cell.get("unit_hint") if cell.get("unit_hint") is not None else table_meta.get("unit_hint"),
                units,
            )
            parse_status_counts[normalized.parse_status] += 1
            if normalized.parse_status == "parsed":
                counts["parsed_numeric_cells"] += 1
            elif len(unable_parse_samples) < MAX_REPORT_SAMPLES:
                unable_parse_samples.append(sample_from_cell(cell, table_meta, normalized, reason=normalized.parse_error))
            if normalized.parse_error:
                parse_error_counts[normalized.parse_error] += 1
            for warning in normalized.warnings:
                normalization_warning_counts[warning] += 1
            if normalized.unit_rule_id:
                counts["unit_conversion_hits"] += 1
                unit_rule_counts[normalized.unit_rule_id] += 1
                if normalized.unit_source:
                    unit_source_counts[normalized.unit_source] += 1
                samples = unit_samples_by_rule[normalized.unit_rule_id]
                if len(samples) < MAX_UNIT_SAMPLES_PER_RULE:
                    samples.append(sample_from_cell(cell, table_meta, normalized))

            metric, ambiguities = metrics.select_metric(cell.get("row_label"), cell.get("column_label"))
            if ambiguities and len(ambiguity_samples) < MAX_REPORT_SAMPLES:
                for ambiguity in ambiguities:
                    if len(ambiguity_samples) >= MAX_REPORT_SAMPLES:
                        break
                    ambiguity_samples.append(sample_from_cell(cell, table_meta, normalized, reason=ambiguity["reason"]))
            if not metric:
                continue

            counts["canonical_metric_hits"] += 1
            canonical_sample_seen[metric.canonical_metric] = True
            metric_counts[metric.canonical_metric]["candidate_rows"] += 1
            if normalized.parse_status == "parsed":
                metric_counts[metric.canonical_metric]["parsed_rows"] += 1
            if normalized.unit_rule_id:
                metric_counts[metric.canonical_metric]["unit_normalized_rows"] += 1
            if context_is_financial:
                metric_counts[metric.canonical_metric]["financial_context_rows"] += 1
            else:
                metric_counts[metric.canonical_metric]["other_context_rows"] += 1

            record = candidate_record(cell, table_meta, metric, normalized, context_is_financial)
            write_jsonl_row(candidate_fh, record)

            if table_id == FLYADA_TABLE_ID:
                row_label = normalize_text(cell.get("row_label"))
                column_label = normalize_text(cell.get("column_label"))
                if row_label == "营业收入（元)" and metric.canonical_metric == "营业收入":
                    spot_checks["flyada_revenue_canonicalized"] = True
                if row_label == "归属于上市公司股东的净利润（元）" and metric.canonical_metric == "归属于上市公司股东的净利润":
                    spot_checks["flyada_parent_net_profit_canonicalized"] = True
                if column_label == "本年比上年增减" and row_label == "营业收入（元)" and normalized.normalized_unit == "ratio":
                    spot_checks["flyada_yoy_percent_is_ratio"] = True

    with open_jsonl(unit_samples_path) as unit_fh:
        for rule_id in sorted(unit_samples_by_rule):
            for sample in unit_samples_by_rule[rule_id]:
                write_jsonl_row(unit_fh, sample)

    missing_core_metrics = [metric for metric, seen in canonical_sample_seen.items() if not seen]
    validation_errors: list[str] = []
    if missing_core_metrics:
        validation_errors.append("missing_core_metric_samples:" + ",".join(missing_core_metrics))
    failed_spot_checks = [name for name, passed in spot_checks.items() if not passed]
    if failed_spot_checks:
        validation_errors.append("failed_spot_checks:" + ",".join(failed_spot_checks))

    report = {
        "schema_version": SCHEMA_REPORT,
        "adapter_version": ADAPTER_VERSION,
        "generated_at": utc_now(),
        "inputs": {
            "table_cells": rel(table_cells_path, root),
            "tabgr_table_index": rel(table_index_path, root),
            "metric_aliases": rel(metric_config_path, root),
            "unit_rules": rel(unit_config_path, root),
        },
        "outputs": {
            "canonical_metric_candidates": rel(candidates_path, root),
            "unit_normalization_samples": rel(unit_samples_path, root),
            "canonical_metric_candidates_schema": rel(schema_path, root),
            "metric_unit_report": rel(report_json_path, root),
            "phase_05_report": rel(report_md_path, root),
        },
        "config_versions": {
            "metric_aliases": metrics.schema_version,
            "unit_rules": units.schema_version,
        },
        "counts": {
            "scanned_data_cells": counts["scanned_data_cells"],
            "parsed_numeric_cells": counts["parsed_numeric_cells"],
            "canonical_metric_hits": counts["canonical_metric_hits"],
            "unit_conversion_hits": counts["unit_conversion_hits"],
            "financial_context_cells": context_counts["financial"],
            "other_context_cells": context_counts["other"],
        },
        "parse_status_counts": dict(sorted(parse_status_counts.items())),
        "parse_error_counts": dict(sorted(parse_error_counts.items())),
        "unit_rule_counts": dict(sorted(unit_rule_counts.items())),
        "unit_source_counts": dict(sorted(unit_source_counts.items())),
        "normalization_warning_counts": dict(sorted(normalization_warning_counts.items())),
        "metric_counts": {
            metric: {
                "candidate_rows": values["candidate_rows"],
                "parsed_rows": values["parsed_rows"],
                "unit_normalized_rows": values["unit_normalized_rows"],
                "financial_context_rows": values["financial_context_rows"],
                "other_context_rows": values["other_context_rows"],
            }
            for metric, values in sorted(metric_counts.items())
        },
        "samples": {
            "unable_parse": unable_parse_samples,
            "ambiguity_or_conflict": ambiguity_samples,
        },
        "validation": {
            "unit_self_tests": unit_self_tests,
            "spot_checks": spot_checks,
            "missing_core_metric_samples": missing_core_metrics,
            "validation_errors": validation_errors,
        },
    }

    write_json(schema_path, validation_schema())
    write_json(report_json_path, report)
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.write_text(make_report_markdown(report), encoding="utf-8")

    if validation_errors:
        raise RuntimeError("Phase 5 validation failed: " + "; ".join(validation_errors))
    return report


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-cells", default=DEFAULT_TABLE_CELLS)
    parser.add_argument("--table-index", default=DEFAULT_TABLE_INDEX)
    parser.add_argument("--metric-aliases", default=DEFAULT_METRIC_ALIASES)
    parser.add_argument("--unit-rules", default=DEFAULT_UNIT_RULES)
    parser.add_argument("--candidates-out", default=DEFAULT_CANDIDATES)
    parser.add_argument("--unit-samples-out", default=DEFAULT_UNIT_SAMPLES)
    parser.add_argument("--schema-out", default=DEFAULT_SCHEMA)
    parser.add_argument("--report-json", default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", default=DEFAULT_REPORT_MD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    report = build_layer(args)
    counts = report["counts"]
    print(
        "Phase 5 complete: "
        f"scanned={counts['scanned_data_cells']} "
        f"parsed={counts['parsed_numeric_cells']} "
        f"canonical_hits={counts['canonical_metric_hits']} "
        f"unit_hits={counts['unit_conversion_hits']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
