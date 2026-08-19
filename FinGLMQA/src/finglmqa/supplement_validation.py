"""Pure fail-closed validation for Phase 9 candidate table cells."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import semantic_sha256


ROOT = Path(__file__).resolve().parents[2]
METRIC_CONFIG = ROOT / "src/config/metric_aliases.json"
UNIT_CONFIG = ROOT / "src/config/unit_rules.json"
VALIDATOR_VERSION = "phase9-supplement-validator-v1"
MIN_CONFIDENCE = Decimal("0.70")

DURATION_METRICS = frozenset({
    "营业收入", "归属于上市公司股东的净利润", "扣除非经常性损益后的净利润",
    "经营活动产生的现金流量净额", "基本每股收益", "稀释每股收益",
    "加权平均净资产收益率",
})
INSTANT_METRICS = frozenset({"总资产", "净资产", "股本"})
COMPARISON_RE = re.compile(r"同比|环比|增减|增长|下降|变动|比例|比上|较上|差异|占比")
QUARTER_RE = re.compile(r"季度|半年度|半年|[1-4一二三四]季|\d+\s*[-至]\s*\d+\s*月")
YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
CURRENT_RE = re.compile(r"本报告期|报告期|本年度|本年|本期|期末|年末")
PREVIOUS_RE = re.compile(r"上年度|上年|上期|期初|年初")
OPENING_DATE_RE = re.compile(r"(?:1月1日|01月01日|01-01|年初|期初)")
YEAR_END_DATE_RE = re.compile(r"(?:12月31日|12-31|年末|期末)")
MONTH_OR_DAY_RE = re.compile(r"\d{1,2}\s*月|\d{1,2}\s*日|\d{1,2}[-/]\d{1,2}")
BRACKET_RE = re.compile(r"[\(（]\s*([^()（）]{1,40})\s*[\)）]")
UNIT_HINT_RE = re.compile(r"(?:单位|金额单位|币种)\s*[:：]\s*([^,，;；。.\s]+)")
NUMBER_RE = re.compile(r"^(?:人民币)?\s*(?P<number>[+-]?(?:(?:\d{1,3}(?:,\d{3})+)|(?:\d+))(?:\.\d+)?|[+-]?\.\d+)\s*(?P<unit>[\u4e00-\u9fffA-Za-z%/／]+)?$")

FINANCIAL_CONTEXT_TERMS = (
    "财务", "会计", "主要会计数据", "财务指标", "资产负债", "利润", "现金流",
    "所有者权益", "股东权益", "营业收入", "股本", "每股收益",
)
MAIN_SCOPE_TERMS = (
    "主要会计数据和财务指标", "主要会计数据", "主要财务指标", "合并资产负债表",
    "合并利润表", "合并现金流量表",
)
NON_COMPANY_SCOPE_TERMS = (
    "分部信息", "分行业", "分地区", "分产品", "主要控股参股公司", "主要子公司",
    "重要子公司", "联营企业", "合营企业", "企业合并", "处置子公司", "非同一控制下",
    "母公司利润表", "母公司资产负债表", "母公司现金流量表", "资产构成重大变动",
    "按行业", "按地区", "按产品",
)
STATEMENT_BY_METRIC = {
    "营业收入": "income_statement",
    "归属于上市公司股东的净利润": "income_statement",
    "扣除非经常性损益后的净利润": "income_statement",
    "经营活动产生的现金流量净额": "cash_flow_statement",
    "总资产": "balance_sheet", "净资产": "balance_sheet", "股本": "balance_sheet",
    "基本每股收益": "financial_indicator", "稀释每股收益": "financial_indicator",
    "加权平均净资产收益率": "financial_indicator",
}

_TRANSLATION = str.maketrans({
    "（": "(", "）": ")", "，": ",", "．": ".", "。": ".", "＋": "+",
    "－": "-", "—": "-", "–": "-", "／": "/", "％": "%", "　": " ",
    **{chr(ord("０") + index): str(index) for index in range(10)},
})


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(value: object) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value or ""))).strip()


def _compact(value: object) -> str:
    return re.sub(r"\s+", "", _text(value).translate(_TRANSLATION))


def _plain(value: Decimal) -> str:
    if value == value.to_integral_value():
        return format(value.quantize(Decimal(1)), "f")
    return format(value.normalize(), "f")


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    failure_code: str | None
    stage: int
    detail: dict[str, Any]


class SupplementValidator:
    def __init__(self, metric_config: Path = METRIC_CONFIG, unit_config: Path = UNIT_CONFIG) -> None:
        self.metric_config = json.loads(metric_config.read_text(encoding="utf-8"))
        self.unit_config = json.loads(unit_config.read_text(encoding="utf-8"))
        self.metric_by_name = {row["canonical_metric"]: row for row in self.metric_config["metrics"]}
        self.unit_aliases: dict[str, dict[str, Any]] = {}
        for rule in self.unit_config["rules"]:
            for alias in rule["aliases"]:
                self.unit_aliases[self._unit_phrase(alias)] = rule
        self.null_markers = {_compact(row) for row in self.unit_config["null_markers"]}
        self.validation_versions = {
            "validator": VALIDATOR_VERSION,
            "metric_aliases_sha256": _file_sha(metric_config),
            "unit_rules_sha256": _file_sha(unit_config),
            "year_rules": "phase6-financial-fact-store-v1",
            "eligibility_rules": "phase6-financial-fact-store-v1",
        }

    def aliases(self, metric: str) -> tuple[str, ...]:
        row = self.metric_by_name[metric]
        return tuple(row["aliases"])

    def _unit_phrase(self, value: object) -> str:
        text = _compact(value)
        text = re.sub(r"^(?:单位|金额单位|币种)[:：]?", "", text)
        return text.strip("：:()[]【】,，;；。.")

    def _resolve_unit_phrase(self, value: object) -> dict[str, Any] | None:
        return self.unit_aliases.get(self._unit_phrase(value))

    def _normalize_metric_label(self, label: object) -> str:
        text = _compact(label)
        text = re.sub(r"^[\(（]?[一二三四五六七八九十百\d]+[、.)）]", "", text)
        text = text.strip(":：,，;；.。")

        def strip_unit(match: re.Match[str]) -> str:
            return "" if self._resolve_unit_phrase(match.group(1)) else match.group(0)

        previous = None
        while previous != text:
            previous = text
            text = BRACKET_RE.sub(strip_unit, text)
        return text.replace("(", "").replace(")", "").replace("（", "").replace("）", "").strip(":：,，;；.。")

    def metric_match(self, request: Mapping[str, Any], cell: Mapping[str, Any]) -> tuple[str | None, str | None, str | None]:
        target = request["canonical_metric"]
        definitions = self.metric_by_name[target]
        alias_set = {self._normalize_metric_label(alias) for alias in definitions["aliases"]}
        matches: list[tuple[str, str]] = []
        for source in ("row_label", "column_label"):
            label = self._normalize_metric_label(cell.get(source))
            if label in alias_set and not any(re.search(pattern, label) for pattern in definitions["exclude_patterns"]):
                matches.append((source, label))
        # A cell may have the metric in both axes only if it is the same exact
        # target. Prefer row labels as Phase 5 does.
        if not matches:
            return None, None, None
        matches.sort(key=lambda row: (0 if row[0] == "row_label" else 1, -len(row[1])))
        return matches[0][0], matches[0][1], "exact_alias"

    def resolve_year(self, request: Mapping[str, Any], cell: Mapping[str, Any], metric_source: str) -> dict[str, Any]:
        metric = request["canonical_metric"]
        label = _text(cell.get("column_label") if metric_source == "row_label" else cell.get("row_label"))
        report_year = int(request["report_year"])
        result: dict[str, Any] = {"period_label": label, "metric_year": None, "year_resolution": None, "year_confidence": None, "error": None}
        if not label:
            result["error"] = "missing_period_label"
            return result
        if COMPARISON_RE.search(label):
            result["error"] = "comparison_or_change_column"
            return result
        if QUARTER_RE.search(label):
            result["error"] = "partial_period_or_quarter"
            return result
        if metric in DURATION_METRICS and (MONTH_OR_DAY_RE.search(label) or "至" in label):
            result["error"] = "partial_duration_date"
            return result
        years = sorted({int(value) for value in YEAR_RE.findall(label)})
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
        year = int(result["metric_year"])
        if year > report_year or year < report_year - 2:
            result["error"] = "metric_year_outside_three_year_window"
        elif year != int(request["metric_year"]):
            result["error"] = "metric_year_mismatch"
        return result

    def _unit_evidence(
        self,
        cell: Mapping[str, Any],
        table: Mapping[str, Any],
    ) -> tuple[dict[str, Any] | None, str | None, list[str]]:
        evidence: list[tuple[str, dict[str, Any]]] = []
        raw_match = NUMBER_RE.fullmatch(_text(cell.get("raw_value")).translate(_TRANSLATION))
        if raw_match and raw_match.group("unit"):
            rule = self._resolve_unit_phrase(raw_match.group("unit"))
            if rule:
                evidence.append(("raw_value", rule))
        sources: list[tuple[str, object]] = [
            ("row_label", cell.get("row_label")),
            ("column_label", cell.get("column_label")),
            ("unit_hint", cell.get("unit_hint") or table.get("unit_hint")),
        ]
        for header in table.get("header") or []:
            sources.append(("table_header", header))
        for section in table.get("section_path") or []:
            sources.append(("section_declaration", section))
        for source, value in sources:
            text = _text(value)
            found: list[dict[str, Any]] = []
            for match in BRACKET_RE.finditer(text):
                rule = self._resolve_unit_phrase(match.group(1))
                if rule:
                    found.append(rule)
            hint = UNIT_HINT_RE.search(text)
            if hint:
                rule = self._resolve_unit_phrase(hint.group(1))
                if rule:
                    found.append(rule)
            direct = self._resolve_unit_phrase(text)
            if direct:
                found.append(direct)
            for rule in found:
                evidence.append((source, rule))
        units = sorted({row[1]["normalized_unit"] for row in evidence})
        if len(units) != 1:
            return None, None, units
        priority = {"raw_value": 0, "row_label": 1, "column_label": 2, "unit_hint": 3, "table_header": 4, "section_declaration": 5}
        source, rule = min((row for row in evidence if row[1]["normalized_unit"] == units[0]), key=lambda row: priority[row[0]])
        return rule, source, units

    def _normalize_value(self, raw: object, unit_rule: Mapping[str, Any]) -> tuple[str | None, list[str]]:
        raw_text = _text(raw)
        if _compact(raw_text) in self.null_markers:
            return None, ["null_marker"]
        translated = raw_text.translate(_TRANSLATION).strip()
        negative = (translated.startswith("(") and translated.endswith(")"))
        if negative:
            translated = translated[1:-1].strip()
        match = NUMBER_RE.fullmatch(translated)
        if not match:
            return None, ["not_numeric"]
        raw_suffix = match.group("unit")
        if raw_suffix:
            raw_rule = self._resolve_unit_phrase(raw_suffix)
            if raw_rule is None:
                return None, ["unknown_unit_suffix"]
            if (
                raw_rule["normalized_unit"] != unit_rule["normalized_unit"]
                or Decimal(str(raw_rule["multiplier"])) != Decimal(str(unit_rule["multiplier"]))
            ):
                return None, ["raw_unit_mismatch"]
        try:
            value = Decimal(match.group("number").replace(",", ""))
        except InvalidOperation:
            return None, ["invalid_decimal"]
        if negative and value > 0:
            value = -value
        value *= Decimal(str(unit_rule["multiplier"]))
        warnings: list[str] = []
        if unit_rule.get("integer") and value != value.to_integral_value():
            warnings.append("integer_unit_has_fraction")
        return _plain(value), warnings

    @staticmethod
    def _plausibility(metric: str, unit: str, value: Decimal) -> str | None:
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

    def confidence(
        self,
        request: Mapping[str, Any],
        cell: Mapping[str, Any],
        table: Mapping[str, Any],
        *,
        metric_source: str,
        match_type: str,
        year: Mapping[str, Any],
        unit_source: str,
    ) -> tuple[Decimal, list[str]]:
        score = Decimal("0")
        reasons: list[str] = []
        score += Decimal("0.32") if match_type == "exact_alias" else Decimal("0.22")
        reasons.append("exact_metric_alias:+0.32" if match_type == "exact_alias" else "regex_metric_match:+0.22")
        score += Decimal("0.25") if metric_source == "row_label" else Decimal("0.12")
        reasons.append("metric_from_row_label:+0.25" if metric_source == "row_label" else "metric_from_column_label:+0.12")
        score += Decimal("0.25") if year.get("year_confidence") == "high" else Decimal("0.18")
        reasons.append("explicit_metric_year:+0.25" if year.get("year_confidence") == "high" else "relative_metric_year:+0.18")
        financial_context = " ".join(str(row) for row in table.get("section_path") or [])
        context = " ".join([financial_context, str(table.get("caption") or ""), str(cell.get("row_label") or ""), str(cell.get("column_label") or "")])
        if any(term in financial_context for term in FINANCIAL_CONTEXT_TERMS):
            score += Decimal("0.10")
            reasons.append("financial_section_context:+0.10")
        if unit_source in {"raw_value", "row_label"}:
            score += Decimal("0.08")
            reasons.append("direct_unit_evidence:+0.08")
        elif unit_source == "unit_hint":
            score += Decimal("0.05")
            reasons.append("table_unit_hint:+0.05")
        elif unit_source == "column_label":
            score += Decimal("0.03")
            reasons.append("column_unit_evidence:+0.03")
        if any(term in context for term in MAIN_SCOPE_TERMS):
            score += Decimal("0.12")
            reasons.append("consolidated_or_summary_scope:+0.12")
        if any(term in context for term in NON_COMPANY_SCOPE_TERMS):
            score -= Decimal("0.30")
            reasons.append("non_company_or_parent_only_scope:-0.30")
        if int(request["metric_year"]) == int(request["report_year"]) - 2:
            score -= Decimal("0.05")
            reasons.append("two_year_lookback:-0.05")
        return min(Decimal("1"), max(Decimal("0"), score)), reasons

    def provenance_matches(
        self,
        cell: Mapping[str, Any],
        table: Mapping[str, Any],
        index: Mapping[str, Any],
    ) -> bool:
        metadata = cell.get("metadata") if isinstance(cell.get("metadata"), dict) else {}
        try:
            row_index = int(cell.get("row_index"))
            col_index = int(cell.get("col_index"))
            matrix = table["matrix"]
            matrix_value = matrix[row_index][col_index]
            matrix_column = matrix[0][col_index]
            matrix_row_label = matrix[row_index][0]
        except (KeyError, TypeError, ValueError, IndexError):
            return False
        return bool(
            index.get("tabgr_ready") is True
            and cell.get("table_id") == table.get("table_id") == index.get("table_id")
            and cell.get("document_id") == table.get("document_id") == index.get("document_id")
            and int(cell.get("table_index")) == int(table.get("table_index")) == int(index.get("table_index"))
            and cell.get("line_range") == table.get("line_range") == index.get("line_range")
            and cell.get("section_path") == table.get("section_path") == index.get("section_path")
            and cell.get("source_markdown") == index.get("source_markdown")
            and metadata.get("raw_markdown_sha1") == table.get("raw_markdown_sha1") == index.get("raw_markdown_sha1")
            and str(cell.get("raw_value")) == str(matrix_value)
            and str(cell.get("column_label") or "") == str(matrix_column or "")
            and str(cell.get("row_label") or "") == str(matrix_row_label or "")
        )

    def validate_cell(
        self,
        request: Mapping[str, Any],
        cell: Mapping[str, Any],
        table: Mapping[str, Any],
        index: Mapping[str, Any],
    ) -> ValidationResult:
        metric_source, metric_label, match_type = self.metric_match(request, cell)
        if metric_source is None:
            return ValidationResult(False, "SUPPLEMENT_CELL_NOT_FOUND", 0, {})
        year = self.resolve_year(request, cell, metric_source)
        if year.get("error"):
            return ValidationResult(False, "SUPPLEMENT_YEAR_UNRESOLVED", 1, {"year": year, "metric_source": metric_source})
        rule, unit_source, unit_candidates = self._unit_evidence(cell, table)
        if rule is None or rule["normalized_unit"] != request["normalized_unit"]:
            return ValidationResult(False, "SUPPLEMENT_UNIT_UNRESOLVED", 2, {"unit_candidates": unit_candidates, "unit_source": unit_source})
        value, warnings = self._normalize_value(cell.get("raw_value"), rule)
        if value is None or warnings:
            return ValidationResult(False, "SUPPLEMENT_VALUE_INVALID", 3, {"warnings": warnings})
        plausibility = self._plausibility(request["canonical_metric"], request["normalized_unit"], Decimal(value))
        if plausibility:
            return ValidationResult(False, "SUPPLEMENT_VALUE_INVALID", 3, {"warnings": [plausibility]})
        score, reasons = self.confidence(
            request, cell, table, metric_source=metric_source,
            match_type=str(match_type), year=year, unit_source=str(unit_source),
        )
        if score < MIN_CONFIDENCE:
            return ValidationResult(False, "SUPPLEMENT_ELIGIBILITY_REJECTED", 4, {"confidence_score": format(score, ".4f"), "score_reasons": reasons})
        if not self.provenance_matches(cell, table, index):
            return ValidationResult(False, "SUPPLEMENT_PROVENANCE_FAILED", 6, {})
        detail = {
            "normalized_value": value,
            "normalized_unit": request["normalized_unit"],
            "statement": STATEMENT_BY_METRIC[request["canonical_metric"]],
            "metric_source": metric_source,
            "metric_label": metric_label,
            "metric_match_type": match_type,
            "period": year,
            "unit_source": unit_source,
            "confidence_score": format(score, ".4f"),
            "score_reasons": reasons,
            "validation_fingerprint": semantic_sha256({
                "versions": self.validation_versions,
                "slot": [request.get(key) for key in ("document_id", "stock_code", "report_year", "metric_year", "canonical_metric", "normalized_unit")],
                "table_id": table["table_id"], "row_index": cell["row_index"], "col_index": cell["col_index"],
                "value": value,
            }),
        }
        return ValidationResult(True, None, 5, detail)


__all__ = [
    "MIN_CONFIDENCE", "STATEMENT_BY_METRIC", "SupplementValidator",
    "VALIDATOR_VERSION", "ValidationResult",
]
