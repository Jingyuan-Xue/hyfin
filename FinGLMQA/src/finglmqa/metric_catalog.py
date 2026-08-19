"""Deterministic metric, metadata, and formula vocabulary for Phase 8.

The Phase 5 alias file remains the source of truth for the ten selected-fact
metrics.  This module adds query-only aliases for those metrics and a small
catalog of formula/metadata names.  Query-only entries never change Phase 5
selection and do not make an unsupported operand executable in Phase 8.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRIC_CONFIG = ROOT / "src/config/metric_aliases.json"


@dataclass(frozen=True)
class MetricDefinition:
    canonical_metric: str
    aliases: tuple[str, ...]
    expected_unit: str | None
    selected_fact_metric: bool


@dataclass(frozen=True)
class FormulaDefinition:
    formula_id: str
    canonical_formula: str
    aliases: tuple[str, ...]
    operands: tuple[tuple[str, str, int], ...]
    normalized_unit: str


def _compact(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", "", text).strip("：:，,。.;；?？")


_SELECTED_UNITS: dict[str, str | None] = {
    "营业收入": "元",
    "归属于上市公司股东的净利润": "元",
    "扣除非经常性损益后的净利润": "元",
    "经营活动产生的现金流量净额": "元",
    "总资产": "元",
    "净资产": "元",
    "基本每股收益": "元/股",
    "稀释每股收益": "元/股",
    "加权平均净资产收益率": "ratio",
    # Selected rows contain both 元 and 股.  The query layer must not guess.
    "股本": None,
}


# Query-only operands and fields.  They are intentionally distinct from the
# selected-fact metric set: recognition is not an assertion of availability.
_QUERY_METRICS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("营业成本", "元", ("营业成本",)),
    ("营业利润", "元", ("营业利润",)),
    ("净利润", "元", ("净利润",)),
    ("利润总额", "元", ("利润总额",)),
    ("综合收益总额", "元", ("综合收益总额",)),
    ("投资收益", "元", ("投资收益",)),
    ("销售费用", "元", ("销售费用",)),
    ("管理费用", "元", ("管理费用",)),
    ("财务费用", "元", ("财务费用",)),
    ("研发费用", "元", ("研发费用", "研发经费")),
    ("营业外收入", "元", ("营业外收入",)),
    ("营业外支出", "元", ("营业外支出",)),
    ("营业总成本", "元", ("营业总成本",)),
    ("利息收入", "元", ("利息收入",)),
    ("利息支出", "元", ("利息支出",)),
    ("所得税费用", "元", ("所得税费用",)),
    ("营业税金及附加", "元", ("营业税金及附加", "营业税金", "税金及附加")),
    ("货币资金", "元", ("货币资金",)),
    ("存货", "元", ("存货",)),
    ("固定资产", "元", ("固定资产",)),
    ("无形资产", "元", ("无形资产",)),
    ("商誉", "元", ("商誉",)),
    ("在建工程", "元", ("在建工程",)),
    ("流动资产", "元", ("流动资产", "流动资产合计")),
    ("非流动资产", "元", ("非流动资产", "非流动资产合计")),
    ("其他流动资产", "元", ("其他流动资产",)),
    ("其他非流动资产", "元", ("其他非流动资产",)),
    ("其他非流动金融资产", "元", ("其他非流动金融资产",)),
    ("衍生金融资产", "元", ("衍生金融资产",)),
    ("流动负债", "元", ("流动负债", "流动负债合计")),
    ("非流动负债", "元", ("非流动负债", "非流动负债合计")),
    ("总负债", "元", ("总负债", "负债合计", "负债总计")),
    ("应收账款", "元", ("应收账款",)),
    ("应付账款", "元", ("应付账款",)),
    ("应收票据", "元", ("应收票据",)),
    ("应付票据", "元", ("应付票据",)),
    ("短期借款", "元", ("短期借款",)),
    ("长期借款", "元", ("长期借款",)),
    ("应交税费", "元", ("应交税费",)),
    ("应付职工薪酬", "元", ("应付职工薪酬",)),
    ("应收款项融资", "元", ("应收款项融资",)),
    ("公允价值变动收益", "元", ("公允价值变动收益",)),
    ("负债和所有者权益总计", "元", ("负债和所有者权益总计", "负债及所有者权益总计")),
    ("归属于母公司所有者权益合计", "元", ("归属于母公司所有者权益合计", "归属于母公司股东权益合计")),
    ("现金及现金等价物", "元", ("现金及现金等价物",)),
    ("研发人员数", "人", ("研发人员数", "研发人员人数")),
    ("职工总数", "人", ("职工总数", "职工总人数", "员工总数")),
    ("技术人员数", "人", ("技术人员数", "技术人员人数")),
    ("硕士人数", "人", ("硕士人数", "硕士员工人数")),
    ("博士及以上人数", "人", ("博士及以上人数", "博士人数", "博士")),
    ("每股净资产", "元/股", ("每股净资产",)),
    ("每股经营现金流量", "元/股", ("每股经营现金流量",)),
)


_QUERY_ONLY_ALIASES: dict[str, tuple[str, ...]] = {
    "营业收入": ("收入",),
    "归属于上市公司股东的净利润": ("归母净利润", "归母利润"),
    "扣除非经常性损益后的净利润": ("扣非净利润", "扣非后净利润"),
    "经营活动产生的现金流量净额": ("经营现金流", "经营性现金流净额"),
    "总资产": ("资产总额", "资产总计"),
    "净资产": ("归母净资产",),
    "加权平均净资产收益率": ("加权ROE", "ROE"),
}


METADATA_FIELDS: dict[str, tuple[str, ...]] = {
    "stock_code": ("证券代码", "股票代码", "证券编号"),
    "stock_name": ("证券简称", "股票简称"),
    "company_full": ("企业名称", "公司名称", "中文名称"),
    "company_english": ("外文名称", "英文名称"),
    "company_english_abbreviation": ("外文名称缩写", "英文简称"),
    "office_address": ("办公地址",),
    "registered_address": ("注册地址", "注册地"),
    "website": ("公司网址", "官方网址", "网站地址"),
    "legal_representative": ("法定代表人", "法人代表"),
    "email": ("电子信箱", "电子邮箱", "公司邮箱"),
}


def _growth_formula(
    formula_id: str,
    label: str,
    metric: str,
    *, expression_metric: str | None = None,
) -> FormulaDefinition:
    shown = expression_metric or metric
    return FormulaDefinition(
        formula_id,
        f"{label}=({shown}-上年{shown})/上年{shown}",
        (label,),
        (("current", metric, 0), ("previous", metric, -1)),
        "ratio",
    )


FORMULAS: tuple[FormulaDefinition, ...] = (
    FormulaDefinition(
        "revenue_growth_rate.v1",
        "营业收入增长率=(营业收入-上年营业收入)/上年营业收入",
        ("营业收入增长率", "收入增长率"),
        (("current", "营业收入", 0), ("previous", "营业收入", -1)),
        "ratio",
    ),
    _growth_formula("total_assets_growth_rate.v1", "总资产增长率", "总资产", expression_metric="资产总额"),
    _growth_formula("net_assets_growth_rate.v1", "净资产增长率", "净资产"),
    _growth_formula("parent_net_profit_growth_rate.v1", "归母净利润增长率", "归属于上市公司股东的净利润"),
    _growth_formula("sales_expense_growth_rate.v1", "销售费用增长率", "销售费用"),
    _growth_formula("operating_profit_growth_rate.v1", "营业利润增长率", "营业利润"),
    _growth_formula("administrative_expense_growth_rate.v1", "管理费用增长率", "管理费用"),
    _growth_formula("current_liabilities_growth_rate.v1", "流动负债增长率", "流动负债"),
    _growth_formula("intangible_assets_growth_rate.v1", "无形资产增长率", "无形资产"),
    _growth_formula("financial_expense_growth_rate.v1", "财务费用增长率", "财务费用"),
    _growth_formula("rd_expense_growth_rate.v1", "研发费用增长率", "研发费用"),
    _growth_formula("net_profit_growth_rate.v1", "净利润增长率", "净利润"),
    _growth_formula("total_liabilities_growth_rate.v1", "总负债增长率", "总负债"),
    _growth_formula("cash_growth_rate.v1", "货币资金增长率", "货币资金"),
    _growth_formula("investment_income_growth_rate.v1", "投资收益增长率", "投资收益"),
    _growth_formula("fixed_assets_growth_rate.v1", "固定资产增长率", "固定资产"),
    _growth_formula(
        "cash_equivalents_growth_rate.v1",
        "现金及现金等价物增长率",
        "现金及现金等价物",
    ),
    FormulaDefinition(
        "gross_margin.v1", "毛利率=(营业收入-营业成本)/营业收入", ("毛利率",),
        (("revenue", "营业收入", 0), ("cost", "营业成本", 0)), "ratio",
    ),
    FormulaDefinition(
        "operating_cost_ratio.v1", "营业成本率=营业成本/营业收入", ("营业成本率",),
        (("numerator", "营业成本", 0), ("denominator", "营业收入", 0)), "ratio",
    ),
    FormulaDefinition(
        "investment_income_revenue_ratio.v1", "投资收益占营业收入比率=投资收益/营业收入",
        ("投资收益占营业收入比率", "投资收益占营业收入的比率", "投资收益占营收比率"),
        (("numerator", "投资收益", 0), ("denominator", "营业收入", 0)), "ratio",
    ),
    FormulaDefinition(
        "administrative_expense_ratio.v1", "管理费用率=管理费用/营业收入", ("管理费用率",),
        (("numerator", "管理费用", 0), ("denominator", "营业收入", 0)), "ratio",
    ),
    FormulaDefinition(
        "financial_expense_ratio.v1", "财务费用率=财务费用/营业收入", ("财务费用率",),
        (("numerator", "财务费用", 0), ("denominator", "营业收入", 0)), "ratio",
    ),
    FormulaDefinition(
        "operating_margin.v1", "营业利润率=营业利润/营业收入", ("营业利润率",),
        (("numerator", "营业利润", 0), ("denominator", "营业收入", 0)), "ratio",
    ),
    FormulaDefinition(
        "net_profit_margin.v1", "净利润率=净利润/营业收入", ("净利润率",),
        (("numerator", "净利润", 0), ("denominator", "营业收入", 0)), "ratio",
    ),
    FormulaDefinition(
        "current_ratio.v1", "流动比率=流动资产/流动负债", ("流动比率",),
        (("numerator", "流动资产", 0), ("denominator", "流动负债", 0)), "ratio",
    ),
    FormulaDefinition(
        "quick_ratio.v1", "速动比率=(流动资产-存货)/流动负债", ("速动比率",),
        (("current_assets", "流动资产", 0), ("inventory", "存货", 0), ("denominator", "流动负债", 0)), "ratio",
    ),
    FormulaDefinition(
        "cash_ratio.v1", "现金比率=货币资金/流动负债", ("现金比率",),
        (("numerator", "货币资金", 0), ("denominator", "流动负债", 0)), "ratio",
    ),
    FormulaDefinition(
        "debt_asset_ratio.v1", "资产负债比率=总负债/资产总额", ("资产负债比率", "资产负债率"),
        (("numerator", "总负债", 0), ("denominator", "总资产", 0)), "ratio",
    ),
    FormulaDefinition(
        "current_liabilities_ratio.v1", "流动负债比率=流动负债/总负债", ("流动负债比率",),
        (("numerator", "流动负债", 0), ("denominator", "总负债", 0)), "ratio",
    ),
    FormulaDefinition(
        "noncurrent_liabilities_ratio.v1", "非流动负债比率=非流动负债/总负债", ("非流动负债比率",),
        (("numerator", "非流动负债", 0), ("denominator", "总负债", 0)), "ratio",
    ),
    FormulaDefinition(
        "rd_revenue_ratio.v1", "企业研发经费与营业收入比值=研发费用/营业收入",
        ("企业研发经费与营业收入比值", "研发经费与营业收入比值", "研发费用占营业收入比例"),
        (("numerator", "研发费用", 0), ("denominator", "营业收入", 0)), "ratio",
    ),
    FormulaDefinition(
        "rd_profit_ratio.v1", "企业研发经费与利润比值=研发费用/净利润",
        ("企业研发经费与利润比值", "研发经费与利润比值"),
        (("numerator", "研发费用", 0), ("denominator", "净利润", 0)), "ratio",
    ),
    FormulaDefinition(
        "three_expense_ratio.v1", "三费比重=(销售费用+管理费用+财务费用)/营业收入", ("三费比重",),
        (("sales", "销售费用", 0), ("administrative", "管理费用", 0), ("financial", "财务费用", 0), ("denominator", "营业收入", 0)), "ratio",
    ),
    FormulaDefinition(
        "rd_expense_share.v1", "企业研发经费占费用比例=研发费用/(销售费用+财务费用+管理费用+研发费用)",
        ("企业研发经费占费用比例", "研发经费占费用比例", "研发经费占费用的比例"),
        (("rd", "研发费用", 0), ("sales", "销售费用", 0), ("financial", "财务费用", 0), ("administrative", "管理费用", 0)), "ratio",
    ),
    FormulaDefinition(
        "rd_staff_ratio.v1", "研发人员占职工人数比例=研发人员数/职工总数",
        ("研发人员占职工人数比例", "研发人员占职工人数的比例", "研发人员占员工人数比例", "研发人员占员工人数的比例"),
        (("numerator", "研发人员数", 0), ("denominator", "职工总数", 0)), "ratio",
    ),
    FormulaDefinition(
        "postgraduate_staff_ratio.v1", "企业硕士及以上人员占职工人数比例=(硕士人数+博士及以上人数)/职工总数",
        ("企业硕士及以上人员占职工人数比例", "硕士及以上人员占职工人数比例"),
        (("master", "硕士人数", 0), ("doctor", "博士及以上人数", 0), ("denominator", "职工总数", 0)), "ratio",
    ),
)


class MetricCatalog:
    """Read-only query vocabulary with deterministic longest-match behavior."""

    version = "phase8-metric-catalog-v1"

    def __init__(self, config_path: str | Path = DEFAULT_METRIC_CONFIG) -> None:
        path = Path(config_path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        definitions: list[MetricDefinition] = []
        for item in raw.get("metrics", []):
            canonical = str(item["canonical_metric"])
            aliases = tuple(dict.fromkeys((canonical, *item.get("aliases", ()), *_QUERY_ONLY_ALIASES.get(canonical, ()))))
            definitions.append(MetricDefinition(canonical, aliases, _SELECTED_UNITS[canonical], True))
        for canonical, unit, aliases in _QUERY_METRICS:
            definitions.append(MetricDefinition(canonical, tuple(dict.fromkeys((canonical, *aliases))), unit, False))

        self._definitions = tuple(definitions)
        self._by_canonical = {item.canonical_metric: item for item in definitions}
        alias_map: dict[str, set[str]] = {}
        for item in definitions:
            for alias in item.aliases:
                alias_map.setdefault(_compact(alias), set()).add(item.canonical_metric)
        self._alias_map = {alias: tuple(sorted(values)) for alias, values in alias_map.items()}
        self._metric_aliases = tuple(sorted(self._alias_map, key=lambda value: (-len(value), value)))

        ambiguous: dict[str, tuple[str, ...]] = {
            "净利润": ("归属于上市公司股东的净利润", "扣除非经常性损益后的净利润", "净利润"),
            "每股收益": ("基本每股收益", "稀释每股收益"),
            "净资产收益率": ("加权平均净资产收益率", "摊薄净资产收益率"),
        }
        self._ambiguous = ambiguous
        self._formula_by_id = {item.formula_id: item for item in FORMULAS}
        formula_aliases: list[tuple[str, FormulaDefinition]] = []
        for formula in FORMULAS:
            for alias in formula.aliases:
                formula_aliases.append((_compact(alias), formula))
        self._formula_aliases = tuple(sorted(formula_aliases, key=lambda row: (-len(row[0]), row[0], row[1].formula_id)))

        metadata_aliases: list[tuple[str, str]] = []
        for field, aliases in METADATA_FIELDS.items():
            metadata_aliases.extend((_compact(alias), field) for alias in aliases)
        self._metadata_aliases = tuple(sorted(metadata_aliases, key=lambda row: (-len(row[0]), row[0], row[1])))

    @property
    def definitions(self) -> tuple[MetricDefinition, ...]:
        return self._definitions

    @property
    def formulas(self) -> tuple[FormulaDefinition, ...]:
        return FORMULAS

    def get(self, canonical_metric: str) -> MetricDefinition | None:
        return self._by_canonical.get(canonical_metric)

    def expected_unit(self, canonical_metric: str) -> str | None:
        definition = self.get(canonical_metric)
        return definition.expected_unit if definition else None

    def is_selected_fact_metric(self, canonical_metric: str) -> bool:
        definition = self.get(canonical_metric)
        return bool(definition and definition.selected_fact_metric)

    def resolve(self, label: object) -> dict[str, Any]:
        normalized = _compact(label)
        if normalized in self._ambiguous:
            return {"status": "ambiguous", "candidates": list(self._ambiguous[normalized])}
        candidates = self._alias_map.get(normalized, ())
        if len(candidates) == 1:
            return {"status": "unique", "candidates": list(candidates)}
        if len(candidates) > 1:
            return {"status": "ambiguous", "candidates": list(candidates)}
        return {"status": "missing", "candidates": []}

    def formula(self, formula_id: str) -> FormulaDefinition | None:
        return self._formula_by_id.get(formula_id)

    def find_formula_mentions(self, text: str) -> list[dict[str, Any]]:
        return self._find_alias_mentions(text, self._formula_aliases, value_name="formula")

    def find_metric_mentions(self, text: str, excluded_spans: Iterable[tuple[int, int]] = ()) -> list[dict[str, Any]]:
        excluded = tuple(excluded_spans)
        rows: list[dict[str, Any]] = []
        occupied: list[tuple[int, int]] = []
        candidates: list[tuple[int, int, str]] = []
        for alias in self._metric_aliases:
            for match in re.finditer(re.escape(alias), text, flags=re.IGNORECASE):
                candidates.append((match.start(), match.end(), match.group(0)))
        # Ambiguous generic terms need to be visible even though they are not
        # necessarily exact aliases in Phase 5.
        for alias in self._ambiguous:
            for match in re.finditer(re.escape(alias), text, flags=re.IGNORECASE):
                candidates.append((match.start(), match.end(), match.group(0)))
        candidates.sort(key=lambda row: (row[0], -(row[1] - row[0]), row[2]))
        for start, end, raw_text in candidates:
            if raw_text == "收入" and re.search(
                r"(?:主营业务|其他业务|利息|手续费及佣金|营业外)$",
                text[max(0, start - 8):start],
            ):
                continue
            if any(start < right and end > left for left, right in excluded):
                continue
            if any(start < right and end > left for left, right in occupied):
                continue
            resolved = self.resolve(raw_text)
            rows.append({"raw_text": raw_text, "span": [start, end], **resolved})
            occupied.append((start, end))
        return sorted(rows, key=lambda row: (row["span"][0], row["span"][1]))

    def find_metadata_mentions(self, text: str, excluded_spans: Iterable[tuple[int, int]] = ()) -> list[dict[str, Any]]:
        excluded = tuple(excluded_spans)
        return [
            {"raw_text": row["raw_text"], "span": row["span"], "metadata_field": row["value"]}
            for row in self._find_alias_mentions(text, self._metadata_aliases, value_name="value")
            if not any(row["span"][0] < right and row["span"][1] > left for left, right in excluded)
        ]

    @staticmethod
    def _find_alias_mentions(text: str, aliases: Iterable[tuple[str, Any]], *, value_name: str) -> list[dict[str, Any]]:
        candidates: list[tuple[int, int, str, Any]] = []
        for alias, value in aliases:
            for match in re.finditer(re.escape(alias), text, flags=re.IGNORECASE):
                candidates.append((match.start(), match.end(), match.group(0), value))
        candidates.sort(key=lambda row: (row[0], -(row[1] - row[0]), row[2]))
        occupied: list[tuple[int, int]] = []
        result: list[dict[str, Any]] = []
        for start, end, raw_text, value in candidates:
            if any(start < right and end > left for left, right in occupied):
                continue
            result.append({"raw_text": raw_text, "span": [start, end], value_name: value})
            occupied.append((start, end))
        return result


__all__ = [
    "DEFAULT_METRIC_CONFIG",
    "FORMULAS",
    "METADATA_FIELDS",
    "FormulaDefinition",
    "MetricCatalog",
    "MetricDefinition",
]
