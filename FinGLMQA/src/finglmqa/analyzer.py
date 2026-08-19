"""Deterministic question analysis for the frozen Phase 8 planner.

This module performs lexical analysis only.  It does not resolve companies or
documents, query facts, inspect benchmark labels, or choose a backend.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import LIMITS, SCHEMA_ANALYSIS, validate_qa_request, validate_question_analysis
from .metric_catalog import MetricCatalog


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_MANIFEST = ROOT / "data/corpus_package/corpus_manifest.json"
DEFAULT_COMPANY_INDEX = ROOT / "data/corpus_package/company_year_index.jsonl"
DEFAULT_COMPANY_ALIASES = ROOT / "src/config/company_aliases.json"
ANALYSIS_VERSION = "phase8-question-analyzer-v1"

_YEAR_RE = re.compile(r"(?<!\d)((?:19|20|21)\d{2})\s*年?")
_YEAR_RANGE_RE = re.compile(
    r"(?<!\d)((?:19|20|21)\d{2})\s*年?\s*(?:至|到|[-—~～])\s*((?:19|20|21)\d{2})\s*年"
)
_GENERIC_COMPANY_RE = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9]{1,48}?(?:集团股份有限公司|股份有限公司|有限责任公司|集团股份公司|股份公司|有限公司|公司)"
)
_SHORT_BEFORE_YEAR_RE = re.compile(r"[\u4e00-\u9fffA-Za-z]{2,24}(?=(?:在)?(?:19|20|21)\d{2}年)")
_SHORT_AFTER_YEAR_RE = re.compile(
    r"(?:19|20|21)\d{2}年(?:的)?([\u4e00-\u9fffA-Za-z]{2,12}?)(?=(?:在|的)?(?:营业|收入|资产|净利润|利润|股本|证券|股票|办公|研发|每股|法定|管理|财务|投资|毛利|流动|现金|负债))"
)
# Period-compare glue after a year boundary ("2020年相比2019年" / "较" / "比"),
# including the optional closing 年 absorbed into the short-before-year span.
_PERIOD_COMPARE_CONNECTOR_MENTION_RE = re.compile(r"^年?(?:相比|相较|较之|较|比)$")
_YEAR_THEN_PERIOD_COMPARE_RE = re.compile(r"(?:19|20|21)\d{2}\s*年?\s*(?:相比|相较|较之|较|比)\s*$")

_INTENT_ORDER = ("lookup", "compare", "rank", "aggregate", "calculate", "narrative")
_COMPARE_RE = re.compile(r"比较|对比|相比|相同|一致|不同|分别.*(?:高|低)|高于|低于|多于|少于|哪个更|孰高|增减")
_RANK_RE = re.compile(r"最高|最低|排名|排行|第[一二三四五六七八九十\d]+|Top\s*\d+|前\s*\d+", re.IGNORECASE)
_AGGREGATE_RE = re.compile(r"平均值|均值|平均|总和|汇总")
_CALCULATE_RE = re.compile(r"计算|算出|增长率|比率|比例|占比|毛利率|费用率|利润率|ROE", re.IGNORECASE)
_EXPLAIN_RE = re.compile(r"为什么|为何|原因|解释|分析|描述|缘由|驱动因素|下降原因|增长原因|变动原因")
_SUMMARIZE_RE = re.compile(r"概述|总结|综述|简述|核心竞争力|主要风险|风险是什么|经营情况|社会责任")
_INTRO_RE = re.compile(r"简要介绍(?:一下)?|介绍一下")
_NARRATIVE_SUBJECT_RE = re.compile(r"情况|详情|状况|原因|说明|事项|展望|是否发生变更")
_NARRATIVE_FALLBACK_RE = re.compile(
    r"经营情况|业务情况|现金流情况|发展情况|行业地位|审计意见|董事长致辞|主要内容|是否涉及利好|"
    r"资产及负债状况|财务状况|重大.{0,16}情况|表现|业务概况|战略|前景|风险|竞争力|社会责任"
)
_TABLE_NEED_RE = re.compile(r"前十大股东|前十名股东|员工构成|职工构成|分红方案|利润分配方案|表格(?:数据)?|表中列示")
_DYNAMIC_CONDITIONAL_RE = re.compile(r"(?:若|如果|假如).*(?:则|就).*(?:否则|不然)")
_DYNAMIC_WINNER_RE = re.compile(r"胜者|第一名|排名第一|最高者|最低者|该年|其原因|该公司")
_DOWNSTREAM_RE = re.compile(r"再|然后|并(?:分别)?解释|并(?:分别)?总结|为什么|原因|解释|概述|总结|列出其|证券代码")
_UNIT_RE = re.compile(r"元\s*/\s*股|亿元|万元|人民币元|元|百分比|%|股(?=$|[?？。.,，;；\s])|人(?=$|[?？。.,，;；\s])")
_SHARED_GROWTH_RE = re.compile(
    r"(?P<items>(?:营业收入|收入|总资产|资产总额|净资产|归母净利润)(?:[、，,和及](?:营业收入|收入|总资产|资产总额|净资产|归母净利润))+?)增长率"
)
_POSSESSIVE_GROWTH_RE = re.compile(r"(?P<base>营业收入|收入|总资产|资产总额|净资产|归母净利润)及其?增长率")

_GENERIC_COMPANY_PREFIXES = (
    "在保留两位小数的情况下请计算出", "在保留两位小数的情况下", "请提供", "请告诉我", "请问", "我想知道",
    "请具体描述一下", "请简要分析", "请简述", "简述", "概述一下", "请计算出", "计算出", "请根据", "能否根据",
    "结合", "在", "根据", "针对", "比较", "对比", "列出", "以及", "和", "与", "及", "截至", "从", "相比",
)
_GENERIC_COMPANY_QUERY_WORDS = (
    "最高公司", "最低公司", "哪些公司", "哪个公司", "全体公司", "所有公司", "上市公司", "收入", "资产",
    "利润", "平均", "排名", "排行", "高于", "低于", "找出", "谁是",
)
_NON_ENTITY_TERMS = {
    "公司", "该公司", "本公司", "股份有限公司", "有限公司", "时候公司", "同行业", "同行业公司", "报告期内公司",
    "母公司", "子公司", "归属于母公司", "主要控股参股公司", "控股股东", "实际控制人", "年至", "年到", "年度报告", "年报",
    "营业", "收入", "资产", "利润", "负债", "证券", "股票", "研发", "管理", "财务", "投资",
}
_NON_ENTITY_WORDS = (
    "描述", "分析", "报告期", "年度报告", "简要", "具体", "控股参股", "控股股东", "实际控制人",
    "法定代表人", "对比", "年报", "披露", "计算",
)


def _normalize_question(question: str) -> str:
    text = unicodedata.normalize("NFKC", question)
    return re.sub(r"\s+", " ", text).strip()


def _overlaps(span: tuple[int, int], occupied: Iterable[tuple[int, int]]) -> bool:
    return any(span[0] < end and span[1] > start for start, end in occupied)


def _stable_concern_id(kind: str, key: str, span: list[int]) -> str:
    body = json.dumps([kind, key, span], ensure_ascii=False, separators=(",", ":"))
    return "concern_" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


class CompanyMentionCatalog:
    """Company aliases built strictly from the production corpus manifest."""

    def __init__(
        self,
        manifest_path: str | Path = DEFAULT_COMPANY_INDEX,
        aliases_path: str | Path = DEFAULT_COMPANY_ALIASES,
    ) -> None:
        path = Path(manifest_path)
        if path.suffix == ".jsonl":
            documents = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            documents = payload.get("documents", [])
        aliases: dict[str, set[str]] = {}
        for document in documents:
            entity_key = str(document.get("stock_code") or document.get("document_id") or "")
            values = [
                *(document.get("aliases") or []),
                document.get("stock_name"),
                document.get("company_full"),
                document.get("stock_code"),
                document.get("stock_symbol"),
            ]
            company_full = str(document.get("company_full") or "")
            derived_short = re.sub(
                r"(?:集团股份有限公司|股份有限公司|有限责任公司|集团股份公司|股份公司|有限公司)$",
                "",
                company_full,
            )
            if derived_short.endswith("集团") and len(derived_short) > 4:
                values.append(derived_short[:-2])
            values.append(derived_short)
            for value in values:
                alias = unicodedata.normalize("NFKC", str(value or "")).strip()
                if len(alias) >= 2:
                    aliases.setdefault(alias, set()).add(entity_key)
        known_entities = {entity for values in aliases.values() for entity in values}
        alias_payload = json.loads(Path(aliases_path).read_text(encoding="utf-8"))
        if alias_payload.get("schema_version") != "finglmqa.phase8.company_aliases.v1":
            raise ValueError("Phase 8 company alias catalog schema mismatch")
        for row in alias_payload.get("aliases", []):
            alias = unicodedata.normalize("NFKC", str(row.get("alias") or "")).strip()
            stock_code = str(row.get("stock_code") or "")
            if len(alias) < 2 or stock_code not in known_entities:
                raise ValueError("Phase 8 company alias catalog contains an unknown identity")
            aliases.setdefault(alias, set()).add(stock_code)
        self._aliases = tuple(sorted(aliases, key=lambda value: (-len(value), value)))
        self._entities = {alias: tuple(sorted(values)) for alias, values in aliases.items()}

    def identity_key(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).strip()
        matches = self._entities.get(normalized, ())
        return matches[0] if len(matches) == 1 else "raw:" + normalized

    def find(self, text: str) -> list[dict[str, Any]]:
        candidates: list[tuple[int, int, str, str, int]] = []
        for alias in self._aliases:
            for match in re.finditer(re.escape(alias), text, flags=re.IGNORECASE):
                candidates.append((match.start(), match.end(), match.group(0), self.identity_key(alias), 0))

        # Preserve explicit company-like mentions that are absent from the
        # corpus so the resolver can fail closed per entity.
        for match in _GENERIC_COMPANY_RE.finditer(text):
            start, end = match.span()
            raw = match.group(0)
            for connector in ("、", "，", ",", "；", ";", "以及", "和", "与", "及"):
                if connector in raw:
                    offset = raw.rfind(connector) + len(connector)
                    start += offset
                    raw = raw[offset:]
            changed = True
            while changed:
                changed = False
                for prefix in _GENERIC_COMPANY_PREFIXES:
                    if raw.startswith(prefix) and len(raw) > len(prefix) + 1:
                        raw = raw[len(prefix):]
                        start += len(prefix)
                        changed = True
                        break
            leading = re.match(r"^(?:(?:在|根据))?(?:19|20|21)\d{2}年(?:度)?(?:的)?", raw)
            if leading:
                raw = raw[leading.end():]
                start += leading.end()
            if (
                len(raw) < 3
                or raw in _NON_ENTITY_TERMS
                or (not raw.startswith(("示例", "未收录")) and any(word in raw for word in _NON_ENTITY_WORDS))
                or any(word in raw for word in _GENERIC_COMPANY_QUERY_WORDS)
            ):
                continue
            candidates.append((start, end, raw, self.identity_key(raw), 1))

        for match in _SHORT_BEFORE_YEAR_RE.finditer(text):
            start, end = match.span()
            raw = match.group(0)
            for prefix in _GENERIC_COMPANY_PREFIXES:
                if raw.startswith(prefix):
                    start += len(prefix)
                    raw = raw[len(prefix):]
                    break
            # "X年相比/较/比Y年": the connector (optionally with the prior 年)
            # is period glue, not a company mention candidate.
            if _PERIOD_COMPARE_CONNECTOR_MENTION_RE.fullmatch(raw) and _YEAR_THEN_PERIOD_COMPARE_RE.search(
                text[:end]
            ):
                continue
            entity_key = self.identity_key(raw)
            if (
                len(raw) >= 2
                and "公司" not in raw
                and raw not in _NON_ENTITY_TERMS
                and not any(word in raw for word in _NON_ENTITY_WORDS)
                and not any(word in raw for word in _GENERIC_COMPANY_QUERY_WORDS)
            ):
                candidates.append((start, end, raw, entity_key, 1))
        for match in _SHORT_AFTER_YEAR_RE.finditer(text):
            raw = match.group(1)
            start, end = match.start(1), match.end(1)
            entity_key = self.identity_key(raw)
            if (
                len(raw) >= 2
                and not entity_key.startswith("raw:")
                and raw not in _NON_ENTITY_TERMS
                and not any(word in raw for word in _NON_ENTITY_WORDS)
                and not any(word in raw for word in _GENERIC_COMPANY_QUERY_WORDS)
            ):
                candidates.append((start, end, raw, entity_key, 1))

        candidates.sort(key=lambda row: (row[4], row[0], -(row[1] - row[0]), row[2], row[3]))
        occupied: list[tuple[int, int]] = []
        seen_entities: set[str] = set()
        result: list[dict[str, Any]] = []
        for start, end, raw_text, entity_key, _source_priority in candidates:
            if _overlaps((start, end), occupied) or entity_key in seen_entities:
                continue
            result.append({
                "raw_text": raw_text,
                "span": [start, end],
                "hint_source": "question",
                "_entity_key": entity_key,
            })
            occupied.append((start, end))
            seen_entities.add(entity_key)
        return sorted(result, key=lambda row: (row["span"][0], row["span"][1], row["raw_text"]))


class QuestionAnalyzer:
    """Produce a contract-valid QuestionAnalysis without execution knowledge."""

    def __init__(
        self,
        *,
        metric_catalog: MetricCatalog | None = None,
        corpus_manifest_path: str | Path = DEFAULT_COMPANY_INDEX,
    ) -> None:
        self.metric_catalog = metric_catalog or MetricCatalog()
        self.company_catalog = CompanyMentionCatalog(corpus_manifest_path)

    def analyze(self, request: Mapping[str, Any]) -> dict[str, Any]:
        request_obj = validate_qa_request(dict(request))
        original_question = str(request_obj["question"])
        question = _normalize_question(original_question)

        company_mentions = self._companies(question, request_obj)
        year_mentions = self._years(question, request_obj, company_mentions)
        output_period_axis = self._output_period_axis(year_mentions)

        raw_formula_mentions = self._contextual_formulas(
            question,
            self.metric_catalog.find_formula_mentions(question),
        )
        formula_spans = [tuple(row["span"]) for row in raw_formula_mentions]
        metric_mentions = self.metric_catalog.find_metric_mentions(question, formula_spans)
        metric_mentions = self._add_metric_hints(metric_mentions, request_obj)
        metadata_mentions = self.metric_catalog.find_metadata_mentions(question, formula_spans)

        narrative_mode, narrative_span = self._narrative(question)
        if (
            narrative_mode is not None
            and _INTRO_RE.search(question)
            and (metric_mentions or raw_formula_mentions or metadata_mentions)
            and not _NARRATIVE_SUBJECT_RE.search(question)
        ):
            narrative_mode, narrative_span = None, None
        table_match = _TABLE_NEED_RE.search(question)
        if table_match and narrative_mode is None:
            narrative_mode, narrative_span = "summarize", list(table_match.span())
        if narrative_mode is not None and self._is_pure_narrative(question):
            metric_mentions = []
            raw_formula_mentions = []
        intents = self._intents(
            question,
            has_structured=bool(metric_mentions or metadata_mentions),
            has_formula=bool(raw_formula_mentions),
            narrative_mode=narrative_mode,
        )
        formula_mentions = self._formulas(
            raw_formula_mentions,
            year_mentions,
            output_period_axis,
        )
        concerns = self._concerns(
            metric_mentions,
            formula_mentions,
            metadata_mentions,
            question,
            narrative_mode,
            narrative_span,
            request_obj,
        )

        output_entities = [row["raw_text"] for row in company_mentions]
        dynamic_target = self._dynamic_target(question, intents, narrative_mode, bool(metadata_mentions))
        unsupported_markers = self._unsupported_markers(
            question,
            output_entities,
            output_period_axis,
            concerns,
            dynamic_target,
            request_obj,
        )
        ambiguity_findings = self._ambiguities(metric_mentions, concerns)

        public_companies: list[dict[str, Any]] = []
        for ordinal, row in enumerate(company_mentions):
            public_companies.append({
                "raw_text": row["raw_text"],
                "span": row["span"],
                "mention_ordinal": ordinal,
                "hint_source": row["hint_source"],
            })
        for ordinal, row in enumerate(year_mentions):
            row["mention_ordinal"] = ordinal
        for ordinal, row in enumerate(metric_mentions):
            row["mention_ordinal"] = ordinal
        for ordinal, row in enumerate(formula_mentions):
            row["mention_ordinal"] = ordinal

        evidence_kinds: list[str] = []
        if metric_mentions or formula_mentions or metadata_mentions:
            evidence_kinds.append("structured_fact")
        if _TABLE_NEED_RE.search(question):
            evidence_kinds.append("table")
        if narrative_mode is not None and not table_match:
            evidence_kinds.append("narrative")

        expanded_years = {year for mention in year_mentions for year in mention["years"]}
        result = {
            "schema_version": SCHEMA_ANALYSIS,
            "analysis_version": ANALYSIS_VERSION,
            "request_id": request_obj["request_id"],
            "question_sha256": hashlib.sha256(original_question.encode("utf-8")).hexdigest(),
            "normalized_question": question,
            "company_mentions": public_companies,
            "year_mentions": year_mentions,
            "metric_mentions": metric_mentions,
            "formula_mentions": formula_mentions,
            "concerns": concerns,
            "cardinalities": {
                "companies": len(public_companies),
                "years": len(expanded_years),
                "metrics": len(metric_mentions),
                "concerns": len(concerns),
            },
            "intents": intents,
            "narrative_mode": narrative_mode,
            "evidence_kinds": evidence_kinds,
            "output_entity_axis": output_entities,
            "output_period_axis": output_period_axis,
            "dynamic_target_dependency": dynamic_target,
            "unsupported_markers": unsupported_markers,
            "ambiguity_findings": ambiguity_findings,
        }
        return validate_question_analysis(result)

    def _contextual_formulas(
        self,
        question: str,
        mentions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        growth_by_metric: dict[str, Any] = {}
        for formula in self.metric_catalog.formulas:
            if formula.formula_id.endswith("growth_rate.v1") and formula.operands:
                growth_by_metric[formula.operands[0][1]] = formula

        result = list(mentions)
        existing = {(row["span"][0], row["span"][1], row["formula"].formula_id) for row in result}

        def canonical_for_base(raw: str) -> str | None:
            resolved = self.metric_catalog.resolve(raw)
            if resolved["status"] == "unique":
                return resolved["candidates"][0]
            if raw == "资产总额":
                return "总资产"
            return None

        # “营业收入及其增长率” explicitly requests the base fact and its
        # formula.  The synthetic formula span covers only the suffix so the
        # base metric remains a separate concern.
        for match in _POSSESSIVE_GROWTH_RE.finditer(question):
            canonical = canonical_for_base(match.group("base"))
            formula = growth_by_metric.get(canonical or "")
            if not formula:
                continue
            suffix_start = question.find("增长率", match.start(), match.end())
            key = (suffix_start, match.end(), formula.formula_id)
            if key not in existing:
                result.append({"raw_text": question[suffix_start:match.end()], "span": [suffix_start, match.end()], "formula": formula})
                existing.add(key)

        # In the frozen General gold, a coordinated list followed by one
        # 增长率 suffix applies that suffix to every listed metric.
        for match in _SHARED_GROWTH_RE.finditer(question):
            items_text = match.group("items")
            items_start = match.start("items")
            cursor = 0
            for raw in re.split(r"[、，,和及]", items_text):
                relative = items_text.find(raw, cursor)
                cursor = relative + len(raw)
                canonical = canonical_for_base(raw)
                formula = growth_by_metric.get(canonical or "")
                if not formula:
                    continue
                start, end = items_start + relative, items_start + relative + len(raw)
                # The final item is already covered by an exact catalog alias;
                # add only missing list members.
                if any(row["formula"].formula_id == formula.formula_id and _overlaps((start, end), [tuple(row["span"])]) for row in result):
                    continue
                key = (start, end, formula.formula_id)
                if key not in existing:
                    result.append({"raw_text": raw + "增长率", "span": [start, end], "formula": formula})
                    existing.add(key)
        return sorted(result, key=lambda row: (row["span"][0], row["span"][1], row["formula"].formula_id))

    def _companies(self, question: str, request: Mapping[str, Any]) -> list[dict[str, Any]]:
        # Request hints are scope guards, not text mentions or output axes.
        # ScopeResolver consumes them separately and can therefore detect a
        # conflict with an explicitly named company.
        return self.company_catalog.find(question)

    def _years(
        self,
        question: str,
        request: Mapping[str, Any],
        company_mentions: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        ranges: list[dict[str, Any]] = []
        occupied: list[tuple[int, int]] = []
        # A year on a corpus rank *or aggregate* constrains the annual-report
        # set. Treating aggregate years as an output axis would fan out a
        # period topology and let comparison columns from later reports leak
        # into the corpus query.
        corpus_operation = bool(_RANK_RE.search(question) or _AGGREGATE_RE.search(question))
        rank_or_corpus = bool(
            corpus_operation
            and (not company_mentions or re.search(r"全体|所有公司|公司中", question))
        )
        formula_present = bool(self.metric_catalog.find_formula_mentions(question))
        # When a corpus query explicitly names an annual report and separately
        # names the value year ("2019年报披露的2018年…"), the latter is a metric
        # column, not a second report-set constraint.
        has_explicit_report_year = bool(re.search(
            r"(?<!\d)(?:19|20|21)\d{2}\s*年?(?:度)?(?:报告|年报|报)", question
        ))

        def precedes_named_company_report(year_end: int) -> bool:
            for company in company_mentions:
                start, end = company["span"]
                if start < year_end or question[year_end:start].strip():
                    continue
                if re.match(r"\s*(?:的)?(?:年度报告|年报)", question[end:]):
                    return True
            return False

        has_explicit_report_year = has_explicit_report_year or any(
            precedes_named_company_report(match.end()) for match in _YEAR_RE.finditer(question)
        )

        for match in _YEAR_RANGE_RE.finditer(question):
            first, last = int(match.group(1)), int(match.group(2))
            years = list(range(min(first, last), max(first, last) + 1))
            role = "corpus_year" if rank_or_corpus else "output_period"
            ranges.append({
                "raw_text": match.group(0), "span": list(match.span()), "years": years,
                "role": role, "is_output_axis": role == "output_period",
            })
            occupied.append(match.span())

        for match in _YEAR_RE.finditer(question):
            if _overlaps(match.span(), occupied):
                continue
            year = int(match.group(1))
            suffix = question[match.end():match.end() + 7]
            prefix = question[max(0, match.start() - 8):match.start()]
            if (
                re.match(r"\s*(?:年)?(?:度)?(?:报告|年报)", suffix)
                or (match.group(0).rstrip().endswith("年") and re.match(r"\s*(?:报|度报告)", suffix))
                or (match.group(0).rstrip().endswith("年") and re.match(r"\s*的(?:年度报告|年报)", suffix))
                or precedes_named_company_report(match.end())
            ):
                role, output = "report_year", False
            elif formula_present and (
                re.match(r"\s*(?:作为)?(?:基期|分母|基数)", suffix)
                or re.search(r"(?:上年|基期|分母)(?:为|是|即|\(|（)?\s*$", prefix)
                # "2020年相比/较/比2019年…增长率": the trailing year is the
                # growth baseline, not a second output-period axis.
                or _YEAR_THEN_PERIOD_COMPARE_RE.search(question[: match.start()])
            ):
                role, output = "formula_operand", False
            elif rank_or_corpus and has_explicit_report_year:
                role, output = "metric_year", False
            elif rank_or_corpus:
                role, output = "corpus_year", False
            else:
                role, output = "output_period", True
            ranges.append({
                "raw_text": match.group(0), "span": list(match.span()), "years": [year],
                "role": role, "is_output_axis": output,
            })

        if not formula_present and _COMPARE_RE.search(question):
            explicit_output_years = [
                year for row in ranges if row["is_output_axis"] for year in row["years"]
            ]
            if explicit_output_years:
                current_year = explicit_output_years[-1]
                for relative in re.finditer(r"上一年|上年", question):
                    previous_year = current_year - 1
                    if previous_year not in {year for row in ranges for year in row["years"]}:
                        ranges.append({
                            "raw_text": relative.group(0),
                            "span": list(relative.span()),
                            "years": [previous_year],
                            "role": "output_period",
                            "is_output_axis": True,
                        })

        ranges.sort(key=lambda row: (row["span"][0], row["span"][1]))

        # report_year/metric_years request hints are kept out of analysis for
        # the same reason as company hints.  Resolver applies them after it has
        # captured question-only years and can fail closed on disagreement.
        return ranges

    @staticmethod
    def _output_period_axis(year_mentions: list[dict[str, Any]]) -> list[int]:
        result: list[int] = []
        for mention in year_mentions:
            if not mention["is_output_axis"]:
                continue
            for year in mention["years"]:
                if year not in result:
                    result.append(year)
        return result

    def _add_metric_hints(self, mentions: list[dict[str, Any]], request: Mapping[str, Any]) -> list[dict[str, Any]]:
        rows = list(mentions)
        present = {candidate for row in rows for candidate in row["candidates"] if row["status"] == "unique"}
        for metric in request.get("canonical_metrics", ()):
            if metric in present:
                continue
            resolved = self.metric_catalog.resolve(metric)
            rows.append({
                "raw_text": metric,
                "span": [0, max(1, len(metric))],
                **resolved,
            })
        return rows

    def _formulas(
        self,
        raw_mentions: list[dict[str, Any]],
        year_mentions: list[dict[str, Any]],
        output_periods: list[int],
    ) -> list[dict[str, Any]]:
        report_years = [year for row in year_mentions if row["role"] == "report_year" for year in row["years"]]
        explicit_operand_years = [year for row in year_mentions if row["role"] == "formula_operand" for year in row["years"]]
        target_years = output_periods or list(dict.fromkeys(report_years))
        result: list[dict[str, Any]] = []
        for row in raw_mentions:
            formula = row["formula"]
            result.append({
                "raw_text": row["raw_text"],
                "span": row["span"],
                "mention_ordinal": 0,
                "formula_id": formula.formula_id,
                "canonical_formula": formula.canonical_formula,
                "status": "unique",
                "target_years": target_years,
                "explicit_operand_years": explicit_operand_years,
                "operands": [
                    {"operand_role": role, "canonical_metric": metric, "year_offset": offset}
                    for role, metric, offset in formula.operands
                ],
                "normalized_unit": formula.normalized_unit,
            })
        return result

    @staticmethod
    def _narrative(question: str) -> tuple[str | None, list[int] | None]:
        if (
            "描述" in question
            and re.search(r"详细信息|详细数据|具体数据|具体数值", question)
            and not re.search(r"原因|分析|情况|状况|风险", question)
        ):
            return None, None
        explain = _EXPLAIN_RE.search(question)
        summarize = _SUMMARIZE_RE.search(question)
        introduction = _INTRO_RE.search(question)
        if explain and (not summarize or explain.start() <= summarize.start()):
            return "explain", [explain.start(), explain.end()]
        if summarize:
            return "summarize", [summarize.start(), summarize.end()]
        if introduction:
            return "summarize", [introduction.start(), introduction.end()]
        fallback = _NARRATIVE_FALLBACK_RE.search(question)
        if fallback and not re.search(r"(?:多少|数值|比率|比例|率是|代码|简称)", question):
            return "summarize", [fallback.start(), fallback.end()]
        return None, None

    @staticmethod
    def _is_pure_narrative(question: str) -> bool:
        if not re.search(r"原因|为什么|为何|描述|分析|概述|总结|情况|状况|风险|竞争力|社会责任", question):
            return False
        return not bool(re.search(
            r"多少|数值|是多少|分别为|计算|比较|对比|比率|比例|率为|代码|简称|Top\s*\d+|最高|最低",
            question,
            re.IGNORECASE,
        ))

    @staticmethod
    def _intents(
        question: str,
        *,
        has_structured: bool,
        has_formula: bool,
        narrative_mode: str | None,
    ) -> list[str]:
        found: set[str] = set()
        if has_structured:
            found.add("lookup")
        if _COMPARE_RE.search(question):
            found.add("compare")
        if _RANK_RE.search(question):
            found.add("rank")
        if _AGGREGATE_RE.search(question):
            found.add("aggregate")
        if has_formula or _CALCULATE_RE.search(question):
            found.add("calculate")
            # A formula mention is one calculation concern, not a second base
            # lookup.  Mixed explicit facts retain lookup through has_structured.
        if narrative_mode is not None:
            found.add("narrative")
        if not found:
            found.add("lookup")
        return [intent for intent in _INTENT_ORDER if intent in found]

    def _concerns(
        self,
        metrics: list[dict[str, Any]],
        formulas: list[dict[str, Any]],
        metadata: list[dict[str, Any]],
        question: str,
        narrative_mode: str | None,
        narrative_span: list[int] | None,
        request: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        candidates: list[tuple[int, int, str, str, str | None, str | None, str | None]] = []
        for row in metrics:
            canonical = row["candidates"][0] if row["status"] == "unique" and len(row["candidates"]) == 1 else None
            candidates.append((row["span"][0], row["span"][1], "metric", row["raw_text"], canonical, None, None))
        for row in formulas:
            candidates.append((row["span"][0], row["span"][1], "formula", row["raw_text"], None, row["formula_id"], None))
        for row in metadata:
            candidates.append((row["span"][0], row["span"][1], "metadata", row["raw_text"], None, None, row["metadata_field"]))
        if narrative_mode is not None and narrative_span is not None:
            candidates.append((narrative_span[0], narrative_span[1], "narrative", question[narrative_span[0]:], None, None, None))

        kind_order = {"metric": 0, "formula": 1, "metadata": 2, "narrative": 3}
        candidates.sort(key=lambda row: (row[0], row[1], kind_order[row[2]], row[3]))
        result: list[dict[str, Any]] = []
        for ordinal, (start, end, kind, raw_text, canonical, formula_id, metadata_field) in enumerate(candidates):
            key = canonical or formula_id or metadata_field or narrative_mode or raw_text
            normalized_unit, unit_source = self._concern_unit(
                kind=kind,
                canonical_metric=canonical,
                formula_id=formula_id,
                span=(start, end),
                question=question,
                request=request,
                structured_concern_count=sum(row[2] != "narrative" for row in candidates),
            )
            result.append({
                "concern_id": _stable_concern_id(kind, key, [start, end]),
                "mention_ordinal": ordinal,
                "kind": kind,
                "raw_text": raw_text,
                "canonical_metric": canonical,
                "formula_id": formula_id,
                "metadata_field": metadata_field,
                "normalized_unit": normalized_unit,
                "unit_source": unit_source,
            })
        return result

    def _concern_unit(
        self,
        *,
        kind: str,
        canonical_metric: str | None,
        formula_id: str | None,
        span: tuple[int, int],
        question: str,
        request: Mapping[str, Any],
        structured_concern_count: int,
    ) -> tuple[str | None, str]:
        if kind in {"metadata", "narrative"}:
            return None, "none"
        expected: str | None
        default_source: str
        if kind == "formula" and formula_id:
            formula = self.metric_catalog.formula(formula_id)
            expected = formula.normalized_unit if formula else None
            default_source = "formula" if expected else "none"
        else:
            expected = self.metric_catalog.expected_unit(canonical_metric or "")
            default_source = "catalog" if expected else ("ambiguous" if canonical_metric == "股本" else "none")

        tail = question[span[1]:span[1] + 18]
        tail = re.split(r"[，,。.;；、]", tail, maxsplit=1)[0]
        match = _UNIT_RE.search(tail)
        explicit = self._normalize_unit(match.group(0)) if match else None
        if explicit is not None:
            if expected is None and canonical_metric == "股本":
                return explicit, "question"
            if expected == explicit:
                return explicit, "question"
            return None, "ambiguous"

        request_unit = request.get("normalized_unit")
        if request_unit is not None:
            hinted = self._normalize_unit(str(request_unit)) or str(request_unit)
            if expected is None and canonical_metric == "股本":
                return hinted, "request_hint"
            if expected == hinted:
                return hinted, "request_hint"
            return None, "ambiguous"
        return expected, default_source

    @staticmethod
    def _normalize_unit(raw_unit: str) -> str | None:
        compact = re.sub(r"\s+", "", raw_unit)
        if compact in {"元", "人民币元", "万元", "亿元"}:
            return "元"
        if compact == "元/股":
            return "元/股"
        if compact in {"%", "百分比"}:
            return "ratio"
        if compact in {"股", "人", "ratio"}:
            return compact
        return None

    @staticmethod
    def _dynamic_target(question: str, intents: list[str], narrative_mode: str | None, has_metadata: bool) -> bool:
        if _DYNAMIC_CONDITIONAL_RE.search(question):
            return True
        if "rank" not in intents:
            if re.search(r"只解释(?:胜者|第一名|最高者|最低者)", question):
                return True
            if (
                (narrative_mode or has_metadata)
                and "aggregate" in intents
                and re.search(r"找出|筛选|高于|低于|超过|不低于|不高于", question)
            ):
                return True
            return False
        downstream = bool(narrative_mode or has_metadata or _DOWNSTREAM_RE.search(question) or _DYNAMIC_WINNER_RE.search(question))
        return downstream

    @staticmethod
    def _unsupported_markers(
        question: str,
        entities: list[str],
        periods: list[int],
        concerns: list[dict[str, Any]],
        dynamic_target: bool,
        request: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        markers: list[dict[str, Any]] = []
        if len(entities) > LIMITS["max_companies"]:
            markers.append({"code": "max_companies_exceeded", "actual": len(entities), "limit": LIMITS["max_companies"]})
        if len(periods) > LIMITS["max_years"]:
            markers.append({"code": "max_years_exceeded", "actual": len(periods), "limit": LIMITS["max_years"]})
        if len(entities) > 1 and len(periods) > 1:
            markers.append({"code": "two_dimensional_output_axis", "companies": len(entities), "years": len(periods)})
        if int(request.get("top_k", 1)) > LIMITS["max_evidence_top_k"]:
            markers.append({
                "code": "evidence_top_k_exceeded", "actual": int(request["top_k"]),
                "limit": LIMITS["max_evidence_top_k"],
            })
        if dynamic_target:
            markers.append({"code": "dynamic_subplan_expansion_required"})
        if _TABLE_NEED_RE.search(question):
            markers.append({"code": "table_evidence_unavailable_phase7"})
        if not concerns:
            markers.append({"code": "no_supported_concern"})
        return markers

    @staticmethod
    def _ambiguities(metrics: list[dict[str, Any]], concerns: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for row in metrics:
            if row["status"] != "unique":
                result.append({
                    "kind": "metric",
                    "raw_text": row["raw_text"],
                    "span": row["span"],
                    "status": row["status"],
                    "candidates": row["candidates"],
                })
        for concern in concerns:
            if concern.get("unit_source") == "ambiguous":
                result.append({
                    "kind": "unit",
                    "concern_id": concern["concern_id"],
                    "raw_text": concern["raw_text"],
                    "status": "ambiguous",
                    "normalized_unit": concern.get("normalized_unit"),
                })
        return result


__all__ = ["ANALYSIS_VERSION", "CompanyMentionCatalog", "QuestionAnalyzer"]
