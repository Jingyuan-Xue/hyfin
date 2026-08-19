#!/usr/bin/env python3
"""Retrieve Phase 7 type3 evidence with mandatory company-year prefiltering.

Run retrieval with the reused A2RAG environment:

    refs/a2rag_runtime/.venv/bin/python scripts/query_type3_evidence.py \
      --company 飞亚达 --year 2019 \
      --question '飞亚达2019年主要经营风险是什么？'

The script is an evidence/fact preparation interface, not an answer generator.
It intentionally does not implement the Phase 8 router or LLM response path.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_QUERY_RESULT = "finglmqa.phase7.type3_evidence_result.v1"
RETRIEVER_VERSION = "phase7-type3-evidence-retriever-v3"
WORKER_PROTOCOL_VERSION = "finglmqa.phase7.a2rag_worker.v1"
A2RAG_LABEL = "qwen3.6-27b-local_BAAI_bge-m3"
MAX_TOP_K = 5
MIN_RERANK_POOL = 40
RERANK_POOL_MULTIPLIER = 10
NUMERIC_INTENT_RE = re.compile(r"多少|金额|数值|数据|增长率|比例|比率|收益率|同比|环比|收入|利润|资产|股本|现金流|每股")
CALCULATION_INTENT_RE = re.compile(r"增长率|增幅|同比|环比|变化率|增长了多少|下降了多少")
YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*年")
REPORT_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*年(?:度报告|报)")
QUERY_ONLY_ALIASES = {
    "归属于上市公司股东的净利润": ("归母净利润",),
}

# These rules deliberately operate only on the question and immutable heading
# metadata.  They are a small retrieval policy, not a classifier or an answer
# generator.  Aliases are grouped by one annual-report concern so a question
# wording can match the conventional heading wording without rewarding an
# incidental mention in chunk content.
HEADING_TOPIC_GROUPS: tuple[tuple[str, ...], ...] = (
    ("破产重整", "破产", "重整"),
    ("重大资产和股权出售", "出售重大资产", "重大资产出售", "股权出售", "出售股权", "资产出售"),
    ("核心竞争力", "核心优势", "竞争优势", "竞争力"),
    ("风险因素", "经营风险", "主要风险", "风险及对策", "风险"),
    ("研发投入", "研发情况", "技术研发", "研发"),
    ("员工情况", "员工数量", "人员构成", "员工"),
    ("公司治理", "治理情况", "治理"),
    ("主要业务", "公司业务", "业务概要", "业务"),
    ("行业情况", "行业发展", "行业地位", "行业"),
    ("社会责任", "环境保护", "环保", "社会责任"),
    ("客户集中度", "主要客户", "客户"),
    ("供应商集中度", "主要供应商", "供应商"),
    ("重大诉讼仲裁事项", "重大诉讼", "重大仲裁", "诉讼仲裁", "诉讼", "仲裁"),
    ("暂停上市和终止上市", "终止上市", "暂停上市", "面临退市", "退市风险警示", "退市"),
    ("处罚及整改", "行政处罚", "监管处罚", "处罚", "整改"),
    ("重大合同", "其他重大合同", "重要合同", "合同履行"),
    ("重大关联交易", "关联交易", "关联方交易", "日常关联交易", "关联方"),
    ("关键审计事项", "审计关键事项"),
    ("资产及负债状况分析", "资产及负债状况", "资产负债状况", "资产构成重大变动", "负债构成重大变动"),
    # Keep the three cash-flow activities in separate groups.  A heading about
    # operating cash flow must never receive the investment or financing bonus
    # merely because all three share ``活动产生的现金流量净额``.
    ("经营活动产生的现金流量净额", "经营活动现金流量净额", "经营活动现金流", "经营活动"),
    ("投资活动产生的现金流量净额", "投资活动现金流量净额", "投资活动现金流", "投资活动"),
    ("筹资活动产生的现金流量净额", "筹资活动现金流量净额", "筹资活动现金流", "筹资活动"),
)
CHECKBOX_ONLY_RE = re.compile(
    r"^[\s　]*(?:[□☐○◯■☑√✓✔×✕☒]\s*)?"
    r"适用\s*(?:[□☐○◯■☑√✓✔×✕☒]\s*)?"
    r"不适用[\s。．.]*$"
)
UNIT_ONLY_RE = re.compile(
    r"^[\s　]*(?:行业和产品分类\s*)?"
    r"单位\s*[：:]?\s*(?:人民币)?\s*(?:元|万元|亿元|股|人|%|％)"
    r"[\s。．.]*$"
)
TRIVIAL_TEMPLATE_VALUES = frozenset({"不适用", "无", "否", "未发生", "无此情况", "报告期内无此情况"})


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} must contain a JSON object")
            rows.append(row)
    return rows


def normalize_alias(value: str) -> str:
    return re.sub(r"\s+", "", str(value)).upper()


def _normalized_heading_text(value: str) -> str:
    """Normalize punctuation that varies across annual-report headings."""
    return re.sub(r"[\s\W_]+", "", str(value), flags=re.UNICODE).upper()


def _heading_values(evidence: dict[str, Any]) -> list[str]:
    """Read both the original A2RAG heading path and aligned section path."""
    values: list[str] = []
    metadata = evidence.get("a2rag_metadata")
    paths = []
    if isinstance(metadata, dict):
        paths.append(metadata.get("heading_path"))
    paths.append(evidence.get("section_path"))
    for path in paths:
        if not isinstance(path, list):
            continue
        for value in path:
            if isinstance(value, str) and value.strip() and value not in values:
                values.append(value)
    return values


def _content_penalty(evidence: dict[str, Any], headings: list[str]) -> float:
    """Return a deterministic penalty for low-information annual-report text."""
    content = str(evidence.get("content") or "").strip()
    compact = re.sub(r"[\s\u3000]+", "", content)
    leaf = _normalized_heading_text(headings[-1]) if headings else ""

    # Standalone applicability controls and one-word placeholders carry no
    # evidence beyond a heading.  A checkbox followed by an explanatory
    # sentence is intentionally *not* penalized.
    if CHECKBOX_ONLY_RE.fullmatch(content) or compact in TRIVIAL_TEMPLATE_VALUES:
        return 0.48
    if UNIT_ONLY_RE.fullmatch(content):
        return 0.50

    # A table of contents can look lexically relevant to almost every question.
    # Require several section markers so ordinary prose mentioning one section
    # is not mistaken for the actual contents page.
    section_marker_count = len(re.findall(r"第[0-9一二三四五六七八九十百]+节", content))
    if leaf == "目录" or section_marker_count >= 4:
        return 0.45
    if leaf == "释义":
        return 0.36
    if "目录" in leaf and "释义" in leaf:
        return 0.16
    return 0.0


def heading_rerank_adjustment(question: str, evidence: dict[str, Any]) -> float:
    """Score immutable heading-topic agreement and low-information penalties.

    The return value is deliberately bounded and added to the BGE-M3 cosine
    score only after dense top-candidate recall.  Content never earns a topic
    bonus, which prevents an incidental phrase in an unrelated section from
    outranking the section whose heading directly names the user's concern.
    """
    normalized_question = _normalized_heading_text(question)
    headings = _heading_values(evidence)
    normalized_headings = [_normalized_heading_text(value) for value in headings]
    heading_bonus = 0.0
    matched_groups = 0
    for group in HEADING_TOPIC_GROUPS:
        aliases = [_normalized_heading_text(value) for value in group]
        if not any(alias and alias in normalized_question for alias in aliases):
            continue
        if any(alias and alias in heading for alias in aliases for heading in normalized_headings):
            matched_groups += 1
    if matched_groups:
        heading_bonus = min(0.30, 0.24 + (matched_groups - 1) * 0.03)
    penalty = _content_penalty(evidence, headings)
    return round(heading_bonus - penalty, 8)


def rerank_dense_candidates(
    question: str,
    candidates: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Rerank a dense-recalled pool with an explicit stable total order.

    Each input row must contain ``chunk_id``, ``document_chunk_ordinal``,
    ``dense_score`` and ``evidence``.  The returned rows additionally contain
    the fixed eight-place decimal ``score`` used by the worker protocol.
    """
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        adjusted = float(candidate["dense_score"]) + heading_rerank_adjustment(
            question, candidate["evidence"]
        )
        ranked.append({**candidate, "score": format(adjusted, ".8f")})
    ranked.sort(
        key=lambda row: (
            -float(row["score"]),
            int(row["document_chunk_ordinal"]),
            str(row["chunk_id"]),
        )
    )
    count = min(max(int(top_k), 1), MAX_TOP_K, len(ranked))
    return ranked[:count]


def portable_path(value: str | Path, root: Path) -> str:
    path = Path(str(value))
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        markdown_ref = root / "refs/source_markdown" / path.name
        if markdown_ref.exists():
            return markdown_ref.relative_to(root).as_posix()
        return f"external:{path.name}"


def resolve_artifact_path(value: str | Path, root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def resolve_device(requested: str) -> str:
    normalized = requested.strip().lower()
    if normalized not in {"auto", "cpu", "cuda"} and not normalized.startswith("cuda:"):
        raise ValueError(f"Unsupported device {requested!r}; use auto, cpu, cuda, or cuda:N")
    if normalized != "auto":
        return normalized
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def embedding_cache_dir(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.expanduser()
    configured = os.environ.get("FINGLMQA_EMBEDDING_CACHE") or os.environ.get("A2RAG_EMBEDDING_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    # The demo workspace keeps the shared Hugging Face cache next to the
    # FinGLMQA checkout. Prefer it when the required model is present so the
    # local command and Phase 10 worker work offline without shell setup.
    workspace_cache = workspace_root().parent / "models"
    if (workspace_cache / "models--BAAI--bge-m3").is_dir():
        return workspace_cache
    return Path.home() / ".cache/huggingface"


class CompanyYearResolver:
    """Exact, fail-closed resolver over the Phase 2 audited index."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def resolve(self, company: str, report_year: int) -> dict[str, Any]:
        needle = normalize_alias(company)
        candidates: list[dict[str, Any]] = []
        for row in self.rows:
            if int(row.get("report_year") or 0) != int(report_year):
                continue
            aliases = row.get("aliases") or []
            if needle in {normalize_alias(value) for value in aliases}:
                candidates.append(row)

        if not candidates:
            return {
                "status": "missing",
                "company_query": company,
                "report_year": int(report_year),
                "candidate_count": 0,
                "candidates": [],
            }
        if len(candidates) > 1:
            return {
                "status": "ambiguous",
                "company_query": company,
                "report_year": int(report_year),
                "candidate_count": len(candidates),
                "candidates": [row.get("document_id") for row in candidates],
            }
        row = candidates[0]
        return {
            "status": "unique",
            "company_query": company,
            "report_year": int(report_year),
            "candidate_count": 1,
            "document_id": row["document_id"],
            "stock_code": row.get("stock_code"),
            "stock_symbol": row.get("stock_symbol"),
            "stock_name": row.get("stock_name"),
            "company_full": row.get("company_full"),
            "source_markdown": row.get("markdown_path"),
        }


def recognize_metrics(question: str, metric_config: dict[str, Any]) -> list[str]:
    """Recognize explicit aliases with longest-match and local exclusions.

    An exclusion only rejects an alias occurrence it actually overlaps. Thus a
    later clause containing ``扣非`` cannot suppress an earlier explicit
    ``归母净利润`` mention.
    """
    normalized_question = normalize_alias(question)
    mentions: list[tuple[int, int, str, str]] = []
    for metric in metric_config.get("metrics") or []:
        canonical = str(metric["canonical_metric"])
        excluded = [normalize_alias(value) for value in metric.get("exclude_patterns") or []]
        aliases = [canonical, *(metric.get("aliases") or []), *QUERY_ONLY_ALIASES.get(canonical, ())]
        for alias in aliases:
            normalized_alias = normalize_alias(alias)
            if not normalized_alias:
                continue
            start = 0
            while True:
                position = normalized_question.find(normalized_alias, start)
                if position < 0:
                    break
                end = position + len(normalized_alias)
                locally_excluded = False
                for excluded_value in excluded:
                    excluded_start = 0
                    while excluded_value:
                        excluded_position = normalized_question.find(excluded_value, excluded_start)
                        if excluded_position < 0:
                            break
                        excluded_end = excluded_position + len(excluded_value)
                        if excluded_position <= position and end <= excluded_end:
                            locally_excluded = True
                            break
                        excluded_start = excluded_position + 1
                    if locally_excluded:
                        break
                if not locally_excluded:
                    mentions.append((position, end, canonical, normalized_alias))
                start = position + 1

    mentions.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
    kept: list[tuple[int, int, str, str]] = []
    for mention in mentions:
        start, end, canonical, _ = mention
        if any(existing[0] <= start and end <= existing[1] and existing[2] != canonical for existing in kept):
            continue
        kept.append(mention)
    ordered: list[str] = []
    for _, _, canonical, _ in kept:
        if canonical not in ordered:
            ordered.append(canonical)
    return ordered


def resolve_question_years(question: str, requested_report_year: int) -> dict[str, Any]:
    report_mentions = [int(match.group(1)) for match in REPORT_YEAR_RE.finditer(question)]
    unique_report_mentions = sorted(set(report_mentions))
    if len(unique_report_mentions) > 1:
        return {
            "status": "report_year_ambiguous",
            "report_year": int(requested_report_year),
            "metric_years": [],
            "report_year_mentions": unique_report_mentions,
        }
    if unique_report_mentions and unique_report_mentions[0] != int(requested_report_year):
        return {
            "status": "report_year_mismatch",
            "report_year": int(requested_report_year),
            "metric_years": [],
            "report_year_mentions": unique_report_mentions,
        }

    report_spans = [match.span() for match in REPORT_YEAR_RE.finditer(question)]
    metric_mentions: list[int] = []
    for match in YEAR_RE.finditer(question):
        if any(start <= match.start() and match.end() <= end for start, end in report_spans):
            continue
        metric_mentions.append(int(match.group(1)))
    metric_years = sorted(set(metric_mentions))
    if len(metric_years) > 1:
        return {
            "status": "metric_year_ambiguous",
            "report_year": int(requested_report_year),
            "metric_years": metric_years,
            "report_year_mentions": unique_report_mentions,
        }
    if not metric_years:
        metric_years = [int(requested_report_year)]
        resolution = "default_report_year"
    else:
        resolution = "explicit_metric_year"
    return {
        "status": "ok",
        "report_year": int(requested_report_year),
        "metric_years": metric_years,
        "metric_year": metric_years[0],
        "year_resolution": resolution,
        "report_year_mentions": unique_report_mentions,
    }


def requested_normalized_unit(question: str) -> dict[str, Any]:
    if re.search(r"单位\s*[：:为]?\s*亿元|多少亿元|以亿元", question):
        return {"status": "unit_conversion_required", "requested_unit": "亿元"}
    if re.search(r"单位\s*[：:为]?\s*万元|多少万元|以万元", question):
        return {"status": "unit_conversion_required", "requested_unit": "万元"}
    if re.search(r"单位\s*[：:为]?\s*(?:人民币)?元|多少元|以元", question):
        return {"status": "ok", "requested_unit": "元"}
    if re.search(r"单位\s*[：:为]?\s*股|多少股|股本[^，。；？?]{0,8}(?:股|股份)$", question):
        return {"status": "ok", "requested_unit": "股"}
    if re.search(r"%|百分比|百分点|单位\s*[：:为]?\s*比率", question):
        return {"status": "ok", "requested_unit": "ratio"}
    return {"status": "unspecified", "requested_unit": None}


def apply_fact_ambiguity_policy(facts: list[dict[str, Any]], question: str) -> dict[str, Any]:
    unit_request = requested_normalized_unit(question)
    if unit_request["status"] == "unit_conversion_required":
        return {**unit_request, "facts": []}
    requested_unit = unit_request["requested_unit"]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for fact in facts:
        key = (str(fact["canonical_metric"]), int(fact["metric_year"]))
        grouped.setdefault(key, []).append(fact)
    selected: list[dict[str, Any]] = []
    for (metric, metric_year), candidates in sorted(grouped.items()):
        units = sorted({str(fact["normalized_unit"]) for fact in candidates})
        if len(units) > 1:
            if requested_unit is None:
                return {
                    "status": "unit_ambiguous",
                    "ambiguous_metric": metric,
                    "metric_year": metric_year,
                    "available_units": units,
                    "facts": [],
                }
            candidates = [fact for fact in candidates if fact["normalized_unit"] == requested_unit]
            if not candidates:
                return {
                    "status": "unit_not_available",
                    "ambiguous_metric": metric,
                    "metric_year": metric_year,
                    "requested_unit": requested_unit,
                    "available_units": units,
                    "facts": [],
                }
        elif requested_unit is not None and units and units[0] != requested_unit:
            return {
                "status": "unit_not_available",
                "ambiguous_metric": metric,
                "metric_year": metric_year,
                "requested_unit": requested_unit,
                "available_units": units,
                "facts": [],
            }
        meanings = {(fact["normalized_value"], fact["normalized_unit"]) for fact in candidates}
        if len(meanings) != 1:
            return {
                "status": "fact_semantics_ambiguous",
                "ambiguous_metric": metric,
                "metric_year": metric_year,
                "candidate_value_units": sorted([list(value) for value in meanings]),
                "facts": [],
            }
        selected.extend(candidates)
    return {
        "status": "ok",
        "requested_unit": requested_unit,
        "facts": selected,
    }


def _fact_helper(payload: dict[str, Any]) -> dict[str, Any]:
    """Executed only by the FinGLMQA Python that owns the duckdb package."""
    import duckdb

    database = Path(str(payload["database"]))
    connection = duckdb.connect(database.as_posix(), read_only=True)
    try:
        metrics = list(payload.get("metrics") or [])
        if not metrics:
            return {"duckdb_version": duckdb.__version__, "facts": [], "connection_mode": "read_only"}
        placeholders = ",".join("?" for _ in metrics)
        query = f"""
            SELECT
                fact_id, document_id, report_year, metric_year, canonical_metric,
                normalized_value_text, normalized_unit, confidence_score,
                statement, source_table_id, source_markdown,
                source_line_start, source_line_end
            FROM selected_financial_facts
            WHERE document_id = ?
              AND metric_year = ?
              AND canonical_metric IN ({placeholders})
            ORDER BY canonical_metric, normalized_value_text, fact_id
        """
        params = [payload["document_id"], int(payload["metric_year"]), *metrics]
        columns = [description[0] for description in connection.execute(query, params).description]
        raw_rows = [dict(zip(columns, row)) for row in connection.fetchall()]
    finally:
        connection.close()

    deduplicated: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in raw_rows:
        key = (row["canonical_metric"], row["normalized_value_text"], row["normalized_unit"])
        target = deduplicated.setdefault(key, {
            "canonical_metric": row["canonical_metric"],
            "metric_year": row["metric_year"],
            "normalized_value": row["normalized_value_text"],
            "normalized_unit": row["normalized_unit"],
            "fact_ids": [],
            "confidence_scores": [],
            "provenance": [],
        })
        target["fact_ids"].append(row["fact_id"])
        target["confidence_scores"].append(str(row["confidence_score"]))
        target["provenance"].append({
            "statement": row["statement"],
            "source_table_id": row["source_table_id"],
            "source_markdown": row["source_markdown"],
            "line_range": [row["source_line_start"], row["source_line_end"]],
        })
    return {
        "duckdb_version": duckdb.__version__,
        "connection_mode": "read_only",
        "source_view": "selected_financial_facts",
        "facts": list(deduplicated.values()),
    }


def query_selected_facts(
    database: Path,
    document_id: str,
    metric_year: int,
    metrics: list[str],
    root: Path,
) -> dict[str, Any]:
    payload = {
        "database": database.as_posix(),
        "document_id": document_id,
        "metric_year": int(metric_year),
        "metrics": metrics,
    }
    command = [
        (root / ".venv/bin/python").as_posix(),
        Path(__file__).resolve().as_posix(),
        "--fact-helper",
    ]
    result = subprocess.run(
        command,
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Read-only fact helper failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


class Type3EvidenceRetriever:
    """A2RAG DPR adapter that applies the resolver before vector scoring."""

    def __init__(
        self,
        root: Path | None = None,
        device: str = "auto",
        model_cache: Path | None = None,
        load_dense: bool = True,
    ) -> None:
        self.root = root or workspace_root()
        self.index_dir = self.root / "data/indexes/a2rag_index"
        self.index_manifest = load_json(self.index_dir / "index_manifest.json")
        self.resolver_rows = load_jsonl(self.root / "data/corpus_package/company_year_index.jsonl")
        self.resolver = CompanyYearResolver(self.resolver_rows)
        self.document_rows = load_jsonl(self.index_dir / "document_chunk_map.jsonl")
        self.document_map = {row["document_id"]: row for row in self.document_rows}
        evidence_path = resolve_artifact_path(self.index_manifest["artifacts"]["evidence_chunks"], self.root)
        self.evidence_rows = load_jsonl(evidence_path)
        self.evidence_by_id = {row["a2rag_chunk_id"]: row for row in self.evidence_rows}
        self.metric_config = load_json(self.root / "src/config/metric_aliases.json")
        self.fact_database = self.root / "data/facts/financial_facts.duckdb"
        self.device = resolve_device(device)
        self.model_cache = embedding_cache_dir(model_cache)
        self._dense_frame = None
        self._dense_by_id = None
        self._embedding_model = None
        if load_dense:
            self._load_dense_vectors()

    def _load_dense_vectors(self) -> None:
        if self._dense_frame is not None:
            return
        import pandas as pd

        parquet_path = resolve_artifact_path(
            self.index_manifest["artifacts"]["runtime_dense_parquet_symlink"], self.root
        )
        frame = pd.read_parquet(parquet_path, columns=["hash_id", "embedding"])
        self._dense_frame = frame
        self._dense_by_id = dict(zip(frame["hash_id"], frame["embedding"]))

    def _load_embedding_model(self):
        if self._embedding_model is not None:
            return self._embedding_model
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["A2RAG_EMBEDDING_PROVIDER"] = "local"
        os.environ["A2RAG_LOCAL_EMBEDDING"] = "true"
        os.environ["A2RAG_EMBEDDING_CACHE_DIR"] = self.model_cache.as_posix()
        os.environ["A2RAG_EMBEDDING_DEVICE"] = self.device

        from a2rag.config import BaseConfig
        from a2rag.providers.local_embedding import LocalSentenceTransformerEmbedding

        config = BaseConfig(
            embedding_provider="local",
            embedding_model="BAAI/bge-m3",
            embedding_device=self.device,
            embedding_cache_dir=self.model_cache.as_posix(),
            embedding_batch_size=16,
            embedding_return_as_normalized=True,
        )
        self._embedding_model = LocalSentenceTransformerEmbedding(
            global_config=config,
            embedding_model="BAAI/bge-m3",
        )
        return self._embedding_model

    def _dense_retrieve(self, question: str, document_id: str, top_k: int) -> dict[str, Any]:
        import numpy as np

        self._load_dense_vectors()
        document = self.document_map[document_id]
        candidate_ids = list(document["chunk_ids"])
        missing_vector_ids = [chunk_id for chunk_id in candidate_ids if chunk_id not in self._dense_by_id]
        if missing_vector_ids:
            raise RuntimeError(f"Evidence IDs are missing dense vectors: {missing_vector_ids[:5]}")

        candidate_matrix = np.vstack([self._dense_by_id[chunk_id] for chunk_id in candidate_ids]).astype(np.float32)
        model = self._load_embedding_model()
        query_vector = np.asarray(model.batch_encode([question])[0], dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vector)) or 1.0
        candidate_norms = np.linalg.norm(candidate_matrix, axis=1)
        candidate_norms[candidate_norms == 0] = 1.0
        scores = (candidate_matrix @ query_vector) / (candidate_norms * query_norm)
        count = min(max(int(top_k), 1), MAX_TOP_K, len(candidate_ids))

        # Heading-aware scoring is intentionally limited to a dense-recalled
        # pool.  It cannot introduce a chunk that BGE-M3 did not first place in
        # the document's top candidates, and the company/year allow-list was
        # already applied before the dense matrix was built.
        dense_ranked_indices = sorted(
            range(len(candidate_ids)),
            key=lambda index: (
                -float(scores[index]),
                index + 1,
                candidate_ids[index],
            ),
        )
        pool_size = min(
            len(candidate_ids),
            max(MIN_RERANK_POOL, count * RERANK_POOL_MULTIPLIER),
        )
        dense_candidates = []
        for index in dense_ranked_indices[:pool_size]:
            chunk_id = candidate_ids[index]
            dense_candidates.append({
                "chunk_id": chunk_id,
                "document_chunk_ordinal": index + 1,
                "dense_score": float(scores[index]),
                "evidence": self.evidence_by_id[chunk_id],
            })
        reranked = rerank_dense_candidates(question, dense_candidates, count)

        results: list[dict[str, Any]] = []
        for rank, candidate in enumerate(reranked, start=1):
            chunk_id = candidate["chunk_id"]
            evidence = candidate["evidence"]
            results.append({
                "rank": rank,
                "score": candidate["score"],
                "evidence_chunk_id": chunk_id,
                "document_id": evidence["document_id"],
                "company_name": evidence["company_name"],
                "stock_code": evidence["stock_code"],
                "report_year": evidence["report_year"],
                "section_path": evidence["section_path"],
                "semantic_tags": evidence["semantic_tags"],
                "line_range": evidence["line_range"],
                "source_markdown": evidence["source_markdown"],
                "content": evidence["content"],
            })
        return {
            "retrieval_method": "a2rag_dpr_bge_m3",
            "reranking_method": "deterministic_heading_section_v1",
            "candidate_prefilter": "company_year_resolver_document_allow_list",
            "prefilter_applied_before_scoring": True,
            "candidate_document_id": document_id,
            "candidate_chunk_count": len(candidate_ids),
            "dense_candidate_pool_size": pool_size,
            "top_k": count,
            "chunks": results,
        }

    def retrieve_for_document(self, document_id: str, question: str, top_k: int = 5) -> dict[str, Any]:
        """Public retrieval-only adapter for a caller that already resolved scope."""
        if document_id not in self.document_map:
            raise KeyError(f"Unknown Phase 7 document_id: {document_id}")
        return self._dense_retrieve(question, document_id, top_k)

    def query(self, company: str, report_year: int, question: str, top_k: int = 5) -> dict[str, Any]:
        resolution = self.resolver.resolve(company, report_year)
        result: dict[str, Any] = {
            "schema_version": SCHEMA_QUERY_RESULT,
            "retriever_version": RETRIEVER_VERSION,
            "question": question,
            "resolver": resolution,
            "retrieval": None,
            "fact_injection": None,
        }
        if resolution["status"] != "unique":
            result["status"] = f"resolver_{resolution['status']}"
            result["fact_injection"] = {
                "status": "not_attempted_resolver_not_unique",
                "numeric_source_policy": "phase6_selected_facts_only",
                "allow_text_numeric_answer": False,
                "facts": [],
            }
            return result

        document_id = resolution["document_id"]
        resolution["source_markdown"] = portable_path(resolution["source_markdown"], self.root)
        retrieval = self.retrieve_for_document(document_id, question, top_k)
        if any(chunk["document_id"] != document_id for chunk in retrieval["chunks"]):
            raise RuntimeError("Company-year isolation violation in retrieved chunks")
        result["retrieval"] = retrieval

        metrics = recognize_metrics(question, self.metric_config)
        numeric_intent = bool(metrics or NUMERIC_INTENT_RE.search(question))
        if not numeric_intent:
            injection = {
                "status": "not_numeric",
                "recognized_metrics": [],
                "numeric_source_policy": "phase6_selected_facts_only",
                "allow_text_numeric_answer": False,
                "facts": [],
            }
        elif not metrics:
            injection = {
                "status": "metric_not_recognized",
                "recognized_metrics": [],
                "numeric_source_policy": "phase6_selected_facts_only",
                "allow_text_numeric_answer": False,
                "facts": [],
            }
        elif CALCULATION_INTENT_RE.search(question):
            injection = {
                "status": "formula_required",
                "reason": "calculation_not_supported_by_phase7_fact_injection",
                "recognized_metrics": metrics,
                "numeric_source_policy": "phase6_selected_facts_only",
                "allow_text_numeric_answer": False,
                "facts": [],
            }
        else:
            year_policy = resolve_question_years(question, int(report_year))
            if year_policy["status"] != "ok":
                injection = {
                    "status": year_policy["status"],
                    "recognized_metrics": metrics,
                    "year_policy": year_policy,
                    "numeric_source_policy": "phase6_selected_facts_only",
                    "allow_text_numeric_answer": False,
                    "facts": [],
                }
                result["fact_injection"] = injection
                result["status"] = "ok"
                return result
            fact_result = query_selected_facts(
                self.fact_database,
                document_id=document_id,
                metric_year=int(year_policy["metric_year"]),
                metrics=metrics,
                root=self.root,
            )
            ambiguity = apply_fact_ambiguity_policy(fact_result["facts"], question)
            facts = ambiguity["facts"]
            for fact in facts:
                for provenance in fact.get("provenance") or []:
                    provenance["source_markdown"] = portable_path(provenance["source_markdown"], self.root)
            ambiguity_status = ambiguity["status"]
            injection = {
                "status": (
                    ambiguity_status
                    if ambiguity_status != "ok"
                    else ("selected_facts_injected" if facts else "no_selected_fact")
                ),
                "recognized_metrics": metrics,
                "year_policy": year_policy,
                "unit_policy": {key: value for key, value in ambiguity.items() if key != "facts"},
                "numeric_source_policy": "phase6_selected_facts_only",
                "source_database": portable_path(self.fact_database, self.root),
                "source_view": fact_result.get("source_view", "selected_financial_facts"),
                "connection_mode": fact_result["connection_mode"],
                "duckdb_version": fact_result["duckdb_version"],
                "allow_text_numeric_answer": False,
                "facts": facts,
            }
        result["fact_injection"] = injection
        result["status"] = "ok"
        return result


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(path)


def emit_worker_message(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
    sys.stdout.flush()


def serve_worker(retriever: Type3EvidenceRetriever) -> int:
    """Serve one request at a time over a flushed JSONL stdin/stdout protocol."""
    emit_worker_message({
        "type": "ready",
        "protocol_version": WORKER_PROTOCOL_VERSION,
        "retriever_version": RETRIEVER_VERSION,
        "device": retriever.device,
        "concurrency": 1,
        "stdout": "jsonl_only",
        "logs": "stderr",
        "commands": ["ping", "query", "shutdown"],
    })
    for raw_line in sys.stdin:
        if not raw_line.strip():
            continue
        request_id: Any = None
        try:
            request = json.loads(raw_line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            request_id = request.get("request_id")
            if request_id is None or str(request_id) == "":
                raise ValueError("request_id is required")
            message_type = request.get("type")
            if message_type == "ping":
                emit_worker_message({
                    "type": "pong",
                    "protocol_version": WORKER_PROTOCOL_VERSION,
                    "request_id": request_id,
                })
                continue
            if message_type == "shutdown":
                emit_worker_message({
                    "type": "shutdown_ack",
                    "protocol_version": WORKER_PROTOCOL_VERSION,
                    "request_id": request_id,
                })
                return 0
            if message_type != "query":
                raise ValueError("type must be one of: ping, query, shutdown")
            with contextlib.redirect_stdout(sys.stderr):
                result = retriever.query(
                    company=str(request["company"]),
                    report_year=int(request["report_year"]),
                    question=str(request["question"]),
                    top_k=int(request.get("top_k", 5)),
                )
            emit_worker_message({
                "type": "result",
                "protocol_version": WORKER_PROTOCOL_VERSION,
                "request_id": request_id,
                "result": result,
            })
        except Exception as exc:
            print(f"worker request failed request_id={request_id!r}: {exc}", file=sys.stderr, flush=True)
            emit_worker_message({
                "type": "error",
                "protocol_version": WORKER_PROTOCOL_VERSION,
                "request_id": request_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
    print("worker stdin closed; exiting cleanly", file=sys.stderr, flush=True)
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--fact-helper":
        payload = json.loads(sys.stdin.read())
        print(json.dumps(_fact_helper(payload), ensure_ascii=False, default=str))
        return 0

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", help="Exact Phase 2 company alias, name, or stock code")
    parser.add_argument("--year", type=int, help="Annual-report year")
    parser.add_argument("--question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default=os.environ.get("FINGLMQA_DEVICE", "auto"))
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument("--serve", action="store_true", help="Warm single-concurrency JSONL worker on stdin/stdout")
    parser.add_argument("--no-dense", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.serve and (not args.company or args.year is None or not args.question):
        parser.error("--company, --year, and --question are required in one-shot mode")
    retriever = Type3EvidenceRetriever(
        device=args.device,
        model_cache=args.model_cache,
        load_dense=not args.no_dense,
    )
    if args.serve:
        return serve_worker(retriever)
    result = retriever.query(args.company, args.year, args.question, top_k=args.top_k)
    if args.output:
        atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["resolver"]["status"] == "unique" else 2


if __name__ == "__main__":
    raise SystemExit(main())
