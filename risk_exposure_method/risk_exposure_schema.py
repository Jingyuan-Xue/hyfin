#!/usr/bin/env python3
"""Schema and validation utilities for annual-report risk exposure JSON."""

from __future__ import annotations

import re
from typing import Any


RISK_CATEGORIES: tuple[str, ...] = (
    "市场价格",
    "需求周期",
    "供应链",
    "政策监管",
    "财务流动性",
    "汇率利率",
    "客户信用/集中度",
    "运营安全环保",
    "技术替代",
    "诉讼合规治理",
)

EXPOSURE_SCORES = {1, 2, 3}
TOP_LEVEL_KEYS = {"Company", "RiskExposures", "CanonicalizationHints", "QualityFlag"}
COMPANY_KEYS = {"company_id", "company_name", "report_year", "source_doc_id"}
EXPOSURE_KEYS = {
    "RiskName",
    "Category",
    "Subcategory",
    "ExposureScore",
    "Evidence",
    "Reason",
    "Mitigants",
    "MetricHints",
    "NeedHumanReview",
}
EVIDENCE_ITEM_KEYS = {"EvidenceQuote", "Interpretation"}
CANONICAL_KEYS = {"CanonicalRisk", "Aliases", "MergeReason"}
QUALITY_KEYS = {"Sufficiency", "NeedHumanReview", "Notes"}
NAME_RE = re.compile(r"^[\w\s\u4e00-\u9fff/（）()、，,.-]{2,40}$")
FORBIDDEN_QUOTE_MARKERS = ("...", "……", "[TRUNCATED]", "[表格见下方风险表格区]")


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def compact_for_trace(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").strip())


def to_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not an integer score")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = normalize_text(value)
    if re.fullmatch(r"[123]", text):
        return int(text)
    raise ValueError(f"unsupported integer value: {value!r}")


def _validate_name(value: Any, field: str) -> tuple[bool, str]:
    text = normalize_text(value)
    if not text:
        return False, f"{field} cannot be empty."
    if not NAME_RE.match(text):
        return False, f"{field} has unsupported characters or length."
    return True, ""


def validate_risk_exposure_schema(data: Any, evidence_text: str | None = None) -> tuple[bool, str]:
    """Return (ok, reason) for the risk exposure model output."""

    if not isinstance(data, dict):
        return False, "Risk exposure output must be a JSON object."
    compact_evidence_text = compact_for_trace(evidence_text) if evidence_text is not None else None
    missing_top = sorted(TOP_LEVEL_KEYS - set(data))
    extra_top = sorted(set(data) - TOP_LEVEL_KEYS)
    if missing_top or extra_top:
        return False, f"Top-level keys invalid. missing={missing_top}, extra={extra_top}"

    company = data.get("Company")
    if not isinstance(company, dict):
        return False, "Company must be an object."
    missing_company = sorted(COMPANY_KEYS - set(company))
    extra_company = sorted(set(company) - COMPANY_KEYS)
    if missing_company or extra_company:
        return False, f"Company keys invalid. missing={missing_company}, extra={extra_company}"
    for key in COMPANY_KEYS:
        if not normalize_text(company.get(key)):
            return False, f"Company.{key} cannot be empty."

    exposures = data.get("RiskExposures")
    if not isinstance(exposures, list) or not exposures:
        return False, "RiskExposures must be a non-empty array."
    if len(exposures) > 12:
        return False, "RiskExposures must contain at most 12 items."

    seen_risks: set[str] = set()
    for index, item in enumerate(exposures):
        if not isinstance(item, dict):
            return False, f"RiskExposures[{index}] must be an object."
        missing = sorted(EXPOSURE_KEYS - set(item))
        extra = sorted(set(item) - EXPOSURE_KEYS)
        if missing or extra:
            return False, f"RiskExposures[{index}] keys invalid. missing={missing}, extra={extra}"

        ok, reason = _validate_name(item.get("RiskName"), f"RiskExposures[{index}].RiskName")
        if not ok:
            return False, reason
        risk_name = normalize_text(item["RiskName"])
        if risk_name in seen_risks:
            return False, f"Duplicated RiskName: {risk_name}"
        seen_risks.add(risk_name)

        category = normalize_text(item.get("Category"))
        if category not in RISK_CATEGORIES:
            return False, f"RiskExposures[{index}].Category must be one of {list(RISK_CATEGORIES)}."
        ok, reason = _validate_name(item.get("Subcategory"), f"RiskExposures[{index}].Subcategory")
        if not ok:
            return False, reason
        try:
            score = to_int(item.get("ExposureScore"))
        except ValueError:
            return False, f"RiskExposures[{index}].ExposureScore must be 1, 2, or 3."
        if score not in EXPOSURE_SCORES:
            return False, f"RiskExposures[{index}].ExposureScore must be 1, 2, or 3."

        evidence_items = item.get("Evidence")
        if not isinstance(evidence_items, list) or not evidence_items or len(evidence_items) > 6:
            return False, f"RiskExposures[{index}].Evidence must contain 1 to 6 items."
        seen_quotes: set[str] = set()
        for evidence_index, evidence_item in enumerate(evidence_items):
            if not isinstance(evidence_item, dict):
                return False, f"RiskExposures[{index}].Evidence[{evidence_index}] must be an object."
            missing = sorted(EVIDENCE_ITEM_KEYS - set(evidence_item))
            extra = sorted(set(evidence_item) - EVIDENCE_ITEM_KEYS)
            if missing or extra:
                return False, (
                    f"RiskExposures[{index}].Evidence[{evidence_index}] keys invalid. "
                    f"missing={missing}, extra={extra}"
                )
            quote = normalize_text(evidence_item.get("EvidenceQuote"))
            interpretation = normalize_text(evidence_item.get("Interpretation"))
            if not quote:
                return False, f"RiskExposures[{index}].Evidence[{evidence_index}].EvidenceQuote cannot be empty."
            if not interpretation:
                return False, f"RiskExposures[{index}].Evidence[{evidence_index}].Interpretation cannot be empty."
            if '"' in quote:
                return False, (
                    f"RiskExposures[{index}].Evidence[{evidence_index}].EvidenceQuote "
                    "must not contain raw ASCII double quotes."
                )
            if any(marker in quote for marker in FORBIDDEN_QUOTE_MARKERS):
                return False, (
                    f"RiskExposures[{index}].Evidence[{evidence_index}].EvidenceQuote "
                    "must not contain ellipsis, truncation, or placeholder markers."
                )
            compact_quote = compact_for_trace(quote)
            if compact_quote in seen_quotes:
                return False, f"RiskExposures[{index}].Evidence contains duplicate EvidenceQuote values."
            seen_quotes.add(compact_quote)
            if compact_evidence_text is not None and compact_quote not in compact_evidence_text:
                return False, (
                    f"RiskExposures[{index}].Evidence[{evidence_index}].EvidenceQuote "
                    "must be an exact quote from the supplied evidence text."
                )

        if not normalize_text(item.get("Reason")):
            return False, f"RiskExposures[{index}].Reason cannot be empty."
        if not isinstance(item.get("Mitigants"), list) or len(item["Mitigants"]) > 6:
            return False, f"RiskExposures[{index}].Mitigants must be an array with <=6 items."
        if not isinstance(item.get("MetricHints"), list) or len(item["MetricHints"]) > 8:
            return False, f"RiskExposures[{index}].MetricHints must be an array with <=8 items."
        if not isinstance(item.get("NeedHumanReview"), bool):
            return False, f"RiskExposures[{index}].NeedHumanReview must be boolean."

    hints = data.get("CanonicalizationHints")
    if not isinstance(hints, list):
        return False, "CanonicalizationHints must be an array."
    seen_canonical: set[str] = set()
    for index, item in enumerate(hints):
        if not isinstance(item, dict):
            return False, f"CanonicalizationHints[{index}] must be an object."
        missing = sorted(CANONICAL_KEYS - set(item))
        extra = sorted(set(item) - CANONICAL_KEYS)
        if missing or extra:
            return False, f"CanonicalizationHints[{index}] keys invalid. missing={missing}, extra={extra}"
        ok, reason = _validate_name(item.get("CanonicalRisk"), f"CanonicalizationHints[{index}].CanonicalRisk")
        if not ok:
            return False, reason
        canonical = normalize_text(item["CanonicalRisk"])
        if canonical in seen_canonical:
            return False, f"Duplicated CanonicalRisk: {canonical}"
        seen_canonical.add(canonical)
        aliases = item.get("Aliases")
        if not isinstance(aliases, list) or len(aliases) > 12:
            return False, f"CanonicalizationHints[{index}].Aliases must be an array with <=12 items."
        normalized_aliases = [normalize_text(alias) for alias in aliases]
        if any(not alias for alias in normalized_aliases):
            return False, f"CanonicalizationHints[{index}].Aliases cannot contain empty values."
        if canonical in normalized_aliases:
            return False, f"CanonicalizationHints[{index}].Aliases must not include CanonicalRisk itself."
        if not normalize_text(item.get("MergeReason")):
            return False, f"CanonicalizationHints[{index}].MergeReason cannot be empty."

    quality = data.get("QualityFlag")
    if not isinstance(quality, dict):
        return False, "QualityFlag must be an object."
    missing_quality = sorted(QUALITY_KEYS - set(quality))
    extra_quality = sorted(set(quality) - QUALITY_KEYS)
    if missing_quality or extra_quality:
        return False, f"QualityFlag keys invalid. missing={missing_quality}, extra={extra_quality}"
    sufficiency = normalize_text(quality.get("Sufficiency")).lower()
    if sufficiency not in {"sufficient", "insufficient"}:
        return False, "QualityFlag.Sufficiency must be sufficient or insufficient."
    if not isinstance(quality.get("NeedHumanReview"), bool):
        return False, "QualityFlag.NeedHumanReview must be boolean."
    if not normalize_text(quality.get("Notes")):
        return False, "QualityFlag.Notes cannot be empty."
    if sufficiency == "insufficient" and quality["NeedHumanReview"] is not True:
        return False, "QualityFlag.NeedHumanReview must be true when evidence is insufficient."

    return True, ""


def risk_categories_markdown() -> str:
    return "\n".join(f"- {category}" for category in RISK_CATEGORIES)
