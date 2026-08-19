"""Read-only API over the validated risk-exposure run artifacts."""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from .risk_exposure_schema import RISK_CATEGORIES, validate_risk_exposure_schema


API_VERSION = "risk-exposure.demo.v1"
DEFAULT_RUN_DIR = (
    Path(__file__).resolve().parent
    / "output"
    / "risk_exposure_sw3_430101_25_strict"
)
COMPANY_ID_RE = re.compile(r"^A\d{6}$")


class ArtifactError(RuntimeError):
    """The configured run is missing or internally inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"JSON artifact must contain an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise ArtifactError(f"Cannot read CSV artifact: {path}") from exc


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_company_id(value: str) -> str:
    normalized = str(value or "").strip().upper()
    if normalized.isdigit() and len(normalized) <= 6:
        normalized = "A" + normalized.zfill(6)
    if not COMPANY_ID_RE.fullmatch(normalized):
        raise HTTPException(status_code=422, detail="company_id must look like A000002 or 000002")
    return normalized


def _normalized_search(value: str) -> str:
    return "".join(str(value or "").casefold().split())


class RiskExposureStore:
    """Validated, immutable projection of one completed pipeline run."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.dataset_dir = self.run_dir / "03_dataset"
        self.stage2_dir = self.run_dir / "02_stage2"
        self.summary = _read_json(self.dataset_dir / "dataset_summary.json")
        self.quote_audit = _read_json(self.dataset_dir / "quote_audit.json")
        self.target_company_ids = self._load_targets()
        self.failed_company_ids = self._load_failures()
        self.matrix = self._load_matrix()
        self.documents = self._load_documents()
        self.risk_factors = self._load_risk_factors()
        self.evaluations = self._load_evaluations()
        self._validate_consistency()

    def _load_targets(self) -> list[str]:
        path = self.run_dir / "target_sources.txt"
        try:
            rows = [Path(line.strip()).stem for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except OSError as exc:
            raise ArtifactError(f"Cannot read target list: {path}") from exc
        return sorted({row.split("_", 1)[0] for row in rows if COMPANY_ID_RE.fullmatch(row.split("_", 1)[0])})

    def _load_failures(self) -> list[str]:
        failed: set[str] = set()
        for name in ("stage2_failed.txt", "stage2_failed_resume.txt"):
            path = self.run_dir / name
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                company_id = line.split("_", 1)[0].strip().upper()
                if COMPANY_ID_RE.fullmatch(company_id):
                    failed.add(company_id)
        return sorted(failed)

    def _load_matrix(self) -> dict[tuple[str, int], dict[str, Any]]:
        rows: dict[tuple[str, int], dict[str, Any]] = {}
        for source in _read_csv(self.dataset_dir / "company_risk_matrix.csv"):
            company_id = str(source.get("company_id") or "").strip().upper()
            year = _integer(source.get("report_year"))
            category_scores = {
                category: _integer(source.get(f"category_{category}"))
                for category in RISK_CATEGORIES
            }
            rows[(company_id, year)] = {
                "company_id": company_id,
                "company_name": str(source.get("company_name") or "").strip(),
                "report_year": year,
                "risk_count": _integer(source.get("risk_count")),
                "max_exposure_score": _integer(source.get("max_exposure_score")),
                "category_scores": category_scores,
            }
        return rows

    def _load_documents(self) -> dict[tuple[str, int], dict[str, Any]]:
        documents: dict[tuple[str, int], dict[str, Any]] = {}
        for path in sorted(self.stage2_dir.rglob("*_risk_exposure.json")):
            data = _read_json(path)
            valid, reason = validate_risk_exposure_schema(data)
            if not valid:
                raise ArtifactError(f"Invalid risk exposure artifact {path}: {reason}")
            company = data["Company"]
            key = (str(company["company_id"]).upper(), _integer(company["report_year"]))
            if key in documents:
                raise ArtifactError(f"Duplicate company-year artifact: {key}")
            documents[key] = data
        if not documents:
            raise ArtifactError(f"No risk exposure artifacts found under {self.stage2_dir}")
        return documents

    def _load_risk_factors(self) -> list[dict[str, Any]]:
        factors: list[dict[str, Any]] = []
        for row in _read_csv(self.dataset_dir / "risk_factor_library.csv"):
            factors.append(
                {
                    "risk_name": row.get("risk_name", ""),
                    "category": row.get("category", ""),
                    "subcategory": row.get("subcategory", ""),
                    "record_count": _integer(row.get("record_count")),
                    "company_count": _integer(row.get("company_count")),
                    "avg_exposure_score": _number(row.get("avg_exposure_score")),
                    "max_exposure_score": _integer(row.get("max_exposure_score")),
                }
            )
        return factors

    def _load_evaluations(self) -> dict[str, dict[str, Any]]:
        evaluations: dict[str, dict[str, Any]] = {}
        for name, folder in (
            ("category", "04_eval"),
            ("hybrid", "04_eval_hybrid"),
            ("risk_name", "04_eval_risk_name"),
        ):
            path = self.run_dir / folder / "summary.json"
            if path.is_file():
                evaluations[name] = _read_json(path)
        return evaluations

    def _validate_consistency(self) -> None:
        expected = _integer(self.summary.get("valid_json_count"), -1)
        if expected != len(self.documents):
            raise ArtifactError(
                f"dataset_summary valid_json_count={expected}, but loaded {len(self.documents)} artifacts"
            )
        if set(self.matrix) != set(self.documents):
            raise ArtifactError("company_risk_matrix and Stage-2 artifacts have different company-year keys")
        matched = _integer(self.quote_audit.get("evidence_matched"), -1)
        total = _integer(self.quote_audit.get("evidence_total"), -1)
        if matched != total:
            raise ArtifactError(f"Evidence audit is incomplete: {matched}/{total}")

    @property
    def coverage(self) -> dict[str, Any]:
        target_count = len(self.target_company_ids)
        available_ids = sorted({key[0] for key in self.documents})
        missing_ids = sorted(set(self.target_company_ids) - set(available_ids))
        return {
            "target_company_count": target_count,
            "available_company_count": len(available_ids),
            "coverage_ratio": round(len(available_ids) / target_count, 4) if target_count else 0.0,
            "missing_company_ids": missing_ids,
            "record_count": _integer(self.summary.get("record_count")),
            "evidence_quote_count": _integer(self.quote_audit.get("evidence_total")),
            "evidence_quote_match_count": _integer(self.quote_audit.get("evidence_matched")),
        }

    def company_summaries(
        self,
        *,
        query: str = "",
        category: str | None = None,
        min_score: int = 0,
    ) -> list[dict[str, Any]]:
        needle = _normalized_search(query)
        rows: list[dict[str, Any]] = []
        for key in sorted(self.matrix):
            summary = self.matrix[key]
            document = self.documents[key]
            exposures = document["RiskExposures"]
            if needle and needle not in _normalized_search(
                summary["company_id"] + summary["company_name"] + str(summary["report_year"])
            ):
                continue
            if category and not any(
                item["Category"] == category and _integer(item["ExposureScore"]) >= min_score
                for item in exposures
            ):
                continue
            if not category and min_score and not any(
                _integer(item["ExposureScore"]) >= min_score for item in exposures
            ):
                continue
            top = sorted(
                exposures,
                key=lambda item: (-_integer(item["ExposureScore"]), item["Category"], item["RiskName"]),
            )[:3]
            rows.append(
                {
                    **summary,
                    "top_risks": [
                        {
                            "risk_name": item["RiskName"],
                            "category": item["Category"],
                            "exposure_score": _integer(item["ExposureScore"]),
                        }
                        for item in top
                    ],
                    "need_human_review": bool(document["QualityFlag"]["NeedHumanReview"]),
                }
            )
        return rows

    def company_detail(self, company_id: str, report_year: int | None = None) -> dict[str, Any]:
        normalized_id = _normalize_company_id(company_id)
        keys = [key for key in self.documents if key[0] == normalized_id]
        if report_year is not None:
            keys = [key for key in keys if key[1] == report_year]
        if not keys:
            raise HTTPException(status_code=404, detail="Risk exposure artifact not found")
        if len(keys) > 1:
            raise HTTPException(status_code=409, detail="Multiple report years found; specify report_year")
        key = keys[0]
        data = self.documents[key]
        evidence_count = sum(len(item["Evidence"]) for item in data["RiskExposures"])
        return {
            "company": data["Company"],
            "summary": self.matrix[key],
            "risk_exposures": data["RiskExposures"],
            "canonicalization_hints": data["CanonicalizationHints"],
            "quality_flag": data["QualityFlag"],
            "evidence_count": evidence_count,
        }


def create_app(run_dir: str | Path | None = None) -> FastAPI:
    configured = run_dir or os.environ.get("RISK_EXPOSURE_RUN_DIR") or DEFAULT_RUN_DIR
    store = RiskExposureStore(configured)
    app = FastAPI(
        title="Risk Exposure Artifact API",
        version="1.0.0",
        description="Read-only, evidence-preserving API for the strict SW3 430101 risk-exposure run.",
    )
    app.state.store = store

    @app.get("/health/ready")
    def health_ready() -> dict[str, Any]:
        return {
            "ready": True,
            "service": "risk-exposure-artifact-api",
            "api_version": API_VERSION,
            "run": store.run_dir.name,
            "coverage": store.coverage,
        }

    @app.get("/api/v1/meta")
    def meta() -> dict[str, Any]:
        return {
            "api_version": API_VERSION,
            "name": "Risk Exposure Method",
            "mode": "artifact-backed-read-only",
            "online_llm": False,
            "run": store.run_dir.name,
            "report_years": sorted({key[1] for key in store.documents}),
            "risk_categories": list(RISK_CATEGORIES),
            "exposure_scores": [1, 2, 3],
            "coverage": store.coverage,
            "evaluation_granularities": sorted(store.evaluations),
        }

    @app.get("/api/v1/companies")
    def companies(
        query: str = Query(default="", max_length=100),
        category: str | None = Query(default=None),
        min_score: int = Query(default=0, ge=0, le=3),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=100),
    ) -> dict[str, Any]:
        if category is not None and category not in RISK_CATEGORIES:
            raise HTTPException(status_code=422, detail="Unknown risk category")
        rows = store.company_summaries(query=query, category=category, min_score=min_score)
        return {
            "api_version": API_VERSION,
            "total": len(rows),
            "offset": offset,
            "limit": limit,
            "companies": rows[offset : offset + limit],
            "coverage": store.coverage,
        }

    @app.get("/api/v1/companies/{company_id}")
    def company_detail(
        company_id: str,
        report_year: int | None = Query(default=None, ge=1900, le=2200),
    ) -> dict[str, Any]:
        return {"api_version": API_VERSION, **store.company_detail(company_id, report_year)}

    @app.get("/api/v1/risk-factors")
    def risk_factors(
        query: str = Query(default="", max_length=100),
        category: str | None = Query(default=None),
        min_company_count: int = Query(default=1, ge=1),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict[str, Any]:
        if category is not None and category not in RISK_CATEGORIES:
            raise HTTPException(status_code=422, detail="Unknown risk category")
        needle = _normalized_search(query)
        rows = [
            row for row in store.risk_factors
            if (not category or row["category"] == category)
            and row["company_count"] >= min_company_count
            and (not needle or needle in _normalized_search(
                row["risk_name"] + row["category"] + row["subcategory"]
            ))
        ]
        return {
            "api_version": API_VERSION,
            "total": len(rows),
            "offset": offset,
            "limit": limit,
            "risk_factors": rows[offset : offset + limit],
        }

    @app.get("/api/v1/evaluation")
    def evaluation() -> dict[str, Any]:
        return {"api_version": API_VERSION, "evaluations": store.evaluations}

    @app.get("/api/v1/compare")
    def compare(company_ids: str = Query(min_length=1, max_length=100)) -> dict[str, Any]:
        requested = []
        for raw in company_ids.split(","):
            company_id = _normalize_company_id(raw)
            if company_id not in requested:
                requested.append(company_id)
        if not 2 <= len(requested) <= 8:
            raise HTTPException(status_code=422, detail="Compare between 2 and 8 unique companies")
        details = [store.company_detail(company_id) for company_id in requested]
        category_presence: Counter[str] = Counter()
        for detail in details:
            category_presence.update(
                category for category, score in detail["summary"]["category_scores"].items() if score > 0
            )
        shared_categories = [
            category for category in RISK_CATEGORIES if category_presence[category] == len(details)
        ]
        return {
            "api_version": API_VERSION,
            "company_ids": requested,
            "companies": [
                {
                    "company": detail["company"],
                    "risk_count": detail["summary"]["risk_count"],
                    "max_exposure_score": detail["summary"]["max_exposure_score"],
                    "category_scores": detail["summary"]["category_scores"],
                }
                for detail in details
            ],
            "shared_categories": shared_categories,
        }

    return app


app = create_app()
