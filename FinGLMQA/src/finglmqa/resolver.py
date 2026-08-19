"""Deterministic company and report scope resolution for Phase 8.

The resolver deliberately performs exact alias resolution.  It never consults
benchmark selections and never guesses a company from a substring.  A uniquely
resolved company can still carry a document-level ambiguity finding; this keeps
the company identity available to structured sibling plans while requiring an
evidence plan to fail closed.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import SCHEMA_SCOPE, semantic_sha256, validate_question_analysis, validate_scope_plan


RESOLVER_VERSION = "phase8-scope-resolver-v1.0.0"

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INDEX = _ROOT / "data/corpus_package/company_year_index.jsonl"
_DEFAULT_MANIFEST = _ROOT / "data/corpus_package/corpus_manifest.json"
_DEFAULT_COMPANY_ALIASES = _ROOT / "src/config/company_aliases.json"
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

_CAPABILITIES: dict[str, dict[str, str]] = {
    "single_document": {
        "fact": "direct",
        "sql": "direct",
        "formula": "direct",
        "evidence": "direct",
    },
    "company_documents": {
        "fact": "direct",
        "sql": "direct",
        "formula": "single_document_subscope",
        "evidence": "single_document_subscope",
    },
    "multi_company_documents": {
        "fact": "entity_subscope",
        "sql": "entity_subscope",
        "formula": "single_document_subscope",
        "evidence": "single_document_subscope",
    },
    "explicit_document_set": {
        "fact": "contract_only",
        "sql": "contract_only",
        "formula": "contract_only",
        "evidence": "contract_only",
    },
    "corpus": {
        "fact": "forbidden",
        "sql": "direct",
        "formula": "forbidden",
        "evidence": "forbidden",
    },
}


def _normalize_alias(value: str) -> str:
    """Return an exact-match key without enabling substring matching."""

    text = unicodedata.normalize("NFKC", value).strip().upper()
    # Spaces and common separators are presentation differences, not fuzzy
    # similarity. Chinese company-name characters are otherwise preserved.
    return re.sub(r"[\s·・._\-—－]+", "", text)


def _ordered_unique(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        number = int(value)
        if number not in seen:
            seen.add(number)
            result.append(number)
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _synthetic_fingerprint(row: Mapping[str, Any]) -> str:
    value = row.get("artifact_fingerprint") or row.get("content_sha256")
    if isinstance(value, str) and _HEX64_RE.fullmatch(value):
        return value
    stable = {
        "document_id": row.get("document_id"),
        "stock_code": row.get("stock_code"),
        "report_year": int(row.get("report_year", 0)),
    }
    return semantic_sha256(stable)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    return rows


class ScopeResolver:
    """Resolve an analysis into the frozen :class:`ScopePlan` dictionary.

    ``records`` is an intentional test/fixture seam for multi-report companies,
    alias collisions, and conflict states that are absent from the current
    one-report-per-company production snapshot.  When supplied, no production
    index rows are mixed into the fixture.
    """

    def __init__(
        self,
        *,
        index_path: str | Path = _DEFAULT_INDEX,
        manifest_path: str | Path = _DEFAULT_MANIFEST,
        aliases_path: str | Path = _DEFAULT_COMPANY_ALIASES,
        records: Sequence[Mapping[str, Any]] | None = None,
        resolver_version: str = RESOLVER_VERSION,
    ) -> None:
        self.resolver_version = resolver_version
        self._index_path = Path(index_path)
        self._manifest_path = Path(manifest_path)

        if records is None:
            raw_rows = _read_jsonl(self._index_path)
            fingerprints = self._manifest_fingerprints(self._manifest_path)
            self._source_fingerprint = semantic_sha256(
                {
                    "company_year_index_sha256": _file_sha256(self._index_path),
                    "corpus_manifest_sha256": _file_sha256(self._manifest_path),
                    "company_aliases_sha256": _file_sha256(Path(aliases_path)),
                }
            )
        else:
            raw_rows = [dict(row) for row in records]
            fingerprints = {}
            self._source_fingerprint = semantic_sha256(raw_rows)

        self._rows = tuple(self._canonical_row(row, fingerprints) for row in raw_rows)
        if records is None:
            aliases = json.loads(Path(aliases_path).read_text(encoding="utf-8"))
            if aliases.get("schema_version") != "finglmqa.phase8.company_aliases.v1":
                raise ValueError("Phase 8 company alias catalog schema mismatch")
            by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in self._rows:
                by_code[row["stock_code"]].append(row)
            for item in aliases.get("aliases", []):
                stock_code = str(item.get("stock_code") or "")
                alias = str(item.get("alias") or "").strip()
                if not alias or stock_code not in by_code:
                    raise ValueError("Phase 8 company alias catalog contains an unknown identity")
                for row in by_code[stock_code]:
                    if alias not in row["aliases"]:
                        row["aliases"].append(alias)
        self._alias_index: dict[str, tuple[dict[str, Any], ...]] = self._build_alias_index(self._rows)
        self._documents_by_id = self._build_document_index(self._rows)

    @staticmethod
    def _manifest_fingerprints(path: Path) -> dict[str, str]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result: dict[str, str] = {}
        for document in payload.get("documents", []):
            document_id = document.get("document_id")
            fingerprint = document.get("content_sha256")
            if isinstance(document_id, str) and isinstance(fingerprint, str) and _HEX64_RE.fullmatch(fingerprint):
                result[document_id] = fingerprint
        return result

    @staticmethod
    def _canonical_row(row: Mapping[str, Any], fingerprints: Mapping[str, str]) -> dict[str, Any]:
        required = ("document_id", "stock_code", "stock_name", "company_full", "report_year")
        missing = [name for name in required if row.get(name) in (None, "")]
        if missing:
            raise ValueError(f"resolver index row missing fields: {missing}")
        document_id = str(row["document_id"])
        aliases = [
            str(alias)
            for alias in row.get("aliases", [])
            if isinstance(alias, (str, int)) and str(alias).strip()
        ]
        aliases.extend(
            [
                str(row["stock_code"]),
                str(row.get("stock_symbol") or ""),
                str(row["stock_name"]),
                str(row["company_full"]),
            ]
        )
        aliases = list(dict.fromkeys(alias for alias in aliases if alias))
        fingerprint = fingerprints.get(document_id) or _synthetic_fingerprint(row)
        return {
            "aliases": aliases,
            "status": str(row.get("status", "unique")),
            "document_id": document_id,
            "stock_code": str(row["stock_code"]),
            "stock_name": str(row["stock_name"]),
            "company_full": str(row["company_full"]),
            "report_year": int(row["report_year"]),
            "artifact_fingerprint": fingerprint,
        }

    @staticmethod
    def _build_alias_index(rows: Sequence[dict[str, Any]]) -> dict[str, tuple[dict[str, Any], ...]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            for alias in row["aliases"]:
                key = _normalize_alias(alias)
                if key:
                    grouped[key].append(row)
        return {
            key: tuple(sorted(value, key=lambda item: (item["stock_code"], item["report_year"], item["document_id"])))
            for key, value in grouped.items()
        }

    @staticmethod
    def _build_document_index(rows: Sequence[dict[str, Any]]) -> dict[str, tuple[dict[str, Any], ...]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row["document_id"]].append(row)
        return {key: tuple(value) for key, value in grouped.items()}

    def resolve(
        self,
        analysis: Mapping[str, Any],
        request: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a deterministic, ``validate_scope_plan`` compatible payload."""

        analysis_obj = validate_question_analysis(dict(analysis))
        request_obj = dict(request or {})

        explicit_ids = self._explicit_document_ids(request_obj)
        if explicit_ids:
            return self._explicit_document_scope(explicit_ids)

        report_years = self._report_year_constraints(analysis_obj, request_obj)
        question_report_years = self._question_report_year_constraints(analysis_obj)
        metric_years = self._metric_years(analysis_obj, request_obj)
        mentions = self._ordered_mentions(analysis_obj, request_obj)

        if not mentions:
            return self._corpus_scope(analysis_obj, report_years, metric_years)

        narrative = "narrative" in analysis_obj["intents"]
        output_periods = _ordered_unique(analysis_obj["output_period_axis"])
        # A narrative period topology names the reports it wants.  In a purely
        # structured period comparison those years remain metric periods and do
        # not silently select report documents.
        effective_report_years = list(report_years)
        if narrative and not effective_report_years and output_periods:
            effective_report_years = output_periods

        entities: list[dict[str, Any]] = []
        seen_entities: set[str] = set()
        for mention in mentions:
            entity = self._resolve_mention(
                mention,
                report_years=effective_report_years,
                output_years=output_periods,
                narrative=narrative,
            )
            dedupe_key = (
                entity["identity"]["stock_code"]
                if entity["status"] == "unique"
                else f'{entity["status"]}:{_normalize_alias(entity["mention"])}'
            )
            if dedupe_key in seen_entities:
                continue
            seen_entities.add(dedupe_key)
            entities.append(entity)

        self._apply_request_company_guard(entities, request_obj)
        self._apply_request_report_year_guard(
            entities,
            request_obj,
            question_report_years=question_report_years,
            output_years=output_periods,
        )

        if len(entities) > 1:
            scope_kind = "multi_company_documents"
        elif entities[0]["status"] == "unique" and len(entities[0]["document_set"]) == 1:
            scope_kind = "single_document"
        else:
            scope_kind = "company_documents"

        findings = [finding for entity in entities for finding in entity["findings"]]
        body: dict[str, Any] = {
            "schema_version": SCHEMA_SCOPE,
            "scope_kind": scope_kind,
            "resolver_version": self.resolver_version,
            "phase8_capabilities": dict(_CAPABILITIES[scope_kind]),
            "entity_resolutions": entities,
            "report_year_constraints": effective_report_years,
            "metric_years": metric_years,
            "corpus_scope": None,
            "explicit_document_ids": [],
            "findings": findings,
            "resolution_skipped_reason": None,
        }
        return self._finalize(body)

    @staticmethod
    def _explicit_document_ids(request: Mapping[str, Any]) -> list[str]:
        raw = request.get("explicit_document_ids", [])
        if not isinstance(raw, list):
            raise ValueError("request.explicit_document_ids must be an array")
        return list(dict.fromkeys(str(value) for value in raw if str(value).strip()))

    @staticmethod
    def _ordered_mentions(analysis: Mapping[str, Any], request: Mapping[str, Any]) -> list[dict[str, Any]]:
        indexed = list(enumerate(analysis["company_mentions"]))
        indexed.sort(key=lambda pair: (pair[1]["mention_ordinal"], pair[0]))
        mentions = [dict(value) for _, value in indexed]
        request_company = request.get("company")
        if request_company and not mentions:
            mentions.append(
                {
                    "raw_text": str(request_company),
                    "span": [0, len(str(request_company))],
                    "mention_ordinal": 0,
                    "hint_source": "request_hint",
                }
            )
        return mentions

    @staticmethod
    def _report_year_constraints(analysis: Mapping[str, Any], request: Mapping[str, Any]) -> list[int]:
        values = ScopeResolver._question_report_year_constraints(analysis)
        if request.get("report_year") is not None:
            values.append(int(request["report_year"]))
        return _ordered_unique(values)

    @staticmethod
    def _question_report_year_constraints(analysis: Mapping[str, Any]) -> list[int]:
        ordered = sorted(
            enumerate(analysis["year_mentions"]),
            key=lambda pair: (pair[1]["mention_ordinal"], pair[0]),
        )
        values: list[int] = []
        for _, mention in ordered:
            if mention["role"] in {"report_year", "corpus_year"}:
                values.extend(mention["years"])
        return _ordered_unique(values)

    @staticmethod
    def _metric_years(analysis: Mapping[str, Any], request: Mapping[str, Any]) -> list[int]:
        ordered = sorted(
            enumerate(analysis["year_mentions"]),
            key=lambda pair: (pair[1]["mention_ordinal"], pair[0]),
        )
        values: list[int] = []
        for _, mention in ordered:
            if mention["role"] in {"metric_year", "formula_operand", "output_period"}:
                values.extend(mention["years"])
        values.extend(analysis["output_period_axis"])
        values.extend(request.get("metric_years", []))
        return _ordered_unique(values)

    def _resolve_mention(
        self,
        mention: Mapping[str, Any],
        *,
        report_years: Sequence[int],
        output_years: Sequence[int],
        narrative: bool,
    ) -> dict[str, Any]:
        raw_text = str(mention["raw_text"])
        alias_key = _normalize_alias(raw_text)
        candidates = list(self._alias_index.get(alias_key, ()))
        unresolved_key = "unresolved:" + hashlib.sha256(alias_key.encode("utf-8")).hexdigest()[:12]
        base = {
            "entity_key": unresolved_key,
            "mention": raw_text,
            "mention_ordinal": int(mention["mention_ordinal"]),
            "status": "missing",
            "identity": None,
            "document_set": [],
            "findings": [],
        }

        if not candidates:
            base["findings"].append(
                self._finding(
                    "RESOLVER_MISSING",
                    "entity",
                    unresolved_key,
                    report_years,
                    output_years,
                    [],
                    mention=raw_text,
                    candidate_stock_codes=[],
                )
            )
            return base

        stock_codes = sorted({row["stock_code"] for row in candidates})
        candidate_doc_ids = sorted({row["document_id"] for row in candidates})
        if len(stock_codes) > 1:
            base["status"] = "ambiguous"
            base["findings"].append(
                self._finding(
                    "RESOLVER_AMBIGUOUS",
                    "entity",
                    unresolved_key,
                    report_years,
                    output_years,
                    candidate_doc_ids,
                    mention=raw_text,
                    candidate_stock_codes=stock_codes,
                )
            )
            return base

        identity_tuples = {
            (row["stock_code"], row["stock_name"], row["company_full"])
            for row in candidates
        }
        conflicting_rows = any(row["status"] not in {"unique", "valid"} for row in candidates)
        if len(identity_tuples) != 1 or conflicting_rows or self._documents_conflict(candidates):
            base["status"] = "conflict"
            base["findings"].append(
                self._finding(
                    "RESOLVER_CONFLICT",
                    "entity",
                    unresolved_key,
                    report_years,
                    output_years,
                    candidate_doc_ids,
                    mention=raw_text,
                    candidate_stock_codes=stock_codes,
                )
            )
            return base

        stock_code, stock_name, company_full = next(iter(identity_tuples))
        all_documents = self._documents(candidates)
        filtered_documents = (
            [document for document in all_documents if document["report_year"] in report_years]
            if report_years
            else all_documents
        )
        entity: dict[str, Any] = {
            "entity_key": stock_code,
            "mention": raw_text,
            "mention_ordinal": int(mention["mention_ordinal"]),
            "status": "unique",
            "identity": {
                "stock_code": stock_code,
                "stock_name": stock_name,
                "company_full": company_full,
            },
            "document_set": filtered_documents,
            "findings": [],
        }

        if report_years:
            missing_years = [
                year for year in report_years if not any(doc["report_year"] == year for doc in all_documents)
            ]
            if missing_years:
                entity["findings"].append(
                    self._finding(
                        "RESOLVER_MISSING",
                        "document",
                        stock_code,
                        missing_years,
                        output_years,
                        [doc["document_id"] for doc in all_documents],
                        mention=raw_text,
                        candidate_stock_codes=[stock_code],
                    )
                )

        ambiguous_documents = self._ambiguous_document_ids(
            filtered_documents,
            report_years=report_years,
            narrative=narrative,
        )
        if ambiguous_documents:
            entity["findings"].append(
                self._finding(
                    "RESOLVER_AMBIGUOUS",
                    "document",
                    stock_code,
                    report_years,
                    output_years,
                    ambiguous_documents,
                    mention=raw_text,
                    candidate_stock_codes=[stock_code],
                )
            )
        return entity

    @staticmethod
    def _documents_conflict(rows: Sequence[Mapping[str, Any]]) -> bool:
        seen: dict[str, tuple[str, int, str]] = {}
        for row in rows:
            current = (row["stock_code"], row["report_year"], row["artifact_fingerprint"])
            previous = seen.setdefault(row["document_id"], current)
            if previous != current:
                return True
        return False

    @staticmethod
    def _documents(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            by_id[row["document_id"]] = {
                "document_id": row["document_id"],
                "stock_code": row["stock_code"],
                "report_year": row["report_year"],
                "artifact_fingerprint": row["artifact_fingerprint"],
            }
        return sorted(by_id.values(), key=lambda item: (item["report_year"], item["document_id"]))

    @staticmethod
    def _ambiguous_document_ids(
        documents: Sequence[Mapping[str, Any]],
        *,
        report_years: Sequence[int],
        narrative: bool,
    ) -> list[str]:
        by_year: dict[int, list[str]] = defaultdict(list)
        for document in documents:
            by_year[int(document["report_year"])].append(str(document["document_id"]))
        duplicate_year_docs = [doc for values in by_year.values() if len(values) > 1 for doc in values]
        if duplicate_year_docs:
            return sorted(duplicate_year_docs)
        if narrative and not report_years and len(documents) > 1:
            return sorted(str(document["document_id"]) for document in documents)
        return []

    @staticmethod
    def _finding(
        failure_code: str,
        subject: str,
        entity_key: str,
        requested_report_years: Sequence[int],
        requested_output_years: Sequence[int],
        candidate_document_ids: Sequence[str],
        *,
        mention: str,
        candidate_stock_codes: Sequence[str],
    ) -> dict[str, Any]:
        return {
            "failure_code": failure_code,
            "subject": subject,
            "entity_key": entity_key,
            "mention": mention,
            "requested_report_years": list(requested_report_years),
            "requested_output_years": list(requested_output_years),
            "candidate_stock_codes": sorted(set(candidate_stock_codes)),
            "candidate_document_ids": sorted(set(candidate_document_ids)),
        }

    def _apply_request_company_guard(
        self,
        entities: list[dict[str, Any]],
        request: Mapping[str, Any],
    ) -> None:
        hint = request.get("company")
        if not hint or any(entity["mention"] == str(hint) for entity in entities):
            return
        hint_rows = self._alias_index.get(_normalize_alias(str(hint)), ())
        hint_codes = sorted({row["stock_code"] for row in hint_rows})
        if len(hint_codes) != 1:
            return
        permitted = hint_codes[0]
        for entity in entities:
            if entity["status"] == "unique" and entity["identity"]["stock_code"] != permitted:
                existing_docs = [doc["document_id"] for doc in entity["document_set"]]
                entity["status"] = "conflict"
                entity["identity"] = None
                entity["document_set"] = []
                entity["findings"].append(
                    self._finding(
                        "RESOLVER_CONFLICT",
                        "request_company_guard",
                        entity["entity_key"],
                        [],
                        [],
                        existing_docs,
                        mention=entity["mention"],
                        candidate_stock_codes=sorted({permitted, entity["entity_key"]}),
                    )
                )

    def _apply_request_report_year_guard(
        self,
        entities: list[dict[str, Any]],
        request: Mapping[str, Any],
        *,
        question_report_years: Sequence[int],
        output_years: Sequence[int],
    ) -> None:
        hint = request.get("report_year")
        if hint is None or not question_report_years or int(hint) in question_report_years:
            return
        conflicting_years = [*question_report_years, int(hint)]
        for entity in entities:
            if entity["status"] != "unique":
                continue
            candidate_ids = [doc["document_id"] for doc in entity["document_set"]]
            stock_code = entity["identity"]["stock_code"]
            entity["status"] = "conflict"
            entity["identity"] = None
            entity["document_set"] = []
            entity["findings"].append(
                self._finding(
                    "RESOLVER_CONFLICT",
                    "report_year_guard",
                    entity["entity_key"],
                    conflicting_years,
                    output_years,
                    candidate_ids,
                    mention=entity["mention"],
                    candidate_stock_codes=[stock_code],
                )
            )

    def _explicit_document_scope(self, document_ids: Sequence[str]) -> dict[str, Any]:
        unknown_ids = sorted(document_id for document_id in document_ids if document_id not in self._documents_by_id)
        findings: list[dict[str, Any]] = []
        if unknown_ids:
            findings.append(
                {
                    "failure_code": "RESOLVER_MISSING",
                    "subject": "explicit_document_set",
                    "unknown_document_ids": unknown_ids,
                }
            )
        body: dict[str, Any] = {
            "schema_version": SCHEMA_SCOPE,
            "scope_kind": "explicit_document_set",
            "resolver_version": self.resolver_version,
            "phase8_capabilities": dict(_CAPABILITIES["explicit_document_set"]),
            "entity_resolutions": [],
            "report_year_constraints": [],
            "metric_years": [],
            "corpus_scope": None,
            "explicit_document_ids": list(document_ids),
            "findings": findings,
            "resolution_skipped_reason": "explicit_document_set_is_contract_only_in_phase8",
        }
        return self._finalize(body)

    def _corpus_scope(
        self,
        analysis: Mapping[str, Any],
        report_years: Sequence[int],
        metric_years: Sequence[int],
    ) -> dict[str, Any]:
        all_documents = self._documents(self._rows)
        selected = (
            [document for document in all_documents if document["report_year"] in report_years]
            if report_years
            else all_documents
        )
        stock_codes = sorted({document["stock_code"] for document in selected})
        corpus_scope = {
            "scope_version": "phase8-corpus-snapshot-v1",
            "source_fingerprint": self._source_fingerprint,
            "document_count": len(selected),
            "company_count": len(stock_codes),
            "report_years": sorted({document["report_year"] for document in selected}),
            "allowed_operations": ["aggregate", "rank"],
        }
        findings: list[dict[str, Any]] = []
        if not selected and report_years:
            findings.append(
                {
                    "failure_code": "RESOLVER_MISSING",
                    "subject": "corpus_documents",
                    "requested_report_years": list(report_years),
                    "candidate_document_ids": [],
                }
            )
        if not set(analysis["intents"]).intersection({"rank", "aggregate"}):
            findings.append(
                {
                    "failure_code": "COMPOSITION_UNSUPPORTED",
                    "subject": "corpus_intent",
                    "intents": list(analysis["intents"]),
                }
            )
        body: dict[str, Any] = {
            "schema_version": SCHEMA_SCOPE,
            "scope_kind": "corpus",
            "resolver_version": self.resolver_version,
            "phase8_capabilities": dict(_CAPABILITIES["corpus"]),
            "entity_resolutions": [],
            "report_year_constraints": list(report_years),
            "metric_years": list(metric_years),
            "corpus_scope": corpus_scope,
            "explicit_document_ids": [],
            "findings": findings,
            "resolution_skipped_reason": None,
        }
        return self._finalize(body)

    @staticmethod
    def _finalize(body: dict[str, Any]) -> dict[str, Any]:
        payload = {"scope_plan_id": "scope_" + semantic_sha256(body)[:20], **body}
        validate_scope_plan(payload)
        return payload


__all__ = ["RESOLVER_VERSION", "ScopeResolver"]
