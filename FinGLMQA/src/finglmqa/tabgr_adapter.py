"""Hash-pinned deterministic adapter for TabGR's QG-PPR scoring chain."""

from __future__ import annotations

import hashlib
import importlib.util
import re
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

from .contracts import semantic_sha256


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABGR_SOURCE = ROOT / "refs/tabgr_runtime/build_graphs/graph_to_text_triple_full.py"
PINNED_TABGR_SHA256 = "7d193807d5f74b3281c8bd52c0d6da76f1f149cd5e92c4c82b47de4b8708d316"
ADAPTER_VERSION = "phase9-tabgr-qg-ppr-adapter-v1"


class TabGRRuntimeUnavailable(RuntimeError):
    failure_code = "SUPPLEMENT_RUNTIME_UNAVAILABLE"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _norm(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _row_number(value: str) -> int | None:
    match = re.fullmatch(r"row0*([0-9]+)", value.strip(), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


class TabGRAdapter:
    """Calls the pinned private scorer and checks the public renderer agrees."""

    def __init__(
        self,
        source_path: str | Path = DEFAULT_TABGR_SOURCE,
        *,
        expected_sha256: str = PINNED_TABGR_SHA256,
    ) -> None:
        self.source_path = Path(source_path)
        try:
            actual = file_sha256(self.source_path)
        except OSError as exc:
            raise TabGRRuntimeUnavailable("TabGR source cannot be read") from exc
        if actual != expected_sha256:
            raise TabGRRuntimeUnavailable("TabGR source hash pin mismatch")
        self.source_sha256 = actual
        self._module = self._load_module()
        self._active_table_id: str | None = None
        self._structural_cache: tuple[Any, Any, Any] | None = None
        self._score_cache: dict[tuple[str, ...], tuple[dict[int, float], str, str]] = {}
        self.trace_fingerprint = semantic_sha256({
            "adapter_version": ADAPTER_VERSION,
            "source_sha256": actual,
            "model": "llama3",
            "alpha": "0.35",
            "w_row": "0.7",
            "w_col": "0.3",
            "iterations": 30,
        })

    def _load_module(self) -> ModuleType:
        spec = importlib.util.spec_from_file_location(
            "finglmqa_phase9_tabgr_runtime_" + self.source_sha256[:12],
            self.source_path,
        )
        if spec is None or spec.loader is None:
            raise TabGRRuntimeUnavailable("TabGR import spec is unavailable")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise TabGRRuntimeUnavailable("TabGR module import failed") from exc
        required = (
            "_parse_triples", "_build_index", "_build_neighbors", "_build_personalization",
            "_run_ppr", "_render_baseline_graph", "_render_merged_graph",
            "_cell_value_key", "grouped_string_with_cell_merges_w",
        )
        if any(not callable(getattr(module, name, None)) for name in required):
            raise TabGRRuntimeUnavailable("TabGR scoring API is incomplete")
        return module

    @staticmethod
    def targets(aliases: Iterable[str], metric_year: int, normalized_unit: str) -> list[str]:
        year_values = {
            str(metric_year), f"{metric_year}年", f"{metric_year}年度",
            f"{metric_year}年12月31日", f"{metric_year}-12-31",
        }
        unit_values = {
            normalized_unit,
            "人民币元" if normalized_unit == "元" else normalized_unit,
            "%" if normalized_unit == "ratio" else normalized_unit,
            "元／股" if normalized_unit == "元/股" else normalized_unit,
        }
        return sorted({str(value).strip() for value in (*aliases, *year_values, *unit_values) if str(value).strip()})

    def rank_table(
        self,
        table: Mapping[str, Any],
        cells: Sequence[Mapping[str, Any]],
        *,
        aliases: Iterable[str],
        metric_year: int,
        normalized_unit: str,
    ) -> dict[str, Any]:
        edge_list = table.get("edge_list")
        if not isinstance(edge_list, list) or any(not isinstance(row, str) for row in edge_list):
            raise ValueError("table.edge_list must be a string array")
        table_id = str(table["table_id"])
        target_values = self.targets(aliases, metric_year, normalized_unit)
        module = self._module
        if self._active_table_id != table_id:
            triples = module._parse_triples(edge_list)
            index = module._build_index(triples)
            neighbors = module._build_neighbors(index, model="llama3", ds="wtq")
            self._active_table_id = table_id
            self._structural_cache = (triples, index, neighbors)
            self._score_cache.clear()
        else:
            assert self._structural_cache is not None
            triples, index, neighbors = self._structural_cache
        target_keys = {module._cell_value_key(value) for value in target_values if module._cell_value_key(value)}
        present_values = set(index.val_key_of.values())
        effective_targets = tuple(sorted(target_keys & present_values))
        cached = self._score_cache.get(effective_targets)
        if cached is None:
            effective_set = set(effective_targets)
            personalization = module._build_personalization(index, None, effective_set)
            scores = module._run_ppr(index, neighbors, personalization, alpha=0.35, iters=30, model="llama3")
            private_baseline = module._render_baseline_graph(index, scores, " ")
            private_merged = module._render_merged_graph(index, scores, effective_set) if effective_set else ""
            private_render = (
                f"{private_merged}\n{private_baseline}"
                if private_merged and private_baseline else private_baseline or private_merged
            )
            public_render = module.grouped_string_with_cell_merges_w(
                edge_list, list(effective_targets), question=None, use_graph=True, sep=" ", model="llama3", ds="wtq",
            )
            if public_render != private_render:
                raise TabGRRuntimeUnavailable("TabGR public/private renderer consistency check failed")
            public_sha = hashlib.sha256(public_render.encode("utf-8")).hexdigest()
            self._score_cache[effective_targets] = (scores, private_render, public_sha)
        else:
            scores, private_render, public_sha = cached

        by_coordinate: dict[tuple[int, str, str], list[Mapping[str, Any]]] = {}
        for cell in cells:
            key = (
                int(cell["row_index"]),
                _norm(cell.get("column_label")),
                _norm(cell.get("raw_value")),
            )
            by_coordinate.setdefault(key, []).append(cell)

        ranked: list[dict[str, Any]] = []
        discarded_nonunique = 0
        discarded_unmapped = 0
        for cell_id in index.cells:
            row_index = _row_number(index.row_of[cell_id])
            if row_index is None:
                discarded_unmapped += 1
                continue
            key = (row_index, _norm(index.col_print_of[cell_id]), _norm(index.val_print_of[cell_id]))
            matches = by_coordinate.get(key, [])
            if len(matches) != 1:
                if len(matches) > 1:
                    discarded_nonunique += 1
                else:
                    discarded_unmapped += 1
                continue
            cell = matches[0]
            score_text = format(float(scores.get(cell_id, 0.0)), ".8f")
            fingerprint = semantic_sha256({
                "table_id": table["table_id"],
                "row_index": int(cell["row_index"]),
                "col_index": int(cell["col_index"]),
                "raw_value": str(cell["raw_value"]),
                "score": score_text,
            })
            ranked.append({
                "table_id": str(table["table_id"]),
                "table_index": int(table["table_index"]),
                "row_index": int(cell["row_index"]),
                "col_index": int(cell["col_index"]),
                "raw_value": str(cell["raw_value"]),
                "row_label": str(cell.get("row_label") or ""),
                "column_label": str(cell.get("column_label") or ""),
                "score": score_text,
                "cell_fingerprint": fingerprint,
            })
        ranked.sort(key=lambda row: (-float(row["score"]), row["row_index"], row["col_index"], row["cell_fingerprint"]))
        return {
            "adapter_version": ADAPTER_VERSION,
            "tabgr_source_sha256": self.source_sha256,
            "trace_fingerprint": self.trace_fingerprint,
            "target_fingerprint": semantic_sha256(target_values),
            "public_renderer_sha256": public_sha,
            "ranked_cells": ranked,
            "audit_counts": {
                "parsed_triples": len(triples),
                "mapped_triples": len(ranked),
                "discarded_nonunique": discarded_nonunique,
                "discarded_unmapped": discarded_unmapped,
            },
        }


__all__ = [
    "ADAPTER_VERSION", "DEFAULT_TABGR_SOURCE", "PINNED_TABGR_SHA256",
    "TabGRAdapter", "TabGRRuntimeUnavailable", "file_sha256",
]
