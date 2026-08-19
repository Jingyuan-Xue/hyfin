"""Corpus-bound online Type 3 A2RAG + TabGR pipeline."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

from finglmqa.type3_a2rag_retriever import Type3A2RAGRetriever
from finglmqa.type3_evidence_fusion import (
    FUSION_VERSION,
    PLANNER_VERSION,
    TRACE_SCHEMA,
    EvidenceCandidate,
    Facet,
    Type3FusionError,
    candidate_from_atom,
    candidate_from_table,
    compose_answer_safe_packet,
    evaluate_shadow_id_selector,
    fuse_evidence,
    plan_facets,
    semantic_sha256,
    validate_generator_input,
)
from finglmqa.type3_tabgr_retriever import Type3TabGRCandidate, Type3TabGRRetriever


PIPELINE_VERSION = "type3-a2rag-tabgr-pipeline-v1"
SUPPORTED_ARMS = frozenset(
    {
        "union",
        "text_only",
        "table_only",
        "legacy_table_only",
        "v2_table_only",
        "no_route_quota",
        "no_adjacency",
        "no_fact_join",
        "no_table_semantics",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8", errors="strict") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Type3FusionError(f"JSON object required: {path}")
    return value


def bind_text_runtime(
    *,
    root: Path,
    a2rag_index_manifest: Path,
    require_pinned_interpreter: bool = True,
) -> dict[str, Any]:
    """Freeze the Phase 2 query-embedding runtime before loading large artifacts."""

    expected_venv = (root / "refs/a2rag_runtime/.venv").resolve()
    expected_interpreter = (expected_venv / "bin/python").resolve()
    actual_interpreter = Path(sys.executable).resolve()
    actual_prefix = Path(sys.prefix).resolve()
    if require_pinned_interpreter and (
        actual_interpreter != expected_interpreter or actual_prefix != expected_venv
    ):
        raise Type3FusionError(
            "Phase 4 text retrieval requires the pinned Phase 2 interpreter: "
            f"{root / 'refs/a2rag_runtime/.venv/bin/python'}"
        )
    try:
        sentence_transformers_version = importlib.metadata.version("sentence-transformers")
        torch_version = importlib.metadata.version("torch")
        import torch
    except (importlib.metadata.PackageNotFoundError, ModuleNotFoundError) as exc:
        raise Type3FusionError(
            "pinned Phase 2 runtime is incomplete; use "
            f"{root / 'refs/a2rag_runtime/.venv/bin/python'} without installing/downloading"
        ) from exc
    if sentence_transformers_version != "5.6.0":
        raise Type3FusionError("sentence-transformers version differs from frozen Phase 2 runtime")
    if not torch.cuda.is_available():
        raise Type3FusionError("frozen Phase 2 runtime requires CUDA for BGE-M3 query encoding")
    index = read_json(a2rag_index_manifest)
    model = dict(index.get("embedding_model") or {})
    model_path = Path(str(model.get("local_path") or ""))
    if not model_path.is_dir():
        raise Type3FusionError("frozen local BGE-M3 snapshot is missing")
    for field in ("config_sha256", "modules_sha256"):
        if len(str(model.get(field) or "")) != 64:
            raise Type3FusionError(f"BGE-M3 runtime lacks frozen {field}")
    unsigned = {
        "expected_interpreter": (root / "refs/a2rag_runtime/.venv/bin/python").as_posix(),
        "actual_interpreter": Path(sys.executable).as_posix(),
        "interpreter_resolved": actual_interpreter.as_posix(),
        "runtime_prefix": actual_prefix.as_posix(),
        "sentence_transformers_version": sentence_transformers_version,
        "torch_version": torch_version,
        "cuda_available": True,
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_device_name": str(torch.cuda.get_device_name(0)),
        "embedding_model": {
            "name": model.get("name"),
            "snapshot_revision": model.get("snapshot_revision"),
            "local_path": model_path.as_posix(),
            "dimension": model.get("dimension"),
            "config_sha256": model.get("config_sha256"),
            "modules_sha256": model.get("modules_sha256"),
        },
    }
    return {**unsigned, "runtime_binding_sha256": semantic_sha256(unsigned)}


@dataclass(frozen=True)
class ManifestPaths:
    corpus_manifest: Path
    question_profile: Path
    questions: Path
    a2rag_package_manifest: Path
    a2rag_index_manifest: Path
    text_atoms: Path
    tabgr_package_manifest: Path
    tabgr_index_manifest: Path
    fact_manifest: Path

    @classmethod
    def defaults(cls, root: Path, corpus_id: str, question_profile_id: str) -> "ManifestPaths":
        package = root / "data/corpus_package/type3" / corpus_id
        return cls(
            corpus_manifest=package / "corpus_manifest.json",
            question_profile=package / "questions" / question_profile_id / "question_profile.json",
            questions=package / "questions" / question_profile_id / "questions.jsonl",
            a2rag_package_manifest=package / "a2rag_text_v1/manifest.json",
            a2rag_index_manifest=(
                root / "data/indexes/type3" / corpus_id / "a2rag/index_manifest.json"
            ),
            text_atoms=package / "a2rag_text_v1/text_atoms.jsonl",
            tabgr_package_manifest=package / "tabgr_table_v2/manifest.json",
            tabgr_index_manifest=root / "data/indexes/type3" / corpus_id / "tabgr/manifest.json",
            fact_manifest=root / "data/facts/type3" / corpus_id / "manifest.json",
        )


@dataclass(frozen=True)
class ManifestBinding:
    corpus_id: str
    question_profile_id: str
    corpus_profile_sha256: str
    question_profile_sha256: str
    manifests: Mapping[str, Mapping[str, str]]
    binding_sha256: str

    def as_mapping(self) -> dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "question_profile_id": self.question_profile_id,
            "corpus_profile_sha256": self.corpus_profile_sha256,
            "question_profile_sha256": self.question_profile_sha256,
            "manifests": {key: dict(value) for key, value in sorted(self.manifests.items())},
            "binding_sha256": self.binding_sha256,
        }


def bind_manifests(
    paths: ManifestPaths,
    *,
    expected_corpus_id: str,
    expected_question_profile_id: str,
) -> ManifestBinding:
    """Bind every Phase 1/2/3 input by path, file hash, and internal profile hash."""

    named_paths = {
        "corpus": paths.corpus_manifest,
        "questions": paths.question_profile,
        "question_records": paths.questions,
        "a2rag_package": paths.a2rag_package_manifest,
        "a2rag_index": paths.a2rag_index_manifest,
        "text_atoms": paths.text_atoms,
        "tabgr_package": paths.tabgr_package_manifest,
        "tabgr_index": paths.tabgr_index_manifest,
        "facts": paths.fact_manifest,
    }
    for path in named_paths.values():
        if not path.is_file():
            raise Type3FusionError(f"required bound artifact missing: {path}")
    corpus = read_json(paths.corpus_manifest)
    questions = read_json(paths.question_profile)
    a2_package = read_json(paths.a2rag_package_manifest)
    a2_index = read_json(paths.a2rag_index_manifest)
    tab_package = read_json(paths.tabgr_package_manifest)
    tab_index = read_json(paths.tabgr_index_manifest)
    facts = read_json(paths.fact_manifest)
    if corpus.get("corpus_id") != expected_corpus_id:
        raise Type3FusionError("corpus manifest corpus_id mismatch")
    if questions.get("corpus_id") != expected_corpus_id:
        raise Type3FusionError("question profile corpus_id mismatch")
    if questions.get("question_profile_id") != expected_question_profile_id:
        raise Type3FusionError("question profile id mismatch")
    corpus_profile = str(corpus.get("profile_sha256") or "")
    question_profile = str(questions.get("profile_sha256") or "")
    if len(corpus_profile) != 64 or len(question_profile) != 64:
        raise Type3FusionError("profile hashes must be frozen SHA-256 values")
    for label, manifest in (
        ("a2rag package", a2_package),
        ("a2rag index", a2_index),
        ("TabGR package", tab_package),
        ("TabGR index", tab_index),
        ("fact store", facts),
    ):
        if manifest.get("corpus_id") != expected_corpus_id:
            raise Type3FusionError(f"{label} corpus_id mismatch")
        if manifest.get("corpus_profile_sha256") != corpus_profile:
            raise Type3FusionError(f"{label} corpus profile hash mismatch")
    if questions.get("questions_sha256") != sha256_file(paths.questions):
        raise Type3FusionError("sanitized question record hash mismatch")
    a2_artifacts = a2_package.get("artifacts") or {}
    if a2_artifacts.get("text_atoms.jsonl") != sha256_file(paths.text_atoms):
        raise Type3FusionError("text atom artifact hash mismatch")
    if (
        a2_index.get("artifacts", {}).get("text_package_manifest_sha256")
        != sha256_file(paths.a2rag_package_manifest)
    ):
        raise Type3FusionError("A2RAG package/index binding mismatch")
    if (
        tab_package.get("artifacts", {}).get("selected_fact_authorizations_sha256")
        != facts.get("authorizations_sha256")
    ):
        raise Type3FusionError("TabGR package/fact binding mismatch")
    if tab_index.get("row_evidence_sha256") != tab_package.get("artifacts", {}).get(
        "table_row_evidence_sha256"
    ):
        raise Type3FusionError("TabGR package/index row evidence mismatch")

    manifests = {
        key: {"path": path.resolve().as_posix(), "sha256": sha256_file(path)}
        for key, path in named_paths.items()
    }
    unsigned = {
        "corpus_id": expected_corpus_id,
        "question_profile_id": expected_question_profile_id,
        "corpus_profile_sha256": corpus_profile,
        "question_profile_sha256": question_profile,
        "manifests": manifests,
    }
    return ManifestBinding(
        corpus_id=expected_corpus_id,
        question_profile_id=expected_question_profile_id,
        corpus_profile_sha256=corpus_profile,
        question_profile_sha256=question_profile,
        manifests=manifests,
        binding_sha256=semantic_sha256(unsigned),
    )


class ExactAtomStore:
    """Read the persisted atom package once and retain the exact online projection fields."""

    def __init__(
        self,
        path: Path,
        *,
        expected_corpus_id: str,
        expected_sha256: str,
        expected_count: int | None = None,
    ) -> None:
        if sha256_file(path) != expected_sha256:
            raise Type3FusionError("text atom store hash changed after manifest binding")
        atoms: dict[str, dict[str, Any]] = {}
        documents: set[str] = set()
        with path.open(encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise Type3FusionError(f"blank atom row: {line_number}")
                row = json.loads(line)
                if row.get("corpus_id") != expected_corpus_id:
                    raise Type3FusionError("cross-corpus atom in store")
                atom_id = str(row.get("atom_id") or "")
                if not atom_id or atom_id in atoms:
                    raise Type3FusionError("empty or duplicate atom id")
                atoms[atom_id] = {
                    key: row[key]
                    for key in (
                        "atom_id",
                        "atom_kind",
                        "corpus_id",
                        "document_id",
                        "content",
                        "content_sha256",
                        "source_markdown",
                        "source_sha256",
                        "line_range",
                        "char_range",
                        "byte_range",
                        "heading_path",
                        "adjacent_table_ids",
                    )
                }
                documents.add(str(row["document_id"]))
        if expected_count is not None and len(atoms) != expected_count:
            raise Type3FusionError("atom store count differs from manifest")
        self.atoms = atoms
        self.document_ids = frozenset(documents)

    def project(self, atom_ids: Sequence[str], *, document_id: str) -> list[Mapping[str, Any]]:
        result = []
        for atom_id in atom_ids:
            row = self.atoms.get(atom_id)
            if row is None:
                raise Type3FusionError(f"retrieval unit references unknown atom: {atom_id}")
            if row["document_id"] != document_id:
                raise Type3FusionError("cross-document atom projection")
            result.append(row)
        return result


@dataclass(frozen=True)
class PipelineConfig:
    arm: str = "union"
    max_facets: int = 6
    text_top_k: int = 15
    table_top_k: int = 12
    max_fused_candidates: int = 18
    max_composed_items: int = 8
    text_mode: str = "hybrid"

    def validate(self) -> "PipelineConfig":
        if self.arm not in SUPPORTED_ARMS:
            raise Type3FusionError(f"unsupported Phase 4 arm: {self.arm}")
        if self.text_mode not in {"hybrid", "sparse"}:
            raise Type3FusionError("Phase 4 text_mode must be hybrid or sparse")
        if not 1 <= self.max_facets <= 6:
            raise Type3FusionError("max_facets must be in [1, 6]")
        return self


class Type3A2RAGTabGRPipeline:
    """A deterministic online pipeline with route caches shared across ablations."""

    def __init__(
        self,
        *,
        paths: ManifestPaths,
        corpus_id: str,
        question_profile_id: str,
        atom_store: ExactAtomStore | None = None,
        a2rag: Type3A2RAGRetriever | None = None,
        tabgr: Type3TabGRRetriever | None = None,
    ) -> None:
        self.paths = paths
        root = paths.corpus_manifest.parents[4]
        self.runtime_info = bind_text_runtime(
            root=root,
            a2rag_index_manifest=paths.a2rag_index_manifest,
        )
        self.binding = bind_manifests(
            paths,
            expected_corpus_id=corpus_id,
            expected_question_profile_id=question_profile_id,
        )
        a2_package = read_json(paths.a2rag_package_manifest)
        self.atom_store = atom_store or ExactAtomStore(
            paths.text_atoms,
            expected_corpus_id=corpus_id,
            expected_sha256=str(a2_package["artifacts"]["text_atoms.jsonl"]),
            expected_count=int(a2_package["atom_count"]),
        )
        self.a2rag = a2rag or Type3A2RAGRetriever(paths.a2rag_index_manifest.parent)
        self.tabgr = tabgr or Type3TabGRRetriever(
            paths.tabgr_index_manifest.parent, expected_corpus_id=corpus_id
        )
        if set(self.a2rag.documents) != set(self.tabgr.document_ids):
            raise Type3FusionError("A2RAG and TabGR document universes differ")
        if set(self.a2rag.documents) != set(self.atom_store.document_ids):
            raise Type3FusionError("A2RAG index and atom store document universes differ")
        self._text_cache: OrderedDict[
            tuple[str, str, tuple[str, ...], str, int, bool],
            tuple[tuple[Mapping[str, Any], ...], ...],
        ] = OrderedDict()
        self._table_cache: OrderedDict[
            tuple[str, str, tuple[str, ...], str, int],
            tuple[tuple[Type3TabGRCandidate, ...], ...],
        ] = OrderedDict()
        self._table_provenance_cache: OrderedDict[
            str, Mapping[str, Mapping[str, Any]]
        ] = OrderedDict()
        self._cache_limit = 512

    @staticmethod
    def _cache_put(cache: OrderedDict, key: object, value: object, limit: int) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    def _retrieve_text(
        self,
        document_id: str,
        facets: Sequence[Facet],
        *,
        mode: str,
        top_k: int,
        include_adjacency: bool,
    ) -> tuple[tuple[Mapping[str, Any], ...], ...]:
        key = (
            self.binding.binding_sha256,
            document_id,
            tuple(value.query for value in facets),
            mode,
            top_k,
            include_adjacency,
        )
        cached = self._text_cache.get(key)
        if cached is not None:
            self._text_cache.move_to_end(key)
            return cached
        results = self.a2rag.retrieve_many(
            [value.query for value in facets],
            document_id=document_id,
            top_k=top_k,
            mode=mode,
            include_heading=True,
            include_adjacency=include_adjacency,
        )
        frozen = tuple(tuple(result["candidates"]) for result in results)
        self._cache_put(self._text_cache, key, frozen, self._cache_limit)
        return frozen

    def _retrieve_table(
        self,
        document_id: str,
        facets: Sequence[Facet],
        *,
        variant: str,
        top_k: int,
    ) -> tuple[tuple[Type3TabGRCandidate, ...], ...]:
        key = (
            self.binding.binding_sha256,
            document_id,
            tuple(value.query for value in facets),
            variant,
            top_k,
        )
        cached = self._table_cache.get(key)
        if cached is not None:
            self._table_cache.move_to_end(key)
            return cached
        ablations = (
            ("legacy_flattened_a0",)
            if variant == "legacy"
            else ("no_unit_period_scope",)
            if variant == "v2_no_semantics"
            else ()
        )
        frozen = tuple(
            tuple(
                self.tabgr.retrieve(
                    facet.query,
                    document_id=document_id,
                    top_k=top_k,
                    ablations=ablations,
                )
            )
            for facet in facets
        )
        self._cache_put(self._table_cache, key, frozen, self._cache_limit)
        return frozen

    def _table_provenance(
        self,
        document_id: str,
        evidence_id: str,
    ) -> Mapping[str, Any]:
        cached = self._table_provenance_cache.get(document_id)
        if cached is None:
            records, _ = self.tabgr._load_document(document_id)
            document = self.a2rag.documents[document_id]
            source_sha256 = str(document.get("source_sha256") or "")
            source_markdown = str(document.get("source_markdown") or "")
            indexed: dict[str, Mapping[str, Any]] = {}
            for record in records:
                if record.get("record_type") != "table_row":
                    continue
                coordinates = []
                origin_hashes = []
                for cell in record.get("cells") or ():
                    coordinate = cell.get("coordinate")
                    origin_hash = str(cell.get("origin_cell_hash") or "")
                    if isinstance(coordinate, list):
                        coordinates.append(coordinate)
                    if origin_hash:
                        origin_hashes.append(origin_hash)
                indexed[str(record["evidence_id"])] = {
                    "source_markdown": source_markdown,
                    "source_sha256": source_sha256,
                    "table_id": str(record.get("table_id") or ""),
                    "table_line_range": list(record.get("table_line_range") or ()),
                    "table_sha256": str(record.get("table_sha256") or ""),
                    "cell_coordinates": coordinates,
                    "origin_cell_hashes": origin_hashes,
                }
            self._table_provenance_cache[document_id] = indexed
            self._table_provenance_cache.move_to_end(document_id)
            while len(self._table_provenance_cache) > 6:
                self._table_provenance_cache.popitem(last=False)
            cached = indexed
        else:
            self._table_provenance_cache.move_to_end(document_id)
        provenance = cached.get(evidence_id)
        if provenance is None:
            raise Type3FusionError("TabGR candidate lacks indexed rich row provenance")
        return provenance

    def _text_candidates(
        self,
        input_row: Mapping[str, str],
        facets: Sequence[Facet],
        *,
        config: PipelineConfig,
    ) -> list[EvidenceCandidate]:
        hybrid = self._retrieve_text(
            input_row["document_id"],
            facets,
            mode=config.text_mode,
            top_k=config.text_top_k,
            include_adjacency=config.arm != "no_adjacency",
        )
        dense = (
            self._retrieve_text(
                input_row["document_id"],
                facets,
                mode="dense",
                top_k=config.text_top_k,
                include_adjacency=config.arm != "no_adjacency",
            )
            if config.text_mode == "hybrid"
            else ()
        )
        result: list[EvidenceCandidate] = []
        for facet_index, facet in enumerate(facets):
            channels: list[tuple[str, int, Sequence[Mapping[str, Any]]]] = [
                ("a2rag_hybrid" if config.text_mode == "hybrid" else "a2rag_sparse", 3, hybrid[facet_index])
            ]
            if dense:
                channels.append(("a2rag_dense", 2, dense[facet_index]))
            for channel, weight, units in channels:
                for unit_rank, unit in enumerate(units, 1):
                    atoms = self.atom_store.project(
                        [str(value) for value in unit["atom_ids"]],
                        document_id=input_row["document_id"],
                    )
                    for atom_ordinal, atom in enumerate(atoms, 1):
                        result.append(
                            candidate_from_atom(
                                atom,
                                facet_id=facet.facet_id,
                                channel=channel,
                                rank=(unit_rank - 1) * 16 + atom_ordinal,
                                weight=weight,
                            )
                        )
        return result

    def _table_candidates(
        self,
        input_row: Mapping[str, str],
        facets: Sequence[Facet],
        *,
        config: PipelineConfig,
    ) -> list[EvidenceCandidate]:
        variants = (
            ("legacy",)
            if config.arm == "legacy_table_only"
            else ("v2",)
            if config.arm == "v2_table_only"
            else ("legacy", "v2_no_semantics")
            if config.arm == "no_table_semantics"
            else ("legacy", "v2")
        )
        result: list[EvidenceCandidate] = []
        for variant in variants:
            retrieved = self._retrieve_table(
                input_row["document_id"],
                facets,
                variant=variant,
                top_k=config.table_top_k,
            )
            channel = (
                "tabgr_legacy_flattened_a0"
                if variant == "legacy"
                else "tabgr_structural_v2_no_semantics"
                if variant == "v2_no_semantics"
                else "tabgr_structural_v2"
            )
            weight = 4 if variant == "legacy" else 2
            for facet_index, facet in enumerate(facets):
                for rank, candidate in enumerate(retrieved[facet_index], 1):
                    result.append(
                        candidate_from_table(
                            candidate,
                            provenance=self._table_provenance(
                                input_row["document_id"], candidate.evidence_id
                            ),
                            facet_id=facet.facet_id,
                            channel=channel,
                            rank=rank,
                            weight=weight,
                            no_fact_join=config.arm == "no_fact_join",
                        )
                    )
        return result

    def run_question(
        self,
        value: Mapping[str, Any],
        *,
        config: PipelineConfig = PipelineConfig(),
        shadow_selector_response: object | None = None,
        shadow_selector_timed_out: bool = False,
    ) -> dict[str, Any]:
        config = config.validate()
        input_row = validate_generator_input(value)
        if input_row["corpus_id"] != self.binding.corpus_id:
            raise Type3FusionError("online request corpus_id differs from bound corpus")
        if input_row["document_id"] not in self.a2rag.documents:
            raise Type3FusionError("online request document_id is outside bound corpus")
        facets = plan_facets(input_row["question"], max_facets=config.max_facets)
        include_text = config.arm not in {
            "table_only",
            "legacy_table_only",
            "v2_table_only",
        }
        include_table = config.arm != "text_only"
        candidates: list[EvidenceCandidate] = []
        if include_text:
            candidates.extend(self._text_candidates(input_row, facets, config=config))
        if include_table:
            candidates.extend(self._table_candidates(input_row, facets, config=config))
        fused = fuse_evidence(
            candidates,
            facets=facets,
            max_candidates=config.max_fused_candidates,
            route_quota=config.arm != "no_route_quota",
            adjacency=config.arm != "no_adjacency",
        )
        packet = compose_answer_safe_packet(fused, max_items=config.max_composed_items)
        trace_unsigned = {
            "schema_version": TRACE_SCHEMA,
            "pipeline_version": PIPELINE_VERSION,
            "planner_version": PLANNER_VERSION,
            "fusion_version": FUSION_VERSION,
            "arm": config.arm,
            "input": {
                "corpus_id": input_row["corpus_id"],
                "document_id": input_row["document_id"],
                "question": input_row["question"],
            },
            "manifest_binding": self.binding.as_mapping(),
            "runtime_binding": getattr(
                self,
                "runtime_info",
                {"mode": "injected_test_runtime", "runtime_binding_sha256": "0" * 64},
            ),
            "facets": [value.as_mapping() for value in facets],
            "retrieval": {
                "text_enabled": include_text,
                "table_enabled": include_table,
                "text_mode": config.text_mode if include_text else None,
                "a2rag_adjacency_terms": (
                    config.arm != "no_adjacency" if include_text else None
                ),
                "text_unit_projection": "exact_persisted_atoms" if include_text else None,
                "table_shortlist": (
                    "legacy_v2_union_no_table_semantics"
                    if include_table and config.arm == "no_table_semantics"
                    else "legacy_v2_union"
                    if include_table and config.arm not in {"legacy_table_only", "v2_table_only"}
                    else config.arm
                    if include_table
                    else None
                ),
                "legacy_rank_anchor_weight": 4 if include_table else None,
                "v2_structural_weight": 2 if include_table else None,
            },
            "fusion": {
                "route_quota": config.arm != "no_route_quota",
                "adjacency": config.arm != "no_adjacency",
                "candidate_count_before_fusion": len(candidates),
                "selected_candidate_ids": [value.candidate_id for value in fused],
                "selected_routes": [value.route for value in fused],
            },
            "numeric_safety": {
                "text_rule": "literal_occurs_verbatim_in_exact_atom",
                "table_rule": (
                    "all_numbers_redacted_no_fact_arm"
                    if config.arm == "no_fact_join"
                    else "exact_selected_fact_authorization"
                ),
                "unsupported_numeric_literals": packet["unsupported_numeric_literals"],
                "conflict_candidates": [
                    value.candidate_id for value in fused if value.conflict_status != "clear"
                ],
            },
            "composer": {
                "version": packet["composer_version"],
                "selected_candidate_ids": packet["selected_candidate_ids"],
            },
        }
        semantic_trace = {
            **trace_unsigned,
            "semantic_trace_sha256": semantic_sha256(trace_unsigned),
        }
        result = {
            "schema_version": "finglmqa.type3.a2rag_tabgr.answer_packet.v1",
            "corpus_id": input_row["corpus_id"],
            "question_id": input_row["question_id"],
            "document_id": input_row["document_id"],
            "question": input_row["question"],
            "arm": config.arm,
            "answer_safe_text": packet["answer_safe_text"],
            "citations": packet["citations"],
            "evidence": [value.as_mapping() for value in fused],
            "semantic_trace": semantic_trace,
        }
        if shadow_selector_response is not None or shadow_selector_timed_out:
            result["qwen_shadow_selector"] = evaluate_shadow_id_selector(
                fused,
                shadow_selector_response,
                max_selected=config.max_composed_items,
                timed_out=shadow_selector_timed_out,
            )
        return result


def load_sanitized_questions(
    path: Path,
    *,
    corpus_id: str,
    expected_count: int | None = None,
) -> list[dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise Type3FusionError(f"blank sanitized question row: {line_number}")
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise Type3FusionError("sanitized question row must be an object")
            rows.append(validate_generator_input({"corpus_id": corpus_id, **dict(value)}))
    if expected_count is not None and len(rows) != expected_count:
        raise Type3FusionError("sanitized question count mismatch")
    if len({value["question_id"] for value in rows}) != len(rows):
        raise Type3FusionError("duplicate sanitized question_id")
    return rows


__all__ = [
    "ExactAtomStore",
    "ManifestBinding",
    "ManifestPaths",
    "PIPELINE_VERSION",
    "PipelineConfig",
    "SUPPORTED_ARMS",
    "Type3A2RAGTabGRPipeline",
    "bind_manifests",
    "bind_text_runtime",
    "load_sanitized_questions",
    "read_json",
    "sha256_file",
]
