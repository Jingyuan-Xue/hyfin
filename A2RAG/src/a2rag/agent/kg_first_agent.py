import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..prompts.prompt_template_manager import PromptTemplateManager
from ..prompts.linking import get_query_instruction
from ..retrieval.knn import retrieve_knn
from ..retrieval.entity_resolution import (
    parse_named_entities,
    build_normalized_entity_index,
    fuzzy_match_entities,
    normalize_entity,
)
from ..utils.chat_utils import fix_broken_generated_json

logger = logging.getLogger(__name__)


@dataclass
class KGFirstResult:
    answer: str
    sufficient: bool
    used_passages: List[str]
    seed_entities: List[str]
    matched_entity_ids: List[str]
    fallback_used: bool
    metadata: Optional[Dict] = None


class KGFirstAgent:
    def __init__(self,
                 a2rag,
                 entity_top_k: int = 5,
                 entity_sim_threshold: float = 0.75,
                 max_passages: int = 8,
                 fuzzy_cutoff: float = 0.9,
                 enable_fuzzy: bool = True):
        """
        KG-first agent:
        1) extract entities from query
        2) align entities to KG
        3) collect 1-hop info around seed entities
        4) chat judges if enough; otherwise fallback to A2RAG pipeline (PPR)
        """
        self.a2rag = a2rag
        self.entity_top_k = entity_top_k
        self.entity_sim_threshold = entity_sim_threshold
        self.max_passages = max_passages
        self.fuzzy_cutoff = fuzzy_cutoff
        self.enable_fuzzy = enable_fuzzy

        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )

        # Ensure retrieval structures are ready
        if not self.a2rag.ready_to_retrieve:
            self.a2rag.prepare_retrieval_objects()

        # Build entity lookup index
        self.entity_id_to_row = self.a2rag.entity_embedding_store.get_all_id_to_rows()
        self.entity_ids = list(self.entity_id_to_row.keys())
        self.entity_id_set = set(self.entity_ids)
        self.passage_id_set = set(self.a2rag.chunk_embedding_store.get_all_ids())
        self.normalized_to_ids, self.normalized_entities = build_normalized_entity_index(self.entity_id_to_row)

        self._entity_embeddings = None

    def _get_entity_embeddings(self) -> np.ndarray:
        if self._entity_embeddings is None:
            self._entity_embeddings = np.array(
                self.a2rag.entity_embedding_store.get_embeddings(self.entity_ids)
            )
        return self._entity_embeddings

    def extract_entities(self, query: str) -> List[str]:
        messages = self.prompt_template_manager.render(name="ner_query", query=query)
        raw_response, metadata = self._infer(messages)
        entities = parse_named_entities(raw_response)
        # Deduplicate while preserving order
        seen = set()
        ordered = []
        for ent in entities:
            ent = ent.strip()
            if not ent or ent in seen:
                continue
            seen.add(ent)
            ordered.append(ent)
        return ordered

    def link_entities(self, entities: List[str]) -> List[str]:
        matched_ids: List[str] = []

        # Exact / normalized match
        for ent in entities:
            norm = normalize_entity(ent)
            if norm in self.normalized_to_ids:
                matched_ids.extend(self.normalized_to_ids[norm])

        # Fuzzy match
        if self.enable_fuzzy:
            for ent in entities:
                if normalize_entity(ent) in self.normalized_to_ids:
                    continue
                matched_ids.extend(
                    fuzzy_match_entities(
                        query_entity=ent,
                        normalized_entities=self.normalized_entities,
                        normalized_to_ids=self.normalized_to_ids,
                        cutoff=self.fuzzy_cutoff,
                    )
                )

        # Embedding-based alignment
        if self.a2rag.embedding_model is not None and self.entity_ids:
            query_texts = []
            for ent in entities:
                norm = normalize_entity(ent)
                if norm not in self.normalized_to_ids:
                    query_texts.append(ent)
            if query_texts:
                query_vecs = self.a2rag.embedding_model.batch_encode(
                    query_texts,
                    instruction=get_query_instruction("ner_to_node"),
                    norm=True,
                )
                key_vecs = self._get_entity_embeddings()
                knn = retrieve_knn(
                    query_ids=query_texts,
                    key_ids=self.entity_ids,
                    query_vecs=query_vecs,
                    key_vecs=key_vecs,
                    k=self.entity_top_k,
                )
                for q in query_texts:
                    if q not in knn:
                        continue
                    candidate_ids, scores = knn[q]
                    for node_id, score in zip(candidate_ids, scores):
                        if score >= self.entity_sim_threshold:
                            matched_ids.append(node_id)

        # Deduplicate
        return list(dict.fromkeys(matched_ids))

    def collect_1hop_passages(self, entity_node_ids: List[str]) -> Tuple[List[str], List[str]]:
        if not entity_node_ids:
            return [], []

        passage_ids: Set[str] = set()
        neighbor_entities: Set[str] = set()

        for node_id in entity_node_ids:
            # Passage neighbors via cached map
            passage_ids.update(self.a2rag.ent_node_to_chunk_ids.get(node_id, set()))

            # Graph neighbors (entity + passage)
            if hasattr(self.a2rag, "node_name_to_vertex_idx"):
                node_idx = self.a2rag.node_name_to_vertex_idx.get(node_id)
                if node_idx is None:
                    continue
                for n_idx in self.a2rag.graph.neighbors(node_idx):
                    neighbor_id = self.a2rag.graph.vs[n_idx]["name"]
                    if neighbor_id in self.passage_id_set:
                        passage_ids.add(neighbor_id)
                    elif neighbor_id in self.entity_id_set:
                        neighbor_entities.add(neighbor_id)

        # Build passage texts
        passages = []
        for pid in passage_ids:
            try:
                passages.append(self.a2rag.chunk_embedding_store.get_row(pid)["content"])
            except Exception:
                continue

        # Limit passages
        passages = passages[: self.max_passages]

        neighbor_entity_texts = []
        for eid in neighbor_entities:
            try:
                neighbor_entity_texts.append(self.entity_id_to_row[eid]["content"])
            except Exception:
                continue

        return passages, neighbor_entity_texts

    def judge_and_answer(self, query: str, passages: List[str]) -> Tuple[bool, str, Dict]:
        evidence = ""
        for idx, passage in enumerate(passages, start=1):
            evidence += f"[Passage {idx}] {passage}\n\n"

        messages = self.prompt_template_manager.render(
            name="kg_first_qa",
            query=query,
            evidence=evidence.strip(),
        )
        raw_response, metadata = self._infer(messages)

        # Parse JSON response
        parsed = None
        try:
            parsed = json.loads(raw_response)
        except Exception:
            try:
                fixed = fix_broken_generated_json(raw_response)
                parsed = json.loads(fixed)
            except Exception:
                parsed = None

        if not isinstance(parsed, dict):
            return False, "", {"raw_response": raw_response}

        sufficient = bool(parsed.get("sufficient", False))
        answer = parsed.get("answer", "") if sufficient else ""
        return sufficient, answer, {"raw_response": raw_response, "parsed": parsed}

    def _infer(self, messages: List[Dict]) -> Tuple[str, Dict]:
        result = self.a2rag.chat_model_client.infer(messages=messages)
        if isinstance(result, tuple) and len(result) == 3:
            response, metadata, _cache_hit = result
            return response, metadata
        if isinstance(result, tuple) and len(result) == 2:
            response, metadata = result
            return response, metadata
        # Fallback for unexpected return format
        return str(result), {}

    def answer(self, query: str) -> KGFirstResult:
        seed_entities = self.extract_entities(query)
        matched_entity_ids = self.link_entities(seed_entities)
        passages, neighbor_entities = self.collect_1hop_passages(matched_entity_ids)

        sufficient = False
        answer = ""
        metadata = {"neighbor_entities": neighbor_entities}

        if passages:
            sufficient, answer, judge_meta = self.judge_and_answer(query, passages)
            metadata.update(judge_meta)

        if sufficient:
            return KGFirstResult(
                answer=answer,
                sufficient=True,
                used_passages=passages,
                seed_entities=seed_entities,
                matched_entity_ids=matched_entity_ids,
                fallback_used=False,
                metadata=metadata,
            )

        # Fallback to A2RAG pipeline (PPR)
        qa_results = self.a2rag.rag_qa(queries=[query])
        # qa_results[0] is QuerySolution list
        query_solution = qa_results[0][0]
        return KGFirstResult(
            answer=query_solution.answer,
            sufficient=False,
            used_passages=query_solution.docs,
            seed_entities=seed_entities,
            matched_entity_ids=matched_entity_ids,
            fallback_used=True,
            metadata=metadata,
        )
