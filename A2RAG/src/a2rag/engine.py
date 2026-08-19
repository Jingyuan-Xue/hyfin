import json
import os
import logging
import ast
from dataclasses import asdict
from typing import Optional, List, Set, Dict, Any, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import igraph as ig
import numpy as np
import re
import time
import copy

from .models.chat import get_chat_model_class
from .models.chat_base import BaseChatModel
from .models.embeddings import get_embedding_model_class
from .models.embedding_base import BaseEmbeddingModel
from .text.splitters import get_text_preprocessor
from .storage.vector_store import EmbeddingStore
from .knowledge_extraction import KnowledgeExtractor
from .prompts.linking import get_query_instruction
from .prompts.prompt_template_manager import PromptTemplateManager
from .retrieval.reranker import DSPyFilter
from .schemas import NerRawOutput, QuerySolution, TripleRawOutput
from .utils.misc_utils import *
from .retrieval.knn import retrieve_knn
from .retrieval.entity_resolution import build_canonical_entity_map, normalize_entity
from .utils.typing import Triple
from .config.settings import BaseConfig

logger = logging.getLogger(__name__)

class A2RAG:

    _PRONOUNS = {
        "he", "she", "it", "they", "them", "him", "her", "his", "hers", "their",
        "theirs", "this", "that", "these", "those", "we", "us", "our", "ours",
        "i", "me", "my", "mine", "you", "your", "yours",
    }

    def __init__(self,
                 global_config=None,
                 save_dir=None,
                 chat_model=None,
                 chat_base_url=None,
                 embedding_model=None,
                 embedding_base_url=None):
        """
        Initializes an instance of the class and its related components.

        Attributes:
            global_config (BaseConfig): The global configuration settings for the instance. An instance
                of BaseConfig is used if no value is provided.
            saving_dir (str): The directory where specific A2RAG instances will be stored. This defaults
                to `outputs` if no value is provided.
            chat_model_client (BaseChatModel): The chat client used for processing based on the global
                configuration settings.
            extraction (KnowledgeExtractor): The knowledge extraction component.
            graph: The graph instance initialized by the `initialize_graph` method.
            embedding_model (BaseEmbeddingModel): The vector service associated with the current
                configuration.
            chunk_embedding_store (EmbeddingStore): The embedding store handling chunk embeddings.
            entity_embedding_store (EmbeddingStore): The embedding store handling entity embeddings.
            fact_embedding_store (EmbeddingStore): The embedding store handling fact embeddings.
            prompt_template_manager (PromptTemplateManager): The manager for handling prompt templates
                and roles mappings.
            extraction_results_path (str): The file path for storing knowledge extraction results
                based on the dataset and chat name in the global configuration.
            rerank_filter (Optional[DSPyFilter]): The filter responsible for reranking information
                when a rerank file path is specified in the global configuration.
            ready_to_retrieve (bool): A flag indicating whether the system is ready for retrieval
                operations.

        Parameters:
            global_config: The global configuration object. Defaults to None, leading to initialization
                of a new BaseConfig object.
            working_dir: The directory for storing working files. Defaults to None, constructing a default
                directory based on the class name and timestamp.
            chat_model: Chat model name.
            embedding_model: Embedding model name.
            chat_base_url: Chat gateway URL.
        """
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        #Overwriting Configuration if Specified
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if chat_model is not None:
            self.global_config.chat_model = chat_model

        if embedding_model is not None:
            self.global_config.embedding_model = embedding_model

        if chat_base_url is not None:
            self.global_config.chat_base_url = chat_base_url

        if embedding_base_url is not None:
            self.global_config.embedding_base_url = embedding_base_url

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"A2RAG init with config:\n  {_print_config}\n")

        #chat and vector service specific working directories are created under every specified saving directories
        chat_label = self.global_config.chat_model.replace("/", "_")
        embedding_label = self.global_config.embedding_model.replace("/", "_")
        self.working_dir = os.path.join(self.global_config.save_dir, f"{chat_label}_{embedding_label}")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self.chat_model_client: BaseChatModel = get_chat_model_class(self.global_config)

        if self.global_config.knowledge_extraction_mode == 'online':
            self.extraction = KnowledgeExtractor(chat_model_client=self.chat_model_client)
        else:
            raise NotImplementedError(
                f"Unsupported knowledge_extraction_mode: {self.global_config.knowledge_extraction_mode}"
            )

        self.graph = self.initialize_graph()

        if self.global_config.knowledge_extraction_mode == 'offline':
            self.embedding_model = None
        else:
            self.embedding_model: BaseEmbeddingModel = get_embedding_model_class(
                embedding_model=self.global_config.embedding_model
            )(global_config=self.global_config, embedding_model=self.global_config.embedding_model)
        self.chunk_embedding_store = EmbeddingStore(self.embedding_model,
                                                    os.path.join(self.working_dir, "chunk_embeddings"),
                                                    self.global_config.embedding_batch_size, 'chunk')
        self.entity_embedding_store = EmbeddingStore(self.embedding_model,
                                                     os.path.join(self.working_dir, "entity_embeddings"),
                                                     self.global_config.embedding_batch_size, 'entity')
        self.fact_embedding_store = EmbeddingStore(self.embedding_model,
                                                   os.path.join(self.working_dir, "fact_embeddings"),
                                                   self.global_config.embedding_batch_size, 'fact')

        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})

        self.text_preprocessor = get_text_preprocessor(self.global_config, embedding_model=self.embedding_model)

        self.extraction_results_path = self.global_config.extraction_results_path or os.path.join(
            self.global_config.save_dir,
            f'knowledge_extraction_results_{self.global_config.chat_model.replace("/", "_")}.json'
        )

        self.rerank_filter = DSPyFilter(self)

        self.ready_to_retrieve = False

        self.ppr_time = 0
        self.rerank_time = 0
        self.all_retrieval_time = 0

        self.ent_node_to_chunk_ids = None
        self._filter_chat_model_client = None


    def initialize_graph(self):
        """
        Initializes a graph using a Pickle file if available or creates a new graph.

        The function attempts to load a pre-existing graph stored in a Pickle file. If the file
        is not present or the graph needs to be created from scratch, it initializes a new directed
        or undirected graph based on the global configuration. If the graph is loaded successfully
        from the file, pertinent information about the graph (number of nodes and edges) is logged.

        Returns:
            ig.Graph: A pre-loaded or newly initialized graph.

        Raises:
            None
        """
        self._graph_pickle_filename = os.path.join(
            self.working_dir, f"graph.pickle"
        )

        preloaded_graph = None

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graph_pickle_filename):
                preloaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_filename)

        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(
                f"Loaded graph from {self._graph_pickle_filename} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges"
            )
            return preloaded_graph

    def prepare_knowledge_extraction(self,  docs: List[str]):
        docs = self.text_preprocessor.preprocess_docs(docs)
        logger.info(f"Indexing Documents")
        logger.info(f"Performing KnowledgeExtractor Offline")

        chunks = self.chunk_embedding_store.get_missing_string_hash_ids(docs)

        all_extraction_info, chunk_keys_to_process = self.load_extraction_cache(chunks.keys())
        new_extraction_rows = {k : chunks[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.extraction.batch_extract_knowledge(new_extraction_rows)
            self.merge_extraction_results(all_extraction_info, new_extraction_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_extraction_results:
            self.save_extraction_cache(all_extraction_info)

        assert False, logger.info('Done with KnowledgeExtractor, run online indexing for future retrieval.')

    def index(self, docs: List[str]):
        """
        Indexes the given documents based on the A2RAG 2 framework which generates an KnowledgeExtractor knowledge graph
        based on the given documents and encodes passages, entities and facts separately for later retrieval.

        Parameters:
            docs : List[str]
                A list of documents to be indexed.
        """

        docs = self.text_preprocessor.preprocess_docs(docs)
        logger.info(f"Indexing Documents")

        logger.info(f"Performing KnowledgeExtractor")

        if self.global_config.knowledge_extraction_mode == 'offline':
            self.prepare_knowledge_extraction(docs)

        self.chunk_embedding_store.insert_strings(docs)
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()

        all_extraction_info, chunk_keys_to_process = self.load_extraction_cache(chunk_to_rows.keys())
        new_extraction_rows = {k : chunk_to_rows[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.extraction.batch_extract_knowledge(new_extraction_rows)
            self.merge_extraction_results(all_extraction_info, new_extraction_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_extraction_results:
            self.save_extraction_cache(all_extraction_info)

        extraction_info_by_id = {info['idx']: info for info in all_extraction_info}

        # prepare data_store
        chunk_ids = list(chunk_to_rows.keys())
        chunk_triples = []
        chunk_triple_confidences = []
        for chunk_id in chunk_ids:
            info = extraction_info_by_id.get(chunk_id, {})
            canonical_triples = info.get('extracted_triples_canonical', None)
            if not canonical_triples:
                raw_triples = info.get('extracted_triples', []) or []
                canonical_triples = [canonicalize_triple(t) for t in raw_triples]
            chunk_triples.append(canonical_triples)

            confidences = info.get('extracted_triple_confidences', None)
            if not confidences or len(confidences) != len(canonical_triples):
                confidences = [1.0 for _ in canonical_triples]
            chunk_triple_confidences.append(confidences)

        entity_nodes, chunk_triple_entities = extract_entity_nodes(
            chunk_triples,
            include_predicates=self.global_config.include_predicate_nodes
        )

        fact_text_map, fact_confidence_map = self._build_fact_maps_from_extractions(all_extraction_info)
        fact_texts = list(fact_text_map.values())

        logger.info(f"Encoding Entities")
        self.entity_embedding_store.insert_strings(entity_nodes)

        logger.info(f"Encoding Facts")
        self.fact_embedding_store.insert_strings(fact_texts)
        self.fact_confidence_map = fact_confidence_map

        # build fact id -> triple mapping
        self.fact_id_to_triple = {}
        for triple_tuple, fact_text in fact_text_map.items():
            fact_id = self.fact_embedding_store.text_to_hash_id.get(fact_text)
            if fact_id:
                self.fact_id_to_triple[fact_id] = triple_tuple
        self.triple_to_fact_id = {v: k for k, v in self.fact_id_to_triple.items()}
        self.triple_to_fact_id = {v: k for k, v in self.fact_id_to_triple.items()}

        logger.info(f"Constructing Graph")

        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}

        self.add_fact_edges(chunk_ids, chunk_triples, chunk_triple_confidences)
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities, chunk_triple_confidences)

        if num_new_chunks > 0:
            logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
            self.add_synonymy_edges()

            self.augment_graph()
            self.save_igraph()

    def delete(self, docs_to_delete: List[str]):
        """
        Deletes the given documents from all data structures within the A2RAG class.
        Note that triples and entities which are indexed from chunks that are not being removed will not be removed.

        Parameters:
            docs : List[str]
                A list of documents to be deleted.
        """

        #Making sure that all the necessary structures have been built.
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        current_docs = set(self.chunk_embedding_store.get_all_texts())
        candidate_chunks = list(docs_to_delete)
        try:
            candidate_chunks.extend(self.text_preprocessor.preprocess_docs(docs_to_delete))
        except Exception as e:
            logger.warning(f"Could not preprocess documents for deletion; falling back to exact chunks: {e}")

        current_chunk_ids = set(self.chunk_embedding_store.get_all_ids())
        chunk_ids_to_delete = set()
        for candidate in candidate_chunks:
            if isinstance(candidate, dict):
                normalized = self.chunk_embedding_store._normalize_record(candidate)
                if normalized["hash_id"] in current_chunk_ids:
                    chunk_ids_to_delete.add(normalized["hash_id"])
                content = normalized["content"]
            else:
                content = candidate
            if content in current_docs:
                chunk_ids_to_delete.update(self.chunk_embedding_store.content_to_hash_ids.get(content, set()))

        if not chunk_ids_to_delete:
            logger.info("No matching chunks found to delete.")
            return

        self.delete_chunk_ids(list(chunk_ids_to_delete))

    def delete_chunk_ids(self, chunk_ids_to_delete: List[str]):
        """
        Deletes indexed chunks by their stable chunk identifiers.

        This is the preferred deletion path for document-level APIs because it
        does not depend on re-splitting source text exactly the same way.
        """

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        current_chunk_ids = set(self.chunk_embedding_store.get_all_ids())
        chunk_ids_to_delete = set(chunk_ids_to_delete).intersection(current_chunk_ids)

        if not chunk_ids_to_delete:
            logger.info("No matching chunks found to delete.")
            return

        #Find triples in chunks to delete
        all_extraction_info, chunk_keys_to_process = self.load_extraction_cache([])
        triples_to_delete = []

        all_extraction_info_with_deletes = []

        for extraction_doc in all_extraction_info:
            if extraction_doc['idx'] in chunk_ids_to_delete:
                triples_to_delete.append(extraction_doc['extracted_triples'])
            else:
                all_extraction_info_with_deletes.append(extraction_doc)

        triples_to_delete = flatten_facts(triples_to_delete)

        #Filter out triples that appear in unaltered chunks
        true_triples_to_delete = []

        for triple in triples_to_delete:
            proc_triple = tuple(canonicalize_triple(list(triple)))

            doc_ids = self.proc_triples_to_docs.get(str(proc_triple), set())

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                true_triples_to_delete.append(triple)

        processed_true_triples_to_delete = [[canonicalize_triple(list(triple)) for triple in true_triples_to_delete]]
        entities_to_delete, _ = extract_entity_nodes(processed_true_triples_to_delete)
        processed_true_triples_to_delete = flatten_facts(processed_true_triples_to_delete)

        triple_ids_to_delete = set()
        for triple in processed_true_triples_to_delete:
            triple = tuple(triple)
            fact_id = None
            if hasattr(self, "triple_to_fact_id"):
                fact_id = self.triple_to_fact_id.get(triple)
            if fact_id is None:
                # Fallback: try to find by formatted fact text without context
                fact_text = format_fact(list(triple), passage=None, max_context_tokens=None)
                fact_id = self.fact_embedding_store.text_to_hash_id.get(fact_text)
            if fact_id:
                triple_ids_to_delete.add(fact_id)

        #Filter out entities that appear in unaltered chunks
        ent_ids_to_delete = [
            self.entity_embedding_store.text_to_hash_id[ent]
            for ent in entities_to_delete
            if ent in self.entity_embedding_store.text_to_hash_id
        ]

        filtered_ent_ids_to_delete = []

        for ent_node in ent_ids_to_delete:
            doc_ids = self.ent_node_to_chunk_ids.get(ent_node, set())

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                filtered_ent_ids_to_delete.append(ent_node)

        logger.info(f"Deleting {len(chunk_ids_to_delete)} Chunks")
        logger.info(f"Deleting {len(triple_ids_to_delete)} Triples")
        logger.info(f"Deleting {len(filtered_ent_ids_to_delete)} Entities")

        self.save_extraction_cache(all_extraction_info_with_deletes)

        self.entity_embedding_store.delete(filtered_ent_ids_to_delete)
        self.fact_embedding_store.delete(triple_ids_to_delete)
        self.chunk_embedding_store.delete(chunk_ids_to_delete)

        #Delete Nodes from Graph
        vertices_to_delete = list(filtered_ent_ids_to_delete) + list(chunk_ids_to_delete)
        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
            vertices_to_delete = [v for v in vertices_to_delete if v in current_graph_nodes]
        if vertices_to_delete:
            self.graph.delete_vertices(vertices_to_delete)
        self.save_igraph()

        self.ent_node_to_chunk_ids = None
        self.ready_to_retrieve = False

    def retrieve(self,
                 queries: List[str],
                 num_to_retrieve: int = None,
                 gold_docs: List[List[str]] = None) -> List[QuerySolution]:
        """
        Performs retrieval using the A2RAG 2 framework, which consists of several steps:
        - Fact Retrieval
        - Recognition Memory for improved fact selection
        - Dense passage scoring
        - Personalized PageRank based re-ranking

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                No longer supported by the core package.

        Returns:
            List[QuerySolution]
                A list of QuerySolution objects, each containing the retrieved documents and their scores.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        if gold_docs is not None:
            raise ValueError("gold_docs is no longer supported by the core package.")

        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            rerank_start = time.time()
            query_fact_scores = self.get_fact_scores(query)
            top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(query, query_fact_scores)
            rerank_end = time.time()

            self.rerank_time += rerank_end - rerank_start

            if len(top_k_facts) == 0:
                logger.info('No facts found after reranking, return DPR results')
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
            else:
                sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(query=query,
                                                                                         link_top_k=self.global_config.linking_top_k,
                                                                                         query_fact_scores=query_fact_scores,
                                                                                         top_k_facts=top_k_facts,
                                                                                         top_k_fact_indices=top_k_fact_indices,
                                                                                         passage_node_weight=self.global_config.passage_node_weight)

            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in sorted_doc_ids[:num_to_retrieve]]

            retrieval_results.append(QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))

        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")
        logger.info(f"Total Recognition Memory Time {self.rerank_time:.2f}s")
        logger.info(f"Total PPR Time {self.ppr_time:.2f}s")
        logger.info(f"Total Misc Time {self.all_retrieval_time - (self.rerank_time + self.ppr_time):.2f}s")

        return retrieval_results

    def rag_qa(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Performs retrieval-augmented generation enhanced QA using the A2RAG 2 framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it runs retrieval first or answers from pre-computed query solutions.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): No longer supported.
            gold_answers (Optional[List[List[str]]]): No longer supported.

        Returns:
            Tuple[List[QuerySolution], List[str], List[Dict]]:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
        """
        if gold_answers is not None:
            raise ValueError("gold_answers is no longer supported by the core package.")
        if gold_docs is not None:
            raise ValueError("gold_docs is no longer supported by the core package.")

        if not isinstance(queries[0], QuerySolution):
            queries = self.retrieve(queries=queries)

        return self.qa(queries)

    def retrieve_dpr(self,
                     queries: List[str],
                     num_to_retrieve: int = None,
                     gold_docs: List[List[str]] = None) -> List[QuerySolution]:
        """
        Performs retrieval using a DPR framework, which consists of several steps:
        - Dense passage scoring

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                No longer supported by the core package.

        Returns:
            List[QuerySolution]
                A list of QuerySolution objects, each containing the retrieved documents and their scores.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        if gold_docs is not None:
            raise ValueError("gold_docs is no longer supported by the core package.")

        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            logger.info('No facts found after reranking, return DPR results')
            sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)

            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in
                          sorted_doc_ids[:num_to_retrieve]]

            retrieval_results.append(
                QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))

        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")

        return retrieval_results

    def rag_qa_dpr(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Performs retrieval-augmented generation enhanced QA using a standard DPR framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it runs retrieval first or answers from pre-computed query solutions.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): No longer supported.
            gold_answers (Optional[List[List[str]]]): No longer supported.

        Returns:
            Tuple[List[QuerySolution], List[str], List[Dict]]:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
        """
        if gold_answers is not None:
            raise ValueError("gold_answers is no longer supported by the core package.")
        if gold_docs is not None:
            raise ValueError("gold_docs is no longer supported by the core package.")

        if not isinstance(queries[0], QuerySolution):
            queries = self.retrieve_dpr(queries=queries)

        return self.qa(queries)

    def qa(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Executes question-answering (QA) inference using a provided set of query solutions and a language model.

        Parameters:
            queries: List[QuerySolution]
                A list of QuerySolution objects that contain the user queries, retrieved documents, and other related information.

        Returns:
            Tuple[List[QuerySolution], List[str], List[Dict]]
                A tuple containing:
                - A list of updated QuerySolution objects with the predicted answers embedded in them.
                - A list of raw response messages from the language model.
                - A list of metadata dictionaries associated with the results.
        """
        #Running inference for QA
        all_qa_messages = []

        for query_solution in tqdm(queries, desc="Collecting QA prompts"):

            # obtain the retrieved docs
            retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]

            if self.global_config.filter_chunk_with_chat_model:
                retrieved_passages = self.filter_chunks_with_chat_model(
                    question=query_solution.question,
                    passages=retrieved_passages
                )
                query_solution.filtered_docs = retrieved_passages

            prompt_user = ''
            for passage in retrieved_passages:
                prompt_user += f'Wikipedia Title: {passage}\n\n'
            prompt_user += 'Question: ' + query_solution.question + '\n'

            if self.prompt_template_manager.is_template_name_valid(name=f'rag_qa_{self.global_config.dataset}'):
                # find the corresponding prompt for this dataset
                prompt_dataset_name = self.global_config.dataset
            else:
                # the dataset does not have a customized prompt template yet
                logger.debug(
                    f"rag_qa_{self.global_config.dataset} does not have a customized prompt template. Using MUSIQUE's prompt template instead.")
                prompt_dataset_name = 'musique'
            all_qa_messages.append(
                self.prompt_template_manager.render(name=f'rag_qa_{prompt_dataset_name}', prompt_user=prompt_user))

        all_qa_results = [self.chat_model_client.infer(qa_messages) for qa_messages in tqdm(all_qa_messages, desc="QA Reading")]

        all_response_message, all_metadata, all_cache_hit = zip(*all_qa_results)
        all_response_message, all_metadata = list(all_response_message), list(all_metadata)

        #Process responses and extract predicted answers.
        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(enumerate(queries), desc="Extraction Answers from chat Response"):
            response_content = all_response_message[query_solution_idx]
            try:
                pred_ans = None
                for line in response_content.splitlines():
                    m = re.match(r"^\s*Answer\s*[:：]\s*(.*)$", line, re.IGNORECASE)
                    if m:
                        pred_ans = m.group(1).strip()
                        if not pred_ans:
                            continue
                        if "Evidence:" in pred_ans:
                            pred_ans = pred_ans.split("Evidence:")[0].strip()
                        break
                if pred_ans is None:
                    # Fallback: try to locate "Answer:" anywhere in the text
                    m = re.search(r"Answer\s*[:：]\s*(.+)", response_content, re.IGNORECASE)
                    pred_ans = m.group(1).strip() if m else response_content.strip()
            except Exception as e:
                logger.warning(f"Error in parsing the answer from the raw chat QA inference response: {str(e)}!")
                pred_ans = response_content

            query_solution.answer = pred_ans
            queries_solutions.append(query_solution)

        return queries_solutions, all_response_message, all_metadata

    def _get_filter_chat_client(self) -> BaseChatModel:
        if self._filter_chat_model_client is not None:
            return self._filter_chat_model_client

        if self.global_config.filter_chunk_model_name or self.global_config.filter_chunk_base_url:
            cfg = copy.deepcopy(self.global_config)
            if self.global_config.filter_chunk_model_name:
                cfg.chat_model = self.global_config.filter_chunk_model_name
            if self.global_config.filter_chunk_base_url:
                cfg.chat_base_url = self.global_config.filter_chunk_base_url
            self._filter_chat_model_client = get_chat_model_class(cfg)
        else:
            self._filter_chat_model_client = self.chat_model_client

        return self._filter_chat_model_client

    def _parse_filter_response(self, response_text: str) -> Optional[bool]:
        if not response_text:
            return None
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, dict) and "relevant" in parsed:
                return bool(parsed["relevant"])
        except Exception:
            pass

        match = re.search(r'\{[^{}]*"relevant"\s*:\s*(true|false)[^{}]*\}', response_text, re.IGNORECASE)
        if match:
            return match.group(1).lower() == "true"

        lowered = response_text.strip().lower()
        if lowered in ("yes", "y", "true", "relevant"):
            return True
        if lowered in ("no", "n", "false", "irrelevant"):
            return False
        return None

    def filter_chunks_with_chat_model(self, question: str, passages: List[str]) -> List[str]:
        if not passages:
            return []

        chat = self._get_filter_chat_client()
        max_workers = max(1, int(self.global_config.filter_chunk_max_workers or 1))
        min_keep = max(0, int(self.global_config.filter_chunk_min_keep or 0))
        strict = True
        q_preview = question.replace("\n", " ").strip()
        if len(q_preview) > 200:
            q_preview = q_preview[:200] + "..."
        logger.info(f"[Filter] Start filtering {len(passages)} chunks for question: {q_preview}")

        system_prompt = (
            "You are a strict retrieval filter. Mark relevant ONLY if the chunk directly "
            "states facts that answer the question or is essential evidence. If the chunk is "
            "generic, navigational, duplicated, or only loosely related, mark it irrelevant. "
            "Reply ONLY with JSON: {\"relevant\": true} or {\"relevant\": false}."
        )

        def _judge(passage: str) -> bool:
            user_prompt = (
                f"Question:\n{question}\n\n"
                f"Chunk:\n{passage}\n\n"
                "Is this chunk relevant to answering the question?"
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            try:
                response_text, _, _ = chat.infer(messages=messages, temperature=0.0, max_completion_tokens=32)
                decision = self._parse_filter_response(response_text)
                if decision is None:
                    return False if strict else True
                return decision
            except Exception as e:
                logger.warning(f"Filter chat failed, keep chunk: {e}")
                return True

        results: List[Tuple[int, bool]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_judge, passage): idx
                for idx, passage in enumerate(passages)
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Filtering chunks"):
                idx = futures[future]
                keep = True
                try:
                    keep = future.result()
                except Exception:
                    keep = True
                results.append((idx, keep))

        results.sort(key=lambda x: x[0])
        kept = [passages[idx] for idx, keep in results if keep]
        kept_indices = [idx for idx, keep in results if keep]

        if min_keep > 0 and len(kept) < min_keep:
            logger.info(f"[Filter] Kept {len(kept)}/{len(passages)} < min_keep={min_keep}, fallback to top-{min_keep}")
            kept = passages[:min_keep]
            kept_indices = list(range(min_keep))

        logger.info(f"[Filter] Kept {len(kept_indices)}/{len(passages)} chunks.")
        for idx in kept_indices:
            preview = passages[idx].replace("\n", " ").strip()
            if len(preview) > 160:
                preview = preview[:160] + "..."
            logger.info(f"[Filter] Kept chunk[{idx}]: {preview}")

        return kept

    def add_fact_edges(self,
                       chunk_ids: List[str],
                       chunk_triples: List[Tuple],
                       chunk_triple_confidences: Optional[List[List[float]]] = None):
        """
        Adds fact edges from given triples to the graph.

        The method processes chunks of triples, computes unique identifiers
        for entities and relations, and updates various internal statistics
        to build and maintain the graph structure. Entities are uniquely
        identified and linked based on their relationships.

        Parameters:
            chunk_ids: List[str]
                A list of unique identifiers for the chunks being processed.
            chunk_triples: List[Tuple]
                A list of tuples representing triples to process. Each triple
                consists of a subject, predicate, and object.

        Raises:
            Does not explicitly raise exceptions within the provided function logic.
        """

        if "name" in self.graph.vs:
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info(f"Adding KnowledgeExtractor triples to graph.")

        for idx, (chunk_key, triples) in enumerate(tqdm(zip(chunk_ids, chunk_triples))):
            entities_in_chunk = set()

            if chunk_key not in current_graph_nodes:
                confs = None
                if chunk_triple_confidences and idx < len(chunk_triple_confidences):
                    confs = chunk_triple_confidences[idx]

                for t_idx, triple in enumerate(triples):
                    triple = tuple(triple)
                    weight = 1.0
                    if confs and t_idx < len(confs):
                        weight = float(confs[t_idx])

                    node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                    node_2_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))

                    self.node_to_node_stats[(node_key, node_2_key)] = self.node_to_node_stats.get(
                        (node_key, node_2_key), 0.0) + weight
                    self.node_to_node_stats[(node_2_key, node_key)] = self.node_to_node_stats.get(
                        (node_2_key, node_key), 0.0) + weight

                    if self.global_config.include_predicate_nodes:
                        pred_key = compute_mdhash_id(content=triple[1], prefix=("entity-"))
                        self.node_to_node_stats[(node_key, pred_key)] = self.node_to_node_stats.get(
                            (node_key, pred_key), 0.0) + weight
                        self.node_to_node_stats[(pred_key, node_key)] = self.node_to_node_stats.get(
                            (pred_key, node_key), 0.0) + weight
                        self.node_to_node_stats[(pred_key, node_2_key)] = self.node_to_node_stats.get(
                            (pred_key, node_2_key), 0.0) + weight
                        self.node_to_node_stats[(node_2_key, pred_key)] = self.node_to_node_stats.get(
                            (node_2_key, pred_key), 0.0) + weight

                    entities_in_chunk.add(node_key)
                    entities_in_chunk.add(node_2_key)
                    if self.global_config.include_predicate_nodes:
                        entities_in_chunk.add(pred_key)

                for node in entities_in_chunk:
                    self.ent_node_to_chunk_ids[node] = self.ent_node_to_chunk_ids.get(node, set()).union(set([chunk_key]))

    def add_passage_edges(self,
                          chunk_ids: List[str],
                          chunk_triple_entities: List[List[str]],
                          chunk_triple_confidences: Optional[List[List[float]]] = None):
        """
        Adds edges connecting passage nodes to phrase nodes in the graph.

        This method is responsible for iterating through a list of chunk identifiers
        and their corresponding triple entities. It calculates and adds new edges
        between the passage nodes (defined by the chunk identifiers) and the phrase
        nodes (defined by the computed unique hash IDs of triple entities). The method
        also updates the node-to-node statistics map and keeps count of newly added
        passage nodes.

        Parameters:
            chunk_ids : List[str]
                A list of identifiers representing passage nodes in the graph.
            chunk_triple_entities : List[List[str]]
                A list of lists where each sublist contains entities (strings) associated
                with the corresponding chunk in the chunk_ids list.

        Returns:
            int
                The number of new passage nodes added to the graph.
        """

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_chunks = 0

        logger.info(f"Connecting passage nodes to phrase nodes.")

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):

            if chunk_key not in current_graph_nodes:
                ent_weights = {}
                if chunk_triple_confidences and idx < len(chunk_triple_confidences):
                    confs = chunk_triple_confidences[idx]
                    max_conf = max(confs) if confs else 1.0
                    for ent in chunk_triple_entities[idx]:
                        ent_weights[ent] = max_conf

                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")

                    weight = ent_weights.get(chunk_ent, 1.0)
                    self.node_to_node_stats[(chunk_key, node_key)] = weight

                num_new_chunks += 1

        return num_new_chunks

    def add_synonymy_edges(self):
        """
        Adds synonymy edges between similar nodes in the graph to enhance connectivity by identifying and linking synonym entities.

        This method performs key operations to compute and add synonymy edges. It first retrieves embeddings for all nodes, then conducts
        a nearest neighbor (KNN) search to find similar nodes. These similar nodes are identified based on a score threshold, and edges
        are added to represent the synonym relationship.

        Attributes:
            entity_id_to_row: dict (populated within the function). Maps each entity ID to its corresponding row data, where rows
                              contain `content` of entities used for comparison.
            entity_embedding_store: Manages retrieval of texts and embeddings for all rows related to entities.
            global_config: Configuration object that defines parameters such as `synonymy_edge_topk`, `synonymy_edge_sim_threshold`,
                           `synonymy_edge_query_batch_size`, and `synonymy_edge_key_batch_size`.
            node_to_node_stats: dict. Stores scores for edges between nodes representing their relationship.

        """
        logger.info(f"Expanding graph with synonymy edges")

        self.entity_id_to_row = self.entity_embedding_store.get_all_id_to_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.info(f"Performing KNN retrieval for each phrase nodes ({len(entity_node_keys)}).")

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        # Here we build synonymy edges only between newly inserted phrase nodes and all phrase nodes in the storage to reduce cost for incremental graph updates
        query_node_key2knn_node_keys = retrieve_knn(query_ids=entity_node_keys,
                                                    key_ids=entity_node_keys,
                                                    query_vecs=entity_embs,
                                                    key_vecs=entity_embs,
                                                    k=self.global_config.synonymy_edge_topk,
                                                    query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                                                    key_batch_size=self.global_config.synonymy_edge_key_batch_size)

        num_synonym_triple = 0
        synonym_candidates = []  # [(node key, [(synonym node key, corresponding score), ...]), ...]

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            synonyms = []

            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                        break

                    nn_phrase = self.entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != '':
                        sim_edge = (node_key, nn)
                        synonyms.append((nn, score))
                        num_synonym_triple += 1

                        self.node_to_node_stats[sim_edge] = score  # Need to seriously discuss on this
                        num_nns += 1

            synonym_candidates.append((node_key, synonyms))

    def load_extraction_cache(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:
        """
        Loads existing KnowledgeExtractor results from the specified file if it exists and combines
        them with new content while standardizing indices. If the file does not exist or
        is configured to be re-initialized from scratch with the flag `force_extraction_refresh`,
        it prepares new entries for processing.

        Args:
            chunk_keys (List[str]): A list of chunk keys that represent identifiers
                                     for the content to be processed.

        Returns:
            Tuple[List[dict], Set[str]]: A tuple where the first element is the existing KnowledgeExtractor
                                         information (if any) loaded from the file, and the
                                         second element is a set of chunk keys that still need to
                                         be saved or processed.
        """

        # combine extraction_results with contents already in file, if file exists
        chunk_keys_to_save = set()

        if not self.global_config.force_extraction_refresh and os.path.isfile(self.extraction_results_path):
            extraction_results = json.load(open(self.extraction_results_path))
            all_extraction_info = extraction_results.get('docs', [])

            #Standardizing indices for KnowledgeExtractor Files.

            renamed_extraction_info = []
            for info in all_extraction_info:
                if not info.get('idx'):
                    info['idx'] = compute_mdhash_id(info['passage'], 'chunk-')
                renamed_extraction_info.append(info)

            all_extraction_info = renamed_extraction_info

            existing_extraction_keys = set([info['idx'] for info in all_extraction_info])

            for chunk_key in chunk_keys:
                if chunk_key not in existing_extraction_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_extraction_info = []
            chunk_keys_to_save = chunk_keys

        return all_extraction_info, chunk_keys_to_save

    def _compute_triple_confidence(self,
                                   triple: List[str],
                                   passage: str,
                                   named_entities: List[str],
                                   metadata: Optional[Dict[str, Any]] = None) -> float:
        if not triple or len(triple) != 3:
            return 0.0

        s, p, o = (str(triple[0]), str(triple[1]), str(triple[2]))
        if not s.strip() or not p.strip() or not o.strip():
            return 0.0

        conf = 1.0
        if len(s.strip()) < 2 or len(o.strip()) < 2:
            conf *= 0.7
        if len(p.strip()) < 2:
            conf *= 0.8

        low_passage = (passage or "").lower()
        if s.lower() not in low_passage:
            conf *= 0.9
        if o.lower() not in low_passage:
            conf *= 0.9

        norm_ents = {normalize_entity(e) for e in (named_entities or [])}
        if normalize_entity(s) in norm_ents:
            conf *= 1.05
        if normalize_entity(o) in norm_ents:
            conf *= 1.05

        if metadata and metadata.get("finish_reason") == "length":
            conf *= 0.85

        return max(min(conf, 1.0), 0.05)

    def _build_fact_maps_from_extractions(self, all_extraction_info: List[dict]) -> Tuple[Dict[Tuple[str, str, str], str], Dict[str, float]]:
        fact_text_map: Dict[Tuple[str, str, str], str] = {}
        fact_confidence_map: Dict[str, float] = {}

        for doc in all_extraction_info:
            passage = doc.get("passage", "")
            raw_triples = doc.get("extracted_triples", []) or []
            canonical_triples = doc.get("extracted_triples_canonical", None)
            if not canonical_triples:
                canonical_triples = [canonicalize_triple(t) for t in raw_triples]
            confidences = doc.get("extracted_triple_confidences", None)
            if not confidences or len(confidences) != len(canonical_triples):
                confidences = [1.0 for _ in canonical_triples]

            for raw, canon, conf in zip(raw_triples, canonical_triples, confidences):
                if not canon or len(canon) != 3:
                    continue
                canon_tuple = tuple(canon)
                if canon_tuple not in fact_text_map:
                    fact_text_map[canon_tuple] = format_fact(
                        raw,
                        passage=passage,
                        max_context_tokens=self.global_config.fact_context_max_tokens,
                        encoder_name=self.global_config.preprocess_encoder_name,
                    )
                key = str(canon_tuple)
                fact_confidence_map[key] = max(conf, fact_confidence_map.get(key, 0.0))

        return fact_text_map, fact_confidence_map

    def merge_extraction_results(self,
                             all_extraction_info: List[dict],
                             chunks_to_save: Dict[str, dict],
                             ner_results_dict: Dict[str, NerRawOutput],
                             triple_results_dict: Dict[str, TripleRawOutput]) -> List[dict]:
        """
        Merges KnowledgeExtractor extraction results with corresponding passage and metadata.

        This function integrates the KnowledgeExtractor extraction results, including named-entity
        recognition (NER) entities and triples, with their respective text passages
        using the provided chunk keys. The resulting merged data is appended to
        the `all_extraction_info` list containing dictionaries with combined and organized
        data for further processing or storage.

        Parameters:
            all_extraction_info (List[dict]): A list to hold dictionaries of merged KnowledgeExtractor
                results and metadata for all chunks.
            chunks_to_save (Dict[str, dict]): A dict of chunk identifiers (keys) to process
                and merge KnowledgeExtractor results to dictionaries with `hash_id` and `content` keys.
            ner_results_dict (Dict[str, NerRawOutput]): A dictionary mapping chunk keys
                to their corresponding NER extraction results.
            triple_results_dict (Dict[str, TripleRawOutput]): A dictionary mapping chunk
                keys to their corresponding KnowledgeExtractor triple extraction results.

        Returns:
            List[dict]: The `all_extraction_info` list containing dictionaries with merged
            KnowledgeExtractor results, metadata, and the passage content for each chunk.

        """

        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            try:
                raw_entities = ner_results_dict[chunk_key].unique_entities
                raw_triples = triple_results_dict[chunk_key].triples

                all_entity_mentions = list(raw_entities)
                for t in raw_triples:
                    if len(t) == 3:
                        all_entity_mentions.extend([t[0], t[2]])
                        if self.global_config.include_predicate_nodes:
                            all_entity_mentions.append(t[1])

                canonical_entities, entity_aliases = build_canonical_entity_map(all_entity_mentions)
                # Filter obvious pronouns and empty strings
                canonical_entities = [
                    e for e in canonical_entities
                    if e and normalize_entity(e) not in self._PRONOUNS
                ]

                canonical_triples = []
                triple_confidences = []
                for t in raw_triples:
                    if len(t) != 3:
                        continue
                    canon = canonicalize_triple(t)
                    canonical_triples.append(canon)
                    triple_confidences.append(
                        self._compute_triple_confidence(
                            t,
                            passage=passage,
                            named_entities=raw_entities,
                            metadata=triple_results_dict[chunk_key].metadata,
                        )
                    )

                chunk_extraction_info = {
                    'idx': chunk_key,
                    'passage': passage,
                    'chunk_metadata': row.get('metadata', {}),
                    'extracted_entities': raw_entities,
                    'extracted_entities_canonical': canonical_entities,
                    'entity_aliases': entity_aliases,
                    'extracted_triples': raw_triples,
                    'extracted_triples_canonical': canonical_triples,
                    'extracted_triple_confidences': triple_confidences,
                }
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_key}: {e}")
                chunk_extraction_info = {'idx': chunk_key, 'passage': passage,
                                 'chunk_metadata': row.get('metadata', {}),
                                 'extracted_entities': [],
                                 'extracted_triples': [],
                                 'extracted_triples_canonical': [],
                                 'extracted_triple_confidences': []}
            all_extraction_info.append(chunk_extraction_info)

        return all_extraction_info

    def save_extraction_cache(self, all_extraction_info: List[dict]):
        """
        Computes statistics on extracted entities from KnowledgeExtractor results and saves the aggregated data in a
        JSON file. The function calculates the average character and word lengths of the extracted entities
        and writes them along with the provided KnowledgeExtractor information to a file.

        Parameters:
            all_extraction_info : List[dict]
                List of dictionaries, where each dictionary represents information from KnowledgeExtractor, including
                extracted entities.
        """

        sum_phrase_chars = sum([len(e) for chunk in all_extraction_info for e in chunk['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for chunk in all_extraction_info for e in chunk['extracted_entities']])
        num_phrases = sum([len(chunk['extracted_entities']) for chunk in all_extraction_info])

        if len(all_extraction_info) > 0:
            # Avoid division by zero if there are no phrases
            if num_phrases > 0:
                avg_ent_chars = round(sum_phrase_chars / num_phrases, 4)
                avg_ent_words = round(sum_phrase_words / num_phrases, 4)
            else:
                avg_ent_chars = 0
                avg_ent_words = 0
                
            extraction_dict = {
                'docs': all_extraction_info,
                'avg_ent_chars': avg_ent_chars,
                'avg_ent_words': avg_ent_words
            }
            
            with open(self.extraction_results_path, 'w') as f:
                json.dump(extraction_dict, f)
            logger.info(f"KnowledgeExtractor results saved to {self.extraction_results_path}")

    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """

        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"Graph construction completed!")
        print(self.get_graph_info())

    def add_new_nodes(self):
        """
        Adds new nodes to the graph from entity and passage embedding stores based on their attributes.

        This method identifies and adds new nodes to the graph by comparing existing nodes
        in the graph and nodes retrieved from the entity embedding store and the passage
        embedding store. The method checks attributes and ensures no duplicates are added.
        New nodes are prepared and added in bulk to optimize graph updates.
        """

        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        entity_to_row = self.entity_embedding_store.get_all_id_to_rows()
        passage_to_row = self.chunk_embedding_store.get_all_id_to_rows()

        node_to_rows = entity_to_row
        node_to_rows.update(passage_to_row)

        new_nodes = {}
        for node_id, node in node_to_rows.items():
            node['name'] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def add_new_edges(self):
        """
        Processes edges from `node_to_node_stats` to add them into a graph object while
        managing adjacency lists, validating edges, and logging invalid edge cases.
        """

        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]: continue
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({
                "weight": weight
            })

        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
        self.graph.add_edges(
            valid_edges,
            attributes=valid_weights
        )

    def save_igraph(self):
        logger.info(
            f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges"
        )
        self.graph.write_pickle(self._graph_pickle_filename)
        logger.info(f"Saving graph completed!")

    def get_graph_info(self) -> Dict:
        """
        Obtains detailed information about the graph such as the number of nodes,
        triples, and their classifications.

        This method calculates various statistics about the graph based on the
        stores and node-to-node relationships, including counts of phrase and
        passage nodes, total nodes, extracted triples, triples involving passage
        nodes, synonymy triples, and total triples.

        Returns:
            Dict
                A dictionary containing the following keys and their respective values:
                - num_phrase_nodes: The number of unique phrase nodes.
                - num_passage_nodes: The number of unique passage nodes.
                - num_total_nodes: The total number of nodes (sum of phrase and passage nodes).
                - num_extracted_triples: The number of unique extracted triples.
                - num_triples_with_passage_node: The number of triples involving at least one
                  passage node.
                - num_synonymy_triples: The number of synonymy triples (distinct from extracted
                  triples and those with passage nodes).
                - num_total_triples: The total number of triples.
        """
        graph_info = {}

        # get # of phrase nodes
        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        # get # of passage nodes
        passage_nodes_keys = self.chunk_embedding_store.get_all_ids()
        graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))

        # get # of total nodes
        graph_info["num_total_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"]

        # get # of extracted triples
        graph_info["num_extracted_triples"] = len(self.fact_embedding_store.get_all_ids())

        num_triples_with_passage_node = 0
        passage_nodes_set = set(passage_nodes_keys)
        num_triples_with_passage_node = sum(
            1 for node_pair in self.node_to_node_stats
            if node_pair[0] in passage_nodes_set or node_pair[1] in passage_nodes_set
        )
        graph_info['num_triples_with_passage_node'] = num_triples_with_passage_node

        graph_info['num_synonymy_triples'] = len(self.node_to_node_stats) - graph_info[
            "num_extracted_triples"] - num_triples_with_passage_node

        # get # of total triples
        graph_info["num_total_triples"] = len(self.node_to_node_stats)

        return graph_info

    def prepare_retrieval_objects(self):
        """
        Prepares various in-memory objects and attributes necessary for fast retrieval processes, such as embedding data and graph relationships, ensuring consistency
        and alignment with the underlying graph structure.
        """

        logger.info("Preparing for fast retrieval.")

        logger.info("Loading keys.")
        self.query_to_embedding: Dict = {'triple': {}, 'passage': {}}

        self.entity_node_keys: List = list(self.entity_embedding_store.get_all_ids()) # a list of phrase node keys
        self.passage_node_keys: List = list(self.chunk_embedding_store.get_all_ids()) # a list of passage node keys
        self.fact_node_keys: List = list(self.fact_embedding_store.get_all_ids())

        # Check if the graph has the expected number of nodes
        expected_node_count = len(self.entity_node_keys) + len(self.passage_node_keys)
        actual_node_count = self.graph.vcount()
        
        if expected_node_count != actual_node_count:
            logger.warning(f"Graph node count mismatch: expected {expected_node_count}, got {actual_node_count}")
            # If the graph is empty but we have nodes, we need to add them
            if actual_node_count == 0 and expected_node_count > 0:
                logger.info(f"Initializing graph with {expected_node_count} nodes")
                self.add_new_nodes()
                self.save_igraph()

        # Create mapping from node name to vertex index
        try:
            igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)} # from node key to the index in the backbone graph
            self.node_name_to_vertex_idx = igraph_name_to_idx
            
            # Check if all entity and passage nodes are in the graph
            missing_entity_nodes = [node_key for node_key in self.entity_node_keys if node_key not in igraph_name_to_idx]
            missing_passage_nodes = [node_key for node_key in self.passage_node_keys if node_key not in igraph_name_to_idx]
            
            if missing_entity_nodes or missing_passage_nodes:
                logger.warning(f"Missing nodes in graph: {len(missing_entity_nodes)} entity nodes, {len(missing_passage_nodes)} passage nodes")
                # If nodes are missing, rebuild the graph
                self.add_new_nodes()
                self.save_igraph()
                # Update the mapping
                igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
                self.node_name_to_vertex_idx = igraph_name_to_idx
            
            self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys] # a list of backbone graph node index
            self.passage_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.passage_node_keys] # a list of backbone passage node index
        except Exception as e:
            logger.error(f"Error creating node index mapping: {str(e)}")
            # Initialize with empty lists if mapping fails
            self.node_name_to_vertex_idx = {}
            self.entity_node_idxs = []
            self.passage_node_idxs = []

        logger.info("Loading embeddings.")
        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))

        self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))

        all_extraction_info, chunk_keys_to_process = self.load_extraction_cache([])

        self.proc_triples_to_docs = {}

        for doc in all_extraction_info:
            raw_triples = doc.get('extracted_triples', []) or []
            canonical_triples = doc.get('extracted_triples_canonical', None)
            if not canonical_triples:
                canonical_triples = [canonicalize_triple(t) for t in raw_triples]
            triples = flatten_facts([canonical_triples])
            for triple in triples:
                if len(triple) == 3:
                    proc_triple = tuple(triple)
                    self.proc_triples_to_docs[str(proc_triple)] = self.proc_triples_to_docs.get(str(proc_triple), set()).union(set([doc['idx']]))

        fact_text_map, fact_confidence_map = self._build_fact_maps_from_extractions(all_extraction_info)
        self.fact_confidence_map = fact_confidence_map
        self.fact_id_to_triple = {}
        for triple_tuple, fact_text in fact_text_map.items():
            fact_id = self.fact_embedding_store.text_to_hash_id.get(fact_text)
            if fact_id:
                self.fact_id_to_triple[fact_id] = triple_tuple

        if self.ent_node_to_chunk_ids is None:
            # Build canonical triples and confidences for graph reconstruction
            extraction_info_by_id = {info['idx']: info for info in all_extraction_info}
            chunk_triples = []
            chunk_triple_confidences = []
            for chunk_id in self.passage_node_keys:
                info = extraction_info_by_id.get(chunk_id, {})
                canonical_triples = info.get('extracted_triples_canonical', None)
                if not canonical_triples:
                    raw_triples = info.get('extracted_triples', []) or []
                    canonical_triples = [canonicalize_triple(t) for t in raw_triples]
                chunk_triples.append(canonical_triples)

                confidences = info.get('extracted_triple_confidences', None)
                if not confidences or len(confidences) != len(canonical_triples):
                    confidences = [1.0 for _ in canonical_triples]
                chunk_triple_confidences.append(confidences)

            self.node_to_node_stats = {}
            self.ent_node_to_chunk_ids = {}
            self.add_fact_edges(self.passage_node_keys, chunk_triples, chunk_triple_confidences)

        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping. The method determines whether each query
        is already present in the `self.query_to_embedding` dictionary under the keys 'triple' and 'passage'. If a query is not present in
        either, it is encoded into embeddings using the vector service and stored.

        Args:
            queries List[str] | List[QuerySolution]: A list of query strings or QuerySolution objects. Each query is checked for
            its presence in the query-to-embedding mappings.
        """

        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                    query.question not in self.query_to_embedding['triple'] or query.question not in
                    self.query_to_embedding['passage']):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['triple'] or query not in self.query_to_embedding['passage']:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            # get all query embeddings
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_fact.")
            query_embeddings_for_triple = self.embedding_model.batch_encode(all_query_strings,
                                                                            instruction=get_query_instruction('query_to_fact'),
                                                                            norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding

            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
                                                                             instruction=get_query_instruction('query_to_passage'),
                                                                             norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

    def get_fact_scores(self, query: str) -> np.ndarray:
        """
        Retrieves and computes normalized similarity scores between the given query and pre-stored fact embeddings.

        Parameters:
        query : str
            The input query text for which similarity scores with fact embeddings
            need to be computed.

        Returns:
        numpy.ndarray
            A normalized array of similarity scores between the query and fact
            embeddings. The shape of the array is determined by the number of
            facts.

        Raises:
        KeyError
            If no embedding is found for the provided query in the stored query
            embeddings dictionary.
        """
        query_embedding = self.query_to_embedding['triple'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_fact'),
                                                                norm=True)

        # Check if there are any facts
        if len(self.fact_embeddings) == 0:
            logger.warning("No facts available for scoring. Returning empty array.")
            return np.array([])
            
        try:
            query_fact_scores = np.dot(self.fact_embeddings, query_embedding.T) # shape: (#facts, )
            query_fact_scores = np.squeeze(query_fact_scores) if query_fact_scores.ndim == 2 else query_fact_scores
            query_fact_scores = min_max_normalize(query_fact_scores)
            return query_fact_scores
        except Exception as e:
            logger.error(f"Error computing fact scores: {str(e)}")
            return np.array([])

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conduct dense passage retrieval to find relevant documents for a query.

        This function processes a given query using a pre-trained vector service
        to generate query embeddings. The similarity scores between the query
        embedding and passage embeddings are computed using dot product, followed
        by score normalization. Finally, the function ranks the documents based
        on their similarity scores and returns the ranked document identifiers
        and their scores.

        Parameters
        ----------
        query : str
            The input query for which relevant passages should be retrieved.

        Returns
        -------
        tuple : Tuple[np.ndarray, np.ndarray]
            A tuple containing two elements:
            - A list of sorted document identifiers based on their relevance scores.
            - A numpy array of the normalized similarity scores for the corresponding
              documents.
        """
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_passage'),
                                                                norm=True)
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores


    def get_top_k_weights(self,
                          link_top_k: int,
                          all_phrase_weights: np.ndarray,
                          linking_score_map: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        This function filters the all_phrase_weights to retain only the weights for the
        top-ranked phrases in terms of the linking_score_map. It also filters linking scores
        to retain only the top `link_top_k` ranked nodes. Non-selected phrases in phrase
        weights are reset to a weight of 0.0.

        Args:
            link_top_k (int): Number of top-ranked nodes to retain in the linking score map.
            all_phrase_weights (np.ndarray): An array representing the phrase weights, indexed
                by phrase ID.
            linking_score_map (Dict[str, float]): A mapping of phrase content to its linking
                score, sorted in descending order of scores.

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: A tuple containing the filtered array
            of all_phrase_weights with unselected weights set to 0.0, and the filtered
            linking_score_map containing only the top `link_top_k` phrases.
        """
        # choose top ranked phrase nodes that are actually present in the graph.
        # The original implementation asserted that the number of retained phrase
        # strings exactly matched the number of non-zero graph weights. In practice
        # extraction can produce duplicate/canonicalized entities, zero confidence
        # weights, or phrases that were not materialized as graph vertices. Those
        # cases should not crash KG/PageRank retrieval; they should simply be
        # filtered out before PPR.
        filtered_linking_score_map = {}
        top_k_phrase_keys = set()
        for phrase, score in sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True):
            if len(filtered_linking_score_map) >= link_top_k:
                break
            phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
            phrase_id = self.node_name_to_vertex_idx.get(phrase_key)
            if phrase_id is None:
                continue
            if phrase_id >= len(all_phrase_weights) or all_phrase_weights[phrase_id] <= 0:
                continue
            filtered_linking_score_map[phrase] = float(score)
            top_k_phrase_keys.add(phrase_key)

        for phrase_key, phrase_id in self.node_name_to_vertex_idx.items():
            if phrase_key not in top_k_phrase_keys and phrase_id < len(all_phrase_weights):
                all_phrase_weights[phrase_id] = 0.0

        nonzero_count = int(np.count_nonzero(all_phrase_weights))
        if nonzero_count != len(filtered_linking_score_map):
            logger.warning(
                "Top-k phrase weight mismatch after filtering: nonzero=%d linked_phrases=%d",
                nonzero_count,
                len(filtered_linking_score_map),
            )
        return all_phrase_weights, filtered_linking_score_map

    def graph_search_with_fact_entities(self, query: str,
                                        link_top_k: int,
                                        query_fact_scores: np.ndarray,
                                        top_k_facts: List[Tuple],
                                        top_k_fact_indices: List[str],
                                        passage_node_weight: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes document scores based on fact-based similarity and relevance using personalized
        PageRank (PPR) and dense retrieval models. This function combines the signal from the relevant
        facts identified with passage similarity and graph-based search for enhanced result ranking.

        Parameters:
            query (str): The input query string for which similarity and relevance computations
                need to be performed.
            link_top_k (int): The number of top phrases to include from the linking score map for
                downstream processing.
            query_fact_scores (np.ndarray): An array of scores representing fact-query similarity
                for each of the provided facts.
            top_k_facts (List[Tuple]): A list of top-ranked facts, where each fact is represented
                as a tuple of its subject, predicate, and object.
            top_k_fact_indices (List[str]): Corresponding indices or identifiers for the top-ranked
                facts in the query_fact_scores array.
            passage_node_weight (float): Default weight to scale passage scores in the graph.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - The first array corresponds to document IDs sorted based on their scores.
                - The second array consists of the PPR scores associated with the sorted document IDs.
        """

        #Assigning phrase weights based on selected facts from previous steps.
        linking_score_map = {}  # from phrase to the average scores of the facts that contain the phrase
        phrase_scores = {}  # store all fact scores for each phrase regardless of whether they exist in the knowledge graph or not
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))
        number_of_occurs = np.zeros(len(self.graph.vs['name']))

        phrases_and_ids = set()

        for rank, f in enumerate(top_k_facts):
            subject_phrase = str(f[0]).lower()
            predicate_phrase = str(f[1]).lower()
            object_phrase = str(f[2]).lower()
            fact_score = query_fact_scores[
                top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores
            canon = canonicalize_triple([subject_phrase, predicate_phrase, object_phrase])
            conf = 1.0
            if hasattr(self, "fact_confidence_map"):
                conf = self.fact_confidence_map.get(str(tuple(canon)), 1.0)

            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(
                    content=phrase,
                    prefix="entity-"
                )
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    weighted_fact_score = fact_score * conf

                    if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        weighted_fact_score /= len(self.ent_node_to_chunk_ids[phrase_key])

                    phrase_weights[phrase_id] += weighted_fact_score
                    number_of_occurs[phrase_id] += 1

                phrases_and_ids.add((phrase, phrase_id))

            if self.global_config.include_predicate_nodes:
                pred_key = compute_mdhash_id(
                    content=predicate_phrase,
                    prefix="entity-"
                )
                pred_id = self.node_name_to_vertex_idx.get(pred_key, None)
                if pred_id is not None:
                    weighted_fact_score = fact_score * conf * self.global_config.predicate_weight
                    if len(self.ent_node_to_chunk_ids.get(pred_key, set())) > 0:
                        weighted_fact_score /= len(self.ent_node_to_chunk_ids[pred_key])
                    phrase_weights[pred_id] += weighted_fact_score
                    number_of_occurs[pred_id] += 1
                phrases_and_ids.add((predicate_phrase, pred_id))

        phrase_weights = np.divide(
            phrase_weights,
            np.where(number_of_occurs == 0, 1.0, number_of_occurs)
        )

        for phrase, phrase_id in phrases_and_ids:
            if phrase_id is None:
                continue
            if phrase not in phrase_scores:
                phrase_scores[phrase] = []

            phrase_scores[phrase].append(phrase_weights[phrase_id])

        # calculate average fact score for each phrase
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k,
                                                                           phrase_weights,
                                                                           linking_score_map)  # at this stage, the length of linking_scope_map is determined by link_top_k

        #Get passage scores according to chosen dense retrieval model
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)["content"]
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight

        #Combining phrase and passage scores into one array for PPR
        node_weights = phrase_weights + passage_weights

        #Recording top 30 facts in linking_score_map
        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        assert sum(node_weights) > 0, f'No phrases found in the graph for the given facts: {top_k_facts}'

        #Running PPR algorithm based on the passage and phrase weights previously assigned
        ppr_start = time.time()
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=self.global_config.damping)
        ppr_end = time.time()

        self.ppr_time += (ppr_end - ppr_start)

        assert len(ppr_sorted_doc_ids) == len(
            self.passage_node_idxs), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.passage_node_idxs)}"

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores


    def rerank_facts(self, query: str, query_fact_scores: np.ndarray) -> Tuple[List[int], List[Tuple], dict]:
        """

        Args:

        Returns:
            top_k_fact_indicies:
            top_k_facts:
            rerank_log (dict): {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
                - candidate_facts (list): list of link_top_k facts (each fact is a relation triple in tuple data type).
                - top_k_facts:


        """
        # load args
        link_top_k: int = self.global_config.linking_top_k
        
        # Check if there are any facts to rerank
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}
            
        try:
            # Get the top k facts by score
            if len(query_fact_scores) <= link_top_k:
                # If we have fewer facts than requested, use all of them
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:
                # Otherwise get the top k
                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
                
            # Get the actual fact IDs
            real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            candidate_facts = []
            for fid in real_candidate_fact_ids:
                triple = None
                if hasattr(self, "fact_id_to_triple") and fid in self.fact_id_to_triple:
                    triple = self.fact_id_to_triple[fid]
                if triple is None:
                    fact_text = fact_row_dict[fid]["content"]
                    parsed = parse_fact_text(fact_text)
                    if parsed:
                        triple = parsed
                if triple is None:
                    try:
                        triple = ast.literal_eval(fact_row_dict[fid]["content"])
                    except Exception:
                        triple = ("", "", "")
                candidate_facts.append(tuple(triple))
            
            # Rerank the facts
            top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter(query,
                                                                                candidate_facts,
                                                                                candidate_fact_indices,
                                                                                len_after_rerank=link_top_k)
            
            rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
            
            return top_k_fact_indices, top_k_facts, rerank_log
            
        except Exception as e:
            logger.error(f"Error in rerank_facts: {str(e)}")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': [], 'error': str(e)}
    
    def run_ppr(self,
                reset_prob: np.ndarray,
                damping: float =0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Personalized PageRank (PPR) on a graph and computes relevance scores for
        nodes corresponding to document passages. The method utilizes a damping
        factor for teleportation during rank computation and can take a reset
        probability array to influence the starting state of the computation.

        Parameters:
            reset_prob (np.ndarray): A 1-dimensional array specifying the reset
                probability distribution for each node. The array must have a size
                equal to the number of nodes in the graph. NaNs or negative values
                within the array are replaced with zeros.
            damping (float): A scalar specifying the damping factor for the
                computation. Defaults to 0.5 if not provided or set to `None`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The
                first array represents the sorted node IDs of document passages based
                on their relevance scores in descending order. The second array
                contains the corresponding relevance scores of each document passage
                in the same order.
        """

        if damping is None: damping = 0.5 # for potential compatibility
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores
