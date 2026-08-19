import os
from dataclasses import dataclass, field
from typing import (
    Literal,
    Union,
    Optional
)

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BaseConfig:
    """One and only configuration."""
    # chat specific attributes 
    chat_model: str = field(
        default="qwen2.5-1.5b-instruct-local",
        metadata={"help": "Class name indicating which chat model to use."}
    )
    chat_base_url: str = field(
        default=None,
        metadata={"help": "Base URL for the chat gateway."}
    )
    embedding_base_url: str = field(
        default=None,
        metadata={"help": "Base URL for the embedding gateway."}
    )
    max_new_tokens: Union[None, int] = field(
        default=2048,
        metadata={"help": "Max new tokens to generate in each inference."}
    )
    num_gen_choices: int = field(
        default=1,
        metadata={"help": "How many chat completion choices to generate for each input message."}
    )
    seed: Union[None, int] = field(
        default=None,
        metadata={"help": "Random seed."}
    )
    temperature: float = field(
        default=0,
        metadata={"help": "Temperature for sampling in each inference."}
    )
    response_format: Union[dict, None] = field(
        default_factory=lambda: { "type": "json_object" },
        metadata={"help": "Specifying the format that the model must output."}
    )
    
    ## chat specific attributes -> Async hyperparameters
    max_retry_attempts: int = field(
        default=5,
        metadata={"help": "Max number of retry attempts for an asynchronous API calling."}
    )
    # Storage specific attributes
    force_extraction_refresh: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing extraction files and rebuild them from scratch."}
    )

    # Storage specific attributes 
    force_index_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing storage files and graph data and will rebuild from scratch."}
    )
    rerank_dspy_file_path: str = field(
        default=None,
        metadata={"help": "Path to the rerank dspy file."}
    )
    passage_node_weight: float = field(
        default=0.05,
        metadata={"help": "Multiplicative factor that modified the passage node weights in PPR."}
    )
    save_extraction_results: bool = field(
        default=True,
        metadata={"help": "If set to True, will save the KnowledgeExtractor model to disk."}
    )
    extraction_results_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path for cached knowledge extraction results."}
    )
    
    # Preprocessing specific attributes
    text_preprocessor_class_name: str = field(
        default="TextPreprocessor",
        metadata={"help": "Deprecated. Use chunking_mode instead."}
    )
    chunking_mode: Literal["text", "markdown"] = field(
        default="markdown",
        metadata={"help": "Document chunking mode. Use 'text' for semantic plain-text chunking or 'markdown' for heading-aware Markdown chunking."}
    )
    chunk_min_token_size: int = field(
        default=800,
        metadata={"help": "Preferred minimum token size for chunks when neighbouring chunks can be merged safely."}
    )
    chunk_target_token_size: int = field(
        default=1000,
        metadata={"help": "Target token size for chunks."}
    )
    chunk_max_token_size: int = field(
        default=1200,
        metadata={"help": "Hard maximum token size for chunks."}
    )
    chunk_overlap_token_size: int = field(
        default=100,
        metadata={"help": "Token overlap for token-bounded fallback splitting."}
    )
    chunk_semantic_threshold: float = field(
        default=0.8,
        metadata={"help": "Semantic similarity threshold for text-mode semantic chunking."}
    )
    preprocess_encoder_name: str = field(
        default="gpt-4o",
        metadata={"help": "Name of the encoder to use in preprocessing (currently implemented specifically for doc chunking)."}
    )
    preprocess_chunk_overlap_token_size: int = field(
        default=128,
        metadata={"help": "Number of overlap tokens between neighbouring chunks."}
    )
    preprocess_chunk_min_token_size: Optional[int] = field(
        default=None,
        metadata={"help": "Minimum token length for a chunk; smaller chunks may be merged when possible."}
    )
    preprocess_chunk_max_token_size: int = field(
        default=None,
        metadata={"help": "Max number of tokens each chunk can contain. If set to None, the whole doc will treated as a single chunk."}
    )
    preprocess_chunk_func: Literal["by_token", "by_word"] = field(default='by_token')

    preprocess_split_markdown: bool = field(
        default=False,
        metadata={"help": "If set to True, split markdown docs into chunks using the split-markdown algorithm."}
    )
    preprocess_markdown_min_len: int = field(
        default=1500,
        metadata={"help": "Minimum char length before emitting a markdown chunk."}
    )
    preprocess_markdown_max_len: int = field(
        default=2000,
        metadata={"help": "Maximum char length for a markdown chunk before recursive splitting."}
    )
    preprocess_markdown_overlap_chars: int = field(
        default=0,
        metadata={"help": "Number of overlapping characters to prepend to each subsequent markdown chunk."}
    )
    preprocess_markdown_use_result: bool = field(
        default=True,
        metadata={"help": "If True, embed the markdown 'result' (summary + content); if False, embed content only."}
    )

    preprocess_markdown_min_token_size: Optional[int] = field(
        default=None,
        metadata={"help": "Minimum token length for markdown chunks when using token-based splitters."}
    )
    preprocess_markdown_max_token_size: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum token length for markdown chunks when using token-based splitters."}
    )
    preprocess_markdown_overlap_token_size: Optional[int] = field(
        default=None,
        metadata={"help": "Token overlap between markdown chunks when using token-based splitters."}
    )
    preprocess_markdown_include_heading_path: bool = field(
        default=True,
        metadata={"help": "Include heading path prefixes in markdown chunks when using token-based splitters."}
    )
    
    
    # Knowledge extraction specific attributes
    knowledge_extraction_model_name: Literal["chat_knowledge_extractor", ] = field(
        default="chat_knowledge_extractor",
        metadata={"help": "Class name indicating which knowledge extraction model to use."}
    )
    knowledge_extraction_mode: Literal["offline", "online"] = field(
        default="online",
        metadata={"help": "Mode of the knowledge extraction model to use."}
    )
    skip_graph: bool = field(
        default=False,
        metadata={"help": "Whether to skip graph construction."}
    )
    
    
    # Embedding specific attributes
    embedding_provider: Literal["gateway", "local", "sentence_transformers"] = field(
        default="gateway",
        metadata={"help": "Embedding backend provider. Use 'gateway' for OpenAI-compatible HTTP, or 'local' for sentence-transformers."}
    )
    embedding_model: str = field(
        default="BAAI/bge-m3",
        metadata={"help": "Class name indicating which vector service to use."}
    )
    embedding_batch_size: int = field(
        default=10,
        metadata={"help": "Batch size of calling vector service."}
    )
    embedding_return_as_normalized: bool = field(
        default=True,
        metadata={"help": "Whether to normalize encoded embeddings not."}
    )
    embedding_max_seq_len: int = field(
        default=2048,
        metadata={"help": "Max sequence length for the vector service."}
    )
    embedding_model_dtype: Literal["float16", "float32", "bfloat16", "auto"] = field(
        default="auto",
        metadata={"help": "Data type for local vector service."}
    )
    embedding_device: str = field(
        default="auto",
        metadata={"help": "Device for local vector service, for example auto, cuda, cuda:0, or cpu."}
    )
    embedding_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional cache directory for local embedding model weights."}
    )
    
    
    
    # Graph construction specific attributes
    synonymy_edge_topk: int = field(
        default=2047,
        metadata={"help": "k for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_query_batch_size: int = field(
        default=1000,
        metadata={"help": "Batch size for query embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_key_batch_size: int = field(
        default=10000,
        metadata={"help": "Batch size for key embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_sim_threshold: float = field(
        default=0.8,
        metadata={"help": "Similarity threshold to include candidate synonymy nodes."}
    )
    is_directed_graph: bool = field(
        default=False,
        metadata={"help": "Whether the graph is directed or not."}
    )

    include_predicate_nodes: bool = field(
        default=True,
        metadata={"help": "Whether to include predicate nodes in the graph and PPR seeding."}
    )
    predicate_weight: float = field(
        default=0.7,
        metadata={"help": "Relative weight of predicates when seeding PPR."}
    )
    
    
    
    # Retrieval specific attributes
    linking_top_k: int = field(
        default=5,
        metadata={"help": "The number of linked nodes at each retrieval step"}
    )
    retrieval_top_k: int = field(
        default=200,
        metadata={"help": "Retrieving k documents at each step"}
    )
    damping: float = field(
        default=0.5,
        metadata={"help": "Damping factor for ppr algorithm."}
    )
    
    
    # QA specific attributes
    max_qa_steps: int = field(
        default=1,
        metadata={"help": "For answering a single question, the max steps that we use to interleave retrieval and reasoning."}
    )
    qa_top_k: int = field(
        default=5,
        metadata={"help": "Feeding top k documents to the QA model for reading."}
    )

    fact_context_max_tokens: Optional[int] = field(
        default=64,
        metadata={"help": "Max tokens of passage context to append when formatting facts for embedding."}
    )

    filter_chunk_with_chat_model: bool = field(
        default=False,
        metadata={"help": "Whether to filter top-k chunks with an chat before QA."}
    )
    filter_chunk_model_name: Optional[str] = field(
        default=None,
        metadata={"help": "Optional model name for chunk filtering. Defaults to chat_model."}
    )
    filter_chunk_base_url: Optional[str] = field(
        default=None,
        metadata={"help": "Optional base URL for chunk filtering model."}
    )
    filter_chunk_max_workers: int = field(
        default=8,
        metadata={"help": "Max parallel workers for chat chunk filtering."}
    )
    filter_chunk_min_keep: int = field(
        default=2,
        metadata={"help": "Minimum number of chunks to keep after filtering."}
    )
    
    # Save dir (highest level directory)
    save_dir: str = field(
        default=None,
        metadata={"help": "Directory to save all related information. If it's given, will overwrite all default save_dir setups. If it's not given, then if we're not running specific datasets, default to `outputs`, otherwise, default to a dataset-customized output dir."}
    )
    
    
    
    # Dataset running specific attributes
    ## Dataset running specific attributes -> General
    dataset: Optional[Literal['hotpotqa', 'hotpotqa_train', 'musique', '2wikimultihopqa']] = field(
        default=None,
        metadata={"help": "Dataset to use. If specified, it means we will run specific datasets. If not specified, it means we're running freely."}
    )
    ## Dataset running specific attributes -> Graph
    graph_type: Literal[
        'dpr_only', 
        'entity', 
        'passage_entity', 'relation_aware_passage_entity',
        'passage_entity_relation', 
        'facts_and_sim_passage_node_unidirectional',
    ] = field(
        default="facts_and_sim_passage_node_unidirectional",
        metadata={"help": "Type of graph to use in the experiment."}
    )
    corpus_len: Optional[int] = field(
        default=None,
        metadata={"help": "Length of the corpus to use."}
    )
    
    
    def __post_init__(self):
        if self.save_dir is None: # If save_dir not given
            if self.dataset is None: self.save_dir = 'outputs' # running freely
            else: self.save_dir = os.path.join('outputs', self.dataset) # customize your dataset's output dir here
        logger.debug(f"Initializing the highest level of save_dir to be {self.save_dir}")
