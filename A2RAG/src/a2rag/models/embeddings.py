import os

from ..providers.embedding_gateway import EmbeddingGateway
from ..providers.local_embedding import LocalSentenceTransformerEmbedding
from .embedding_base import BaseEmbeddingModel, EmbeddingConfig

__all__ = ["EmbeddingConfig", "BaseEmbeddingModel", "get_embedding_model_class"]


def get_embedding_model_class(embedding_model: str = "BAAI/bge-m3"):
    provider = (os.getenv("A2RAG_EMBEDDING_PROVIDER") or "").strip().lower()
    local_flag = (os.getenv("A2RAG_LOCAL_EMBEDDING") or "").strip().lower()
    if provider in {"local", "sentence-transformers", "sentence_transformers"} or local_flag in {"1", "true", "yes", "y"}:
        return LocalSentenceTransformerEmbedding
    return EmbeddingGateway
