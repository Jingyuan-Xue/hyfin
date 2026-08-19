from copy import deepcopy
import os
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from ..config.settings import BaseConfig
from ..models.embedding_base import BaseEmbeddingModel, EmbeddingConfig
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class LocalSentenceTransformerEmbedding(BaseEmbeddingModel):
    """Local sentence-transformers embedding backend."""

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)
        if embedding_model is not None:
            self.embedding_model = embedding_model
            logger.debug(f"Overriding {self.__class__.__name__}'s embedding_model with: {self.embedding_model}")

        self.device = self._resolve_device()
        self.cache_folder = (
            self.global_config.embedding_cache_dir
            or os.getenv("A2RAG_EMBEDDING_CACHE_DIR")
            or None
        )
        self._model = None
        self.embedding_dim = 0
        self._init_embedding_config()

    def _resolve_device(self) -> str:
        requested = (self.global_config.embedding_device or os.getenv("A2RAG_EMBEDDING_DEVICE") or "auto").strip()
        if requested and requested.lower() != "auto":
            return requested
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _init_embedding_config(self) -> None:
        model_kwargs = {"trust_remote_code": True}
        dtype = self.global_config.embedding_model_dtype
        if dtype and dtype != "auto":
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }
            model_kwargs["torch_dtype"] = dtype_map[dtype]

        config_dict = {
            "embedding_model": self.embedding_model,
            "norm": self.global_config.embedding_return_as_normalized,
            "model_init_params": {
                "pretrained_model_name_or_path": self.embedding_model,
                "device": self.device,
                "cache_folder": self.cache_folder,
                "model_kwargs": model_kwargs,
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "show_progress_bar": False,
            },
        }
        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.info(
            "Init local embedding model=%s device=%s cache=%s",
            self.embedding_model,
            self.device,
            self.cache_folder,
        )

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Run scripts/setup_local_bge_m3_env.sh or install sentence-transformers."
            ) from exc

        init_params = deepcopy(self.embedding_config.model_init_params)
        init_params = {k: v for k, v in init_params.items() if v is not None}
        model_name = init_params.pop("pretrained_model_name_or_path")
        self._model = SentenceTransformer(model_name, **init_params)
        self._model.max_seq_length = self.global_config.embedding_max_seq_len
        if hasattr(self._model, "get_embedding_dimension"):
            self.embedding_dim = int(self._model.get_embedding_dimension() or 0)
        else:
            self.embedding_dim = int(self._model.get_sentence_embedding_dimension() or 0)
        return self._model

    def encode(self, texts: List[str], **kwargs):
        model = self._load_model()
        texts = [t.replace("\n", " ") or " " for t in texts]
        reserved = {"convert_to_numpy", "convert_to_tensor", "normalize_embeddings"}
        allowed = {
            "batch_size",
            "show_progress_bar",
            "output_value",
            "precision",
            "device",
            "prompt_name",
            "prompt",
            "truncate_dim",
            "pool",
            "chunk_size",
        }
        model_kwargs = set()
        if hasattr(model, "get_model_kwargs"):
            model_kwargs = set(model.get_model_kwargs())
        encode_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in reserved and (key in allowed or key in model_kwargs)
        }
        return model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            **encode_kwargs,
        )

    def batch_encode(self, texts: List[str], **kwargs):
        if isinstance(texts, str):
            texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        params.update(kwargs)

        if params.get("instruction"):
            instruction = f"Instruct: {params['instruction']}\nQuery: "
            texts = [instruction + text for text in texts]

        batch_size = int(params.pop("batch_size", 16) or 16)
        params.pop("instruction", None)
        params.pop("max_length", None)
        params.pop("num_workers", None)
        params.pop("norm", None)

        if len(texts) <= batch_size:
            results = self.encode(texts, batch_size=batch_size, **params)
        else:
            batches = []
            with tqdm(total=len(texts), desc="Batch Encoding") as pbar:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    batches.append(self.encode(batch, batch_size=batch_size, **params))
                    pbar.update(len(batch))
            results = np.concatenate(batches)

        if isinstance(results, torch.Tensor):
            results = results.cpu().numpy()
        results = np.asarray(results, dtype=np.float32)
        if self.embedding_config.norm:
            norms = np.linalg.norm(results, axis=1)
            norms[norms == 0] = 1.0
            results = (results.T / norms).T
        return results
