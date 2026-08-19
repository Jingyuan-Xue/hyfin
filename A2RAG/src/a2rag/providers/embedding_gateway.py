from copy import deepcopy
import os
from typing import List, Optional

import httpx
import numpy as np
import torch
from tqdm import tqdm

from ..config.settings import BaseConfig
from ..models.embedding_base import BaseEmbeddingModel, EmbeddingConfig
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class EmbeddingGateway(BaseEmbeddingModel):
    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)
        if embedding_model is not None:
            self.embedding_model = embedding_model
            logger.debug(f"Overriding {self.__class__.__name__}'s embedding_model with: {self.embedding_model}")

        self.client = httpx.Client(timeout=httpx.Timeout(5 * 60, read=5 * 60))
        self._init_embedding_config()

    def _init_embedding_config(self) -> None:
        config_dict = {
            "embedding_model": self.embedding_model,
            "norm": self.global_config.embedding_return_as_normalized,
            "model_init_params": {
                "pretrained_model_name_or_path": self.embedding_model,
                "trust_remote_code": True,
                "device_map": "auto",
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32,
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def encode(self, texts: List[str]):
        if not self.global_config.embedding_base_url:
            raise ValueError("embedding_base_url is required for EmbeddingGateway.")

        texts = [t.replace("\n", " ") or " " for t in texts]
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv("A2RAG_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = self.client.post(
            self.global_config.embedding_base_url.rstrip("/") + "/embeddings",
            headers=headers,
            json={"input": texts, "model": self.embedding_model, "encoding_format": "float"},
        )
        response.raise_for_status()
        payload = response.json()
        return np.array([item["embedding"] for item in payload["data"]])

    def batch_encode(self, texts: List[str], **kwargs):
        if isinstance(texts, str):
            texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        params.update(kwargs)

        if params.get("instruction"):
            instruction = f"Instruct: {params['instruction']}\nQuery: "
            texts = [instruction + text for text in texts]

        logger.debug(f"Calling {self.__class__.__name__} with:\n{params}")
        batch_size = params.pop("batch_size", 16)

        if len(texts) <= batch_size:
            results = self.encode(texts)
        else:
            pbar = tqdm(total=len(texts), desc="Batch Encoding")
            batches = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batches.append(self.encode(batch))
                pbar.update(len(batch))
            pbar.close()
            results = np.concatenate(batches)

        if isinstance(results, torch.Tensor):
            results = results.cpu().numpy()
        if self.embedding_config.norm:
            norms = np.linalg.norm(results, axis=1)
            norms[norms == 0] = 1.0
            results = (results.T / norms).T

        return results
