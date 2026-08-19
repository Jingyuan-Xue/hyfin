import json
import numpy as np
from tqdm import tqdm
import os
from typing import Union, Optional, List, Dict, Set, Any, Tuple, Literal
import logging
from copy import deepcopy
import pandas as pd

from ..utils.misc_utils import compute_mdhash_id
from ..utils.typing import ChunkRecord

logger = logging.getLogger(__name__)


def _metadata_to_json(metadata: Dict[str, Any]) -> str:
    return json.dumps(metadata or {}, sort_keys=True, ensure_ascii=False, default=str)


def _metadata_from_value(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, str):
        if not value.strip():
            return {}
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _chunk_identity(content: str, metadata: Dict[str, Any]) -> str:
    identity = {
        "content": content,
        "chunking_mode": metadata.get("chunking_mode"),
        "source_id": metadata.get("source_id"),
        "doc_index": metadata.get("doc_index"),
        "heading_path": metadata.get("heading_path"),
        "chunk_index": metadata.get("chunk_index"),
    }
    return json.dumps(identity, sort_keys=True, ensure_ascii=False, default=str)

class EmbeddingStore:
    def __init__(self, embedding_model, db_filename, batch_size, namespace):
        """
        Initializes the class with necessary configurations and sets up the working directory.

        Parameters:
        embedding_model: The model used for embeddings.
        db_filename: The directory path where data will be stored or retrieved.
        batch_size: The batch size used for processing.
        namespace: A unique identifier for data segregation.

        Functionality:
        - Assigns the provided parameters to instance variables.
        - Checks if the directory specified by `db_filename` exists.
          - If not, creates the directory and logs the operation.
        - Constructs the filename for storing data in a parquet file format.
        - Calls the method `_load_data()` to initialize the data loading process.
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace

        if not os.path.exists(db_filename):
            logger.info(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)

        self.filename = os.path.join(
            db_filename, f"vdb_{self.namespace}.parquet"
        )
        self._load_data()

    def _normalize_record(self, item: Union[str, ChunkRecord, Dict[str, Any]]) -> ChunkRecord:
        if isinstance(item, str):
            content = item
            embedding_text = item
            metadata: Dict[str, Any] = {}
        elif isinstance(item, dict):
            content = str(item.get("content", ""))
            embedding_text = str(item.get("embedding_text") or content)
            metadata = _metadata_from_value(item.get("metadata", {}))
        else:
            content = str(item)
            embedding_text = content
            metadata = {}

        if metadata:
            hash_source = _chunk_identity(content, metadata)
        else:
            hash_source = content

        hash_id = compute_mdhash_id(hash_source, prefix=self.namespace + "-")
        return {
            "hash_id": hash_id,
            "content": content,
            "embedding_text": embedding_text,
            "metadata": metadata,
        }

    def _records_by_hash_id(self, items: List[Union[str, ChunkRecord, Dict[str, Any]]]) -> Dict[str, ChunkRecord]:
        nodes_dict: Dict[str, ChunkRecord] = {}
        for item in items:
            row = self._normalize_record(item)
            if row["content"].strip():
                nodes_dict[row["hash_id"]] = row
        return nodes_dict

    def get_missing_string_hash_ids(self, texts: List[Union[str, ChunkRecord, Dict[str, Any]]]):
        nodes_dict = self._records_by_hash_id(texts)

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  {}

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
        return {h: nodes_dict[h] for h in missing_ids}

    def insert_strings(self, texts: List[Union[str, ChunkRecord, Dict[str, Any]]]):
        nodes_dict = self._records_by_hash_id(texts)

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  # Nothing to insert.

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]

        logger.info(
            f"Inserting {len(missing_ids)} new records, {len(all_hash_ids) - len(missing_ids)} records already exist.")

        if not missing_ids:
            return  {}# All records already exist.

        # Prepare the texts to encode from the contextual embedding text field.
        texts_to_encode = [nodes_dict[hash_id]["embedding_text"] for hash_id in missing_ids]

        missing_embeddings = self.embedding_model.batch_encode(texts_to_encode)

        missing_rows = [nodes_dict[hash_id] for hash_id in missing_ids]
        self._upsert(missing_rows, missing_embeddings)

    def _load_data(self):
        if os.path.exists(self.filename):
            df = pd.read_parquet(self.filename)
            self.hash_ids = df["hash_id"].values.tolist()
            self.texts = df["content"].values.tolist()
            if "embedding_text" in df.columns:
                self.embedding_texts = df["embedding_text"].values.tolist()
            else:
                self.embedding_texts = list(self.texts)
            if "metadata" in df.columns:
                self.metadatas = [_metadata_from_value(v) for v in df["metadata"].values.tolist()]
            else:
                self.metadatas = [{} for _ in self.hash_ids]
            self.embeddings = df["embedding"].values.tolist()
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_row = {
                h: {"hash_id": h, "content": t, "embedding_text": et, "metadata": m}
                for h, t, et, m in zip(self.hash_ids, self.texts, self.embedding_texts, self.metadatas)
            }
            self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
            self.text_to_hash_id = {self.texts[idx]: h  for idx, h in enumerate(self.hash_ids)}
            self.content_to_hash_ids = {}
            for h, t in zip(self.hash_ids, self.texts):
                self.content_to_hash_ids.setdefault(t, set()).add(h)
            assert len(self.hash_ids) == len(self.texts) == len(self.embeddings)
            logger.info(f"Loaded {len(self.hash_ids)} records from {self.filename}")
        else:
            self.hash_ids, self.texts, self.embedding_texts, self.metadatas, self.embeddings = [], [], [], [], []
            self.hash_id_to_idx, self.hash_id_to_row = {}, {}
            self.hash_id_to_text, self.text_to_hash_id, self.content_to_hash_ids = {}, {}, {}

    def _save_data(self):
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "content": self.texts,
            "embedding_text": self.embedding_texts,
            "metadata": [_metadata_to_json(m) for m in self.metadatas],
            "embedding": self.embeddings
        })
        data_to_save.to_parquet(self.filename, index=False)
        self.hash_id_to_row = {
            h: {"hash_id": h, "content": t, "embedding_text": et, "metadata": m}
            for h, t, et, m in zip(self.hash_ids, self.texts, self.embedding_texts, self.metadatas)
        }
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        self.content_to_hash_ids = {}
        for h, t in zip(self.hash_ids, self.texts):
            self.content_to_hash_ids.setdefault(t, set()).add(h)
        logger.info(f"Saved {len(self.hash_ids)} records to {self.filename}")

    def _upsert(self, rows, embeddings):
        self.embeddings.extend(embeddings)
        self.hash_ids.extend([row["hash_id"] for row in rows])
        self.texts.extend([row["content"] for row in rows])
        self.embedding_texts.extend([row["embedding_text"] for row in rows])
        self.metadatas.extend([row["metadata"] for row in rows])

        logger.info(f"Saving new records.")
        self._save_data()

    def delete(self, hash_ids):
        indices = []

        for hash in hash_ids:
            indices.append(self.hash_id_to_idx[hash])

        sorted_indices = np.sort(indices)[::-1]

        for idx in sorted_indices:
            self.hash_ids.pop(idx)
            self.texts.pop(idx)
            self.embedding_texts.pop(idx)
            self.metadatas.pop(idx)
            self.embeddings.pop(idx)

        logger.info(f"Saving record after deletion.")
        self._save_data()

    def get_row(self, hash_id):
        return self.hash_id_to_row[hash_id]

    def get_hash_id(self, text):
        return self.text_to_hash_id[text]

    def get_rows(self, hash_ids, dtype=np.float32):
        if not hash_ids:
            return {}

        results = {id : self.hash_id_to_row[id] for id in hash_ids}

        return results

    def get_all_ids(self):
        return deepcopy(self.hash_ids)

    def get_all_id_to_rows(self):
        return deepcopy(self.hash_id_to_row)

    def get_all_texts(self):
        return set(row['content'] for row in self.hash_id_to_row.values())

    def get_embedding(self, hash_id, dtype=np.float32) -> np.ndarray:
        return np.asarray(self.embeddings[self.hash_id_to_idx[hash_id]], dtype=dtype)
    
    def get_embeddings(self, hash_ids, dtype=np.float32) -> list[np.ndarray]:
        if not hash_ids:
            return []

        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.embeddings, dtype=dtype)[indices]

        return embeddings
