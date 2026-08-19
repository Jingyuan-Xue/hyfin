from typing import Dict, Any, TypedDict, Tuple

Triple = Tuple[str, str, str]


class ChunkRecord(TypedDict, total=False):
    hash_id: str
    content: str
    embedding_text: str
    metadata: Dict[str, Any]
