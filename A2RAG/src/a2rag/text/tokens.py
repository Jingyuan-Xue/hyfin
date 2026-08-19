from __future__ import annotations

from typing import Iterable, List, Optional

import tiktoken


_DEFAULT_ENCODING = "cl100k_base"


def get_token_encoder(model_name: Optional[str] = None):
    if model_name:
        try:
            return tiktoken.encoding_for_model(model_name)
        except Exception:
            pass
    return tiktoken.get_encoding(_DEFAULT_ENCODING)


def count_tokens(text: str, encoder=None, model_name: Optional[str] = None) -> int:
    if text is None:
        return 0
    if encoder is None:
        encoder = get_token_encoder(model_name=model_name)
    return len(encoder.encode(text))


def truncate_text_by_tokens(text: str,
                            max_tokens: Optional[int],
                            encoder=None,
                            model_name: Optional[str] = None) -> str:
    if text is None:
        return ""
    if max_tokens is None:
        return text
    if encoder is None:
        encoder = get_token_encoder(model_name=model_name)
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoder.decode(tokens[:max_tokens])


def split_text_by_tokens(text: str,
                         max_tokens: Optional[int],
                         overlap_tokens: int = 0,
                         encoder=None,
                         model_name: Optional[str] = None) -> List[str]:
    if text is None:
        return []
    text = text.strip()
    if not text:
        return []
    if max_tokens is None:
        return [text]

    if encoder is None:
        encoder = get_token_encoder(model_name=model_name)

    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    if overlap_tokens < 0:
        overlap_tokens = 0
    step = max_tokens - overlap_tokens
    if step <= 0:
        step = max_tokens

    chunks = []
    for i in range(0, len(tokens), step):
        window = tokens[i:i + max_tokens]
        if not window:
            break
        chunks.append(encoder.decode(window))
    return chunks
