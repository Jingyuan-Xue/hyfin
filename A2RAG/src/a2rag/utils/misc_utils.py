from argparse import ArgumentTypeError
from hashlib import md5
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import re
import logging

from ..schemas import LinkingOutput, NerRawOutput, QuerySolution, TripleRawOutput
from .typing import Triple
from .chat_utils import filter_invalid_triples
from ..retrieval.entity_resolution import normalize_entity
from ..text.tokens import truncate_text_by_tokens

logger = logging.getLogger(__name__)

def text_processing(text):
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub('[^A-Za-z0-9 ]', ' ', text.lower()).strip()


def canonicalize_entity_text(text: str) -> str:
    return normalize_entity(text or "")


def canonicalize_triple(triple: List[str]) -> List[str]:
    if not triple or len(triple) != 3:
        return [canonicalize_entity_text(str(t)) for t in (triple or [])]
    return [canonicalize_entity_text(triple[0]),
            canonicalize_entity_text(triple[1]),
            canonicalize_entity_text(triple[2])]


def format_fact(triple: List[str],
                passage: Optional[str] = None,
                max_context_tokens: Optional[int] = None,
                encoder_name: Optional[str] = None) -> str:
    s = str(triple[0]) if len(triple) > 0 else ""
    p = str(triple[1]) if len(triple) > 1 else ""
    o = str(triple[2]) if len(triple) > 2 else ""
    fact_line = f"fact: {s} ||| {p} ||| {o}"
    if passage and max_context_tokens:
        ctx = truncate_text_by_tokens(passage, max_context_tokens, model_name=encoder_name)
        ctx = ctx.strip()
        if ctx:
            return fact_line + "\ncontext: " + ctx
    return fact_line


def parse_fact_text(fact_text: str) -> Optional[Tuple[str, str, str]]:
    if not fact_text:
        return None
    first_line = fact_text.splitlines()[0].strip()
    if not first_line.lower().startswith("fact:"):
        return None
    payload = first_line[len("fact:"):].strip()
    parts = [p.strip() for p in payload.split("|||")]
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]

def reformat_extraction_results(corpus_extraction_results,
                            use_canonical: bool = True) -> (Dict[str, NerRawOutput], Dict[str, TripleRawOutput]):

    ner_output_dict = {}
    triple_output_dict = {}
    for chunk_item in corpus_extraction_results:
        entities = chunk_item.get('extracted_entities', [])
        canonical_entities = chunk_item.get('extracted_entities_canonical', None)
        if use_canonical and canonical_entities:
            entities = canonical_entities

        triples = chunk_item.get('extracted_triples', [])
        canonical_triples = chunk_item.get('extracted_triples_canonical', None)
        if use_canonical and canonical_triples:
            triples = canonical_triples

        ner_output_dict[chunk_item['idx']] = NerRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            unique_entities=list(np.unique(entities))
        )
        triple_output_dict[chunk_item['idx']] = TripleRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            triples=filter_invalid_triples(triples=triples)
        )

    return ner_output_dict, triple_output_dict

def extract_entity_nodes(chunk_triples: List[List[Triple]], include_predicates: bool = False) -> (List[str], List[List[str]]):
    chunk_triple_entities = []  # a list of lists of unique entities from each chunk's triples
    for triples in chunk_triples:
        triple_entities = set()
        for t in triples:
            if len(t) == 3:
                triple_entities.update([t[0], t[2]])
                if include_predicates:
                    triple_entities.add(t[1])
            else:
                logger.warning(f"During graph construction, invalid triple is found: {t}")
        chunk_triple_entities.append(list(triple_entities))
    graph_nodes = list(np.unique([ent for ents in chunk_triple_entities for ent in ents]))
    return graph_nodes, chunk_triple_entities

def flatten_facts(chunk_triples: List[Triple]) -> List[Triple]:
    graph_triples = []  # a list of unique relation triple (in tuple) from all chunks
    for triples in chunk_triples:
        graph_triples.extend([tuple(t) for t in triples])
    graph_triples = list(set(graph_triples))
    return graph_triples

def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    
    # Handle the case where all values are the same (range is zero)
    if range_val == 0:
        return np.ones_like(x)  # Return an array of ones with the same shape as x
    
    return (x - min_val) / range_val

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute the MD5 hash of the given content string and optionally prepend a prefix.

    Args:
        content (str): The input string to be hashed.
        prefix (str, optional): A string to prepend to the resulting hash. Defaults to an empty string.

    Returns:
        str: A string consisting of the prefix followed by the hexadecimal representation of the MD5 hash.
    """
    return prefix + md5(content.encode()).hexdigest()


def all_values_of_same_length(data: dict) -> bool:
    """
    Return True if all values in 'data' have the same length or data is an empty dict,
    otherwise return False.
    """
    # Get an iterator over the dictionary's values
    value_iter = iter(data.values())

    # Get the length of the first sequence (handle empty dict case safely)
    try:
        first_length = len(next(value_iter))
    except StopIteration:
        # If the dictionary is empty, treat it as all having "the same length"
        return True

    # Check that every remaining sequence has this same length
    return all(len(seq) == first_length for seq in value_iter)


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )
