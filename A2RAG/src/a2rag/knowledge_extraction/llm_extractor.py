import ast
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, TypedDict

from tqdm import tqdm

from ..models.chat_base import BaseChatModel
from ..prompts import PromptTemplateManager
from ..utils.chat_utils import filter_invalid_triples, fix_broken_generated_json
from ..utils.logging_utils import get_logger
from ..schemas import NerRawOutput, TripleRawOutput

logger = get_logger(__name__)


class ChunkInfo(TypedDict, total=False):
    num_tokens: int
    content: str
    chunk_order: List[Tuple]
    full_doc_ids: List[str]


@dataclass
class ChatInput:
    chunk_id: str
    input_message: List[Dict]


def _extract_json_field(response_text: str, field_name: str):
    if not response_text:
        return []

    candidates = [response_text]
    match = re.search(r'\{[^{}]*"' + re.escape(field_name) + r'"\s*:\s*\[[\s\S]*?\][^{}]*\}', response_text)
    if match:
        candidates.insert(0, match.group())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            value = parsed.get(field_name, [])
            return value if isinstance(value, list) else []
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(candidate)
            value = parsed.get(field_name, []) if isinstance(parsed, dict) else []
            return value if isinstance(value, list) else []
        except Exception:
            pass

    return []


class KnowledgeExtractor:
    def __init__(self, chat_model_client: BaseChatModel):
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        self.chat_model_client = chat_model_client

    def ner(self, chunk_key: str, passage: str) -> NerRawOutput:
        messages = self.prompt_template_manager.render(name="ner", passage=passage)
        raw_response = ""
        metadata = {}
        try:
            raw_response, metadata, cache_hit = self.chat_model_client.infer(messages=messages)
            metadata["cache_hit"] = cache_hit
            response_text = fix_broken_generated_json(raw_response) if metadata.get("finish_reason") == "length" else raw_response
            extracted_entities = _extract_json_field(response_text, "named_entities")
            unique_entities = [entity for entity in dict.fromkeys(extracted_entities) if isinstance(entity, str)]
        except Exception as e:
            logger.warning(e)
            metadata.update({"error": str(e)})
            unique_entities = []

        return NerRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            unique_entities=unique_entities,
            metadata=metadata,
        )

    def triple_extraction(self, chunk_key: str, passage: str, named_entities: List[str]) -> TripleRawOutput:
        messages = self.prompt_template_manager.render(
            name="triple_extraction",
            passage=passage,
            named_entity_json=json.dumps({"named_entities": named_entities}),
        )

        raw_response = ""
        metadata = {}
        try:
            raw_response, metadata, cache_hit = self.chat_model_client.infer(messages=messages)
            metadata["cache_hit"] = cache_hit
            response_text = fix_broken_generated_json(raw_response) if metadata.get("finish_reason") == "length" else raw_response
            triples = filter_invalid_triples(_extract_json_field(response_text, "triples"))
        except Exception as e:
            logger.warning(f"Exception for chunk {chunk_key}: {e}")
            metadata.update({"error": str(e)})
            triples = []

        return TripleRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            metadata=metadata,
            triples=triples,
        )

    def extract_knowledge(self, chunk_key: str, passage: str) -> Dict[str, Any]:
        ner_output = self.ner(chunk_key=chunk_key, passage=passage)
        triple_output = self.triple_extraction(
            chunk_key=chunk_key,
            passage=passage,
            named_entities=ner_output.unique_entities,
        )
        return {"ner": ner_output, "triplets": triple_output}

    def batch_extract_knowledge(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}

        ner_results_list = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        with ThreadPoolExecutor() as executor:
            ner_futures = {
                executor.submit(self.ner, chunk_key, passage): chunk_key
                for chunk_key, passage in chunk_passages.items()
            }
            pbar = tqdm(as_completed(ner_futures), total=len(ner_futures), desc="NER")
            for future in pbar:
                result = future.result()
                ner_results_list.append(result)
                metadata = result.metadata
                total_prompt_tokens += metadata.get("prompt_tokens", 0)
                total_completion_tokens += metadata.get("completion_tokens", 0)
                num_cache_hit += 1 if metadata.get("cache_hit") else 0
                pbar.set_postfix({
                    "total_prompt_tokens": total_prompt_tokens,
                    "total_completion_tokens": total_completion_tokens,
                    "num_cache_hit": num_cache_hit,
                })

        triple_results_list = []
        total_prompt_tokens = total_completion_tokens = num_cache_hit = 0
        with ThreadPoolExecutor() as executor:
            triple_futures = {
                executor.submit(
                    self.triple_extraction,
                    ner_result.chunk_id,
                    chunk_passages[ner_result.chunk_id],
                    ner_result.unique_entities,
                ): ner_result.chunk_id
                for ner_result in ner_results_list
            }
            pbar = tqdm(as_completed(triple_futures), total=len(triple_futures), desc="Extracting triples")
            for future in pbar:
                result = future.result()
                triple_results_list.append(result)
                metadata = result.metadata
                total_prompt_tokens += metadata.get("prompt_tokens", 0)
                total_completion_tokens += metadata.get("completion_tokens", 0)
                num_cache_hit += 1 if metadata.get("cache_hit") else 0
                pbar.set_postfix({
                    "total_prompt_tokens": total_prompt_tokens,
                    "total_completion_tokens": total_completion_tokens,
                    "num_cache_hit": num_cache_hit,
                })

        return (
            {res.chunk_id: res for res in ner_results_list},
            {res.chunk_id: res for res in triple_results_list},
        )
