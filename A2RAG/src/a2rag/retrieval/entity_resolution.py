import json
import re
from difflib import get_close_matches
from typing import Dict, List, Tuple, Set

from ..utils.chat_utils import fix_broken_generated_json


_NER_PATTERN = re.compile(r'\{[^{}]*"named_entities"\s*:\s*\[[^\]]*\][^{}]*\}', re.DOTALL)


def normalize_entity(text: str) -> str:
    if text is None:
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return text


_ALIAS_SPLIT_RE = re.compile(r"\b(?:aka|a\.k\.a\.|also known as)\b", re.IGNORECASE)
_COMMON_SUFFIXES = (
    "act",
    "regulation",
    "regulations",
    "code",
    "standard",
    "standards",
    "policy",
    "policies",
)


def _extract_parenthetical_aliases(text: str) -> List[str]:
    aliases: List[str] = []
    if not text:
        return aliases
    # Example: "Design and Building Practitioners Act 2020 (NSW)"
    m = re.search(r"\(([^)]+)\)", text)
    if m:
        inside = m.group(1).strip()
        outer = re.sub(r"\s*\([^)]+\)\s*", "", text).strip()
        if inside:
            aliases.append(inside)
        if outer:
            aliases.append(outer)
    return aliases


def _extract_acronym(text: str) -> str:
    if not text:
        return ""
    words = re.findall(r"[A-Za-z0-9]+", text)
    if len(words) < 2:
        return ""
    acronym = "".join([w[0] for w in words if w and w[0].isalnum()]).upper()
    return acronym if len(acronym) >= 2 else ""


def _strip_common_suffixes(text: str) -> List[str]:
    aliases: List[str] = []
    if not text:
        return aliases
    lower = text.lower()
    for suf in _COMMON_SUFFIXES:
        if lower.endswith(" " + suf):
            aliases.append(text[: -(len(suf) + 1)].strip())
    return aliases


def extract_entity_aliases(entity: str) -> List[str]:
    if not entity:
        return []

    aliases: List[str] = []
    entity = entity.strip()
    if not entity:
        return []

    aliases.append(entity)
    aliases.extend(_extract_parenthetical_aliases(entity))

    # Split on aka / also known as
    parts = [p.strip() for p in _ALIAS_SPLIT_RE.split(entity) if p.strip()]
    if len(parts) > 1:
        aliases.extend(parts)

    # Split on slash if it looks like name variants (e.g., "A/B")
    if "/" in entity and len(entity) <= 80:
        for p in entity.split("/"):
            p = p.strip()
            if p:
                aliases.append(p)

    aliases.extend(_strip_common_suffixes(entity))

    acronym = _extract_acronym(entity)
    if acronym:
        aliases.append(acronym)

    # De-duplicate while preserving order
    seen: Set[str] = set()
    out: List[str] = []
    for a in aliases:
        a = a.strip()
        if not a or a in seen:
            continue
        seen.add(a)
        out.append(a)
    return out


def build_canonical_entity_map(entities: List[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    canonical_to_aliases: Dict[str, List[str]] = {}
    for ent in entities:
        for alias in extract_entity_aliases(ent):
            norm = normalize_entity(alias)
            if not norm:
                continue
            canonical_to_aliases.setdefault(norm, [])
            if alias not in canonical_to_aliases[norm]:
                canonical_to_aliases[norm].append(alias)

    canonical_entities = list(canonical_to_aliases.keys())
    return canonical_entities, canonical_to_aliases


def parse_named_entities(raw_response: str) -> List[str]:
    if not raw_response:
        return []
    # Try strict JSON first
    try:
        parsed = json.loads(raw_response)
        ents = parsed.get("named_entities", [])
        return [e for e in ents if isinstance(e, str) and e.strip()]
    except Exception:
        pass

    # Try to extract a JSON object from a larger response
    match = _NER_PATTERN.search(raw_response)
    if match is not None:
        try:
            parsed = json.loads(match.group())
            ents = parsed.get("named_entities", [])
            return [e for e in ents if isinstance(e, str) and e.strip()]
        except Exception:
            pass

    # Try to fix broken JSON if needed
    try:
        fixed = fix_broken_generated_json(raw_response)
        parsed = json.loads(fixed)
        ents = parsed.get("named_entities", [])
        return [e for e in ents if isinstance(e, str) and e.strip()]
    except Exception:
        return []


def build_normalized_entity_index(entity_id_to_row: Dict[str, Dict]) -> Tuple[Dict[str, List[str]], List[str]]:
    normalized_to_ids: Dict[str, List[str]] = {}
    normalized_list: List[str] = []
    for node_id, row in entity_id_to_row.items():
        content = row.get("content", "")
        norm = normalize_entity(content)
        if not norm:
            continue
        if norm not in normalized_to_ids:
            normalized_to_ids[norm] = []
            normalized_list.append(norm)
        normalized_to_ids[norm].append(node_id)
    return normalized_to_ids, normalized_list


def fuzzy_match_entities(query_entity: str,
                         normalized_entities: List[str],
                         normalized_to_ids: Dict[str, List[str]],
                         cutoff: float = 0.9) -> List[str]:
    if not query_entity:
        return []
    norm = normalize_entity(query_entity)
    if not norm:
        return []
    matches = get_close_matches(norm, normalized_entities, n=1, cutoff=cutoff)
    if not matches:
        return []
    return normalized_to_ids.get(matches[0], [])
