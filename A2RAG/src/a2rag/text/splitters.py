import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from .tokens import (
    get_token_encoder,
    count_tokens,
    split_text_by_tokens,
)

from ..utils.typing import ChunkRecord
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class OutlineItem:
    level: int
    title: str
    position: int


@dataclass
class HeadingRef:
    heading: str
    level: int
    position: int


@dataclass
class Section:
    heading: Optional[str]
    level: int
    content: str
    position: int
    headings: List[HeadingRef] = field(default_factory=list)


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)(?:\s*\{#[\w-]+\})?\s*$", re.M)
_SENTENCE_RE = re.compile(r"[^.!?。！？]+[.!?。！？]+", re.U)


def extract_outline(text: str) -> List[OutlineItem]:
    outline: List[OutlineItem] = []
    for m in _HEADING_RE.finditer(text):
        level = len(m.group(1))
        title = m.group(2).strip()
        outline.append(OutlineItem(level=level, title=title, position=m.start()))
    return outline


def split_by_headings(text: str, outline: List[OutlineItem]) -> List[Section]:
    if not outline:
        return [Section(heading=None, level=0, content=text, position=0)]

    sections: List[Section] = []

    if outline[0].position > 0:
        front = text[: outline[0].position].strip()
        if front:
            sections.append(Section(heading=None, level=0, content=front, position=0))

    for i, cur in enumerate(outline):
        nxt = outline[i + 1] if i + 1 < len(outline) else None
        heading_line = text[cur.position:].split("\n", 1)[0]
        start_pos = cur.position + len(heading_line) + 1
        end_pos = nxt.position if nxt else len(text)
        content = text[start_pos:end_pos].strip()
        sections.append(
            Section(
                heading=cur.title,
                level=cur.level,
                content=content,
                position=cur.position,
            )
        )
    return sections


def _find_outline_index(outline: List[OutlineItem], title: str, level: int) -> int:
    for i, it in enumerate(outline):
        if it.title == title and it.level == level:
            return i
    return -1


def generate_enhanced_summary(
    section: Section,
    outline: List[OutlineItem],
    part_index: Optional[int] = None,
    total_parts: Optional[int] = None,
) -> str:
    if ((section.heading is None and section.level == 0) or (not section.headings and section.heading is None)):
        doc_title = outline[0].title if outline and outline[0].level == 1 else "文档"
        return f"{doc_title} 前言"

    if section.headings:
        sorted_headings = sorted(section.headings, key=lambda h: (h.level, h.position))

        paths_set: Dict[str, str] = {}
        for h in sorted_headings:
            if not h.heading:
                continue
            idx = _find_outline_index(outline, h.heading, h.level)
            if idx == -1:
                paths_set[h.heading] = h.heading
                continue

            path_parts: List[str] = []
            parent_level = h.level - 1
            j = idx - 1
            while j >= 0 and parent_level > 0:
                if outline[j].level == parent_level:
                    path_parts.insert(0, outline[j].title)
                    parent_level -= 1
                j -= 1
            path_parts.append(h.heading)
            full_path = " > ".join(path_parts)
            paths_set[full_path] = full_path

        paths = list(paths_set.values())

        def depth(p: str) -> int:
            return p.count(">")

        paths.sort(key=lambda p: (depth(p), p))

        if not paths:
            return section.heading or "未命名段落"

        if len(paths) == 1:
            summary = paths[0]
            if part_index is not None and total_parts and total_parts > 1:
                summary += f" - Part {part_index}/{total_parts}"
            return summary

        summary = ""
        first_segments = paths[0].split(" > ")
        for i in range(len(first_segments) - 1):
            prefix = " > ".join(first_segments[: i + 1])
            if all(p.startswith(prefix + " > ") for p in paths[1:]):
                uniques = []
                for p in paths:
                    uniques.append(p[len(prefix) + 3 :])
                summary = prefix + " > [" + ", ".join(uniques) + "]"
                break

        if not summary:
            summary = ", ".join(paths)

        if part_index is not None and total_parts and total_parts > 1:
            summary += f" - Part {part_index}/{total_parts}"
        return summary

    if section.heading is None and section.level == 0:
        return "文档前言"

    idx = _find_outline_index(outline, section.heading or "", section.level)
    if idx == -1:
        return section.heading or "未命名段落"

    parents: List[str] = []
    parent_level = section.level - 1
    j = idx - 1
    while j >= 0 and parent_level > 0:
        if outline[j].level == parent_level:
            parents.insert(0, outline[j].title)
            parent_level -= 1
        j -= 1

    summary = (" > ".join(parents) + " > " if parents else "") + (section.heading or "")
    if part_index is not None and total_parts and total_parts > 1:
        summary += f" - Part {part_index}/{total_parts}"
    return summary


def split_long_section(content: str, max_len: int) -> List[str]:
    paragraphs = re.split(r"\n\n+", content)
    result: List[str] = []
    cur = ""

    for p in paragraphs:
        if len(p) > max_len:
            if cur:
                result.append(cur)
                cur = ""

            sentences = _SENTENCE_RE.findall(p) or [p]
            sent_chunk = ""
            for s in sentences:
                if len(sent_chunk) + len(s) <= max_len:
                    sent_chunk += s
                else:
                    if sent_chunk:
                        result.append(sent_chunk)

                    if len(s) > max_len:
                        for i in range(0, len(s), max_len):
                            result.append(s[i:i + max_len])
                        sent_chunk = ""
                    else:
                        sent_chunk = s

            if sent_chunk:
                cur = sent_chunk

        else:
            candidate = (cur + "\n\n" + p) if cur else p
            if len(candidate) <= max_len:
                cur = candidate
            else:
                if cur:
                    result.append(cur)
                cur = p

    if cur:
        result.append(cur)
    return result


def process_sections(
    sections: List[Section],
    outline: List[OutlineItem],
    min_len: int,
    max_len: int,
) -> List[Dict[str, str]]:
    preprocessed: List[Section] = []
    current: Optional[Section] = None

    for sec in sections:
        content_len = len(sec.content.strip())
        if content_len < min_len and current is not None:
            heading_line = f"{'#' * sec.level} {sec.heading}\n" if sec.heading else ""
            merged = current.content + "\n\n" + heading_line + sec.content
            if len(merged) <= max_len:
                current.content = merged
                if sec.heading:
                    current.headings.append(HeadingRef(sec.heading, sec.level, sec.position))
                continue

        if current is not None:
            preprocessed.append(current)

        current = Section(
            heading=sec.heading,
            level=sec.level,
            content=sec.content,
            position=sec.position,
            headings=[HeadingRef(sec.heading, sec.level, sec.position)] if sec.heading else [],
        )

    if current is not None:
        preprocessed.append(current)

    result: List[Dict[str, str]] = []
    accumulated: Optional[Section] = None

    for sec in preprocessed:
        content_len = len(sec.content.strip())

        if content_len < min_len:
            if accumulated is None:
                accumulated = Section(
                    heading=sec.heading,
                    level=sec.level,
                    content=sec.content,
                    position=sec.position,
                    headings=[HeadingRef(sec.heading, sec.level, sec.position)] if sec.heading else [],
                )
            else:
                heading_line = f"{'#' * sec.level} {sec.heading}\n" if sec.heading else ""
                accumulated.content += "\n\n" + heading_line + sec.content
                if sec.heading:
                    accumulated.headings.append(HeadingRef(sec.heading, sec.level, sec.position))

            if len(accumulated.content.strip()) >= min_len:
                summ = generate_enhanced_summary(accumulated, outline)
                if len(accumulated.content.strip()) > max_len:
                    subs = split_long_section(accumulated.content, max_len)
                    for j, sub in enumerate(subs):
                        result.append({"summary": f"{summ} - Part {j+1}/{len(subs)}", "content": sub})
                else:
                    result.append({"summary": summ, "content": accumulated.content})
                accumulated = None
            continue

        if accumulated is not None:
            summ = generate_enhanced_summary(accumulated, outline)
            if len(accumulated.content.strip()) <= max_len:
                result.append({"summary": summ, "content": accumulated.content})
            else:
                subs = split_long_section(accumulated.content, max_len)
                for j, sub in enumerate(subs):
                    result.append({"summary": f"{summ} - Part {j+1}/{len(subs)}", "content": sub})
            accumulated = None

        if content_len > max_len:
            subs = split_long_section(sec.content, max_len)
            if not sec.headings and sec.heading:
                sec.headings = [HeadingRef(sec.heading, sec.level, sec.position)]
            for i, sub in enumerate(subs):
                summ = generate_enhanced_summary(sec, outline, i + 1, len(subs))
                result.append({"summary": summ, "content": sub})
        else:
            if not sec.headings and sec.heading:
                sec.headings = [HeadingRef(sec.heading, sec.level, sec.position)]
            summ = generate_enhanced_summary(sec, outline)
            heading_line = f"{'#' * sec.level} {sec.heading}\n" if sec.heading else ""
            result.append({"summary": summ, "content": heading_line + sec.content})

    if accumulated is not None:
        if result:
            merged = result[-1]["content"] + "\n\n" + accumulated.content
            if len(merged) <= max_len:
                merged_sec = Section(
                    heading=accumulated.heading,
                    level=accumulated.level,
                    content=merged,
                    position=accumulated.position,
                    headings=accumulated.headings,
                )
                result[-1] = {"summary": generate_enhanced_summary(merged_sec, outline), "content": merged}
            else:
                summ = generate_enhanced_summary(accumulated, outline)
                heading_line = f"{'#' * accumulated.level} {accumulated.heading}\n" if accumulated.heading else ""
                result.append({"summary": summ, "content": heading_line + accumulated.content})
        else:
            summ = generate_enhanced_summary(accumulated, outline)
            heading_line = f"{'#' * accumulated.level} {accumulated.heading}\n" if accumulated.heading else ""
            result.append({"summary": summ, "content": heading_line + accumulated.content})

    return result


def split_markdown(markdown_text: str, min_len: int = 1500, max_len: int = 2000, overlap_chars: int = 0) -> List[Dict[str, str]]:
    outline = extract_outline(markdown_text)
    sections = split_by_headings(markdown_text, outline)
    chunks = process_sections(sections, outline, min_len, max_len)

    if overlap_chars and overlap_chars > 0:
        overlapped = []
        prev_content = ""
        for idx, ch in enumerate(chunks):
            content = ch["content"]
            if idx == 0 or not prev_content:
                new_content = content
            else:
                tail = prev_content[-overlap_chars:]
                new_content = tail + "\n\n" + content if tail else content
            overlapped.append({**ch, "content": new_content})
            prev_content = content
        chunks = overlapped

    out = []
    for r in chunks:
        result_md = f"> ** Summarization：** *{r['summary']}*\n\n---\n\n{r['content']}"
        out.append({"summary": r["summary"], "content": r["content"], "result": result_md})
    return out


class TextPreprocessor:
    def __init__(self, config=None):
        self.config = config

    def preprocess_docs(self, docs: List[str]) -> List[str]:
        return [d for d in docs if d and d.strip()]


class MarkdownSplitterPreprocessor(TextPreprocessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.min_len = getattr(config, "preprocess_markdown_min_len", 1500)
        self.max_len = getattr(config, "preprocess_markdown_max_len", 2000)
        self.overlap_chars = getattr(config, "preprocess_markdown_overlap_chars", 0)
        self.use_result = getattr(config, "preprocess_markdown_use_result", True)

    def preprocess_docs(self, docs: List[str]) -> List[str]:
        out: List[str] = []
        for text in docs:
            if not text or not text.strip():
                continue
            chunks = split_markdown(text, self.min_len, self.max_len, self.overlap_chars)
            for ch in chunks:
                out.append(ch["result"] if self.use_result else ch["content"])
        return out


def _build_heading_path(outline: List[OutlineItem], section: Section) -> str:
    if section.heading is None:
        return ""

    # Find this heading in outline, then walk backward to build parents.
    idx = _find_outline_index(outline, section.heading, section.level)
    if idx == -1:
        return section.heading

    parents: List[str] = []
    parent_level = section.level - 1
    j = idx - 1
    while j >= 0 and parent_level > 0:
        if outline[j].level == parent_level:
            parents.insert(0, outline[j].title)
            parent_level -= 1
        j -= 1

    return " > ".join(parents + [section.heading])


class TokenSplitterPreprocessor(TextPreprocessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.max_tokens = getattr(config, "preprocess_chunk_max_token_size", None)
        self.overlap_tokens = getattr(config, "preprocess_chunk_overlap_token_size", 0)
        self.min_tokens = getattr(config, "preprocess_chunk_min_token_size", None)
        self.encoder_name = getattr(config, "preprocess_encoder_name", None)
        self.encoder = get_token_encoder(self.encoder_name)

    def preprocess_docs(self, docs: List[str]) -> List[str]:
        out: List[str] = []
        for text in docs:
            if not text or not text.strip():
                continue
            chunks = split_text_by_tokens(
                text,
                max_tokens=self.max_tokens,
                overlap_tokens=self.overlap_tokens,
                encoder=self.encoder,
                model_name=self.encoder_name,
            )
            if self.min_tokens:
                merged = []
                buffer = ""
                buffer_tokens = 0
                for ch in chunks:
                    ch_tokens = count_tokens(ch, encoder=self.encoder)
                    if buffer and buffer_tokens + ch_tokens <= self.max_tokens:
                        buffer = buffer + "\n\n" + ch
                        buffer_tokens += ch_tokens
                    elif ch_tokens < self.min_tokens and not buffer:
                        buffer = ch
                        buffer_tokens = ch_tokens
                    else:
                        if buffer:
                            merged.append(buffer)
                            buffer = ""
                            buffer_tokens = 0
                        merged.append(ch)
                if buffer:
                    merged.append(buffer)
                chunks = merged
            out.extend(chunks)
        return out


class MarkdownTokenSplitterPreprocessor(TextPreprocessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.max_tokens = getattr(config, "preprocess_markdown_max_token_size", None)
        if self.max_tokens is None:
            self.max_tokens = getattr(config, "preprocess_chunk_max_token_size", None)
        self.overlap_tokens = getattr(config, "preprocess_markdown_overlap_token_size", None)
        if self.overlap_tokens is None:
            self.overlap_tokens = getattr(config, "preprocess_chunk_overlap_token_size", 0)
        self.min_tokens = getattr(config, "preprocess_markdown_min_token_size", None)
        if self.min_tokens is None:
            self.min_tokens = getattr(config, "preprocess_chunk_min_token_size", None)
        if self.min_tokens is None and self.max_tokens:
            self.min_tokens = max(50, self.max_tokens // 4)

        self.include_heading_path = getattr(config, "preprocess_markdown_include_heading_path", True)
        self.encoder_name = getattr(config, "preprocess_encoder_name", None)
        self.encoder = get_token_encoder(self.encoder_name)

    def _split_section(self, section: Section, outline: List[OutlineItem]) -> List[str]:
        heading_line = f"{'#' * section.level} {section.heading}" if section.heading else ""
        heading_path = _build_heading_path(outline, section) if self.include_heading_path else ""

        prefix_parts = []
        if heading_path:
            prefix_parts.append(f"Section: {heading_path}")
        if heading_line:
            prefix_parts.append(heading_line)
        prefix = "\n".join(prefix_parts).strip()

        content = (section.content or "").strip()
        if not content and not prefix:
            return []
        if self.max_tokens is None:
            return ["\n\n".join([p for p in [prefix, content] if p])]

        if not content:
            return [prefix]

        prefix_tokens = count_tokens(prefix, encoder=self.encoder) if prefix else 0
        available = max(self.max_tokens - prefix_tokens, 1)
        body_chunks = split_text_by_tokens(
            content,
            max_tokens=available,
            overlap_tokens=self.overlap_tokens,
            encoder=self.encoder,
            model_name=self.encoder_name,
        )

        results = []
        for ch in body_chunks:
            if prefix:
                results.append(prefix + "\n\n" + ch)
            else:
                results.append(ch)
        return results

    def preprocess_docs(self, docs: List[str]) -> List[str]:
        out: List[str] = []
        for text in docs:
            if not text or not text.strip():
                continue
            outline = extract_outline(text)
            sections = split_by_headings(text, outline)
            chunks: List[str] = []
            for sec in sections:
                chunks.extend(self._split_section(sec, outline))

            if self.min_tokens:
                merged: List[str] = []
                buffer = ""
                buffer_tokens = 0
                for ch in chunks:
                    ch_tokens = count_tokens(ch, encoder=self.encoder)
                    if buffer and buffer_tokens + ch_tokens <= self.max_tokens:
                        buffer = buffer + "\n\n" + ch
                        buffer_tokens += ch_tokens
                    elif ch_tokens < self.min_tokens and not buffer:
                        buffer = ch
                        buffer_tokens = ch_tokens
                    else:
                        if buffer:
                            merged.append(buffer)
                            buffer = ""
                            buffer_tokens = 0
                        merged.append(ch)
                if buffer:
                    merged.append(buffer)
                chunks = merged

            out.extend(chunks)
        return out


def _as_numpy_vector(value):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _make_chonkie_embedding_adapter(embedding_model, encoder):
    from chonkie.embeddings import BaseEmbeddings

    class A2RAGChonkieEmbeddings(BaseEmbeddings):
        def embed(self, text: str):
            return self.embed_batch([text])[0]

        def embed_batch(self, texts: List[str]):
            embeddings = embedding_model.batch_encode(texts)
            return [_as_numpy_vector(emb) for emb in embeddings]

        def similarity(self, u, v):
            import numpy as np

            denom = np.linalg.norm(u) * np.linalg.norm(v)
            if denom == 0:
                return np.float32(0.0)
            return np.float32(np.dot(u, v.T) / denom)

        @property
        def dimension(self) -> int:
            return int(getattr(embedding_model, "embedding_dim", 0) or 0)

        def get_tokenizer(self):
            return encoder

    return A2RAGChonkieEmbeddings()


def _load_langchain_splitter_classes():
    # Load only the splitter modules we need; the package __init__ imports many optional NLP backends.
    import importlib
    import importlib.util
    import sys
    import types

    package_name = "langchain_text_splitters"
    pkg = sys.modules.get(package_name)
    if pkg is None or not hasattr(pkg, "__path__"):
        spec = importlib.util.find_spec(package_name)
        if spec is None or spec.submodule_search_locations is None:
            raise ImportError("langchain-text-splitters is required for markdown chunking.")
        pkg = types.ModuleType(package_name)
        pkg.__path__ = list(spec.submodule_search_locations)
        sys.modules[package_name] = pkg

    markdown_mod = importlib.import_module("langchain_text_splitters.markdown")
    character_mod = importlib.import_module("langchain_text_splitters.character")
    pkg.MarkdownHeaderTextSplitter = markdown_mod.MarkdownHeaderTextSplitter
    pkg.MarkdownTextSplitter = markdown_mod.MarkdownTextSplitter
    pkg.RecursiveCharacterTextSplitter = character_mod.RecursiveCharacterTextSplitter
    return markdown_mod.MarkdownHeaderTextSplitter, character_mod.RecursiveCharacterTextSplitter


class ChunkingPreprocessor(TextPreprocessor):
    def __init__(self, config, embedding_model=None):
        super().__init__(config=config)
        self.embedding_model = embedding_model
        self.chunking_mode = getattr(config, "chunking_mode", "markdown")
        self.min_tokens = getattr(config, "chunk_min_token_size", 800)
        self.target_tokens = getattr(config, "chunk_target_token_size", 1000)
        self.max_tokens = getattr(config, "chunk_max_token_size", 1200)
        self.overlap_tokens = getattr(config, "chunk_overlap_token_size", 100)
        self.semantic_threshold = getattr(config, "chunk_semantic_threshold", 0.8)
        self.encoder_name = getattr(config, "preprocess_encoder_name", None)
        self.encoder = get_token_encoder(self.encoder_name)
        self._semantic_chunker = None

    def preprocess_docs(self, docs: List[Any]) -> List[ChunkRecord]:
        out: List[ChunkRecord] = []
        for doc_index, raw_doc in enumerate(docs):
            text, base_metadata = self._normalize_input_doc(raw_doc, doc_index)
            if not text:
                continue
            if self.chunking_mode == "text":
                out.extend(self._chunk_text_doc(text, base_metadata))
            elif self.chunking_mode == "markdown":
                out.extend(self._chunk_markdown_doc(text, base_metadata))
            else:
                raise ValueError(f"Unsupported chunking_mode: {self.chunking_mode}")
        return out

    def _normalize_input_doc(self, raw_doc: Any, doc_index: int) -> Tuple[str, Dict[str, Any]]:
        if isinstance(raw_doc, dict):
            text = raw_doc.get("content", raw_doc.get("text", ""))
            metadata = dict(raw_doc.get("metadata") or {})
            if raw_doc.get("source_id"):
                metadata.setdefault("source_id", raw_doc["source_id"])
            if raw_doc.get("source"):
                metadata.setdefault("source", raw_doc["source"])
        else:
            text = raw_doc
            metadata = {}

        if text is None:
            text = ""
        text = str(text).strip()
        metadata["doc_index"] = doc_index
        metadata.setdefault("source_id", metadata.get("source") or f"doc-{doc_index}")
        return text, metadata

    def _record(
        self,
        content: str,
        embedding_text: str,
        metadata: Dict[str, Any],
        chunk_index: int,
    ) -> ChunkRecord:
        content = content.strip()
        metadata = dict(metadata)
        metadata["chunk_index"] = chunk_index
        metadata["token_count"] = count_tokens(content, encoder=self.encoder)
        return {
            "content": content,
            "embedding_text": embedding_text.strip() or content,
            "metadata": metadata,
        }

    def _enforce_max_tokens(self, text: str) -> List[str]:
        return split_text_by_tokens(
            text,
            max_tokens=self.max_tokens,
            overlap_tokens=self._effective_overlap_tokens(),
            encoder=self.encoder,
            model_name=self.encoder_name,
        )

    def _effective_overlap_tokens(self) -> int:
        chunk_size = self._target_chunk_size()
        if chunk_size is None:
            return max(0, int(self.overlap_tokens or 0))
        return max(0, min(int(self.overlap_tokens or 0), max(int(chunk_size) - 1, 0)))

    def _target_chunk_size(self) -> Optional[int]:
        if self.max_tokens is None:
            return self.target_tokens
        if self.target_tokens is None:
            return self.max_tokens
        return min(int(self.target_tokens), int(self.max_tokens))

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        if not self.min_tokens or self.max_tokens is None:
            return chunks
        merged: List[str] = []
        buffer = ""
        buffer_tokens = 0

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            chunk_tokens = count_tokens(chunk, encoder=self.encoder)
            if buffer and buffer_tokens < self.min_tokens and buffer_tokens + chunk_tokens <= self.max_tokens:
                buffer = buffer + "\n\n" + chunk
                buffer_tokens += chunk_tokens
                continue
            if buffer:
                merged.append(buffer)
            buffer = chunk
            buffer_tokens = chunk_tokens

        if buffer:
            merged.append(buffer)
        return merged

    def _get_semantic_chunker(self):
        if self._semantic_chunker is not None:
            return self._semantic_chunker
        if self.embedding_model is None:
            raise ValueError("chunking_mode='text' requires an embedding model for semantic chunking.")
        from chonkie import SemanticChunker

        self._semantic_chunker = SemanticChunker(
            embedding_model=_make_chonkie_embedding_adapter(self.embedding_model, self.encoder),
            threshold=self.semantic_threshold,
            chunk_size=self._target_chunk_size() or self.max_tokens,
        )
        return self._semantic_chunker

    def _chunk_text_doc(self, text: str, base_metadata: Dict[str, Any]) -> List[ChunkRecord]:
        chunker = self._get_semantic_chunker()
        raw_chunks = chunker.chunk(text)
        chunks = [getattr(ch, "text", str(ch)).strip() for ch in raw_chunks]
        split_chunks: List[str] = []
        for chunk in chunks:
            split_chunks.extend(self._enforce_max_tokens(chunk))
        split_chunks = self._merge_small_chunks(split_chunks)

        records: List[ChunkRecord] = []
        for idx, chunk in enumerate(split_chunks):
            metadata = {
                **base_metadata,
                "chunking_mode": "text",
            }
            records.append(self._record(chunk, chunk, metadata, idx))
        return records

    def _build_recursive_splitter(self):
        _, RecursiveCharacterTextSplitter = _load_langchain_splitter_classes()

        kwargs = {
            "chunk_size": self._target_chunk_size() or self.max_tokens,
            "chunk_overlap": self._effective_overlap_tokens(),
            "separators": ["\n\n", "\n", ". ", "。", " ", ""],
        }
        try:
            return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                model_name=self.encoder_name,
                **kwargs,
            )
        except Exception:
            return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                **kwargs,
            )

    def _heading_path_from_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        return [
            metadata[key]
            for key in [f"Header {i}" for i in range(1, 7)]
            if key in metadata and metadata[key]
        ]

    def _embedding_text_for_markdown(self, content: str, metadata: Dict[str, Any]) -> str:
        prefix_parts = []
        heading_path = metadata.get("heading_path") or []
        source = metadata.get("source")
        if source:
            prefix_parts.append(f"Source: {source}")
        if heading_path:
            prefix_parts.append("Section: " + " > ".join(heading_path))
        if prefix_parts:
            return "\n".join(prefix_parts) + "\n\n" + content
        return content

    def _chunk_markdown_doc(self, text: str, base_metadata: Dict[str, Any]) -> List[ChunkRecord]:
        MarkdownHeaderTextSplitter, _ = _load_langchain_splitter_classes()

        headers_to_split_on = [(("#" * i), f"Header {i}") for i in range(1, 7)]
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=True,
        )
        docs = md_splitter.split_text(text)
        if not docs:
            return []

        recursive_splitter = self._build_recursive_splitter()
        split_docs = recursive_splitter.split_documents(docs)

        records: List[ChunkRecord] = []
        for idx, doc in enumerate(split_docs):
            content = (doc.page_content or "").strip()
            if not content:
                continue
            heading_metadata = dict(doc.metadata or {})
            heading_path = self._heading_path_from_metadata(heading_metadata)
            metadata = {
                **base_metadata,
                "chunking_mode": "markdown",
                "heading_path": heading_path,
                "headers": heading_metadata,
            }
            embedding_text = self._embedding_text_for_markdown(content, metadata)
            records.append(self._record(content, embedding_text, metadata, idx))
        return records


def get_text_preprocessor(config, embedding_model=None) -> TextPreprocessor:
    name = (getattr(config, "text_preprocessor_class_name", None) or "").lower()
    if name and name not in ("textpreprocessor", "none", "noop", "identity"):
        logger.warning(
            "text_preprocessor_class_name is deprecated; use chunking_mode instead."
        )
    if getattr(config, "preprocess_split_markdown", False):
        logger.warning("preprocess_split_markdown is deprecated; use chunking_mode='markdown'.")
    return ChunkingPreprocessor(config, embedding_model=embedding_model)
