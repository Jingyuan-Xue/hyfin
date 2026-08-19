"""Document-scoped Phase 8 adapter for the warm Phase 7 A2RAG worker.

The provider deliberately accepts only a single frozen ``document_id``.  It
maps that document to the audited Phase 7 company/year identity before calling
the legacy worker, then validates every returned chunk against the immutable
document chunk map.  A worker cannot widen scope by returning a plausible
chunk from another report.
"""

from __future__ import annotations

import hashlib
import json
import selectors
import subprocess
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_DIR = ROOT / "data/indexes/a2rag_index"
DEFAULT_WORKER_SCRIPT = ROOT / "scripts/query_type3_evidence.py"
DEFAULT_WORKER_PYTHON = ROOT / "refs/a2rag_runtime/.venv/bin/python"

EVIDENCE_PROVIDER_RESULT_SCHEMA = "finglmqa.phase8.evidence_provider_result.v1"
PHASE7_WORKER_PROTOCOL = "finglmqa.phase7.a2rag_worker.v1"
MAX_TOP_K = 5
_SCORE_QUANTUM = Decimal("0.00000001")


class EvidenceProviderError(RuntimeError):
    """Fail-closed Phase 8 evidence provider boundary error."""


@runtime_checkable
class EvidenceWorkerTransport(Protocol):
    """Transport used by :class:`DocumentScopedEvidenceProvider`.

    ``query`` returns the inner Phase 7 query result, not its JSONL envelope.
    This small boundary makes unit tests independent of a model process.
    """

    def query(self, request: Mapping[str, Any]) -> dict[str, Any]: ...


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvidenceProviderError(f"{path.name} must contain an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise EvidenceProviderError(f"{path.name}:{line_number} must contain an object")
            rows.append(value)
    return rows


def _portable_path(value: Any, root: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceProviderError("evidence source_markdown must be a non-empty string")
    path = Path(value)
    if not path.is_absolute():
        portable = path.as_posix()
    else:
        try:
            portable = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            mirrored = root / "refs/source_markdown" / path.name
            portable = (
                mirrored.relative_to(root).as_posix()
                if mirrored.is_file()
                else f"external:{path.name}"
            )
    if Path(portable).is_absolute():  # defensive: never expose a host path
        raise EvidenceProviderError("evidence source path was not made portable")
    return portable


def _validate_request(request: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise EvidenceProviderError("EvidenceProvider request must be an object")
    value = dict(request)
    expected = {"document_id", "question", "top_k"}
    if set(value) != expected:
        raise EvidenceProviderError("EvidenceProvider request fields do not match the frozen boundary")
    for field in ("document_id", "question"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise EvidenceProviderError(f"EvidenceProvider request {field} must be a non-empty string")
    top_k = value["top_k"]
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
        raise EvidenceProviderError(f"EvidenceProvider top_k must be between 1 and {MAX_TOP_K}")
    return value


def _score_text(value: Any) -> str:
    if isinstance(value, bool):
        raise EvidenceProviderError("evidence score must be numeric")
    try:
        score = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise EvidenceProviderError("evidence score must be numeric") from exc
    if not score.is_finite():
        raise EvidenceProviderError("evidence score must be finite")
    return format(score.quantize(_SCORE_QUANTUM, rounding=ROUND_HALF_EVEN), ".8f")


class A2RAGWarmWorkerTransport:
    """Single-concurrency JSONL transport for the existing Phase 7 worker.

    The expensive retriever process is started once and reused.  Runtime
    details and request counters remain inside the transport and are never
    copied into deterministic Phase 8 provider results.
    """

    def __init__(
        self,
        *,
        python_executable: str | Path = DEFAULT_WORKER_PYTHON,
        worker_script: str | Path = DEFAULT_WORKER_SCRIPT,
        device: str = "auto",
        model_cache: str | Path | None = None,
        load_dense: bool = True,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.python_executable = Path(python_executable)
        self.worker_script = Path(worker_script)
        self.device = device
        self.model_cache = Path(model_cache) if model_cache is not None else None
        self.load_dense = bool(load_dense)
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._process: subprocess.Popen[str] | None = None
        self._request_ordinal = 0
        self._ready: dict[str, Any] | None = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _read_message(self) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise EvidenceProviderError("evidence worker is not running")
        selector = selectors.DefaultSelector()
        try:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(self.timeout_seconds):
                raise EvidenceProviderError("evidence worker response timed out")
            raw = process.stdout.readline()
        finally:
            selector.close()
        if not raw:
            raise EvidenceProviderError(
                f"evidence worker exited before replying (exit_code={process.poll()})"
            )
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EvidenceProviderError("evidence worker emitted invalid JSONL") from exc
        if not isinstance(message, dict):
            raise EvidenceProviderError("evidence worker message must be an object")
        if message.get("protocol_version") != PHASE7_WORKER_PROTOCOL:
            raise EvidenceProviderError("evidence worker protocol version mismatch")
        return message

    def start(self) -> None:
        if self.is_running:
            return
        if self._process is not None:
            # A prior worker may have exited between requests.  Close its file
            # descriptors before starting a clean protocol lifecycle.
            self.close(force=True)
        if not self.python_executable.is_file() or not self.worker_script.is_file():
            raise EvidenceProviderError("evidence worker executable or script is missing")
        command = [
            str(self.python_executable), str(self.worker_script), "--serve", "--device", self.device,
        ]
        if self.model_cache is not None:
            command.extend(["--model-cache", str(self.model_cache)])
        if not self.load_dense:
            command.append("--no-dense")
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                # Inherit stderr so model/runtime diagnostics cannot fill an
                # unread pipe and deadlock the JSONL protocol.
                stderr=None,
                text=True,
                bufsize=1,
            )
            ready = self._read_message()
        except Exception:
            self.close(force=True)
            raise
        if ready.get("type") != "ready" or ready.get("concurrency") != 1:
            self.close(force=True)
            raise EvidenceProviderError("evidence worker did not advertise the frozen ready protocol")
        commands = ready.get("commands")
        if not isinstance(commands, list) or not {"ping", "query", "shutdown"}.issubset(commands):
            self.close(force=True)
            raise EvidenceProviderError("evidence worker command set is incomplete")
        self._ready = ready

    def _exchange(self, message_type: str, body: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self.start()
        process = self._process
        if process is None or process.stdin is None:
            raise EvidenceProviderError("evidence worker stdin is unavailable")
        self._request_ordinal += 1
        request_id = f"phase8-evidence-{self._request_ordinal:08d}"
        payload = {"type": message_type, "request_id": request_id, **dict(body or {})}
        try:
            process.stdin.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise EvidenceProviderError("evidence worker pipe failed") from exc
        response = self._read_message()
        if response.get("request_id") != request_id:
            raise EvidenceProviderError("evidence worker response request_id mismatch")
        if response.get("type") == "error":
            raise EvidenceProviderError(
                f"evidence worker rejected request ({response.get('error_type', 'unknown')})"
            )
        return response

    def ping(self) -> None:
        response = self._exchange("ping")
        if response.get("type") != "pong":
            raise EvidenceProviderError("evidence worker ping protocol mismatch")

    def query(self, request: Mapping[str, Any]) -> dict[str, Any]:
        response = self._exchange("query", request)
        if response.get("type") != "result" or not isinstance(response.get("result"), dict):
            raise EvidenceProviderError("evidence worker query protocol mismatch")
        return response["result"]

    def close(self, *, force: bool = False) -> None:
        process = self._process
        if process is None:
            return
        try:
            if process.poll() is None and not force:
                try:
                    response = self._exchange("shutdown")
                    if response.get("type") != "shutdown_ack":
                        raise EvidenceProviderError("evidence worker shutdown protocol mismatch")
                except Exception:
                    force = True
            if process.poll() is None:
                if force:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        finally:
            for stream in (process.stdin, process.stdout):
                if stream is not None and not stream.closed:
                    stream.close()
            self._process = None
            self._ready = None

    def __enter__(self) -> "A2RAGWarmWorkerTransport":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort interpreter cleanup
        try:
            self.close(force=True)
        except Exception:
            pass


class DocumentScopedEvidenceProvider:
    """Validate and normalize one-document Phase 7 dense retrieval."""

    def __init__(
        self,
        transport: EvidenceWorkerTransport,
        *,
        root: str | Path = ROOT,
        index_dir: str | Path | None = None,
    ) -> None:
        if not isinstance(transport, EvidenceWorkerTransport):
            raise TypeError("transport must implement EvidenceWorkerTransport")
        self.transport = transport
        self.root = Path(root).resolve()
        self.index_dir = (
            Path(index_dir).resolve() if index_dir is not None else self.root / "data/indexes/a2rag_index"
        )
        manifest_path = self.index_dir / "index_manifest.json"
        map_path = self.index_dir / "document_chunk_map.jsonl"
        self._manifest = _read_json(manifest_path)
        expected_map_hash = self._manifest.get("hashes", {}).get("document_chunk_map_sha256")
        actual_map_hash = _sha256_file(map_path)
        if expected_map_hash != actual_map_hash:
            raise EvidenceProviderError("Phase 7 document chunk map hash mismatch")
        evidence_artifact = self._manifest.get("artifacts", {}).get("evidence_chunks")
        if not isinstance(evidence_artifact, str) or not evidence_artifact:
            raise EvidenceProviderError("Phase 7 evidence artifact path is missing")
        evidence_path = Path(evidence_artifact)
        if not evidence_path.is_absolute():
            evidence_path = self.root / evidence_path
        actual_evidence_hash = _sha256_file(evidence_path)
        if actual_evidence_hash != self._manifest.get("hashes", {}).get("evidence_chunks_sha256"):
            raise EvidenceProviderError("Phase 7 evidence chunks hash mismatch")

        # The worker is trusted only for ranking scores.  Every returned
        # evidence field is checked against this hash-pinned Phase 7 artifact
        # before it can reach the executor or citation builder.
        self._evidence_chunks: dict[str, dict[str, Any]] = {}
        for row in _read_jsonl(evidence_path):
            chunk_id = row.get("evidence_chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id or chunk_id in self._evidence_chunks:
                raise EvidenceProviderError("Phase 7 evidence artifact has an invalid chunk ID")
            content = row.get("content")
            if not isinstance(content, str) or not content.strip():
                raise EvidenceProviderError("Phase 7 evidence artifact has invalid content")
            content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if row.get("content_sha256") != content_sha256:
                raise EvidenceProviderError("Phase 7 evidence artifact content hash mismatch")
            source = _portable_path(row.get("source_markdown"), self.root)
            if source.startswith("external:"):
                raise EvidenceProviderError("Phase 7 evidence source is outside the workspace projection")
            self._evidence_chunks[chunk_id] = {
                "document_id": row.get("document_id"),
                "company_name": row.get("company_name"),
                "stock_code": row.get("stock_code"),
                "report_year": row.get("report_year"),
                "section_path": row.get("section_path"),
                "semantic_tags": row.get("semantic_tags"),
                "line_range": row.get("line_range"),
                "source_markdown": source,
                "content": content,
                "content_sha256": content_sha256,
            }
        if len(self._evidence_chunks) != self._manifest.get("counts", {}).get("evidence_chunks"):
            raise EvidenceProviderError("Phase 7 evidence chunk count mismatch")

        self._documents: dict[str, dict[str, Any]] = {}
        for row in _read_jsonl(map_path):
            document_id = row.get("document_id")
            chunk_ids = row.get("chunk_ids")
            if not isinstance(document_id, str) or not document_id:
                raise EvidenceProviderError("Phase 7 document map has an invalid document_id")
            if document_id in self._documents:
                raise EvidenceProviderError("Phase 7 document map contains duplicate document_id")
            if not isinstance(chunk_ids, list) or any(not isinstance(item, str) for item in chunk_ids):
                raise EvidenceProviderError("Phase 7 document map has invalid chunk_ids")
            if len(chunk_ids) != len(set(chunk_ids)) or row.get("chunk_count") != len(chunk_ids):
                raise EvidenceProviderError("Phase 7 document chunk cardinality is invalid")
            source_markdown = _portable_path(row.get("source_markdown"), self.root)
            if source_markdown.startswith("external:"):
                raise EvidenceProviderError("Phase 7 source markdown is outside the workspace projection")
            self._documents[document_id] = {
                "document_id": document_id,
                "company": row.get("company_name"),
                "company_full": row.get("company_full"),
                "stock_code": row.get("stock_code"),
                "report_year": row.get("report_year"),
                "source_markdown": source_markdown,
                "chunk_ordinals": {chunk_id: index for index, chunk_id in enumerate(chunk_ids, 1)},
            }
            for chunk_id in chunk_ids:
                frozen = self._evidence_chunks.get(chunk_id)
                if frozen is None or frozen["document_id"] != document_id:
                    raise EvidenceProviderError("Phase 7 document map and evidence artifact disagree")
        if len(self._documents) != self._manifest.get("counts", {}).get("evidence_documents"):
            raise EvidenceProviderError("Phase 7 evidence document count mismatch")

        fingerprint_payload = {
            "schema_version": self._manifest.get("schema_version"),
            "builder_version": self._manifest.get("builder_version"),
            "document_chunk_map_sha256": actual_map_hash,
            "evidence_chunks_sha256": actual_evidence_hash,
        }
        encoded = json.dumps(
            fingerprint_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.provider_fingerprint = hashlib.sha256(encoded).hexdigest()

    def retrieve(self, request: Mapping[str, Any]) -> dict[str, Any]:
        checked = _validate_request(request)
        document = self._documents.get(checked["document_id"])
        if document is None:
            raise EvidenceProviderError("unknown Phase 7 evidence document_id")
        for field in ("company", "stock_code"):
            if not isinstance(document[field], str) or not document[field]:
                raise EvidenceProviderError(f"Phase 7 document identity has invalid {field}")
        if isinstance(document["report_year"], bool) or not isinstance(document["report_year"], int):
            raise EvidenceProviderError("Phase 7 document identity has invalid report_year")

        worker_request = {
            "company": document["company"],
            "report_year": document["report_year"],
            "question": checked["question"],
            "top_k": checked["top_k"],
        }
        try:
            raw = self.transport.query(worker_request)
        except Exception as exc:
            if isinstance(exc, EvidenceProviderError):
                raise
            raise EvidenceProviderError("evidence worker request failed") from exc
        return self._normalize_result(raw, document, checked["top_k"])

    def _normalize_result(
        self, raw: Any, document: Mapping[str, Any], top_k: int,
    ) -> dict[str, Any]:
        if not isinstance(raw, dict) or raw.get("status") != "ok":
            raise EvidenceProviderError("evidence worker did not return an ok result")
        resolver = raw.get("resolver")
        if not isinstance(resolver, dict) or resolver.get("status") != "unique":
            raise EvidenceProviderError("evidence worker resolver was not unique")
        identity_checks = {
            "document_id": document["document_id"],
            "report_year": document["report_year"],
            "stock_code": document["stock_code"],
        }
        for field, expected in identity_checks.items():
            if resolver.get(field) != expected:
                raise EvidenceProviderError(f"evidence worker resolver {field} widened scope")

        retrieval = raw.get("retrieval")
        if not isinstance(retrieval, dict):
            raise EvidenceProviderError("evidence worker retrieval is missing")
        if retrieval.get("candidate_document_id") != document["document_id"]:
            raise EvidenceProviderError("evidence worker candidate scope widened")
        if retrieval.get("candidate_prefilter") != "company_year_resolver_document_allow_list":
            raise EvidenceProviderError("evidence worker candidate prefilter contract mismatch")
        if retrieval.get("prefilter_applied_before_scoring") is not True:
            raise EvidenceProviderError("evidence worker did not prove pre-scoring document filtering")
        if retrieval.get("candidate_chunk_count") != len(document["chunk_ordinals"]):
            raise EvidenceProviderError("evidence worker candidate chunk allow-list is incomplete")
        retrieval_method = retrieval.get("retrieval_method")
        if not isinstance(retrieval_method, str) or not retrieval_method:
            raise EvidenceProviderError("evidence worker retrieval_method is invalid")
        raw_chunks = retrieval.get("chunks")
        if not isinstance(raw_chunks, list) or len(raw_chunks) > top_k:
            raise EvidenceProviderError("evidence worker returned invalid chunk cardinality")
        if retrieval.get("top_k") != len(raw_chunks):
            raise EvidenceProviderError("evidence worker reported inconsistent top_k")

        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        ordinals = document["chunk_ordinals"]
        for item in raw_chunks:
            if not isinstance(item, dict):
                raise EvidenceProviderError("evidence worker chunk must be an object")
            chunk_id = item.get("evidence_chunk_id")
            if not isinstance(chunk_id, str) or chunk_id not in ordinals:
                raise EvidenceProviderError("evidence worker returned a chunk outside the document allow-list")
            if chunk_id in seen:
                raise EvidenceProviderError("evidence worker returned a duplicate chunk")
            seen.add(chunk_id)
            frozen = self._evidence_chunks[chunk_id]
            chunk_identity = {
                "document_id": document["document_id"],
                "company_name": document["company"],
                "stock_code": document["stock_code"],
                "report_year": document["report_year"],
            }
            for field, expected in chunk_identity.items():
                if item.get(field) != expected:
                    raise EvidenceProviderError(f"evidence chunk {field} widened scope")
            source = _portable_path(item.get("source_markdown"), self.root)
            if source != document["source_markdown"]:
                raise EvidenceProviderError("evidence chunk source path does not match its document")
            section_path = item.get("section_path")
            tags = item.get("semantic_tags")
            line_range = item.get("line_range")
            content = item.get("content")
            if not isinstance(section_path, list) or any(not isinstance(value, str) for value in section_path):
                raise EvidenceProviderError("evidence chunk section_path is invalid")
            if not isinstance(tags, list) or any(not isinstance(value, str) for value in tags):
                raise EvidenceProviderError("evidence chunk semantic_tags is invalid")
            if (
                not isinstance(line_range, list) or len(line_range) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in line_range)
                or line_range[0] < 1 or line_range[1] < line_range[0]
            ):
                raise EvidenceProviderError("evidence chunk line_range is invalid")
            if not isinstance(content, str) or not content.strip():
                raise EvidenceProviderError("evidence chunk content is invalid")
            immutable_fields = {
                "document_id": document["document_id"],
                "company_name": document["company"],
                "stock_code": document["stock_code"],
                "report_year": document["report_year"],
                "section_path": section_path,
                "semantic_tags": tags,
                "line_range": line_range,
                "source_markdown": source,
                "content": content,
            }
            for field, actual in immutable_fields.items():
                if frozen[field] != actual:
                    raise EvidenceProviderError(
                        f"evidence worker chunk {field} differs from immutable Phase 7 evidence"
                    )
            normalized.append({
                "chunk_id": chunk_id,
                "document_chunk_ordinal": ordinals[chunk_id],
                "score": _score_text(item.get("score")),
                "document_id": document["document_id"],
                "company": document["company"],
                "stock_code": document["stock_code"],
                "report_year": document["report_year"],
                "section_path": list(frozen["section_path"]),
                "semantic_tags": list(frozen["semantic_tags"]),
                "line_range": list(frozen["line_range"]),
                "source_markdown": frozen["source_markdown"],
                "content": frozen["content"],
            })

        normalized.sort(
            key=lambda row: (-Decimal(row["score"]), row["document_chunk_ordinal"], row["chunk_id"])
        )
        return {
            "schema_version": EVIDENCE_PROVIDER_RESULT_SCHEMA,
            "status": "ok",
            "document_id": document["document_id"],
            "company": document["company"],
            "stock_code": document["stock_code"],
            "report_year": document["report_year"],
            "retrieval_method": retrieval_method,
            "provider_fingerprint": self.provider_fingerprint,
            "chunks": normalized,
        }


__all__ = [
    "A2RAGWarmWorkerTransport",
    "DocumentScopedEvidenceProvider",
    "EVIDENCE_PROVIDER_RESULT_SCHEMA",
    "EvidenceProviderError",
    "EvidenceWorkerTransport",
    "MAX_TOP_K",
]
