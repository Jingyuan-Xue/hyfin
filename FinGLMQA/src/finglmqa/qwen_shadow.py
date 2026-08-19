"""Evidence-grounded OpenAI-compatible GeneratorPort and local vLLM lifecycle.

Online and local model output must pass the unchanged Phase 8 EvidenceExecutor
before it is counted as an accepted claim.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import httpx


ROOT = Path(__file__).resolve().parents[2]

CLAIM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["claims"],
    "properties": {
        "claims": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "evidence_chunk_ids"],
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "evidence_chunk_ids": {
                        "type": "array", "minItems": 1, "maxItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
            },
        }
    },
}


class QwenGeneratorError(RuntimeError):
    pass


class QwenShadowGenerator:
    """Strict OpenAI-compatible GeneratorPort with no prompt logging."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8011/v1",
        model: str = "finglmqa-qwen3.6-27b",
        api_key: str | None = None,
        timeout_seconds: float = 130.0,
        max_tokens: int = 4096,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.last_outcome = "not_called"

    @staticmethod
    def _chunk_aliases(chunks: list[Any]) -> dict[str, str]:
        """Use short stable aliases so models do not corrupt long chunk UUIDs."""

        return {
            f"E{index}": row["chunk_id"]
            for index, row in enumerate(chunks, start=1)
            if isinstance(row, dict) and isinstance(row.get("chunk_id"), str)
        }

    @staticmethod
    def _messages(request: Mapping[str, Any]) -> list[dict[str, str]]:
        chunks = request.get("chunks")
        if not isinstance(chunks, list):
            raise QwenGeneratorError("claim-builder chunks are invalid")
        aliases = QwenShadowGenerator._chunk_aliases(chunks)
        chunk_by_id = {
            row["chunk_id"]: row
            for row in chunks
            if isinstance(row, dict) and isinstance(row.get("chunk_id"), str)
        }
        # This string exists only in process memory and is never logged.
        evidence = "\n\n".join(
            f"<chunk id={json.dumps(alias, ensure_ascii=False)}>\n"
            f"{chunk_by_id[chunk_id]['content']}\n</chunk>"
            for alias, chunk_id in aliases.items()
        )
        return [
            {
                "role": "system",
                "content": (
                    "你是有证据约束的年报回答生成器。选择最多五条最能回答问题的证据句。每条 text 必须从一个给定 "
                    "chunk 中连续逐字复制，并在 evidence_chunk_ids 中只返回该 chunk_id；不得改写、跨 chunk 拼接、"
                    "补充常识或数字。覆盖问题涉及的不同方面，避免重复。没有直接证据时返回空 claims。"
                    "证据编号只有 E1、E2 等短编号，必须原样返回。不要解释或分析，立即输出结果。"
                    "输出必须严格为以下 json 对象：{\"claims\":[{\"text\":\"原文句子\",\"evidence_chunk_ids\":[\"E1\"]}]}，"
                    "不得增加任何其他字段。"
                ),
            },
            {
                "role": "user",
                "content": f"问题：{request['question']}\n\n候选证据：\n{evidence}",
            },
        ]

    @staticmethod
    def _validate_response(
        value: Any,
        allowed_chunk_ids: set[str],
        chunk_aliases: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(value, dict) or set(value) != {"claims"} or not isinstance(value["claims"], list):
            raise QwenGeneratorError("Qwen response shape is invalid")
        if len(value["claims"]) > 5:
            raise QwenGeneratorError("LLM returned too many claims")
        claims: list[dict[str, Any]] = []
        for row in value["claims"]:
            if not isinstance(row, dict) or set(row) != {"text", "evidence_chunk_ids"}:
                raise QwenGeneratorError("Qwen claim shape is invalid")
            text = row["text"]
            chunk_ids = row["evidence_chunk_ids"]
            if not isinstance(text, str) or not text.strip():
                raise QwenGeneratorError("Qwen claim text is invalid")
            if not isinstance(chunk_ids, list) or len(chunk_ids) != 1 or not isinstance(chunk_ids[0], str):
                raise QwenGeneratorError("Qwen cited a chunk outside the supplied set")
            cited_id = chunk_ids[0]
            resolved_id = (chunk_aliases or {}).get(cited_id, cited_id)
            if resolved_id not in allowed_chunk_ids:
                raise QwenGeneratorError("Qwen cited a chunk outside the supplied set")
            claims.append({"text": text.strip(), "evidence_chunk_ids": [resolved_id]})
        return {"claims": claims}

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def ping(self) -> None:
        """Verify the configured OpenAI-compatible endpoint without inference."""

        try:
            with httpx.Client(timeout=min(self.timeout_seconds, 15.0), headers=self._headers()) as client:
                response = client.get(f"{self.base_url}/models")
                response.raise_for_status()
                value = response.json()
            if not isinstance(value, dict) or not isinstance(value.get("data"), list):
                raise TypeError("model catalog is invalid")
        except Exception as exc:
            raise QwenGeneratorError("online generator endpoint is not ready") from exc

    def generate_claims(self, request: Mapping[str, Any]) -> dict[str, Any]:
        chunks = request.get("chunks")
        aliases = self._chunk_aliases(chunks) if isinstance(chunks, list) else {}
        allowed = {
            row["chunk_id"] for row in chunks
            if isinstance(row, dict) and isinstance(row.get("chunk_id"), str)
        } if isinstance(chunks, list) else set()
        body = {
            "model": self.model,
            "messages": self._messages(request),
            "temperature": 0,
            "top_p": 1,
            # Reasoning-capable endpoints may consume part of this budget
            # before emitting the JSON content.
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        try:
            with httpx.Client(timeout=self.timeout_seconds, headers=self._headers()) as client:
                response = client.post(f"{self.base_url}/chat/completions", json=body)
                response.raise_for_status()
                envelope = response.json()
            content = envelope["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            checked = self._validate_response(parsed, allowed, aliases)
        except Exception as exc:
            self.last_outcome = "generator_invalid_output"
            raise QwenGeneratorError("Qwen did not return valid structured extractive output") from exc
        self.last_outcome = "generator_refused" if not checked["claims"] else "proposed"
        return checked


class VLLMShadowServer:
    """Own one local vLLM process group and always stop it explicitly."""

    def __init__(self) -> None:
        self.host = os.environ.get("FINGLMQA_QWEN_HOST", "127.0.0.1")
        self.port = int(os.environ.get("FINGLMQA_QWEN_PORT", "8011"))
        self.binary = Path(os.environ.get(
            "FINGLMQA_VLLM_BIN", "/home/coder/demo/exposure_pipeline_workspace/.venv-vllm-auto/bin/vllm"
        ))
        self.model = Path(os.environ.get("FINGLMQA_QWEN_MODEL", ROOT / "refs/qwen_model"))
        self.served_name = os.environ.get("FINGLMQA_QWEN_SERVED_NAME", "finglmqa-qwen3.6-27b")
        self.process: subprocess.Popen[bytes] | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def _healthy(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/models", timeout=2)
            return response.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    def start(self, timeout_seconds: float = 600.0) -> None:
        if self.process is not None or self._healthy():
            raise RuntimeError("Qwen shadow port is already occupied")
        if not self.binary.is_file() or not self.model.exists():
            raise RuntimeError("pinned vLLM executable or Qwen snapshot is missing")
        command = [
            str(self.binary), "serve", str(self.model),
            "--host", self.host, "--port", str(self.port),
            "--served-model-name", self.served_name,
            "--dtype", "bfloat16", "--max-model-len", "16384",
            "--gpu-memory-utilization", "0.80", "--max-num-seqs", "1",
            "--seed", "0", "--generation-config", "vllm",
            "--language-model-only", "--no-trust-remote-code",
            "--no-enable-log-requests", "--disable-uvicorn-access-log",
            "--disable-log-stats", "--uvicorn-log-level", "warning",
        ]
        env = {
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "VLLM_NO_USAGE_STATS": "1",
            # The installed CUDA runtime cannot report Blackwell SM 12.x to
            # FlashInfer correctly.  vLLM's native sampler is deterministic
            # for the frozen seed and avoids that optional JIT path.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
        self.process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            # vLLM request/prompt logs are disabled; all remaining library
            # diagnostics are discarded so paths cannot enter release logs.
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.stop()
                raise RuntimeError("Qwen vLLM exited during startup")
            if self._healthy():
                return
            time.sleep(1)
        self.stop()
        raise TimeoutError("Qwen vLLM startup timed out")

    def stop(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
        self.process = None

    def __enter__(self) -> "VLLMShadowServer":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


__all__ = ["CLAIM_SCHEMA", "QwenGeneratorError", "QwenShadowGenerator", "VLLMShadowServer"]
