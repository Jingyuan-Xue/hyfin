import functools
import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from typing import List, Tuple

import httpx
from filelock import FileLock
from tenacity import retry, stop_after_attempt, wait_fixed

from ..config.settings import BaseConfig
from ..models.chat_base import BaseChatModel, ChatConfig
from ..utils.chat_utils import TextChatMessage
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        messages = args[0] if args else kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        gen_params = getattr(self, "chat_config", {}).generate_params if hasattr(self, "chat_config") else {}
        key_data = {
            "messages": messages,
            "model": kwargs.get("model", gen_params.get("model")),
            "seed": kwargs.get("seed", gen_params.get("seed")),
            "temperature": kwargs.get("temperature", gen_params.get("temperature")),
        }
        key_hash = hashlib.sha256(json.dumps(key_data, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        lock_file = self.cache_file_name + ".lock"

        with FileLock(lock_file):
            with sqlite3.connect(self.cache_file_name) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        message TEXT,
                        metadata TEXT
                    )
                    """
                )
                cursor.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
                row = cursor.fetchone()
                if row is not None:
                    message, metadata_str = row
                    return message, json.loads(metadata_str), True

        message, metadata = func(self, *args, **kwargs)

        with FileLock(lock_file):
            with sqlite3.connect(self.cache_file_name) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        message TEXT,
                        metadata TEXT
                    )
                    """
                )
                cursor.execute(
                    "INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                    (key_hash, message, json.dumps(metadata)),
                )
                conn.commit()

        return message, metadata, False

    return wrapper


def dynamic_retry_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        decorated_func = retry(
            stop=stop_after_attempt(getattr(self, "max_retries", 5)),
            wait=wait_fixed(1),
        )(func)
        return decorated_func(self, *args, **kwargs)

    return wrapper


class ChatGateway(BaseChatModel):
    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "ChatGateway":
        cache_dir = os.path.join(global_config.save_dir, "chat_cache")
        return cls(cache_dir=cache_dir, global_config=global_config)

    def __init__(self, cache_dir, global_config, cache_filename: str = None, high_throughput: bool = True, **kwargs):
        super().__init__(global_config=global_config)
        self.cache_dir = cache_dir
        self.global_config = global_config
        self.chat_model = global_config.chat_model
        self.chat_base_url = global_config.chat_base_url
        self.max_retries = kwargs.get("max_retries", global_config.max_retry_attempts)

        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{self.chat_model.replace('/', '_')}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)

        limits = httpx.Limits(max_connections=500, max_keepalive_connections=100) if high_throughput else None
        self.client = httpx.Client(limits=limits, timeout=httpx.Timeout(5 * 60, read=5 * 60))
        self.api_key = os.getenv("A2RAG_API_KEY")
        self._init_chat_config()

    def _init_chat_config(self) -> None:
        config_dict = dict(self.global_config.__dict__)
        config_dict["chat_model"] = self.global_config.chat_model
        config_dict["chat_base_url"] = self.global_config.chat_base_url
        config_dict["generate_params"] = {
            "model": self.global_config.chat_model,
            "max_completion_tokens": config_dict.get("max_new_tokens", 400),
            "n": config_dict.get("num_gen_choices", 1),
            "seed": config_dict.get("seed", 0),
            "temperature": config_dict.get("temperature", 0.0),
        }
        self.chat_config = ChatConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s chat_config: {self.chat_config}")

    @cache_response
    @dynamic_retry_decorator
    def infer(self, messages: List[TextChatMessage], **kwargs) -> Tuple[str, dict]:
        if not self.chat_base_url:
            raise ValueError("chat_base_url is required for ChatGateway.")

        params = deepcopy(self.chat_config.generate_params)
        params.update(kwargs)
        params["messages"] = messages
        if "gpt" not in str(params.get("model", "")) and "max_completion_tokens" in params:
            params["max_tokens"] = params.pop("max_completion_tokens")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = self.client.post(
            self.chat_base_url.rstrip("/") + "/chat/completions",
            headers=headers,
            json=params,
        )
        response.raise_for_status()
        payload = response.json()

        response_message = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage", {})
        metadata = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "finish_reason": payload["choices"][0].get("finish_reason"),
        }
        return response_message, metadata
