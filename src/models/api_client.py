"""OpenAI-compatible API LLM and Embedding clients."""

from __future__ import annotations

import logging
from typing import AsyncIterator, List, Optional

import httpx

from .base import EmbeddingClient, LLMClient

logger = logging.getLogger(__name__)


class APILLMClient(LLMClient):
    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _get_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def agenerate(self, prompt: str, **params) -> str:
        temperature = params.get("temperature", self.temperature)
        max_tokens = params.get("max_tokens", self.max_tokens)
        model = params.get("model", self.model_name)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return ""

    async def astream(self, prompt: str, **params) -> AsyncIterator[str]:
        temperature = params.get("temperature", self.temperature)
        max_tokens = params.get("max_tokens", self.max_tokens)
        model = params.get("model", self.model_name)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()
                import json
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue


class APIEmbeddingClient(EmbeddingClient):
    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model_name: str = "text-embedding-3-small",
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout

    def _get_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def aembed(self, texts: List[str]) -> List[List[float]]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers=self._get_headers(),
                json={
                    "model": self.model_name,
                    "input": texts,
                },
            )
            response.raise_for_status()
            data = response.json()
            embeddings_data = data.get("data", [])
            embeddings_data.sort(key=lambda x: x.get("index", 0))
            return [item.get("embedding", []) for item in embeddings_data]
