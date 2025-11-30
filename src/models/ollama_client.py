"""Ollama-based LLM and Embedding clients."""

from __future__ import annotations

import logging
from typing import AsyncIterator, List, Optional

import httpx

from .base import EmbeddingClient, LLMClient

logger = logging.getLogger(__name__)


class OllamaLLMClient(LLMClient):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "qwen2.5:7b",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    async def agenerate(self, prompt: str, **params) -> str:
        temperature = params.get("temperature", self.temperature)
        max_tokens = params.get("max_tokens", self.max_tokens)
        model = params.get("model", self.model_name)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")

    async def astream(self, prompt: str, **params) -> AsyncIterator[str]:
        temperature = params.get("temperature", self.temperature)
        max_tokens = params.get("max_tokens", self.max_tokens)
        model = params.get("model", self.model_name)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            ) as response:
                response.raise_for_status()
                import json
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]


class OllamaEmbeddingClient(EmbeddingClient):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "nomic-embed-text",
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

    async def aembed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for text in texts:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.model_name,
                        "prompt": text,
                    },
                )
                response.raise_for_status()
                data = response.json()
                embeddings.append(data.get("embedding", []))
        return embeddings
