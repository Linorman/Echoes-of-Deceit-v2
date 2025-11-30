"""Base interfaces for model providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, List


class LLMClient(ABC):
    @abstractmethod
    async def agenerate(self, prompt: str, **params) -> str:
        ...

    @abstractmethod
    async def astream(self, prompt: str, **params) -> AsyncIterator[str]:
        ...

    def generate(self, prompt: str, **params) -> str:
        import asyncio
        return asyncio.run(self.agenerate(prompt, **params))


class EmbeddingClient(ABC):
    @abstractmethod
    async def aembed(self, texts: List[str]) -> List[List[float]]:
        ...

    def embed(self, texts: List[str]) -> List[List[float]]:
        import asyncio
        return asyncio.run(self.aembed(texts))
