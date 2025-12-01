"""LangChain-based LLM and Embedding clients."""

from __future__ import annotations

import logging
from typing import AsyncIterator, List, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage

from .base import EmbeddingClient, LLMClient

logger = logging.getLogger(__name__)


class LangChainLLMClient(LLMClient):
    def __init__(
        self,
        chat_model: BaseChatModel,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self._chat_model = chat_model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def chat_model(self) -> BaseChatModel:
        return self._chat_model

    async def agenerate(self, prompt: str, **params) -> str:
        messages = [HumanMessage(content=prompt)]
        response = await self._chat_model.ainvoke(messages)
        return str(response.content)

    async def astream(self, prompt: str, **params) -> AsyncIterator[str]:
        messages = [HumanMessage(content=prompt)]
        async for chunk in self._chat_model.astream(messages):
            if chunk.content:
                yield str(chunk.content)


class LangChainEmbeddingClient(EmbeddingClient):
    def __init__(self, embeddings: Embeddings):
        self._embeddings = embeddings

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    async def aembed(self, texts: List[str]) -> List[List[float]]:
        return await self._embeddings.aembed_documents(texts)


def create_ollama_llm_client(
    base_url: str = "http://localhost:11434",
    model_name: str = "qwen2.5:7b",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    **kwargs,
) -> LangChainLLMClient:
    from langchain_ollama import ChatOllama

    chat_model = ChatOllama(
        base_url=base_url,
        model=model_name,
        temperature=temperature,
        num_predict=max_tokens,
        **kwargs,
    )
    return LangChainLLMClient(chat_model, temperature=temperature, max_tokens=max_tokens)


def create_ollama_embedding_client(
    base_url: str = "http://localhost:11434",
    model_name: str = "nomic-embed-text",
    **kwargs,
) -> LangChainEmbeddingClient:
    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(
        base_url=base_url,
        model=model_name,
        **kwargs,
    )
    return LangChainEmbeddingClient(embeddings)


def create_openai_llm_client(
    base_url: str = "https://api.openai.com/v1",
    api_key: str = "",
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout: float = 120.0,
    **kwargs,
) -> LangChainLLMClient:
    from langchain_openai import ChatOpenAI

    chat_model = ChatOpenAI(
        base_url=base_url,
        api_key=api_key or None,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        **kwargs,
    )
    return LangChainLLMClient(chat_model, temperature=temperature, max_tokens=max_tokens)


def create_openai_embedding_client(
    base_url: str = "https://api.openai.com/v1",
    api_key: str = "",
    model_name: str = "text-embedding-3-small",
    timeout: float = 60.0,
    **kwargs,
) -> LangChainEmbeddingClient:
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        base_url=base_url,
        api_key=api_key or None,
        model=model_name,
        timeout=timeout,
        **kwargs,
    )
    return LangChainEmbeddingClient(embeddings)
