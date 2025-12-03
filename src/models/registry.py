"""Model provider registry for managing LLM and Embedding clients."""

from __future__ import annotations

import logging
from typing import Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from config import ConfigLoader, ModelsConfig
from models.base import EmbeddingClient, LLMClient
from models.langchain_client import (
    LangChainLLMClient,
    LangChainEmbeddingClient,
    create_ollama_llm_client,
    create_ollama_embedding_client,
    create_openai_llm_client,
    create_openai_embedding_client,
)

logger = logging.getLogger(__name__)


class ModelProviderRegistry:
    def __init__(self, config: Optional[ModelsConfig] = None):
        if config is None:
            loader = ConfigLoader()
            config = loader.load_models_config()
        self._config = config
        self._llm_client: Optional[LangChainLLMClient] = None
        self._embedding_client: Optional[LangChainEmbeddingClient] = None
        self._chat_model: Optional[BaseChatModel] = None
        self._embeddings: Optional[Embeddings] = None

    @property
    def provider(self) -> str:
        """Legacy property - returns the default provider."""
        return self._config.provider

    @property
    def llm_provider(self) -> str:
        """Get the LLM provider name."""
        return self._config.get_llm_config().provider

    @property
    def embedding_provider(self) -> str:
        """Get the Embedding provider name."""
        return self._config.get_embedding_config().provider

    def get_llm_client(self) -> LLMClient:
        if self._llm_client is not None:
            return self._llm_client

        llm_cfg = self._config.get_llm_config()
        
        if llm_cfg.provider == "ollama":
            self._llm_client = create_ollama_llm_client(
                base_url=llm_cfg.base_url,
                model_name=llm_cfg.model_name,
                temperature=llm_cfg.temperature,
                max_tokens=llm_cfg.max_tokens,
                api_key=llm_cfg.api_key,
            )
        else:
            self._llm_client = create_openai_llm_client(
                base_url=llm_cfg.base_url,
                api_key=llm_cfg.api_key,
                model_name=llm_cfg.model_name,
                temperature=llm_cfg.temperature,
                max_tokens=llm_cfg.max_tokens,
            )

        logger.info(
            "Initialized LLM client: provider=%s, model=%s",
            llm_cfg.provider,
            llm_cfg.model_name,
        )
        return self._llm_client

    def get_embedding_client(self) -> EmbeddingClient:
        if self._embedding_client is not None:
            return self._embedding_client

        emb_cfg = self._config.get_embedding_config()
        
        if emb_cfg.provider == "ollama":
            self._embedding_client = create_ollama_embedding_client(
                base_url=emb_cfg.base_url,
                model_name=emb_cfg.model_name,
                api_key=emb_cfg.api_key,
            )
        else:
            self._embedding_client = create_openai_embedding_client(
                base_url=emb_cfg.base_url,
                api_key=emb_cfg.api_key,
                model_name=emb_cfg.model_name,
            )

        logger.info(
            "Initialized Embedding client: provider=%s, model=%s",
            emb_cfg.provider,
            emb_cfg.model_name,
        )
        return self._embedding_client

    def get_chat_model(self) -> BaseChatModel:
        llm_client = self.get_llm_client()
        if isinstance(llm_client, LangChainLLMClient):
            return llm_client.chat_model
        raise TypeError("LLM client is not a LangChain-based client")

    def get_embeddings(self) -> Embeddings:
        embedding_client = self.get_embedding_client()
        if isinstance(embedding_client, LangChainEmbeddingClient):
            return embedding_client.embeddings
        raise TypeError("Embedding client is not a LangChain-based client")

    def get_provider_options(self) -> dict:
        """Get provider options for RAG and other integrations."""
        llm_cfg = self._config.get_llm_config()
        emb_cfg = self._config.get_embedding_config()
        
        return {
            "llm_host": llm_cfg.base_url,
            "llm_model_name": llm_cfg.model_name,
            "llm_api_key": llm_cfg.api_key,
            "llm_provider": llm_cfg.provider,
            "embedding_host": emb_cfg.base_url,
            "embedding_model_name": emb_cfg.model_name,
            "embedding_api_key": emb_cfg.api_key,
            "embedding_provider": emb_cfg.provider,
            "embedding_dim": emb_cfg.embedding_dim,
        }
