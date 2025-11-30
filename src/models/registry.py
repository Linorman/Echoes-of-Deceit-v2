"""Model provider registry for managing LLM and Embedding clients."""

from __future__ import annotations

import logging
from typing import Optional

from config import ConfigLoader, ModelsConfig
from models.api_client import APIEmbeddingClient, APILLMClient
from models.base import EmbeddingClient, LLMClient
from models.ollama_client import OllamaEmbeddingClient, OllamaLLMClient

logger = logging.getLogger(__name__)


class ModelProviderRegistry:
    def __init__(self, config: Optional[ModelsConfig] = None):
        if config is None:
            loader = ConfigLoader()
            config = loader.load_models_config()
        self._config = config
        self._llm_client: Optional[LLMClient] = None
        self._embedding_client: Optional[EmbeddingClient] = None

    @property
    def provider(self) -> str:
        return self._config.provider

    def get_llm_client(self) -> LLMClient:
        if self._llm_client is not None:
            return self._llm_client

        if self._config.provider == "ollama":
            cfg = self._config.ollama
            self._llm_client = OllamaLLMClient(
                base_url=cfg.base_url,
                model_name=cfg.llm_model_name,
                temperature=cfg.default_temperature,
                max_tokens=cfg.max_tokens,
            )
        else:
            cfg = self._config.api
            self._llm_client = APILLMClient(
                base_url=cfg.base_url,
                api_key=cfg.api_key,
                model_name=cfg.llm_model_name,
                temperature=cfg.default_temperature,
                max_tokens=cfg.max_tokens,
            )

        logger.info(
            "Initialized LLM client: provider=%s, model=%s",
            self._config.provider,
            cfg.llm_model_name,
        )
        return self._llm_client

    def get_embedding_client(self) -> EmbeddingClient:
        if self._embedding_client is not None:
            return self._embedding_client

        if self._config.provider == "ollama":
            cfg = self._config.ollama
            self._embedding_client = OllamaEmbeddingClient(
                base_url=cfg.base_url,
                model_name=cfg.embedding_model_name,
            )
        else:
            cfg = self._config.api
            self._embedding_client = APIEmbeddingClient(
                base_url=cfg.base_url,
                api_key=cfg.api_key,
                model_name=cfg.embedding_model_name,
            )

        logger.info(
            "Initialized Embedding client: provider=%s, model=%s",
            self._config.provider,
            cfg.embedding_model_name,
        )
        return self._embedding_client

    def get_provider_options(self) -> dict:
        if self._config.provider == "ollama":
            cfg = self._config.ollama
            return {
                "llm_host": cfg.base_url,
                "llm_model_name": cfg.llm_model_name,
                "embedding_host": cfg.base_url,
                "embedding_model_name": cfg.embedding_model_name,
                "embedding_dim": cfg.embedding_dim,
            }
        else:
            cfg = self._config.api
            return {
                "llm_host": cfg.base_url,
                "llm_model_name": cfg.llm_model_name,
                "llm_api_key": cfg.api_key,
                "embedding_host": cfg.base_url,
                "embedding_model_name": cfg.embedding_model_name,
                "embedding_api_key": cfg.api_key,
                "embedding_dim": cfg.embedding_dim,
            }
