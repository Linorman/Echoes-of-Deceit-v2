"""Factory for RAG providers."""

from __future__ import annotations

from typing import Dict, Type

from .base_provider import BaseRAGProvider, RAGConfig
from .lightrag_provider import LightRAGProvider
from .minirag_provider import MiniRAGProvider


class RAGProviderFactory:
    """Factory for constructing RAG providers with a consistent interface."""

    _providers: Dict[str, Type[BaseRAGProvider]] = {
        "lightrag": LightRAGProvider,
        "minirag": MiniRAGProvider
    }

    @classmethod
    def create_provider(
        cls,
        provider_type: str,
        config: RAGConfig,
        **kwargs,
    ) -> BaseRAGProvider:
        if provider_type not in cls._providers:
            raise ValueError(
                f"Unknown RAG provider type: {provider_type}. Available: {list(cls._providers.keys())}"
            )

        provider_class = cls._providers[provider_type]
        try:
            provider = provider_class(config)
        except Exception as exc:  # pragma: no cover - instantiation issues
            raise RuntimeError(f"Failed to create RAG provider '{provider_type}': {exc}") from exc

        if not provider.is_available():  # pragma: no cover - runtime check depends on deps
            raise RuntimeError(
                f"Provider '{provider_type}' is not available. Ensure required services and dependencies are ready."
            )

        return provider

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseRAGProvider]) -> None:
        if not issubclass(provider_class, BaseRAGProvider):
            raise ValueError("Provider class must inherit from BaseRAGProvider")
        cls._providers[name] = provider_class

    @classmethod
    def list_providers(cls) -> list[str]:
        return list(cls._providers.keys())
