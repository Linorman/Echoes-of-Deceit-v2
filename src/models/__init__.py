"""Model provider abstractions for LLM and Embedding clients."""

from models.base import EmbeddingClient, LLMClient
from models.ollama_client import OllamaEmbeddingClient, OllamaLLMClient
from models.api_client import APIEmbeddingClient, APILLMClient
from models.registry import ModelProviderRegistry

__all__ = [
    "LLMClient",
    "EmbeddingClient",
    "OllamaLLMClient",
    "OllamaEmbeddingClient",
    "APILLMClient",
    "APIEmbeddingClient",
    "ModelProviderRegistry",
]
