"""Tests for model providers module."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from models.base import EmbeddingClient, LLMClient
from models.ollama_client import OllamaEmbeddingClient, OllamaLLMClient
from models.api_client import APIEmbeddingClient, APILLMClient
from models.registry import ModelProviderRegistry
from config.models import ModelsConfig, OllamaConfig, APIConfig


class TestLLMClientInterface:
    def test_llm_client_is_abstract(self):
        with pytest.raises(TypeError):
            LLMClient()


class TestEmbeddingClientInterface:
    def test_embedding_client_is_abstract(self):
        with pytest.raises(TypeError):
            EmbeddingClient()


class TestOllamaLLMClient:
    def test_initialization_defaults(self):
        client = OllamaLLMClient()
        assert client.base_url == "http://localhost:11434"
        assert client.model_name == "qwen2.5:7b"
        assert client.temperature == 0.7
        assert client.max_tokens == 2048

    def test_initialization_custom(self):
        client = OllamaLLMClient(
            base_url="http://custom:11434",
            model_name="llama2",
            temperature=0.5,
            max_tokens=1024,
        )
        assert client.base_url == "http://custom:11434"
        assert client.model_name == "llama2"
        assert client.temperature == 0.5
        assert client.max_tokens == 1024

    def test_base_url_trailing_slash_removed(self):
        client = OllamaLLMClient(base_url="http://localhost:11434/")
        assert client.base_url == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_agenerate_makes_correct_request(self):
        client = OllamaLLMClient()
        
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Test response"}
        mock_response.raise_for_status = MagicMock()
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client
            
            result = await client.agenerate("Test prompt")
            
            assert result == "Test response"
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert "api/generate" in call_args[0][0]


class TestOllamaEmbeddingClient:
    def test_initialization_defaults(self):
        client = OllamaEmbeddingClient()
        assert client.base_url == "http://localhost:11434"
        assert client.model_name == "nomic-embed-text"

    def test_initialization_custom(self):
        client = OllamaEmbeddingClient(
            base_url="http://custom:11434",
            model_name="custom-embed",
        )
        assert client.base_url == "http://custom:11434"
        assert client.model_name == "custom-embed"

    @pytest.mark.asyncio
    async def test_aembed_makes_correct_requests(self):
        client = OllamaEmbeddingClient()
        
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_response.raise_for_status = MagicMock()
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client
            
            result = await client.aembed(["text1", "text2"])
            
            assert len(result) == 2
            assert result[0] == [0.1, 0.2, 0.3]
            assert mock_client.post.call_count == 2


class TestAPILLMClient:
    def test_initialization_defaults(self):
        client = APILLMClient()
        assert client.base_url == "https://api.openai.com/v1"
        assert client.model_name == "gpt-4o-mini"
        assert client.api_key == ""

    def test_initialization_custom(self):
        client = APILLMClient(
            base_url="https://custom.api.com/v1",
            api_key="sk-test",
            model_name="gpt-4",
        )
        assert client.base_url == "https://custom.api.com/v1"
        assert client.api_key == "sk-test"
        assert client.model_name == "gpt-4"

    def test_get_headers_with_api_key(self):
        client = APILLMClient(api_key="sk-test")
        headers = client._get_headers()
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["Content-Type"] == "application/json"

    def test_get_headers_without_api_key(self):
        client = APILLMClient(api_key="")
        headers = client._get_headers()
        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_agenerate_makes_correct_request(self):
        client = APILLMClient(api_key="sk-test")
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client
            
            result = await client.agenerate("Test prompt")
            
            assert result == "Test response"
            mock_client.post.assert_called_once()


class TestAPIEmbeddingClient:
    def test_initialization_defaults(self):
        client = APIEmbeddingClient()
        assert client.base_url == "https://api.openai.com/v1"
        assert client.model_name == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_aembed_makes_correct_request(self):
        client = APIEmbeddingClient(api_key="sk-test")
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2]},
                {"index": 1, "embedding": [0.3, 0.4]},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client
            
            result = await client.aembed(["text1", "text2"])
            
            assert len(result) == 2
            assert result[0] == [0.1, 0.2]
            assert result[1] == [0.3, 0.4]


class TestModelProviderRegistry:
    def test_initialization_with_ollama_config(self):
        config = ModelsConfig(provider="ollama")
        registry = ModelProviderRegistry(config)
        assert registry.provider == "ollama"

    def test_initialization_with_api_config(self):
        config = ModelsConfig(provider="api")
        registry = ModelProviderRegistry(config)
        assert registry.provider == "api"

    def test_get_llm_client_ollama(self):
        config = ModelsConfig(provider="ollama")
        registry = ModelProviderRegistry(config)
        client = registry.get_llm_client()
        assert isinstance(client, OllamaLLMClient)

    def test_get_llm_client_api(self):
        config = ModelsConfig(provider="api")
        registry = ModelProviderRegistry(config)
        client = registry.get_llm_client()
        assert isinstance(client, APILLMClient)

    def test_get_embedding_client_ollama(self):
        config = ModelsConfig(provider="ollama")
        registry = ModelProviderRegistry(config)
        client = registry.get_embedding_client()
        assert isinstance(client, OllamaEmbeddingClient)

    def test_get_embedding_client_api(self):
        config = ModelsConfig(provider="api")
        registry = ModelProviderRegistry(config)
        client = registry.get_embedding_client()
        assert isinstance(client, APIEmbeddingClient)

    def test_client_caching(self):
        config = ModelsConfig(provider="ollama")
        registry = ModelProviderRegistry(config)
        client1 = registry.get_llm_client()
        client2 = registry.get_llm_client()
        assert client1 is client2

    def test_get_provider_options_ollama(self):
        config = ModelsConfig(
            provider="ollama",
            ollama=OllamaConfig(
                base_url="http://localhost:11434",
                llm_model_name="llama2",
                embedding_model_name="nomic-embed",
            ),
        )
        registry = ModelProviderRegistry(config)
        options = registry.get_provider_options()
        
        assert options["llm_host"] == "http://localhost:11434"
        assert options["llm_model_name"] == "llama2"
        assert options["embedding_host"] == "http://localhost:11434"
        assert options["embedding_model_name"] == "nomic-embed"

    def test_get_provider_options_api(self):
        config = ModelsConfig(
            provider="api",
            api=APIConfig(
                base_url="https://api.example.com/v1",
                api_key="sk-test",
                llm_model_name="gpt-4",
                embedding_model_name="text-embed-3",
            ),
        )
        registry = ModelProviderRegistry(config)
        options = registry.get_provider_options()
        
        assert options["llm_host"] == "https://api.example.com/v1"
        assert options["llm_model_name"] == "gpt-4"
        assert options["llm_api_key"] == "sk-test"
        assert options["embedding_model_name"] == "text-embed-3"
