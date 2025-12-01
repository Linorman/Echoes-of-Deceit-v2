"""Tests for model providers module."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from models.base import EmbeddingClient, LLMClient
from models.langchain_client import (
    LangChainLLMClient,
    LangChainEmbeddingClient,
    create_ollama_llm_client,
    create_ollama_embedding_client,
    create_openai_llm_client,
    create_openai_embedding_client,
)
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


class TestLangChainLLMClient:
    def test_langchain_llm_client_wraps_chat_model(self):
        mock_chat_model = MagicMock()
        client = LangChainLLMClient(mock_chat_model, temperature=0.7, max_tokens=2048)
        
        assert client.chat_model is mock_chat_model
        assert client._temperature == 0.7
        assert client._max_tokens == 2048

    @pytest.mark.asyncio
    async def test_agenerate_calls_ainvoke(self):
        mock_chat_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_chat_model.ainvoke = AsyncMock(return_value=mock_response)
        
        client = LangChainLLMClient(mock_chat_model)
        result = await client.agenerate("Test prompt")
        
        assert result == "Test response"
        mock_chat_model.ainvoke.assert_called_once()


class TestLangChainEmbeddingClient:
    def test_langchain_embedding_client_wraps_embeddings(self):
        mock_embeddings = MagicMock()
        client = LangChainEmbeddingClient(mock_embeddings)
        
        assert client.embeddings is mock_embeddings

    @pytest.mark.asyncio
    async def test_aembed_calls_aembed_documents(self):
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_documents = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        
        client = LangChainEmbeddingClient(mock_embeddings)
        result = await client.aembed(["text1", "text2"])
        
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_embeddings.aembed_documents.assert_called_once_with(["text1", "text2"])


class TestCreateOllamaClients:
    @patch("models.langchain_client.ChatOllama")
    def test_create_ollama_llm_client(self, mock_chat_ollama):
        mock_instance = MagicMock()
        mock_chat_ollama.return_value = mock_instance
        
        client = create_ollama_llm_client(
            base_url="http://localhost:11434",
            model_name="llama2",
            temperature=0.5,
            max_tokens=1024,
        )
        
        assert isinstance(client, LangChainLLMClient)
        mock_chat_ollama.assert_called_once_with(
            base_url="http://localhost:11434",
            model="llama2",
            temperature=0.5,
            num_predict=1024,
        )

    @patch("models.langchain_client.OllamaEmbeddings")
    def test_create_ollama_embedding_client(self, mock_ollama_embeddings):
        mock_instance = MagicMock()
        mock_ollama_embeddings.return_value = mock_instance
        
        client = create_ollama_embedding_client(
            base_url="http://localhost:11434",
            model_name="nomic-embed-text",
        )
        
        assert isinstance(client, LangChainEmbeddingClient)
        mock_ollama_embeddings.assert_called_once_with(
            base_url="http://localhost:11434",
            model="nomic-embed-text",
        )


class TestCreateOpenAIClients:
    @patch("models.langchain_client.ChatOpenAI")
    def test_create_openai_llm_client(self, mock_chat_openai):
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance
        
        client = create_openai_llm_client(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model_name="gpt-4o-mini",
            temperature=0.7,
            max_tokens=2048,
        )
        
        assert isinstance(client, LangChainLLMClient)
        mock_chat_openai.assert_called_once_with(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=2048,
            timeout=120.0,
        )

    @patch("models.langchain_client.OpenAIEmbeddings")
    def test_create_openai_embedding_client(self, mock_openai_embeddings):
        mock_instance = MagicMock()
        mock_openai_embeddings.return_value = mock_instance
        
        client = create_openai_embedding_client(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model_name="text-embedding-3-small",
        )
        
        assert isinstance(client, LangChainEmbeddingClient)
        mock_openai_embeddings.assert_called_once_with(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model="text-embedding-3-small",
            timeout=60.0,
        )


class TestModelProviderRegistry:
    def test_initialization_with_ollama_config(self):
        config = ModelsConfig(provider="ollama")
        registry = ModelProviderRegistry(config)
        assert registry.provider == "ollama"

    def test_initialization_with_api_config(self):
        config = ModelsConfig(provider="api")
        registry = ModelProviderRegistry(config)
        assert registry.provider == "api"

    @patch("models.langchain_client.ChatOllama")
    def test_get_llm_client_ollama(self, mock_chat_ollama):
        mock_chat_ollama.return_value = MagicMock()
        config = ModelsConfig(provider="ollama")
        registry = ModelProviderRegistry(config)
        client = registry.get_llm_client()
        assert isinstance(client, LangChainLLMClient)

    @patch("models.langchain_client.ChatOpenAI")
    def test_get_llm_client_api(self, mock_chat_openai):
        mock_chat_openai.return_value = MagicMock()
        config = ModelsConfig(provider="api")
        registry = ModelProviderRegistry(config)
        client = registry.get_llm_client()
        assert isinstance(client, LangChainLLMClient)

    @patch("models.langchain_client.OllamaEmbeddings")
    def test_get_embedding_client_ollama(self, mock_ollama_embeddings):
        mock_ollama_embeddings.return_value = MagicMock()
        config = ModelsConfig(provider="ollama")
        registry = ModelProviderRegistry(config)
        client = registry.get_embedding_client()
        assert isinstance(client, LangChainEmbeddingClient)

    @patch("models.langchain_client.OpenAIEmbeddings")
    def test_get_embedding_client_api(self, mock_openai_embeddings):
        mock_openai_embeddings.return_value = MagicMock()
        config = ModelsConfig(provider="api")
        registry = ModelProviderRegistry(config)
        client = registry.get_embedding_client()
        assert isinstance(client, LangChainEmbeddingClient)

    @patch("models.langchain_client.ChatOllama")
    def test_client_caching(self, mock_chat_ollama):
        mock_chat_ollama.return_value = MagicMock()
        config = ModelsConfig(provider="ollama")
        registry = ModelProviderRegistry(config)
        client1 = registry.get_llm_client()
        client2 = registry.get_llm_client()
        assert client1 is client2

    @patch("models.langchain_client.ChatOllama")
    def test_get_chat_model(self, mock_chat_ollama):
        mock_instance = MagicMock()
        mock_chat_ollama.return_value = mock_instance
        config = ModelsConfig(provider="ollama")
        registry = ModelProviderRegistry(config)
        chat_model = registry.get_chat_model()
        assert chat_model is mock_instance

    @patch("models.langchain_client.OllamaEmbeddings")
    def test_get_embeddings(self, mock_ollama_embeddings):
        mock_instance = MagicMock()
        mock_ollama_embeddings.return_value = mock_instance
        config = ModelsConfig(provider="ollama")
        registry = ModelProviderRegistry(config)
        embeddings = registry.get_embeddings()
        assert embeddings is mock_instance

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
