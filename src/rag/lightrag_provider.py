"""LightRAG provider implementation."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

from .base_provider import (
    BaseRAGProvider,
    IngestionResult,
    ProviderStatus,
    QueryResult,
    RAGConfig,
    RAGDocument,
)

try:
    from lightrag import LightRAG, QueryParam
    from lightrag.kg.shared_storage import initialize_pipeline_status
    from lightrag.llm.ollama import ollama_embed, ollama_model_complete
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc
except ImportError as exc:  # pragma: no cover - optional dependency
    LightRAG = None  # type: ignore[assignment]
    QueryParam = None  # type: ignore[assignment]
    initialize_pipeline_status = None  # type: ignore[assignment]
    ollama_embed = None  # type: ignore[assignment]
    ollama_model_complete = None  # type: ignore[assignment]
    openai_complete_if_cache = None  # type: ignore[assignment]
    openai_embed = None  # type: ignore[assignment]
    EmbeddingFunc = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

if TYPE_CHECKING:
    from lightrag import LightRAG as LightRAGType
else:
    LightRAGType = Any


logger = logging.getLogger(__name__)


@dataclass
class LightRAGProviderOptions:
    """Provider-specific configuration overrides."""

    llm_backend: Optional[str] = None
    embedding_backend: Optional[str] = None
    llm_model_name: Optional[str] = None
    summary_max_tokens: Optional[int] = None
    llm_model_kwargs: Dict[str, Any] = field(default_factory=dict)
    default_mode: Optional[str] = None
    query_defaults: Dict[str, Any] = field(default_factory=dict)
    lightrag_kwargs: Dict[str, Any] = field(default_factory=dict)
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_dim: Optional[int] = None
    embedding_api_key: Optional[str] = None
    embedding_base_url: Optional[str] = None


class LightRAGProvider(BaseRAGProvider):
    """RAG provider backed by LightRAG."""

    _SUPPORTED_PARAM_KEYS: List[str] = [
        "mode",
        "rerank_mode",
        "user_prompt",
        "metadata_filters",
        "top_k",
        "rerank_top_k",
        "temperature",
        "stream",
    ]
    _ALLOWED_MODES: Tuple[str, ...] = (
        "local",
        "global",
        "hybrid",
        "naive",
        "mix",
        "bypass",
    )

    def __init__(
        self,
        config: RAGConfig,
        *,
        llm_model_func: Optional[Any] = None,
        embedding_func: Optional[Any] = None,
        options: Optional[LightRAGProviderOptions] = None,
    ) -> None:
        super().__init__(config)
        self._llm_model_func = llm_model_func
        self._embedding_func = embedding_func
        self._options = options or LightRAGProviderOptions()
        self._rag: Optional[LightRAGType] = None
        
        self._llm_backend = (
            self._options.llm_backend
            or self.config.options.get("llm_backend")
            or os.getenv("LLM_BACKEND", "ollama")
        ).lower()
        
        self._embedding_backend = (
            self._options.embedding_backend
            or self.config.options.get("embedding_backend")
            or os.getenv("EMBEDDING_BACKEND", self._llm_backend)
        ).lower()
        
        self._default_mode = (
            self._options.default_mode
            or self.config.options.get("default_mode")
            or os.getenv("LIGHTRAG_DEFAULT_MODE", "hybrid")
        )

        raw_defaults: Dict[str, Any] = {}
        config_defaults = self.config.options.get("query_defaults")
        if isinstance(config_defaults, dict):
            raw_defaults.update(config_defaults)
        raw_defaults.update(self._options.query_defaults)
        self._query_defaults = {
            key: value
            for key, value in raw_defaults.items()
            if key in self._SUPPORTED_PARAM_KEYS and value is not None
        }

    def is_available(self) -> bool:  # pragma: no cover - depends on optional dependency
        return _IMPORT_ERROR is None

    async def _ainitialize_impl(self) -> None:
        if LightRAG is None or QueryParam is None:
            raise RuntimeError(
                "LightRAG is not available. Ensure 'lightrag-hku[api]' is installed"
            ) from _IMPORT_ERROR

        os.makedirs(self.config.working_dir, exist_ok=True)

        llm_model_func = self._llm_model_func
        if llm_model_func is None:
            if self._llm_backend == "openai":
                if openai_complete_if_cache is None:
                    raise RuntimeError("OpenAI LLM function is unavailable")
                llm_model_func = self._build_openai_llm_func()
            else:
                if ollama_model_complete is None:
                    raise RuntimeError("Ollama LLM function is unavailable")
                llm_model_func = ollama_model_complete
        
        embedding_func = self._embedding_func or self._build_default_embedding()

        lightrag_kwargs = dict(self.config.options.get("lightrag_kwargs") or {})
        lightrag_kwargs.update(self._options.lightrag_kwargs)

        llm_model_name = (
            self._options.llm_model_name
            or self.config.options.get("llm_model_name")
            or os.getenv("LLM_MODEL", self._get_default_llm_model())
        )

        summary_max_tokens = self._resolve_int_option(
            "summary_max_tokens",
            "SUMMARY_MAX_TOKENS",
            8192,
        )

        rag_kwargs: Dict[str, Any] = {
            "working_dir": self.config.working_dir,
            "llm_model_func": llm_model_func,
            "llm_model_kwargs": self._build_llm_model_kwargs(),
            "embedding_func": embedding_func,
            "llm_model_name": llm_model_name,
            "summary_max_tokens": summary_max_tokens,
        }
        rag_kwargs.update(lightrag_kwargs)

        self._rag = LightRAG(**rag_kwargs)
        logger.info("LightRAG instance created with working_dir=%s", self.config.working_dir)

        await self._rag.initialize_storages()
        if initialize_pipeline_status is not None:
            await initialize_pipeline_status()
        logger.info("LightRAG storages initialized")
    
    def _build_openai_llm_func(self) -> Any:
        if openai_complete_if_cache is None:
            raise RuntimeError("OpenAI LLM function is unavailable")
        
        llm_kwargs = self._build_llm_model_kwargs()
        llm_model_name = (
            self._options.llm_model_name
            or self.config.options.get("llm_model_name")
            or os.getenv("LLM_MODEL", self._get_default_llm_model())
        )
        
        async def wrapped_llm_func(
            prompt: str,
            system_prompt: Optional[str] = None,
            history_messages: Optional[List[Dict[str, str]]] = None,
            **kwargs: Any
        ) -> str:
            merged_kwargs = dict(llm_kwargs)
            merged_kwargs.update(kwargs)
            
            return await openai_complete_if_cache(
                llm_model_name,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                **merged_kwargs
            )
        
        return wrapped_llm_func
    
    def _get_default_llm_model(self) -> str:
        if self._llm_backend == "openai":
            return "gpt-4o-mini"
        else:
            return "qwen3:4b-instruct-2507-q8_0"

    async def _ainsert_impl(
        self,
        documents: Iterable[RAGDocument],
        **kwargs: Any,
    ) -> IngestionResult:
        if self._rag is None:
            raise RuntimeError("LightRAG is not initialized")

        docs = list(documents) if not isinstance(documents, list) else documents
        if not docs:
            return IngestionResult(accepted=0, rejected=0)

        insert_fn = getattr(self._rag, "ainsert", None)
        if insert_fn is None:
            raise RuntimeError("LightRAG instance does not expose 'ainsert'")

        signature = inspect.signature(insert_fn)
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        accepts_metadata = accepts_kwargs or "metadata" in signature.parameters
        accepts_doc_id = accepts_kwargs or "doc_id" in signature.parameters

        max_parallel = self._resolve_int_option("max_parallel_insert", "MAX_PARALLEL_INSERT", 2)
        max_attempts = self._resolve_int_option("insert_max_attempts", "INSERT_MAX_ATTEMPTS", 3)
        backoff = float(self._resolve_option("insert_retry_backoff", os.getenv("INSERT_RETRY_BACKOFF", 0.5)))
        
        sem = asyncio.Semaphore(max(1, max_parallel))
        accepted_flags: List[bool] = [False] * len(docs)
        errors: List[Dict[str, Any]] = []

        async def _insert_one(index: int, document: RAGDocument) -> None:
            payload_kwargs = dict(kwargs)
            metadata = dict(document.metadata)
            
            if document.doc_id:
                metadata.setdefault("doc_id", document.doc_id)
                if accepts_doc_id:
                    payload_kwargs["doc_id"] = document.doc_id

            if metadata and accepts_metadata:
                payload_kwargs["metadata"] = metadata

            for attempt in range(1, max_attempts + 1):
                try:
                    async with sem:
                        await insert_fn(document.content, **payload_kwargs)
                    accepted_flags[index] = True
                    return
                except Exception as exc:
                    if attempt == max_attempts:
                        errors.append({
                            "doc_id": document.doc_id,
                            "error": str(exc),
                            "attempts": attempt,
                        })
                        logger.warning("Failed to insert document %s after %d attempts", document.doc_id, attempt)
                    else:
                        await asyncio.sleep(backoff * attempt)

        await asyncio.gather(*[_insert_one(idx, doc) for idx, doc in enumerate(docs)])

        accepted = sum(accepted_flags)
        rejected = len(docs) - accepted
        return IngestionResult(
            accepted=accepted, 
            rejected=rejected, 
            metadata={"errors": errors} if errors else {}
        )

    async def _aquery_impl(self, query: str, **kwargs: Any) -> QueryResult:
        if self._rag is None:
            raise RuntimeError("LightRAG is not initialized")
        param, call_kwargs = self._prepare_query_params(stream=False, kwargs=kwargs)
        response = await self._rag.aquery(query, param=param, **call_kwargs)
        return self._to_query_result(response)

    async def abatch_query(self, queries: Iterable[str], **kwargs: Any) -> List[QueryResult]:
        await self.ensure_initialized()
        if self._rag is None:
            raise RuntimeError("LightRAG is not initialized")
        if kwargs.get("stream"):
            raise RuntimeError("Streaming is not supported in batch mode")

        per_query_params = kwargs.pop("per_query_params", None)
        queries_list = list(queries)
        concurrency = self._resolve_int_option("batch_query_concurrency", "BATCH_QUERY_CONCURRENCY", 4)
        sem = asyncio.Semaphore(max(1, concurrency))
        results: List[Optional[QueryResult]] = [None] * len(queries_list)

        async def _query_one(index: int, query: str) -> None:
            local_kwargs = dict(kwargs)
            if per_query_params and index < len(per_query_params):
                local_kwargs.update(per_query_params[index])
            param, call_kwargs = self._prepare_query_params(stream=False, kwargs=local_kwargs)
            async with sem:
                response = await self._rag.aquery(query, param=param, **call_kwargs)
            results[index] = self._to_query_result(response)

        gathered = await asyncio.gather(*[_query_one(idx, q) for idx, q in enumerate(queries_list)], return_exceptions=True)
        
        errors = [{"index": idx, "error": str(err)} for idx, err in enumerate(gathered) if isinstance(err, Exception)]
        if errors:
            for err in errors:
                logger.error("Batch query failed at index %d: %s", err["index"], err["error"])
            raise RuntimeError(f"Batch query failed: {errors}")
        
        return [r for r in results if r is not None]

    async def _astream_query_impl(self, query: str, **kwargs: Any) -> AsyncIterator[str]:
        if self._rag is None:
            raise RuntimeError("LightRAG is not initialized")
        param, call_kwargs = self._prepare_query_params(stream=True, kwargs=kwargs)
        response = await self._rag.aquery(query, param=param, **call_kwargs)

        if hasattr(response, "__aiter__"):
            async def _stream() -> AsyncIterator[str]:
                async for chunk in response:  # type: ignore[async-for]
                    yield chunk
            return _stream()

        if isinstance(response, str):
            async def _single() -> AsyncIterator[str]:
                yield response
            return _single()

        raise TypeError(f"Expected async iterator or string, got {type(response).__name__}")

    async def _aclose_impl(self) -> None:
        if self._rag is None:
            return

        cache = getattr(self._rag, "llm_response_cache", None)
        if cache is not None:
            callback = getattr(cache, "index_done_callback", None)
            if callable(callback):
                maybe = callback()
                if inspect.isawaitable(maybe):
                    await maybe

        finalize = getattr(self._rag, "finalize_storages", None)
        if callable(finalize):
            result = finalize()
            if inspect.isawaitable(result):
                await result

        self._rag = None
        logger.info("LightRAG resources released")

    async def ahealth_check(self) -> ProviderStatus:
        base_status = await super().ahealth_check()
        if not base_status.is_ready or self._rag is None:
            return ProviderStatus(is_ready=False, details={"initialized": False})

        details = dict(base_status.details)
        details["working_dir"] = self.config.working_dir
        details["storages"] = {
            name: getattr(self._rag, name, None).__class__.__name__ if getattr(self._rag, name, None) else None
            for name in ("kv_storage", "vector_storage", "graph_storage", "doc_status_storage")
        }

        health_fn = getattr(self._rag, "health_check", None)
        if callable(health_fn):
            try:
                result = health_fn()
                if inspect.isawaitable(result):
                    result = await result
                if isinstance(result, dict):
                    details.update(result)
                elif isinstance(result, bool):
                    return ProviderStatus(is_ready=result, details=details)
                else:
                    details["health_raw"] = result
            except Exception as exc:
                logger.exception("Health check failed: %s", exc)
                return ProviderStatus(is_ready=False, details={"error": str(exc)})

        return ProviderStatus(is_ready=True, details=details)

    def _build_default_embedding(self) -> Any:
        if EmbeddingFunc is None:
            raise RuntimeError("LightRAG utilities are unavailable")
        
        if self._embedding_backend == "openai":
            return self._build_openai_embedding()
        else:
            return self._build_ollama_embedding()
    
    def _build_openai_embedding(self) -> Any:
        if openai_embed is None:
            raise RuntimeError("OpenAI embedding function is unavailable")
        
        embed_model = (
            self._options.embedding_model
            or self._resolve_option("embedding_model", os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"))
        )
        
        embedding_dim = self._resolve_int_option("embedding_dim", "EMBEDDING_DIM", 3072)
        max_token_size = self._resolve_int_option("max_embed_tokens", "MAX_EMBED_TOKENS", 8192)
        
        api_key = (
            self._options.embedding_api_key
            or self.config.options.get("embedding_api_key")
            or os.getenv("OPENAI_API_KEY")
        )
        
        base_url = (
            self._options.embedding_base_url
            or self.config.options.get("embedding_base_url")
            or os.getenv("OPENAI_BASE_URL")
        )
        
        embed_kwargs: Dict[str, Any] = {"model": embed_model}
        if api_key:
            embed_kwargs["api_key"] = api_key
        if base_url:
            embed_kwargs["base_url"] = base_url
        
        user_specified_dim = "embedding_dim" in self.config.options or os.getenv("EMBEDDING_DIM")
        if not user_specified_dim:
            dim_map = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
            }
            if embed_model in dim_map:
                detected = dim_map[embed_model]
                if detected != embedding_dim:
                    logger.info("Auto-detected embedding dim=%d for model=%s", detected, embed_model)
                    embedding_dim = detected
                    self.config.options["embedding_dim"] = embedding_dim
        
        return EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=max_token_size,
            func=lambda texts: openai_embed(texts, **embed_kwargs),
        )
    
    def _build_ollama_embedding(self) -> Any:
        if ollama_embed is None:
            raise RuntimeError("Ollama embedding function is unavailable")
        
        embedding_dim = self._resolve_int_option("embedding_dim", "EMBEDDING_DIM", 1024)
        max_token_size = self._resolve_int_option("max_embed_tokens", "MAX_EMBED_TOKENS", 8192)
        embed_model = self._resolve_option("embedding_model", os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b"))
        host = self._resolve_option("embedding_host", os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"))

        user_specified_dim = "embedding_dim" in self.config.options or os.getenv("EMBEDDING_DIM")
        if not user_specified_dim:
            try:
                probe = ollama_embed(["dimension probe"], embed_model=embed_model, host=host)
                if inspect.iscoroutine(probe):
                    probe = asyncio.get_event_loop().run_until_complete(probe)  # type: ignore[assignment]
                if isinstance(probe, list) and probe and isinstance(probe[0], (list, tuple)):
                    detected = len(probe[0])
                    if detected != embedding_dim:
                        logger.info("Auto-detected embedding dim=%d (was %d) for model=%s", detected, embedding_dim, embed_model)
                        embedding_dim = detected
                        self.config.options["embedding_dim"] = embedding_dim
            except Exception as exc:
                logger.warning("Failed to auto-detect embedding dimension (using %d): %s", embedding_dim, exc)

        return EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=max_token_size,
            func=lambda texts: ollama_embed(texts, embed_model=embed_model, host=host),
        )

    def _build_llm_model_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        
        if self._llm_backend == "ollama":
            kwargs = {
                "host": self._resolve_option("llm_host", os.getenv("LLM_BINDING_HOST", "http://localhost:11434")),
                "options": {"num_ctx": self._resolve_int_option("llm_num_ctx", "LLM_NUM_CTX", 8192)},
                "timeout": self._resolve_int_option("timeout", "TIMEOUT", 600),
            }
        elif self._llm_backend == "openai":
            api_key = (
                self._options.llm_api_key
                or self.config.options.get("llm_api_key")
                or os.getenv("OPENAI_API_KEY")
            )
            if api_key:
                kwargs["api_key"] = api_key
            
            base_url = (
                self._options.llm_base_url
                or self.config.options.get("llm_base_url")
                or os.getenv("OPENAI_BASE_URL")
            )
            if base_url:
                kwargs["base_url"] = base_url
        
        kwargs.update(self.config.options.get("llm_model_kwargs") or {})
        kwargs.update(self._options.llm_model_kwargs)
        return kwargs

    def _prepare_query_params(self, *, stream: bool, kwargs: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        if QueryParam is None:
            raise RuntimeError("QueryParam is unavailable; check LightRAG installation")

        call_kwargs = dict(kwargs)
        param = call_kwargs.pop("param", None)

        param_values = dict(self._query_defaults)
        for key in self._SUPPORTED_PARAM_KEYS:
            if key in call_kwargs:
                param_values[key] = call_kwargs.pop(key)
        param_values = {k: v for k, v in param_values.items() if v is not None}

        if param is None:
            mode = param_values.pop("mode", None) or self._default_mode
            if mode not in self._ALLOWED_MODES:
                mode = self._default_mode
            param_values["stream"] = stream
            param = QueryParam(mode=mode, **param_values)  # type: ignore[arg-type]
        elif stream and hasattr(param, "stream"):
            param.stream = True

        return param, call_kwargs

    def _to_query_result(self, response: Any) -> QueryResult:
        if isinstance(response, str):
            return QueryResult(answer=response)

        if isinstance(response, dict):
            answer = response.get("response") or response.get("answer") or response.get("output") or str(response)
            sources = response.get("sources") or response.get("citations") or []
            metadata = {k: v for k, v in response.items() if k not in {"response", "answer", "output", "sources", "citations"}}
            metadata["raw"] = response
            return QueryResult(answer=answer, sources=sources, metadata=metadata)

        return QueryResult(answer=str(response), metadata={"raw": response})

    def _resolve_option(self, key: str, fallback: Any) -> Any:
        return self.config.options.get(key, fallback)

    def _resolve_int_option(self, key: str, env_key: str, default: Any) -> int:
        candidate = self._options.__dict__.get(key) or self.config.options.get(key) or os.getenv(env_key, default)
        try:
            return int(candidate)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid integer value for {key}: {candidate}") from exc
