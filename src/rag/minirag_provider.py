"""MiniRAG provider implementation."""

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
    from minirag import MiniRAG, QueryParam
    from minirag.utils import EmbeddingFunc
except ImportError as exc:
    MiniRAG = None  # type: ignore[assignment]
    QueryParam = None  # type: ignore[assignment]
    EmbeddingFunc = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

try:
    from lightrag.llm.ollama import ollama_embed, ollama_model_complete
except ImportError:
    ollama_embed = None  # type: ignore[assignment]
    ollama_model_complete = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from minirag import MiniRAG as MiniRAGType
else:
    MiniRAGType = Any


logger = logging.getLogger(__name__)


@dataclass
class MiniRAGProviderOptions:
    """Provider-specific configuration overrides."""

    llm_model_name: Optional[str] = None
    llm_model_max_token_size: Optional[int] = None
    llm_model_kwargs: Dict[str, Any] = field(default_factory=dict)
    default_mode: Optional[str] = None
    query_defaults: Dict[str, Any] = field(default_factory=dict)
    minirag_kwargs: Dict[str, Any] = field(default_factory=dict)


class MiniRAGProvider(BaseRAGProvider):
    """RAG provider backed by MiniRAG."""

    _SUPPORTED_PARAM_KEYS: List[str] = [
        "mode",
        "user_prompt",
        "metadata_filters",
        "top_k",
        "temperature",
        "stream",
    ]
    _ALLOWED_MODES: Tuple[str, ...] = (
        "mini",
        "naive",
    )

    def __init__(
        self,
        config: RAGConfig,
        *,
        llm_model_func: Optional[Any] = None,
        embedding_func: Optional[Any] = None,
        options: Optional[MiniRAGProviderOptions] = None,
    ) -> None:
        super().__init__(config)
        self._llm_model_func = llm_model_func
        self._embedding_func = embedding_func
        self._options = options or MiniRAGProviderOptions()
        self._rag: Optional[MiniRAGType] = None
        self._default_mode = (
            self._options.default_mode
            or self.config.options.get("default_mode")
            or os.getenv("MINIRAG_DEFAULT_MODE", "mini")
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

    def is_available(self) -> bool:
        if _IMPORT_ERROR is not None:
            logger.error(
                "MiniRAG is not available due to import error: %s. "
                "Please install it with: pip install git+https://github.com/HKUDS/MiniRAG.git",
                _IMPORT_ERROR
            )
        return _IMPORT_ERROR is None

    async def _ainitialize_impl(self) -> None:
        if MiniRAG is None or QueryParam is None:
            raise RuntimeError(
                "MiniRAG is not available. Please install it with: "
                "pip install git+https://github.com/HKUDS/MiniRAG.git"
            ) from _IMPORT_ERROR

        os.makedirs(self.config.working_dir, exist_ok=True)

        llm_model_func = self._llm_model_func or self._build_default_llm_func()
        embedding_func = self._embedding_func or self._build_default_embedding()

        minirag_kwargs = dict(self.config.options.get("minirag_kwargs") or {})
        minirag_kwargs.update(self._options.minirag_kwargs)

        llm_model_name = (
            self._options.llm_model_name
            or self.config.options.get("llm_model_name")
            or os.getenv("LLM_MODEL", "qwen3:4b-instruct-2507-q8_0")
        )

        llm_model_max_token_size = self._resolve_int_option(
            "llm_model_max_token_size",
            "LLM_MODEL_MAX_TOKEN_SIZE",
            8192,
        )

        rag_kwargs: Dict[str, Any] = {
            "working_dir": self.config.working_dir,
            "llm_model_func": llm_model_func,
            "llm_model_name": llm_model_name,
            "llm_model_max_token_size": llm_model_max_token_size,
            "embedding_func": embedding_func,
        }
        
        # Only add llm_model_kwargs if not using custom llm_model_func
        if self._llm_model_func is None:
            rag_kwargs["llm_model_kwargs"] = self._build_llm_model_kwargs()
        
        rag_kwargs.update(minirag_kwargs)

        passthrough_keys = {
            "vector_db_storage_cls_kwargs",
            "kv_storage_cls_kwargs",
            "graph_storage_cls_kwargs",
            "doc_status_storage",
            "addon_params",
        }

        for key in passthrough_keys:
            if key in rag_kwargs:
                continue
            value = self.config.options.get(key)
            if value is not None:
                rag_kwargs[key] = value

        self._rag = MiniRAG(**rag_kwargs)
        logger.info("MiniRAG instance created with working_dir=%s", self.config.working_dir)

    async def _ainsert_impl(
        self,
        documents: Iterable[RAGDocument],
        **kwargs: Any,
    ) -> IngestionResult:
        if self._rag is None:
            raise RuntimeError("MiniRAG is not initialized")

        docs = list(documents) if not isinstance(documents, list) else documents
        if not docs:
            return IngestionResult(accepted=0, rejected=0)

        max_parallel = self._resolve_int_option("max_parallel_insert", "MAX_PARALLEL_INSERT", 2)
        max_attempts = self._resolve_int_option("insert_max_attempts", "INSERT_MAX_ATTEMPTS", 3)
        backoff = float(self._resolve_option("insert_retry_backoff", os.getenv("INSERT_RETRY_BACKOFF", 0.5)))
        
        sem = asyncio.Semaphore(max(1, max_parallel))
        accepted_flags: List[bool] = [False] * len(docs)
        errors: List[Dict[str, Any]] = []

        async def _insert_one(index: int, document: RAGDocument) -> None:
            for attempt in range(1, max_attempts + 1):
                try:
                    async with sem:
                        await asyncio.to_thread(self._rag.insert, document.content)
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
            raise RuntimeError("MiniRAG is not initialized")
        param, call_kwargs = self._prepare_query_params(stream=False, kwargs=kwargs)
        response = await asyncio.to_thread(self._rag.query, query, param=param, **call_kwargs)
        return self._to_query_result(response)

    async def abatch_query(self, queries: Iterable[str], **kwargs: Any) -> List[QueryResult]:
        await self.ensure_initialized()
        if self._rag is None:
            raise RuntimeError("MiniRAG is not initialized")
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
                response = await asyncio.to_thread(self._rag.query, query, param=param, **call_kwargs)
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
            raise RuntimeError("MiniRAG is not initialized")
        param, call_kwargs = self._prepare_query_params(stream=True, kwargs=kwargs)
        response = await asyncio.to_thread(self._rag.query, query, param=param, **call_kwargs)

        async def _single() -> AsyncIterator[str]:
            yield str(response) if not isinstance(response, str) else response
        return _single()

    async def _aclose_impl(self) -> None:
        if self._rag is None:
            return
        self._rag = None
        logger.info("MiniRAG resources released")

    async def ahealth_check(self) -> ProviderStatus:
        base_status = await super().ahealth_check()
        if not base_status.is_ready or self._rag is None:
            return ProviderStatus(is_ready=False, details={"initialized": False})

        details = dict(base_status.details)
        details["working_dir"] = self.config.working_dir
        return ProviderStatus(is_ready=True, details=details)

    def _build_default_llm_func(self) -> Any:
        """Build default LLM function with proper host configuration."""
        if ollama_model_complete is None:
            raise RuntimeError("ollama_model_complete is unavailable")
        
        llm_model_kwargs = self._build_llm_model_kwargs()
        
        # Create a wrapper function that passes the host configuration
        def llm_func(prompt: str, **kwargs: Any) -> Any:
            merged_kwargs = dict(llm_model_kwargs)
            merged_kwargs.update(kwargs)
            return ollama_model_complete(prompt, **merged_kwargs)
        
        return llm_func

    def _build_llm_model_kwargs(self) -> Dict[str, Any]:
        """Build LLM model kwargs with host configuration."""
        kwargs = {
            "host": self._resolve_option("llm_host", os.getenv("LLM_BINDING_HOST", "http://localhost:11434")),
            "options": {"num_ctx": self._resolve_int_option("llm_num_ctx", "LLM_NUM_CTX", 8192)},
            "timeout": self._resolve_int_option("timeout", "TIMEOUT", 300),
        }
        kwargs.update(self.config.options.get("llm_model_kwargs") or {})
        kwargs.update(self._options.llm_model_kwargs)
        return kwargs

    def _build_default_embedding(self) -> Any:
        if EmbeddingFunc is None or ollama_embed is None:
            raise RuntimeError("MiniRAG utilities are unavailable")
        
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

    def _prepare_query_params(self, *, stream: bool, kwargs: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        if QueryParam is None:
            raise RuntimeError("QueryParam is unavailable; check MiniRAG installation")

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
            param = QueryParam(mode=mode, **param_values)  # type: ignore[arg-type]

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

