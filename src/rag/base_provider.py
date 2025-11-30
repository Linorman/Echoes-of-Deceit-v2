"""Base interfaces for RAG providers."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional


@dataclass
class RAGConfig:
	"""Configuration shared by all RAG providers."""

	working_dir: str
	auto_initialize: bool = True
	options: Dict[str, Any] = field(default_factory=dict)

	def __post_init__(self) -> None:
		if not self.working_dir:
			raise ValueError("working_dir is required")


@dataclass
class RAGDocument:
	"""Container for documents ingested by a provider."""

	content: str
	doc_id: Optional[str] = None
	metadata: Dict[str, Any] = field(default_factory=dict)

	def __post_init__(self) -> None:
		if not isinstance(self.content, str) or not self.content.strip():
			raise ValueError("Document content must be a non-empty string")


@dataclass
class IngestionResult:
	"""Represents the ingestion outcome."""

	accepted: int
	rejected: int = 0
	metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
	"""Represents a retrieval result."""

	answer: str
	sources: Iterable[Dict[str, Any]] = field(default_factory=list)
	metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderStatus:
	"""Represents a provider health snapshot."""

	is_ready: bool
	details: Dict[str, Any] = field(default_factory=dict)


class BaseRAGProvider(ABC):
	"""Abstract base class for all RAG providers."""

	def __init__(self, config: RAGConfig):
		self.config = config
		self._init_lock = asyncio.Lock()
		self._initialized = False

	async def __aenter__(self) -> "BaseRAGProvider":
		await self.ensure_initialized()
		return self

	async def __aexit__(self, exc_type, exc, tb) -> None:
		await self.aclose()

	async def ensure_initialized(self) -> None:
		if self._initialized:
			return
		async with self._init_lock:
			if not self._initialized:
				await self._initialize()
				self._initialized = True

	async def ainsert(self, documents: Iterable[RAGDocument], **kwargs: Any) -> IngestionResult:
		await self.ensure_initialized()
		return await self._ainsert_impl(list(documents), **kwargs)

	async def aquery(
		self,
		query: str,
		*,
		stream: bool = False,
		**kwargs: Any,
	) -> QueryResult | AsyncIterator[str]:
		await self.ensure_initialized()
		if stream:
			return await self._astream_query_impl(query, **kwargs)
		return await self._aquery_impl(query, **kwargs)

	async def abatch_query(self, queries: Iterable[str], **kwargs: Any) -> List[QueryResult]:
		await self.ensure_initialized()
		if kwargs.get("stream"):
			raise RuntimeError("Streaming is not supported in batch mode")
		results: List[QueryResult] = []
		for item in queries:
			results.append(await self._aquery_impl(item, **kwargs))
		return results

	async def aclose(self) -> None:
		if not self._initialized:
			return
		await self._aclose_impl()
		self._initialized = False

	async def ahealth_check(self) -> ProviderStatus:
		await self.ensure_initialized()
		return ProviderStatus(is_ready=self._initialized)

	def insert(self, documents: Iterable[RAGDocument], **kwargs: Any) -> IngestionResult:
		return self._run_sync(self.ainsert(documents, **kwargs))

	def query(self, query: str, **kwargs: Any) -> QueryResult:
		if kwargs.get("stream"):
			raise RuntimeError("Use aquery with stream=True to consume async streams")
		return self._run_sync(self.aquery(query, **kwargs))

	def batch_query(self, queries: Iterable[str], **kwargs: Any) -> List[QueryResult]:
		return self._run_sync(self.abatch_query(list(queries), **kwargs))

	def close(self) -> None:
		self._run_sync(self.aclose())

	def health_check(self) -> ProviderStatus:
		return self._run_sync(self.ahealth_check())

	def is_available(self) -> bool:
		return True

	async def _initialize(self) -> None:
		await self._ainitialize_impl()

	@abstractmethod
	async def _ainitialize_impl(self) -> None:
		...

	@abstractmethod
	async def _ainsert_impl(self, documents: Iterable[RAGDocument], **kwargs: Any) -> IngestionResult:
		...

	@abstractmethod
	async def _aquery_impl(self, query: str, **kwargs: Any) -> QueryResult:
		...

	async def _astream_query_impl(self, query: str, **kwargs: Any) -> AsyncIterator[str]:
		raise NotImplementedError("Streaming is not supported by this provider")

	async def _aclose_impl(self) -> None:
		return

	def update_config(self, **kwargs: Any) -> None:
		if "working_dir" in kwargs and not kwargs["working_dir"]:
			raise ValueError("working_dir cannot be empty")
		for key, value in kwargs.items():
			if hasattr(self.config, key):
				setattr(self.config, key, value)
			else:
				self.config.options[key] = value

	def _run_sync(self, awaitable: Any) -> Any:
		try:
			loop = asyncio.get_running_loop()
		except RuntimeError:
			return asyncio.run(awaitable)
		raise RuntimeError("Synchronous invocation is not supported when an event loop is running")
