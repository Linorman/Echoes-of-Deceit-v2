"""Abstract base class for memory stores.

This module defines the BaseMemoryStore interface aligned with LangGraph's Store abstraction,
providing put/get/search capabilities for memory documents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from game.memory.entities import MemoryDocument


@dataclass
class MemorySearchResult:
    document: MemoryDocument
    score: float = 0.0

    @property
    def key(self) -> str:
        return self.document.key

    @property
    def value(self) -> Dict[str, Any]:
        return self.document.value

    @property
    def namespace(self) -> str:
        return self.document.namespace


class BaseMemoryStore(ABC):
    @abstractmethod
    def put(
        self,
        namespace: str | Tuple[str, ...],
        key: str,
        value: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        index: bool = True,
    ) -> MemoryDocument:
        pass

    @abstractmethod
    def get(
        self,
        namespace: str | Tuple[str, ...],
        key: str,
    ) -> Optional[MemoryDocument]:
        pass

    @abstractmethod
    def delete(
        self,
        namespace: str | Tuple[str, ...],
        key: str,
    ) -> bool:
        pass

    @abstractmethod
    def search(
        self,
        namespace: str | Tuple[str, ...],
        query: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[MemorySearchResult]:
        pass

    @abstractmethod
    def list_namespaces(self) -> List[str]:
        pass

    @abstractmethod
    def list_keys(self, namespace: str | Tuple[str, ...]) -> List[str]:
        pass

    def _normalize_namespace(self, namespace: str | Tuple[str, ...]) -> str:
        if isinstance(namespace, tuple):
            return ":".join(namespace)
        return namespace

    def put_document(self, document: MemoryDocument) -> MemoryDocument:
        return self.put(
            namespace=document.namespace,
            key=document.key,
            value=document.value,
            metadata=document.metadata,
        )

    def batch_put(
        self,
        documents: List[MemoryDocument],
    ) -> List[MemoryDocument]:
        return [self.put_document(doc) for doc in documents]

    def batch_get(
        self,
        items: List[Tuple[str, str]],
    ) -> List[Optional[MemoryDocument]]:
        return [self.get(namespace, key) for namespace, key in items]

    def clear_namespace(self, namespace: str | Tuple[str, ...]) -> int:
        keys = self.list_keys(namespace)
        count = 0
        for key in keys:
            if self.delete(namespace, key):
                count += 1
        return count
