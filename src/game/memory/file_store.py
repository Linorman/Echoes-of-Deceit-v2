"""File-based memory store implementation.

This module provides a file-based implementation of the BaseMemoryStore,
storing memory documents as JSON files with simple search capabilities.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from game.memory.base_store import BaseMemoryStore, MemorySearchResult
from game.memory.entities import MemoryDocument

logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def datetime_decoder(dct: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in dct.items():
        if isinstance(value, str):
            try:
                if "T" in value and len(value) >= 19:
                    dct[key] = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass
    return dct


class FileMemoryStore(BaseMemoryStore):
    def __init__(
        self,
        base_dir: Path,
        score_fn: Optional[Callable[[str, Dict[str, Any]], float]] = None,
    ):
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._score_fn = score_fn or self._default_score_fn

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def _namespace_dir(self, namespace: str) -> Path:
        safe_namespace = namespace.replace(":", "_").replace("/", "_")
        return self._base_dir / safe_namespace

    def _document_file(self, namespace: str, key: str) -> Path:
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self._namespace_dir(namespace) / f"{safe_key}.json"

    def _ensure_namespace_dir(self, namespace: str) -> Path:
        ns_dir = self._namespace_dir(namespace)
        ns_dir.mkdir(parents=True, exist_ok=True)
        return ns_dir

    def put(
        self,
        namespace: str | Tuple[str, ...],
        key: str,
        value: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        index: bool = True,
    ) -> MemoryDocument:
        ns = self._normalize_namespace(namespace)
        self._ensure_namespace_dir(ns)

        existing = self.get(ns, key)
        now = datetime.now()

        document = MemoryDocument(
            namespace=ns,
            key=key,
            value=value,
            metadata=metadata or {},
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )

        doc_file = self._document_file(ns, key)
        with open(doc_file, "w", encoding="utf-8") as f:
            json.dump(
                document.model_dump(mode="json"),
                f,
                cls=DateTimeEncoder,
                ensure_ascii=False,
                indent=2,
            )

        logger.debug("Stored document %s/%s", ns, key)
        return document

    def get(
        self,
        namespace: str | Tuple[str, ...],
        key: str,
    ) -> Optional[MemoryDocument]:
        ns = self._normalize_namespace(namespace)
        doc_file = self._document_file(ns, key)

        if not doc_file.exists():
            return None

        try:
            with open(doc_file, "r", encoding="utf-8") as f:
                data = json.load(f, object_hook=datetime_decoder)
            return MemoryDocument.model_validate(data)
        except Exception as e:
            logger.error("Failed to load document %s/%s: %s", ns, key, e)
            return None

    def delete(
        self,
        namespace: str | Tuple[str, ...],
        key: str,
    ) -> bool:
        ns = self._normalize_namespace(namespace)
        doc_file = self._document_file(ns, key)

        if doc_file.exists():
            doc_file.unlink()
            logger.debug("Deleted document %s/%s", ns, key)
            return True

        return False

    def search(
        self,
        namespace: str | Tuple[str, ...],
        query: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[MemorySearchResult]:
        ns = self._normalize_namespace(namespace)
        ns_dir = self._namespace_dir(ns)

        if not ns_dir.exists():
            return []

        results: List[MemorySearchResult] = []

        for doc_file in ns_dir.glob("*.json"):
            doc = self.get(ns, doc_file.stem)
            if doc is None:
                continue

            if filter and not self._matches_filter(doc, filter):
                continue

            score = self._score_fn(query, doc.value) if query else 1.0
            results.append(MemorySearchResult(document=doc, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def list_namespaces(self) -> List[str]:
        namespaces = []
        for item in self._base_dir.iterdir():
            if item.is_dir():
                ns = item.name.replace("_", ":", 1)
                namespaces.append(ns)
        return namespaces

    def list_keys(self, namespace: str | Tuple[str, ...]) -> List[str]:
        ns = self._normalize_namespace(namespace)
        ns_dir = self._namespace_dir(ns)

        if not ns_dir.exists():
            return []

        return [f.stem for f in ns_dir.glob("*.json")]

    def _matches_filter(self, doc: MemoryDocument, filter: Dict[str, Any]) -> bool:
        for key, expected_value in filter.items():
            actual_value = doc.metadata.get(key) or doc.value.get(key)
            if actual_value != expected_value:
                return False
        return True

    def _default_score_fn(self, query: Optional[str], value: Dict[str, Any]) -> float:
        if not query:
            return 1.0

        query_lower = query.lower()
        content = str(value).lower()

        query_words = query_lower.split()
        if not query_words:
            return 0.0

        matches = sum(1 for word in query_words if word in content)
        return matches / len(query_words)

    def get_all_in_namespace(
        self,
        namespace: str | Tuple[str, ...],
    ) -> List[MemoryDocument]:
        ns = self._normalize_namespace(namespace)
        keys = self.list_keys(ns)
        documents = []
        for key in keys:
            doc = self.get(ns, key)
            if doc is not None:
                documents.append(doc)
        return documents

    def append_to_namespace(
        self,
        namespace: str | Tuple[str, ...],
        value: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        key_prefix: str = "item",
    ) -> MemoryDocument:
        ns = self._normalize_namespace(namespace)
        existing_keys = self.list_keys(ns)

        max_index = 0
        for key in existing_keys:
            if key.startswith(key_prefix + "_"):
                try:
                    idx = int(key.split("_")[-1])
                    max_index = max(max_index, idx)
                except ValueError:
                    pass

        new_key = f"{key_prefix}_{max_index + 1}"
        return self.put(ns, new_key, value, metadata)
