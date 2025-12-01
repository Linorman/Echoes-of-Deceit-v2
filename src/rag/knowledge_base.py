"""Multi-knowledge base management for RAG system."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .base_provider import (
    BaseRAGProvider,
    IngestionResult,
    ProviderStatus,
    QueryResult,
    RAGConfig,
    RAGDocument,
)
from .provider_factory import RAGProviderFactory

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeBaseConfig:
    kb_id: str
    name: str
    description: str
    working_dir: str
    provider_type: str
    created_at: str
    updated_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "active"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KnowledgeBaseConfig:
        return cls(**data)

    def resolve_working_dir(self, base_dir: Path) -> str:
        working_path = Path(self.working_dir)
        if working_path.is_absolute():
            return str(working_path)
        return str((base_dir / working_path).resolve())

    def resolve_metadata_paths(self, base_dir: Path) -> Dict[str, Any]:
        resolved = dict(self.metadata)
        if "source_files" in resolved:
            resolved_files = []
            for file_path in resolved["source_files"]:
                p = Path(file_path)
                if p.is_absolute():
                    resolved_files.append(str(p))
                else:
                    resolved_files.append(str((base_dir / p).resolve()))
            resolved["source_files"] = resolved_files
        return resolved


class KnowledgeBase:
    """Manages multiple isolated knowledge bases for puzzle/scenario games."""

    _REGISTRY_FILE = "kb_registry.json"
    _DEFAULT_PROVIDER = "lightrag"

    def __init__(
        self,
        base_storage_dir: str,
        default_provider_type: str = _DEFAULT_PROVIDER,
        provider_kwargs=None,
    ):
        self.base_storage_dir = Path(base_storage_dir)
        self.default_provider_type = default_provider_type
        self.provider_kwargs = dict(provider_kwargs or {})
        self._registry_path = self.base_storage_dir / self._REGISTRY_FILE
        self._registry: Dict[str, KnowledgeBaseConfig] = {}
        self._active_providers: Dict[str, BaseRAGProvider] = {}
        self._lock = asyncio.Lock()

        self.base_storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_registry()

    def _load_registry(self) -> None:
        if not self._registry_path.exists():
            self._registry = {}
            self._save_registry()
            return

        try:
            with open(self._registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._registry = {
                kb_id: KnowledgeBaseConfig.from_dict(kb_data)
                for kb_id, kb_data in data.items()
            }
            logger.info("Loaded %d knowledge bases from registry", len(self._registry))
        except Exception as exc:
            logger.error("Failed to load registry: %s", exc)
            self._registry = {}

    def _save_registry(self) -> None:
        try:
            with open(self._registry_path, "w", encoding="utf-8") as f:
                data = {
                    kb_id: kb_config.to_dict()
                    for kb_id, kb_config in self._registry.items()
                }
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error("Failed to save registry: %s", exc)
            raise

    def _collect_provider_options(self, kb_config: KnowledgeBaseConfig) -> Dict[str, Any]:
        options: Dict[str, Any] = {}

        metadata_options = kb_config.metadata.get("provider_options")
        if isinstance(metadata_options, dict):
            options.update(metadata_options)

        if isinstance(self.provider_kwargs, dict):
            direct_overrides = {
                key: value
                for key, value in self.provider_kwargs.items()
                if key not in {"config_options", "init_kwargs", "provider_options"}
            }
            options.update(direct_overrides)

            nested = self.provider_kwargs.get("config_options")
            if isinstance(nested, dict):
                options.update(nested)

            provider_specific = self.provider_kwargs.get("provider_options")
            if isinstance(provider_specific, dict):
                options.update(provider_specific)

        return self._normalize_provider_options(kb_config.provider_type, options)

    def _normalize_provider_options(
        self,
        provider_type: str,
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for raw_key, value in options.items():
            if not isinstance(raw_key, str):
                continue
            key = raw_key.strip()
            if not key:
                continue
            canonical = key.lower().replace("-", "_")
            normalized[canonical] = value

        if provider_type == "lightrag" or provider_type == "minirag":
            return self._apply_lightrag_aliases(normalized)
        return normalized

    @staticmethod
    def _apply_lightrag_aliases(options: Dict[str, Any]) -> Dict[str, Any]:
        resolved = dict(options)
        alias_map = {
            "llm_model": "llm_model_name",
            "llm_binding_host": "llm_host",
            "llm_endpoint": "llm_host",
            "embedding_binding_host": "embedding_host",
            "embedding_model_name": "embedding_model",
        }

        for source, target in alias_map.items():
            if source in options and target not in resolved:
                resolved[target] = options[source]

        return resolved

    def _collect_provider_init_kwargs(self, kb_config: KnowledgeBaseConfig) -> Dict[str, Any]:
        init_kwargs: Dict[str, Any] = {}

        metadata_init = kb_config.metadata.get("provider_init_kwargs")
        if isinstance(metadata_init, dict):
            init_kwargs.update(metadata_init)

        if isinstance(self.provider_kwargs, dict):
            nested = self.provider_kwargs.get("init_kwargs")
            if isinstance(nested, dict):
                init_kwargs.update(nested)

        return init_kwargs

    def list_knowledge_bases(
        self,
        status_filter: Optional[str] = None,
    ) -> List[KnowledgeBaseConfig]:
        """List all registered knowledge bases."""
        kbs = list(self._registry.values())
        if status_filter:
            kbs = [kb for kb in kbs if kb.status == status_filter]
        return sorted(kbs, key=lambda kb: kb.created_at, reverse=True)

    def get_knowledge_base(self, kb_id: str) -> Optional[KnowledgeBaseConfig]:
        """Retrieve knowledge base configuration by ID."""
        return self._registry.get(kb_id)

    def create_knowledge_base(
        self,
        kb_id: str,
        name: str,
        description: str = "",
        provider_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeBaseConfig:
        """Create a new knowledge base."""
        if kb_id in self._registry:
            raise ValueError(f"Knowledge base '{kb_id}' already exists")

        provider_type = provider_type or self.default_provider_type
        working_dir = kb_id
        abs_working_dir = self.base_storage_dir / kb_id
        os.makedirs(abs_working_dir, exist_ok=True)

        now = datetime.now().isoformat(sep=" ", timespec="seconds")
        kb_config = KnowledgeBaseConfig(
            kb_id=kb_id,
            name=name,
            description=description,
            working_dir=working_dir,
            provider_type=provider_type,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
            status="active",
        )

        self._registry[kb_id] = kb_config
        self._save_registry()
        logger.info("Created knowledge base: %s", kb_id)
        return kb_config

    async def delete_knowledge_base(
        self,
        kb_id: str,
        remove_files: bool = False,
    ) -> bool:
        """Delete a knowledge base."""
        if kb_id not in self._registry:
            logger.warning("Knowledge base '%s' not found", kb_id)
            return False

        async with self._lock:
            if kb_id in self._active_providers:
                provider = self._active_providers.pop(kb_id)
                await provider.aclose()

            kb_config = self._registry.pop(kb_id)
            self._save_registry()

            if remove_files:
                try:
                    abs_working_dir = kb_config.resolve_working_dir(self.base_storage_dir)
                    shutil.rmtree(abs_working_dir, ignore_errors=True)
                    logger.info("Removed files for knowledge base: %s", kb_id)
                except Exception as exc:
                    logger.error("Failed to remove files: %s", exc)

        logger.info("Deleted knowledge base: %s", kb_id)
        return True

    async def get_provider(self, kb_id: str) -> BaseRAGProvider:
        """Get or create a provider for the specified knowledge base."""
        if kb_id not in self._registry:
            raise ValueError(f"Knowledge base '{kb_id}' not found")

        async with self._lock:
            if kb_id in self._active_providers:
                return self._active_providers[kb_id]

            kb_config = self._registry[kb_id]
            provider_options = self._collect_provider_options(kb_config)

            abs_working_dir = kb_config.resolve_working_dir(self.base_storage_dir)

            rag_config = RAGConfig(
                working_dir=abs_working_dir,
                auto_initialize=True,
                options=provider_options,
            )

            init_kwargs = self._collect_provider_init_kwargs(kb_config)

            provider = RAGProviderFactory.create_provider(
                kb_config.provider_type,
                rag_config,
                **init_kwargs,
            )

            await provider.ensure_initialized()
            self._active_providers[kb_id] = provider
            logger.info("Initialized provider for knowledge base: %s", kb_id)
            return provider

    async def insert_documents(
        self,
        kb_id: str,
        documents: Iterable[RAGDocument],
        **kwargs: Any,
    ) -> IngestionResult:
        """Insert documents into a knowledge base."""
        provider = await self.get_provider(kb_id)
        result = await provider.ainsert(documents, **kwargs)

        kb_config = self._registry[kb_id]
        kb_config.updated_at = datetime.utcnow().isoformat()
        kb_config.metadata.setdefault("document_count", 0)
        kb_config.metadata["document_count"] += result.accepted
        self._save_registry()

        return result

    async def query(
        self,
        kb_id: str,
        query: str,
        **kwargs: Any,
    ) -> QueryResult:
        """Query a knowledge base."""
        provider = await self.get_provider(kb_id)
        result = await provider.aquery(query, **kwargs)
        if isinstance(result, QueryResult):
            return result
        raise TypeError(
            "Streaming query requested without stream=True; received async iterator instead."
        )

    async def health_check(self, kb_id: str) -> ProviderStatus:
        """Check health of a knowledge base."""
        provider = await self.get_provider(kb_id)
        return await provider.ahealth_check()

    async def close_all(self) -> None:
        """Close all active providers."""
        async with self._lock:
            for kb_id, provider in list(self._active_providers.items()):
                try:
                    await provider.aclose()
                    logger.info("Closed provider for knowledge base: %s", kb_id)
                except Exception as exc:
                    logger.error(
                        "Error closing provider for %s: %s",
                        kb_id,
                        exc,
                    )
            self._active_providers.clear()

    def update_metadata(
        self,
        kb_id: str,
        metadata_updates: Dict[str, Any],
    ) -> KnowledgeBaseConfig:
        """Update knowledge base metadata."""
        if kb_id not in self._registry:
            raise ValueError(f"Knowledge base '{kb_id}' not found")

        kb_config = self._registry[kb_id]
        kb_config.metadata.update(metadata_updates)
        kb_config.updated_at = datetime.utcnow().isoformat()
        self._save_registry()
        return kb_config

    def set_status(self, kb_id: str, status: str) -> KnowledgeBaseConfig:
        """Update knowledge base status."""
        if kb_id not in self._registry:
            raise ValueError(f"Knowledge base '{kb_id}' not found")

        kb_config = self._registry[kb_id]
        kb_config.status = status
        kb_config.updated_at = datetime.utcnow().isoformat()
        self._save_registry()
        return kb_config

    def migrate_to_relative_paths(self, project_root: Optional[Path] = None) -> int:
        """Migrate existing absolute paths to relative paths."""
        if project_root is None:
            project_root = self.base_storage_dir.parent

        migrated_count = 0

        for kb_id, kb_config in self._registry.items():
            changed = False

            working_path = Path(kb_config.working_dir)
            if working_path.is_absolute():
                try:
                    rel_path = working_path.relative_to(self.base_storage_dir)
                    kb_config.working_dir = str(rel_path)
                    changed = True
                except ValueError:
                    logger.warning(
                        "Cannot convert working_dir to relative: %s",
                        kb_config.working_dir,
                    )

            if "source_files" in kb_config.metadata:
                new_source_files = []
                for file_path in kb_config.metadata["source_files"]:
                    fp = Path(file_path)
                    if fp.is_absolute():
                        try:
                            rel_fp = fp.relative_to(project_root)
                            new_source_files.append(str(rel_fp))
                            changed = True
                        except ValueError:
                            new_source_files.append(file_path)
                    else:
                        new_source_files.append(file_path)
                kb_config.metadata["source_files"] = new_source_files

            if changed:
                migrated_count += 1
                logger.info("Migrated paths for knowledge base: %s", kb_id)

        if migrated_count > 0:
            self._save_registry()
            logger.info("Migrated %d knowledge bases to relative paths", migrated_count)

        return migrated_count

