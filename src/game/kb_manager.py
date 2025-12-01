"""Knowledge Base Manager wrapping RAG subsystem for game usage."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from config import ConfigLoader, GameConfig
from models import ModelProviderRegistry
from rag import KnowledgeBase
from rag.base_provider import ProviderStatus, QueryResult, RAGDocument
from rag.tools.kb_loader import GameDataLoader

logger = logging.getLogger(__name__)


DOCUMENT_TYPE_PUBLIC = {"puzzle_statement", "public_fact"}
DOCUMENT_TYPE_HINT = {"hint"}
DOCUMENT_TYPE_SECRET = {"puzzle_answer", "additional_info"}
DOCUMENT_TYPE_ALL = DOCUMENT_TYPE_PUBLIC | DOCUMENT_TYPE_HINT | DOCUMENT_TYPE_SECRET


@dataclass
class PuzzleInfo:
    puzzle_id: str
    kb_id: str
    title: str
    description: str
    game_type: str
    source_files: List[str]
    document_count: int


class KnowledgeBaseManager:
    def __init__(
        self,
        config: Optional[GameConfig] = None,
        model_registry: Optional[ModelProviderRegistry] = None,
        base_dir: Optional[Path] = None,
    ):
        if config is None:
            loader = ConfigLoader()
            config = loader.load_game_config()
        
        self._config = config
        self._model_registry = model_registry or ModelProviderRegistry()
        
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent
        
        self._base_dir = base_dir
        self._rag_storage_dir = base_dir / config.directories.rag_storage_dir
        self._data_base_dir = base_dir / config.directories.data_base_dir
        self._kb_id_prefix = config.puzzle.kb_id_prefix
        self._default_provider = config.rag.default_provider
        
        provider_options = self._model_registry.get_provider_options()
        
        self._knowledge_base = KnowledgeBase(
            base_storage_dir=str(self._rag_storage_dir),
            default_provider_type=self._default_provider,
            provider_kwargs={"config_options": provider_options},
        )

    def _to_relative_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self._base_dir))
        except ValueError:
            return str(path)

    @property
    def knowledge_base(self) -> KnowledgeBase:
        return self._knowledge_base

    def _puzzle_to_kb_id(self, puzzle_id: str) -> str:
        return f"{self._kb_id_prefix}{puzzle_id}"

    def _kb_id_to_puzzle(self, kb_id: str) -> str:
        if kb_id.startswith(self._kb_id_prefix):
            return kb_id[len(self._kb_id_prefix):]
        return kb_id

    def discover_puzzles(self) -> List[tuple[str, Path]]:
        if not self._data_base_dir.exists():
            logger.warning("Data base directory not found: %s", self._data_base_dir)
            return []
        return GameDataLoader.discover_games(self._data_base_dir)

    def get_puzzle_kb_id(self, puzzle_id: str) -> str:
        return self._puzzle_to_kb_id(puzzle_id)

    def kb_exists(self, puzzle_id: str) -> bool:
        kb_id = self._puzzle_to_kb_id(puzzle_id)
        return self._knowledge_base.get_knowledge_base(kb_id) is not None

    async def ensure_puzzle_kb(self, puzzle_id: str, puzzle_dir: Path) -> str:
        kb_id = self._puzzle_to_kb_id(puzzle_id)
        
        existing_kb = self._knowledge_base.get_knowledge_base(kb_id)
        if existing_kb is not None:
            logger.info("Knowledge base already exists: %s", kb_id)
            return kb_id
        
        game_info, file_paths, documents = GameDataLoader.load_game_directory(
            puzzle_dir, game_type="situation_puzzle"
        )
        
        if not documents:
            raise ValueError(f"No documents found for puzzle: {puzzle_id}")
        
        metadata = {
            "game_id": puzzle_id,
            "game_type": game_info.get("game_type", "situation_puzzle"),
            "source_files": [self._to_relative_path(p) for p in file_paths],
            "document_count": len(documents),
        }
        
        self._knowledge_base.create_knowledge_base(
            kb_id=kb_id,
            name=game_info.get("title", puzzle_id),
            description=game_info.get("description", ""),
            provider_type=self._default_provider,
            metadata=metadata,
        )
        
        await self._knowledge_base.insert_documents(kb_id, documents)
        logger.info(
            "Created and populated knowledge base: %s with %d documents",
            kb_id,
            len(documents),
        )
        
        return kb_id

    async def query_public(
        self,
        kb_id: str,
        query: str,
        **kwargs: Any,
    ) -> QueryResult:
        return await self._filtered_query(kb_id, query, DOCUMENT_TYPE_PUBLIC, **kwargs)

    async def query_with_hints(
        self,
        kb_id: str,
        query: str,
        **kwargs: Any,
    ) -> QueryResult:
        allowed_types = DOCUMENT_TYPE_PUBLIC | DOCUMENT_TYPE_HINT
        return await self._filtered_query(kb_id, query, allowed_types, **kwargs)

    async def query_full(
        self,
        kb_id: str,
        query: str,
        **kwargs: Any,
    ) -> QueryResult:
        return await self._filtered_query(kb_id, query, DOCUMENT_TYPE_ALL, **kwargs)

    async def _filtered_query(
        self,
        kb_id: str,
        query: str,
        allowed_types: Set[str],
        **kwargs: Any,
    ) -> QueryResult:
        result = await self._knowledge_base.query(kb_id, query, **kwargs)
        
        if hasattr(result, 'sources') and result.sources:
            filtered_sources = []
            for source in result.sources:
                doc_type = source.get("metadata", {}).get("type", "")
                if doc_type in allowed_types:
                    filtered_sources.append(source)
            
            return QueryResult(
                answer=result.answer,
                sources=filtered_sources,
                metadata=result.metadata,
            )
        
        return result

    async def health_check(self, puzzle_id: str) -> ProviderStatus:
        kb_id = self._puzzle_to_kb_id(puzzle_id)
        return await self._knowledge_base.health_check(kb_id)

    def list_puzzle_kbs(self) -> List[PuzzleInfo]:
        kbs = self._knowledge_base.list_knowledge_bases(status_filter="active")
        puzzles = []
        
        for kb in kbs:
            if not kb.kb_id.startswith(self._kb_id_prefix):
                continue
            
            puzzle_id = self._kb_id_to_puzzle(kb.kb_id)
            metadata = kb.metadata or {}
            
            puzzles.append(
                PuzzleInfo(
                    puzzle_id=puzzle_id,
                    kb_id=kb.kb_id,
                    title=kb.name,
                    description=kb.description,
                    game_type=metadata.get("game_type", "situation_puzzle"),
                    source_files=metadata.get("source_files", []),
                    document_count=metadata.get("document_count", 0),
                )
            )
        
        return puzzles

    async def close(self) -> None:
        await self._knowledge_base.close_all()
