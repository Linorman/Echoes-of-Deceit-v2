"""Game Engine for managing game lifecycle and resources."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import ConfigLoader, AgentsConfig, GameConfig, ModelsConfig
from game.domain.entities import GameSession, GameState, PlayerProfile, PuzzleSummary
from game.kb_manager import KnowledgeBaseManager
from game.memory.manager import MemoryManager
from game.repository.puzzle_repository import PuzzleRepository
from game.storage.session_store import GameSessionStore, PlayerProfileStore
from models import ModelProviderRegistry
from rag.base_provider import ProviderStatus

logger = logging.getLogger(__name__)


class GameEngine:
    def __init__(
        self,
        config_dir: Optional[Path] = None,
        base_dir: Optional[Path] = None,
    ):
        self._config_loader = ConfigLoader(config_dir)
        self._game_config: GameConfig = self._config_loader.load_game_config()
        self._models_config: ModelsConfig = self._config_loader.load_models_config()
        self._agents_config: AgentsConfig = self._config_loader.load_agents_config()

        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent

        self._base_dir = base_dir
        self._model_registry = ModelProviderRegistry(self._models_config)
        self._kb_manager = KnowledgeBaseManager(
            config=self._game_config,
            model_registry=self._model_registry,
            base_dir=base_dir,
        )
        self._puzzle_repository = PuzzleRepository(
            config=self._game_config,
            base_dir=base_dir,
        )
        self._session_store = GameSessionStore(
            config=self._game_config,
            base_dir=base_dir,
        )
        self._profile_store = PlayerProfileStore(
            config=self._game_config,
            base_dir=base_dir,
        )
        self._memory_manager = MemoryManager(
            config=self._game_config,
            base_dir=base_dir,
        )

        self._initialized = False
        logger.info("GameEngine initialized with base_dir=%s", base_dir)

    @property
    def game_config(self) -> GameConfig:
        return self._game_config

    @property
    def models_config(self) -> ModelsConfig:
        return self._models_config

    @property
    def agents_config(self) -> AgentsConfig:
        return self._agents_config

    @property
    def model_registry(self) -> ModelProviderRegistry:
        return self._model_registry

    @property
    def kb_manager(self) -> KnowledgeBaseManager:
        return self._kb_manager

    @property
    def puzzle_repository(self) -> PuzzleRepository:
        return self._puzzle_repository

    @property
    def session_store(self) -> GameSessionStore:
        return self._session_store

    @property
    def profile_store(self) -> PlayerProfileStore:
        return self._profile_store

    @property
    def memory_manager(self) -> MemoryManager:
        return self._memory_manager

    def list_puzzles(self) -> List[PuzzleSummary]:
        return self._puzzle_repository.list_puzzles()

    def get_puzzle(self, puzzle_id: str):
        return self._puzzle_repository.get_puzzle(puzzle_id)

    async def ensure_puzzle_kb(self, puzzle_id: str) -> str:
        puzzle_dir = self._puzzle_repository.get_puzzle_dir(puzzle_id)
        return await self._kb_manager.ensure_puzzle_kb(puzzle_id, puzzle_dir)

    async def health_check(self, puzzle_id: str) -> ProviderStatus:
        return await self._kb_manager.health_check(puzzle_id)

    async def create_session(
        self,
        puzzle_id: str,
        player_id: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> GameSession:
        options = options or {}

        self._puzzle_repository.get_puzzle(puzzle_id)

        kb_id = await self.ensure_puzzle_kb(puzzle_id)

        profile = self._profile_store.get_or_create_profile(player_id)

        session = self._session_store.create_session(
            puzzle_id=puzzle_id,
            player_ids=[player_id],
            kb_id=kb_id,
            **options,
        )

        logger.info(
            "Created session %s for puzzle %s, player %s",
            session.session_id,
            puzzle_id,
            player_id,
        )
        return session

    def get_session(self, session_id: str) -> GameSession:
        return self._session_store.load_session(session_id)

    def save_session(self, session: GameSession) -> None:
        self._session_store.save_session(session)

    def list_sessions(
        self,
        state_filter: Optional[GameState] = None,
        puzzle_id: Optional[str] = None,
        player_id: Optional[str] = None,
    ) -> List[GameSession]:
        return self._session_store.list_sessions(
            state_filter=state_filter,
            puzzle_id=puzzle_id,
            player_id=player_id,
        )

    def get_player_profile(self, player_id: str) -> PlayerProfile:
        return self._profile_store.get_or_create_profile(player_id)

    async def close(self) -> None:
        await self._kb_manager.close()
        logger.info("GameEngine closed")
