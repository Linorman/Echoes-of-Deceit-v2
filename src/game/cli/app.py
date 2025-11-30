"""CLI Application for the Turtle Soup game system.

This module provides the main CLI application that integrates with the GameEngine
and provides user-facing commands for listing puzzles, managing sessions, and playing games.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from game.engine import GameEngine
from game.session_runner import GameSessionRunner
from game.domain.entities import GameSession, GameState, PuzzleSummary

logger = logging.getLogger(__name__)


class GameCLIApp:
    def __init__(
        self,
        config_dir: Optional[Path] = None,
        base_dir: Optional[Path] = None,
    ):
        self._engine = GameEngine(config_dir=config_dir, base_dir=base_dir)
        self._active_runner: Optional[GameSessionRunner] = None

    @property
    def engine(self) -> GameEngine:
        return self._engine

    def list_puzzles(self) -> list[PuzzleSummary]:
        return self._engine.list_puzzles()

    async def create_session(
        self,
        puzzle_id: str,
        player_name: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> GameSession:
        return await self._engine.create_session(puzzle_id, player_name, options)

    def get_session(self, session_id: str) -> GameSession:
        return self._engine.get_session(session_id)

    def list_sessions(
        self,
        state_filter: Optional[GameState] = None,
        puzzle_id: Optional[str] = None,
        player_id: Optional[str] = None,
    ) -> list[GameSession]:
        return self._engine.list_sessions(
            state_filter=state_filter,
            puzzle_id=puzzle_id,
            player_id=player_id,
        )

    def create_runner(self, session: GameSession) -> GameSessionRunner:
        puzzle = self._engine.get_puzzle(session.puzzle_id)
        runner = GameSessionRunner(
            session=session,
            puzzle=puzzle,
            kb_manager=self._engine.kb_manager,
            memory_manager=self._engine.memory_manager,
            session_store=self._engine.session_store,
            llm_client=self._engine.model_registry.get_llm_client(),
            agents_config=self._engine.agents_config,
        )
        self._active_runner = runner
        return runner

    def get_session_status(self, session_id: str) -> dict:
        session = self._engine.get_session(session_id)
        puzzle = self._engine.get_puzzle(session.puzzle_id)

        question_count = sum(
            1 for event in session.turn_history
            if "question" in event.tags
        )

        return {
            "session_id": session.session_id,
            "puzzle_id": session.puzzle_id,
            "puzzle_title": puzzle.title,
            "state": session.state.value,
            "turn_count": session.turn_count,
            "question_count": question_count,
            "hint_count": session.hint_count,
            "max_hints": puzzle.constraints.max_hints,
            "score": session.score,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
        }

    async def close(self) -> None:
        await self._engine.close()
        logger.info("GameCLIApp closed")
