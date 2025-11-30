"""LangGraph Game Runner.

This module provides the GameGraphRunner class that wraps
the compiled LangGraph and provides a simple API for game sessions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver

from game.graph.state import (
    GameGraphState,
    GameGraphOutput,
    GamePhase,
    TurnEvent,
)
from game.graph.builder import GameGraphBuilder

if TYPE_CHECKING:
    from config import AgentsConfig
    from game.kb_manager import KnowledgeBaseManager
    from game.memory.manager import MemoryManager
    from game.domain.entities import GameSession, Puzzle
    from models.base import LLMClient

logger = logging.getLogger(__name__)


class GameGraphRunner:
    def __init__(
        self,
        session: "GameSession",
        puzzle: "Puzzle",
        llm_client: Optional["LLMClient"] = None,
        kb_manager: Optional["KnowledgeBaseManager"] = None,
        memory_manager: Optional["MemoryManager"] = None,
        agents_config: Optional["AgentsConfig"] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        self._session = session
        self._puzzle = puzzle
        self._llm_client = llm_client
        self._kb_manager = kb_manager
        self._memory_manager = memory_manager
        self._agents_config = agents_config

        if checkpointer is None:
            checkpointer = MemorySaver()
        self._checkpointer = checkpointer

        builder = GameGraphBuilder(
            llm_client=llm_client,
            kb_manager=kb_manager,
            memory_manager=memory_manager,
            agents_config=agents_config,
            checkpointer=checkpointer,
        )
        self._graph = builder.compile()

        self._initialized = False
        self._current_state: Optional[GameGraphState] = None

    @property
    def session(self) -> "GameSession":
        return self._session

    @property
    def puzzle(self) -> "Puzzle":
        return self._puzzle

    @property
    def is_active(self) -> bool:
        if self._current_state:
            return self._current_state.game_phase not in (
                GamePhase.COMPLETED,
                GamePhase.ABORTED,
            )
        return self._session.is_active

    @property
    def thread_id(self) -> str:
        return self._session.session_id

    def _get_config(self) -> Dict[str, Any]:
        return {"configurable": {"thread_id": self.thread_id}}

    def _create_initial_state(self) -> Dict[str, Any]:
        player_id = self._session.player_ids[0] if self._session.player_ids else "unknown"

        return {
            "session_id": self._session.session_id,
            "kb_id": self._session.kb_id or "",
            "player_id": player_id,
            "puzzle_id": self._session.puzzle_id,
            "puzzle_statement": self._puzzle.puzzle_statement,
            "puzzle_answer": self._puzzle.answer,
            "puzzle_title": self._puzzle.title,
            "max_hints": self._puzzle.constraints.max_hints,
            "game_phase": GamePhase.INTRO,
            "turn_index": 0,
            "hint_count": 0,
            "turn_history": [],
            "last_user_message": "",
            "last_dm_response": "",
        }

    async def start_game(self) -> GameGraphOutput:
        config = self._get_config()
        initial_state = self._create_initial_state()

        result = await self._graph.ainvoke(initial_state, config)

        self._current_state = GameGraphState(**result)
        self._initialized = True

        return GameGraphOutput(
            message=result.get("last_dm_response", ""),
            game_phase=result.get("game_phase", GamePhase.PLAYING),
            turn_index=result.get("turn_index", 0),
        )

    async def process_input(self, user_message: str) -> GameGraphOutput:
        if not self._initialized:
            return await self.start_game()

        config = self._get_config()

        input_state = {
            "last_user_message": user_message,
        }

        result = await self._graph.ainvoke(input_state, config)

        self._current_state = GameGraphState(**result)

        game_phase = result.get("game_phase", GamePhase.PLAYING)
        game_over = game_phase in (GamePhase.COMPLETED, GamePhase.ABORTED)

        return GameGraphOutput(
            message=result.get("last_dm_response", ""),
            verdict=result.get("last_verdict"),
            game_over=game_over,
            game_phase=game_phase,
            turn_index=result.get("turn_index", 0),
            metadata={
                "hint_count": result.get("hint_count", 0),
                "score": result.get("score"),
            },
        )

    def get_state(self) -> Optional[GameGraphState]:
        return self._current_state

    async def get_checkpoint_state(self) -> Optional[Dict[str, Any]]:
        config = self._get_config()
        try:
            checkpoint = self._checkpointer.get(config)
            if checkpoint:
                return checkpoint.get("channel_values", {})
        except Exception as e:
            logger.warning("Failed to get checkpoint state: %s", e)
        return None

    def get_turn_history(self) -> list[TurnEvent]:
        if self._current_state:
            return self._current_state.turn_history
        return []


class GameGraphRunnerFactory:
    def __init__(
        self,
        llm_client: Optional["LLMClient"] = None,
        kb_manager: Optional["KnowledgeBaseManager"] = None,
        memory_manager: Optional["MemoryManager"] = None,
        agents_config: Optional["AgentsConfig"] = None,
    ):
        self._llm_client = llm_client
        self._kb_manager = kb_manager
        self._memory_manager = memory_manager
        self._agents_config = agents_config
        self._runners: Dict[str, GameGraphRunner] = {}

    def create_runner(
        self,
        session: "GameSession",
        puzzle: "Puzzle",
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ) -> GameGraphRunner:
        runner = GameGraphRunner(
            session=session,
            puzzle=puzzle,
            llm_client=self._llm_client,
            kb_manager=self._kb_manager,
            memory_manager=self._memory_manager,
            agents_config=self._agents_config,
            checkpointer=checkpointer,
        )
        self._runners[session.session_id] = runner
        return runner

    def get_runner(self, session_id: str) -> Optional[GameGraphRunner]:
        return self._runners.get(session_id)

    def remove_runner(self, session_id: str) -> None:
        self._runners.pop(session_id, None)
