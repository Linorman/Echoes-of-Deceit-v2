"""LangGraph Game Graph Builder.

This module provides the GameGraphBuilder class that constructs
the complete game flow graph with nodes and conditional edges.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional, TYPE_CHECKING

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver

from game.graph.state import GameGraphState, GamePhase, GameMode, MessageType
from game.graph.nodes import (
    PlayerMessageNode,
    PlayerAgentNode,
    IntroNode,
    DMQuestionNode,
    DMHypothesisNode,
    CommandHandlerNode,
    MemoryUpdateNode,
    RevealSolutionNode,
)

if TYPE_CHECKING:
    from config import AgentsConfig
    from game.kb_manager import KnowledgeBaseManager
    from game.memory.manager import MemoryManager
    from models.base import LLMClient

logger = logging.getLogger(__name__)


NODE_PLAYER_MESSAGE = "player_message"
NODE_PLAYER_AGENT = "player_agent"
NODE_INTRO = "intro"
NODE_DM_QUESTION = "dm_question"
NODE_DM_HYPOTHESIS = "dm_hypothesis"
NODE_COMMAND_HANDLER = "command_handler"
NODE_MEMORY_UPDATE = "memory_update"
NODE_REVEAL_SOLUTION = "reveal_solution"


def route_by_phase(state: GameGraphState) -> str:
    if state.game_phase == GamePhase.INTRO:
        return NODE_INTRO
    elif state.game_phase in (GamePhase.COMPLETED, GamePhase.ABORTED):
        return END
    else:
        if state.player_agent_enabled and state.awaiting_player_agent:
            return NODE_PLAYER_AGENT
        return NODE_PLAYER_MESSAGE


def route_by_message_type(state: GameGraphState) -> str:
    if state.message_type == MessageType.COMMAND:
        return NODE_COMMAND_HANDLER
    elif state.message_type == MessageType.HYPOTHESIS:
        return NODE_DM_HYPOTHESIS
    else:
        return NODE_DM_QUESTION


def route_after_dm(state: GameGraphState) -> str:
    if state.game_phase == GamePhase.COMPLETED:
        return NODE_REVEAL_SOLUTION
    if state.player_agent_enabled:
        return NODE_MEMORY_UPDATE
    return NODE_MEMORY_UPDATE


def route_after_command(state: GameGraphState) -> str:
    if state.game_phase == GamePhase.ABORTED:
        return END
    return NODE_MEMORY_UPDATE


def route_after_player_agent(state: GameGraphState) -> str:
    if state.message_type == MessageType.HYPOTHESIS:
        return NODE_DM_HYPOTHESIS
    return NODE_DM_QUESTION


def route_after_memory_update(state: GameGraphState) -> str:
    if state.player_agent_enabled and state.game_phase == GamePhase.PLAYING:
        return NODE_PLAYER_AGENT
    return END


class GameGraphBuilder:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        kb_manager: Optional[KnowledgeBaseManager] = None,
        memory_manager: Optional[MemoryManager] = None,
        agents_config: Optional[AgentsConfig] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        self._llm_client = llm_client
        self._kb_manager = kb_manager
        self._memory_manager = memory_manager
        self._agents_config = agents_config
        self._checkpointer = checkpointer

        self._nodes = self._create_nodes()

    def _create_nodes(self) -> Dict[str, Any]:
        node_kwargs = {
            "llm_client": self._llm_client,
            "kb_manager": self._kb_manager,
            "memory_manager": self._memory_manager,
            "agents_config": self._agents_config,
        }

        return {
            NODE_PLAYER_MESSAGE: PlayerMessageNode(**node_kwargs),
            NODE_PLAYER_AGENT: PlayerAgentNode(**node_kwargs),
            NODE_INTRO: IntroNode(**node_kwargs),
            NODE_DM_QUESTION: DMQuestionNode(**node_kwargs),
            NODE_DM_HYPOTHESIS: DMHypothesisNode(**node_kwargs),
            NODE_COMMAND_HANDLER: CommandHandlerNode(**node_kwargs),
            NODE_MEMORY_UPDATE: MemoryUpdateNode(**node_kwargs),
            NODE_REVEAL_SOLUTION: RevealSolutionNode(**node_kwargs),
        }

    def build(self) -> StateGraph:
        builder = StateGraph(GameGraphState)

        for name, node in self._nodes.items():
            builder.add_node(name, node)

        builder.add_conditional_edges(
            START,
            route_by_phase,
            {
                NODE_INTRO: NODE_INTRO,
                NODE_PLAYER_MESSAGE: NODE_PLAYER_MESSAGE,
                NODE_PLAYER_AGENT: NODE_PLAYER_AGENT,
                END: END,
            },
        )

        builder.add_edge(NODE_INTRO, NODE_MEMORY_UPDATE)

        builder.add_conditional_edges(
            NODE_PLAYER_MESSAGE,
            route_by_message_type,
            {
                NODE_COMMAND_HANDLER: NODE_COMMAND_HANDLER,
                NODE_DM_HYPOTHESIS: NODE_DM_HYPOTHESIS,
                NODE_DM_QUESTION: NODE_DM_QUESTION,
            },
        )

        builder.add_conditional_edges(
            NODE_PLAYER_AGENT,
            route_after_player_agent,
            {
                NODE_DM_QUESTION: NODE_DM_QUESTION,
                NODE_DM_HYPOTHESIS: NODE_DM_HYPOTHESIS,
            },
        )

        builder.add_conditional_edges(
            NODE_DM_QUESTION,
            route_after_dm,
            {
                NODE_MEMORY_UPDATE: NODE_MEMORY_UPDATE,
                NODE_REVEAL_SOLUTION: NODE_REVEAL_SOLUTION,
            },
        )

        builder.add_conditional_edges(
            NODE_DM_HYPOTHESIS,
            route_after_dm,
            {
                NODE_MEMORY_UPDATE: NODE_MEMORY_UPDATE,
                NODE_REVEAL_SOLUTION: NODE_REVEAL_SOLUTION,
            },
        )

        builder.add_conditional_edges(
            NODE_COMMAND_HANDLER,
            route_after_command,
            {
                NODE_MEMORY_UPDATE: NODE_MEMORY_UPDATE,
                END: END,
            },
        )

        builder.add_conditional_edges(
            NODE_MEMORY_UPDATE,
            route_after_memory_update,
            {
                NODE_PLAYER_AGENT: NODE_PLAYER_AGENT,
                END: END,
            },
        )

        builder.add_edge(NODE_REVEAL_SOLUTION, NODE_MEMORY_UPDATE)

        return builder

    def compile(self) -> Any:
        builder = self.build()

        if self._checkpointer:
            return builder.compile(checkpointer=self._checkpointer)
        else:
            return builder.compile()

    @classmethod
    def create_with_memory_saver(
        cls,
        llm_client: Optional[LLMClient] = None,
        kb_manager: Optional[KnowledgeBaseManager] = None,
        memory_manager: Optional[MemoryManager] = None,
        agents_config: Optional[AgentsConfig] = None,
    ) -> "GameGraphBuilder":
        checkpointer = MemorySaver()
        return cls(
            llm_client=llm_client,
            kb_manager=kb_manager,
            memory_manager=memory_manager,
            agents_config=agents_config,
            checkpointer=checkpointer,
        )
