"""LangChain Tools for RAG and Memory operations.

This module exposes domain services as tools usable inside LangGraph nodes:
- RAG tools: query_public, query_full
- Memory tools: append_event, get_recent_events, summarize_session, 
  get_player_profile, update_player_profile
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from game.kb_manager import KnowledgeBaseManager
    from game.memory.manager import MemoryManager
    from game.memory.entities import SessionEventRecord

logger = logging.getLogger(__name__)


class ToolResult(BaseModel):
    success: bool = True
    data: Any = None
    error: Optional[str] = None


class RAGQueryPublicTool:
    name: str = "rag_query_public"
    description: str = "Query the knowledge base for public puzzle information (puzzle statement and public facts only)"

    def __init__(self, kb_manager: KnowledgeBaseManager):
        self._kb_manager = kb_manager

    async def __call__(
        self,
        kb_id: str,
        query: str,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            result = await self._kb_manager.query_public(kb_id, query, **kwargs)
            return ToolResult(
                success=True,
                data={
                    "answer": result.answer if result else None,
                    "sources": result.sources if result else [],
                },
            )
        except Exception as e:
            logger.error("RAG query_public failed: %s", e)
            return ToolResult(success=False, error=str(e))


class RAGQueryFullTool:
    name: str = "rag_query_full"
    description: str = "Query the knowledge base with full access (including puzzle answer and hints) - DM/Judge only"

    def __init__(self, kb_manager: KnowledgeBaseManager):
        self._kb_manager = kb_manager

    async def __call__(
        self,
        kb_id: str,
        query: str,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            result = await self._kb_manager.query_full(kb_id, query, **kwargs)
            return ToolResult(
                success=True,
                data={
                    "answer": result.answer if result else None,
                    "sources": result.sources if result else [],
                },
            )
        except Exception as e:
            logger.error("RAG query_full failed: %s", e)
            return ToolResult(success=False, error=str(e))


class AppendEventTool:
    name: str = "append_event"
    description: str = "Append a session event to the turn history and memory store"

    def __init__(self, memory_manager: MemoryManager):
        self._memory_manager = memory_manager

    def __call__(
        self,
        session_id: str,
        event: "SessionEventRecord",
    ) -> ToolResult:
        try:
            doc = self._memory_manager.append_session_event(session_id, event)
            return ToolResult(
                success=True,
                data={"namespace": doc.namespace, "key": doc.key},
            )
        except Exception as e:
            logger.error("Append event failed: %s", e)
            return ToolResult(success=False, error=str(e))


class GetRecentEventsTool:
    name: str = "get_recent_events"
    description: str = "Get the most recent session events from memory"

    def __init__(self, memory_manager: MemoryManager):
        self._memory_manager = memory_manager

    def __call__(
        self,
        session_id: str,
        limit: int = 10,
    ) -> ToolResult:
        try:
            events = self._memory_manager.get_recent_events(session_id, limit=limit)
            return ToolResult(
                success=True,
                data=[
                    {
                        "turn_index": e.turn_index,
                        "role": e.role,
                        "message": e.message,
                        "tags": e.tags,
                    }
                    for e in events
                ],
            )
        except Exception as e:
            logger.error("Get recent events failed: %s", e)
            return ToolResult(success=False, error=str(e))


class SummarizeSessionTool:
    name: str = "summarize_session"
    description: str = "Generate a summary of the session and update player profile"

    def __init__(self, memory_manager: MemoryManager):
        self._memory_manager = memory_manager

    def __call__(
        self,
        session_id: str,
        player_id: Optional[str] = None,
        puzzle_id: Optional[str] = None,
    ) -> ToolResult:
        try:
            summary = self._memory_manager.summarize_session(
                session_id=session_id,
                player_id=player_id,
                puzzle_id=puzzle_id,
            )
            return ToolResult(success=True, data={"summary": summary})
        except Exception as e:
            logger.error("Summarize session failed: %s", e)
            return ToolResult(success=False, error=str(e))


class GetPlayerProfileTool:
    name: str = "get_player_profile"
    description: str = "Retrieve player profile and memory for DM/Judge prompts"

    def __init__(self, memory_manager: MemoryManager):
        self._memory_manager = memory_manager

    def __call__(
        self,
        player_id: str,
        limit: int = 3,
    ) -> ToolResult:
        try:
            profiles = self._memory_manager.retrieve_player_profile(
                player_id=player_id,
                limit=limit,
            )
            return ToolResult(
                success=True,
                data=[
                    {
                        "summary_type": p.summary_type,
                        "content": p.content,
                    }
                    for p in profiles
                ],
            )
        except Exception as e:
            logger.error("Get player profile failed: %s", e)
            return ToolResult(success=False, error=str(e))


class UpdatePlayerProfileTool:
    name: str = "update_player_profile"
    description: str = "Update player long-term memory profile"

    def __init__(self, memory_manager: MemoryManager):
        self._memory_manager = memory_manager

    def __call__(
        self,
        player_id: str,
        new_summary: str,
        summary_type: str = "performance_summary",
    ) -> ToolResult:
        try:
            self._memory_manager.update_player_profile(
                player_id=player_id,
                new_summary=new_summary,
            )
            return ToolResult(success=True, data={"player_id": player_id})
        except Exception as e:
            logger.error("Update player profile failed: %s", e)
            return ToolResult(success=False, error=str(e))


class GameToolkit:
    def __init__(
        self,
        kb_manager: Optional[KnowledgeBaseManager] = None,
        memory_manager: Optional[MemoryManager] = None,
    ):
        self._kb_manager = kb_manager
        self._memory_manager = memory_manager
        self._tools: Dict[str, Any] = {}
        self._build_tools()

    def _build_tools(self) -> None:
        if self._kb_manager:
            self._tools["rag_query_public"] = RAGQueryPublicTool(self._kb_manager)
            self._tools["rag_query_full"] = RAGQueryFullTool(self._kb_manager)

        if self._memory_manager:
            self._tools["append_event"] = AppendEventTool(self._memory_manager)
            self._tools["get_recent_events"] = GetRecentEventsTool(self._memory_manager)
            self._tools["summarize_session"] = SummarizeSessionTool(self._memory_manager)
            self._tools["get_player_profile"] = GetPlayerProfileTool(self._memory_manager)
            self._tools["update_player_profile"] = UpdatePlayerProfileTool(self._memory_manager)

    def get_tool(self, name: str) -> Any:
        return self._tools.get(name)

    def get_all_tools(self) -> Dict[str, Any]:
        return self._tools.copy()

    @property
    def rag_query_public(self) -> Optional[RAGQueryPublicTool]:
        return self._tools.get("rag_query_public")

    @property
    def rag_query_full(self) -> Optional[RAGQueryFullTool]:
        return self._tools.get("rag_query_full")

    @property
    def append_event(self) -> Optional[AppendEventTool]:
        return self._tools.get("append_event")

    @property
    def get_recent_events(self) -> Optional[GetRecentEventsTool]:
        return self._tools.get("get_recent_events")

    @property
    def summarize_session(self) -> Optional[SummarizeSessionTool]:
        return self._tools.get("summarize_session")

    @property
    def get_player_profile(self) -> Optional[GetPlayerProfileTool]:
        return self._tools.get("get_player_profile")

    @property
    def update_player_profile(self) -> Optional[UpdatePlayerProfileTool]:
        return self._tools.get("update_player_profile")
