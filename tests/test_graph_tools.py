"""Tests for LangGraph Tools."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from game.graph.tools import (
    ToolResult,
    RAGQueryPublicTool,
    RAGQueryFullTool,
    AppendEventTool,
    GetRecentEventsTool,
    SummarizeSessionTool,
    GetPlayerProfileTool,
    UpdatePlayerProfileTool,
    GameToolkit,
)
from game.memory.entities import SessionEventRecord, PlayerMemoryRecord, MemorySummaryType


@pytest.fixture
def mock_kb_manager():
    manager = MagicMock()
    manager.query_public = AsyncMock(
        return_value=MagicMock(answer="Public info", sources=["source1"])
    )
    manager.query_full = AsyncMock(
        return_value=MagicMock(answer="Full info with answer", sources=["source1", "source2"])
    )
    return manager


@pytest.fixture
def mock_memory_manager():
    manager = MagicMock()
    manager.append_session_event = MagicMock(
        return_value=MagicMock(namespace="session:test", key="event_0001")
    )
    manager.get_recent_events = MagicMock(
        return_value=[
            SessionEventRecord(
                session_id="test",
                turn_index=0,
                role="player",
                message="Q1",
                tags=["question"],
            )
        ]
    )
    manager.summarize_session = MagicMock(return_value="Session summary")
    manager.retrieve_player_profile = MagicMock(
        return_value=[
            PlayerMemoryRecord(
                player_id="player1",
                summary_type=MemorySummaryType.STYLE_PROFILE,
                content="Player tends to ask direct questions",
            )
        ]
    )
    manager.update_player_profile = MagicMock()
    return manager


class TestToolResult:
    def test_success_result(self):
        result = ToolResult(success=True, data={"key": "value"})
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None

    def test_error_result(self):
        result = ToolResult(success=False, error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"


class TestRAGQueryPublicTool:
    @pytest.mark.asyncio
    async def test_query_public_success(self, mock_kb_manager):
        tool = RAGQueryPublicTool(mock_kb_manager)
        
        result = await tool(kb_id="game_puzzle1", query="What is the puzzle?")
        
        assert result.success is True
        assert result.data["answer"] == "Public info"
        mock_kb_manager.query_public.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_public_error(self, mock_kb_manager):
        mock_kb_manager.query_public = AsyncMock(side_effect=Exception("KB error"))
        tool = RAGQueryPublicTool(mock_kb_manager)
        
        result = await tool(kb_id="game_puzzle1", query="What is the puzzle?")
        
        assert result.success is False
        assert "KB error" in result.error


class TestRAGQueryFullTool:
    @pytest.mark.asyncio
    async def test_query_full_success(self, mock_kb_manager):
        tool = RAGQueryFullTool(mock_kb_manager)
        
        result = await tool(kb_id="game_puzzle1", query="What is the answer?")
        
        assert result.success is True
        assert result.data["answer"] == "Full info with answer"
        mock_kb_manager.query_full.assert_called_once()


class TestAppendEventTool:
    def test_append_event_success(self, mock_memory_manager):
        tool = AppendEventTool(mock_memory_manager)
        event = SessionEventRecord(
            session_id="test",
            turn_index=0,
            role="player",
            message="Q1",
            tags=["question"],
        )
        
        result = tool(session_id="test", event=event)
        
        assert result.success is True
        assert result.data["namespace"] == "session:test"
        mock_memory_manager.append_session_event.assert_called_once()

    def test_append_event_error(self, mock_memory_manager):
        mock_memory_manager.append_session_event = MagicMock(
            side_effect=Exception("Storage error")
        )
        tool = AppendEventTool(mock_memory_manager)
        event = SessionEventRecord(
            session_id="test",
            turn_index=0,
            role="player",
            message="Q1",
            tags=["question"],
        )
        
        result = tool(session_id="test", event=event)
        
        assert result.success is False
        assert "Storage error" in result.error


class TestGetRecentEventsTool:
    def test_get_recent_events_success(self, mock_memory_manager):
        tool = GetRecentEventsTool(mock_memory_manager)
        
        result = tool(session_id="test", limit=10)
        
        assert result.success is True
        assert len(result.data) == 1
        assert result.data[0]["role"] == "player"


class TestSummarizeSessionTool:
    def test_summarize_session_success(self, mock_memory_manager):
        tool = SummarizeSessionTool(mock_memory_manager)
        
        result = tool(
            session_id="test",
            player_id="player1",
            puzzle_id="puzzle1",
        )
        
        assert result.success is True
        assert result.data["summary"] == "Session summary"


class TestGetPlayerProfileTool:
    def test_get_player_profile_success(self, mock_memory_manager):
        tool = GetPlayerProfileTool(mock_memory_manager)
        
        result = tool(player_id="player1", limit=3)
        
        assert result.success is True
        assert len(result.data) == 1
        assert "direct questions" in result.data[0]["content"]


class TestUpdatePlayerProfileTool:
    def test_update_player_profile_success(self, mock_memory_manager):
        tool = UpdatePlayerProfileTool(mock_memory_manager)
        
        result = tool(
            player_id="player1",
            new_summary="Updated profile info",
        )
        
        assert result.success is True
        assert result.data["player_id"] == "player1"
        mock_memory_manager.update_player_profile.assert_called_once()


class TestGameToolkit:
    def test_toolkit_with_all_managers(self, mock_kb_manager, mock_memory_manager):
        toolkit = GameToolkit(
            kb_manager=mock_kb_manager,
            memory_manager=mock_memory_manager,
        )
        
        assert toolkit.rag_query_public is not None
        assert toolkit.rag_query_full is not None
        assert toolkit.append_event is not None
        assert toolkit.get_recent_events is not None
        assert toolkit.summarize_session is not None
        assert toolkit.get_player_profile is not None
        assert toolkit.update_player_profile is not None

    def test_toolkit_with_kb_only(self, mock_kb_manager):
        toolkit = GameToolkit(kb_manager=mock_kb_manager)
        
        assert toolkit.rag_query_public is not None
        assert toolkit.rag_query_full is not None
        assert toolkit.append_event is None

    def test_toolkit_with_memory_only(self, mock_memory_manager):
        toolkit = GameToolkit(memory_manager=mock_memory_manager)
        
        assert toolkit.rag_query_public is None
        assert toolkit.append_event is not None

    def test_get_all_tools(self, mock_kb_manager, mock_memory_manager):
        toolkit = GameToolkit(
            kb_manager=mock_kb_manager,
            memory_manager=mock_memory_manager,
        )
        
        all_tools = toolkit.get_all_tools()
        
        assert "rag_query_public" in all_tools
        assert "rag_query_full" in all_tools
        assert "append_event" in all_tools

    def test_get_tool_by_name(self, mock_kb_manager):
        toolkit = GameToolkit(kb_manager=mock_kb_manager)
        
        tool = toolkit.get_tool("rag_query_public")
        
        assert tool is not None
        assert tool.name == "rag_query_public"

    def test_get_nonexistent_tool(self, mock_kb_manager):
        toolkit = GameToolkit(kb_manager=mock_kb_manager)
        
        tool = toolkit.get_tool("nonexistent")
        
        assert tool is None
