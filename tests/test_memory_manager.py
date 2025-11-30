"""Tests for MemoryManager."""

import pytest
from pathlib import Path
import tempfile
import shutil

from game.memory.manager import MemoryManager, default_summarizer
from game.memory.entities import (
    EventTag,
    MemorySummaryType,
    SessionEventRecord,
)
from game.memory.file_store import FileMemoryStore


@pytest.fixture
def temp_dir():
    dir_path = Path(tempfile.mkdtemp())
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture
def store(temp_dir):
    return FileMemoryStore(base_dir=temp_dir)


@pytest.fixture
def manager(store):
    return MemoryManager(store=store)


class TestMemoryManagerSessionEvents:
    def test_append_and_get_session_event(self, manager):
        event = SessionEventRecord(
            session_id="session1",
            turn_index=0,
            role="dm",
            message="Welcome to the puzzle!",
            tags=["intro"],
        )

        manager.append_session_event("session1", event)

        history = manager.get_session_history("session1")
        assert len(history) == 1
        assert history[0].message == "Welcome to the puzzle!"
        assert history[0].turn_index == 0

    def test_multiple_events_in_order(self, manager):
        for i in range(5):
            event = SessionEventRecord(
                session_id="session1",
                turn_index=i,
                role="player" if i % 2 == 0 else "dm",
                message=f"Message {i}",
            )
            manager.append_session_event("session1", event)

        history = manager.get_session_history("session1")
        assert len(history) == 5
        for i, event in enumerate(history):
            assert event.turn_index == i

    def test_get_recent_events_with_limit(self, manager):
        for i in range(10):
            event = SessionEventRecord(
                session_id="session1",
                turn_index=i,
                role="player",
                message=f"Message {i}",
            )
            manager.append_session_event("session1", event)

        recent = manager.get_recent_events("session1", limit=3)
        assert len(recent) == 3
        assert recent[0].turn_index == 7
        assert recent[-1].turn_index == 9

    def test_create_event_helper(self, manager):
        event = manager.create_event(
            session_id="session1",
            turn_index=0,
            role="dm",
            message="Hello",
            tags=["greeting"],
        )

        assert event.session_id == "session1"
        assert event.role == "dm"
        assert "greeting" in event.tags


class TestMemoryManagerSummarization:
    def test_summarize_empty_session(self, manager):
        summary = manager.summarize_session("empty_session")
        assert "No events recorded" in summary

    def test_summarize_session_with_events(self, manager):
        events = [
            SessionEventRecord(
                session_id="session1",
                turn_index=0,
                role="player",
                message="Is the man dead?",
                tags=["question"],
            ),
            SessionEventRecord(
                session_id="session1",
                turn_index=1,
                role="dm",
                message="Yes",
                tags=["answer"],
            ),
            SessionEventRecord(
                session_id="session1",
                turn_index=2,
                role="player",
                message="The man was murdered",
                tags=["hypothesis", "final_verdict"],
            ),
        ]

        for event in events:
            manager.append_session_event("session1", event)

        summary = manager.summarize_session("session1")

        assert "3 total events" in summary
        assert "Questions asked: 1" in summary

    def test_summarize_updates_player_memory(self, manager):
        event = SessionEventRecord(
            session_id="session1",
            turn_index=0,
            role="player",
            message="Question",
            tags=["question"],
        )
        manager.append_session_event("session1", event)

        manager.summarize_session("session1", player_id="player1")

        results = manager.retrieve_player_profile("player1")
        assert len(results) > 0

    def test_summarize_updates_puzzle_stats(self, manager):
        events = [
            SessionEventRecord(
                session_id="s1",
                turn_index=0,
                role="player",
                message="Q1",
                tags=["question"],
            ),
            SessionEventRecord(
                session_id="s1",
                turn_index=1,
                role="player",
                message="Q2",
                tags=["question"],
            ),
        ]
        for e in events:
            manager.append_session_event("s1", e)

        manager.summarize_session("s1", puzzle_id="puzzle1")

        stats = manager.get_puzzle_stats("puzzle1")
        assert stats is not None
        assert stats["total_sessions"] == 1
        assert stats["total_questions"] == 2


class TestMemoryManagerPlayerProfile:
    def test_update_player_profile_new(self, manager):
        record = manager.update_player_profile(
            player_id="player1",
            new_info="Player prefers direct questions",
            summary_type=MemorySummaryType.STYLE_PROFILE,
        )

        assert record.player_id == "player1"
        assert record.content == "Player prefers direct questions"
        assert record.version == 1

    def test_update_player_profile_merge(self, manager):
        manager.update_player_profile(
            player_id="player1",
            new_info="Initial observation",
        )

        record = manager.update_player_profile(
            player_id="player1",
            new_info="New observation",
        )

        assert record.version == 2
        assert "Initial observation" in record.content
        assert "New observation" in record.content

    def test_update_with_custom_merger(self, manager):
        def custom_merger(existing: str, new: str) -> str:
            return f"[MERGED] {existing} + {new}"

        manager.update_player_profile("p1", "First")
        record = manager.update_player_profile(
            "p1",
            "Second",
            custom_merger=custom_merger,
        )

        assert "[MERGED]" in record.content

    def test_retrieve_player_profile(self, manager):
        manager.update_player_profile(
            "player1",
            "Analytical player",
            MemorySummaryType.STYLE_PROFILE,
        )
        manager.update_player_profile(
            "player1",
            "Good at puzzles",
            MemorySummaryType.PERFORMANCE_SUMMARY,
        )

        results = manager.retrieve_player_profile("player1", limit=5)
        assert len(results) == 2

    def test_get_player_memory_specific_type(self, manager):
        manager.update_player_profile(
            "player1",
            "Test content",
            MemorySummaryType.STYLE_PROFILE,
        )

        record = manager.get_player_memory("player1", MemorySummaryType.STYLE_PROFILE)
        assert record is not None
        assert record.content == "Test content"

        missing = manager.get_player_memory("player1", MemorySummaryType.PERFORMANCE_SUMMARY)
        assert missing is None


class TestMemoryManagerGlobalMemory:
    def test_store_global_memory(self, manager):
        record = manager.store_global_memory(
            record_id="hint_strategy_1",
            content="Keep hints vague for new players",
            summary_type=MemorySummaryType.HINT_STRATEGY,
        )

        assert record.record_id == "hint_strategy_1"
        assert record.content == "Keep hints vague for new players"

    def test_get_global_memory(self, manager):
        manager.store_global_memory(
            record_id="test_record",
            content="Test content",
            summary_type=MemorySummaryType.HINT_STRATEGY,
        )

        record = manager.get_global_memory("test_record")
        assert record is not None
        assert record.content == "Test content"

    def test_search_global_memory(self, manager):
        manager.store_global_memory(
            "hint1",
            "Hint about food",
            MemorySummaryType.HINT_STRATEGY,
        )
        manager.store_global_memory(
            "hint2",
            "Hint about travel",
            MemorySummaryType.HINT_STRATEGY,
        )

        results = manager.search_global_memory(query="food")
        assert len(results) > 0

    def test_global_memory_with_puzzle_id(self, manager):
        manager.store_global_memory(
            record_id="puzzle1_hint",
            content="Specific puzzle hint",
            summary_type=MemorySummaryType.HINT_STRATEGY,
            puzzle_id="puzzle1",
        )

        record = manager.get_global_memory("puzzle1_hint")
        assert record.puzzle_id == "puzzle1"


class TestMemoryManagerClear:
    def test_clear_session_memory(self, manager):
        for i in range(5):
            event = SessionEventRecord(
                session_id="session1",
                turn_index=i,
                role="player",
                message=f"Message {i}",
            )
            manager.append_session_event("session1", event)

        count = manager.clear_session_memory("session1")
        assert count == 5

        history = manager.get_session_history("session1")
        assert len(history) == 0

    def test_clear_player_memory(self, manager):
        manager.update_player_profile("player1", "Info1", MemorySummaryType.STYLE_PROFILE)
        manager.update_player_profile("player1", "Info2", MemorySummaryType.PERFORMANCE_SUMMARY)

        count = manager.clear_player_memory("player1")
        assert count == 2

        results = manager.retrieve_player_profile("player1")
        assert len(results) == 0


class TestDefaultSummarizer:
    def test_empty_events(self):
        summary = default_summarizer([])
        assert "No events recorded" in summary

    def test_with_questions_and_hints(self):
        events = [
            SessionEventRecord(
                session_id="s1",
                turn_index=0,
                role="player",
                message="Q1",
                tags=["question"],
            ),
            SessionEventRecord(
                session_id="s1",
                turn_index=1,
                role="dm",
                message="Hint",
                tags=["hint"],
            ),
            SessionEventRecord(
                session_id="s1",
                turn_index=2,
                role="player",
                message="Q2",
                tags=["question"],
            ),
        ]

        summary = default_summarizer(events)

        assert "3 total events" in summary
        assert "Questions asked: 2" in summary
        assert "Hints used: 1" in summary

    def test_with_final_verdict(self):
        events = [
            SessionEventRecord(
                session_id="s1",
                turn_index=0,
                role="player",
                message="The answer is X",
                tags=["hypothesis", "final_verdict"],
            ),
        ]

        summary = default_summarizer(events)
        assert "Final verdict:" in summary
