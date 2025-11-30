"""Tests for memory entities."""

import pytest
from datetime import datetime

from game.memory.entities import (
    EventTag,
    GlobalMemoryRecord,
    MemoryDocument,
    MemorySummaryType,
    PlayerMemoryRecord,
    SessionEventRecord,
)


class TestSessionEventRecord:
    def test_create_basic_event(self):
        event = SessionEventRecord(
            session_id="test-session",
            turn_index=0,
            role="player",
            message="Is the man alive?",
        )

        assert event.session_id == "test-session"
        assert event.turn_index == 0
        assert event.role == "player"
        assert event.message == "Is the man alive?"
        assert event.event_type == "message"
        assert event.tags == []

    def test_event_with_tags(self):
        event = SessionEventRecord(
            session_id="test-session",
            turn_index=1,
            role="dm",
            message="No",
            tags=["answer", "yes_no"],
        )

        assert "answer" in event.tags
        assert event.has_tag("answer")
        assert event.has_tag(EventTag.ANSWER)

    def test_add_tag(self):
        event = SessionEventRecord(
            session_id="test-session",
            turn_index=0,
            role="player",
            message="Test",
        )

        event.add_tag(EventTag.QUESTION)
        assert event.has_tag(EventTag.QUESTION)

        event.add_tag("question")
        assert len([t for t in event.tags if t == "question"]) == 1

    def test_event_with_metadata(self):
        event = SessionEventRecord(
            session_id="test-session",
            turn_index=0,
            role="dm",
            message="Welcome to the puzzle",
            metadata={"puzzle_id": "puzzle1", "phase": "intro"},
        )

        assert event.metadata["puzzle_id"] == "puzzle1"
        assert event.metadata["phase"] == "intro"


class TestPlayerMemoryRecord:
    def test_create_player_memory(self):
        record = PlayerMemoryRecord(
            player_id="player1",
            summary_type=MemorySummaryType.STYLE_PROFILE,
            content="Player asks direct yes/no questions",
        )

        assert record.player_id == "player1"
        assert record.summary_type == MemorySummaryType.STYLE_PROFILE
        assert record.content == "Player asks direct yes/no questions"
        assert record.version == 1

    def test_update_content(self):
        record = PlayerMemoryRecord(
            player_id="player1",
            summary_type=MemorySummaryType.PERFORMANCE_SUMMARY,
            content="Initial content",
        )

        original_updated = record.updated_at
        record.update_content("Updated content")

        assert record.content == "Updated content"
        assert record.version == 2
        assert record.updated_at >= original_updated


class TestGlobalMemoryRecord:
    def test_create_global_memory(self):
        record = GlobalMemoryRecord(
            record_id="hint_strategy_1",
            summary_type=MemorySummaryType.HINT_STRATEGY,
            content="For new players, keep hints vague",
        )

        assert record.record_id == "hint_strategy_1"
        assert record.summary_type == MemorySummaryType.HINT_STRATEGY
        assert record.puzzle_id is None

    def test_global_memory_with_puzzle(self):
        record = GlobalMemoryRecord(
            record_id="puzzle1_stats",
            summary_type=MemorySummaryType.PUZZLE_STATS,
            content="Average 15 questions, 40% success rate",
            puzzle_id="puzzle1",
        )

        assert record.puzzle_id == "puzzle1"


class TestMemoryDocument:
    def test_create_from_session_event(self):
        event = SessionEventRecord(
            session_id="session1",
            turn_index=5,
            role="player",
            message="Is it related to death?",
            tags=["question"],
        )

        doc = MemoryDocument.from_session_event(event)

        assert doc.namespace == "session:session1"
        assert doc.key == "event_5"
        assert doc.value["turn_index"] == 5
        assert doc.value["message"] == "Is it related to death?"
        assert "question" in doc.metadata["tags"]

    def test_create_from_player_memory(self):
        record = PlayerMemoryRecord(
            player_id="player1",
            summary_type=MemorySummaryType.STYLE_PROFILE,
            content="Analytical player",
            version=3,
        )

        doc = MemoryDocument.from_player_memory(record)

        assert doc.namespace == "player:player1"
        assert doc.key == "style_profile"
        assert doc.value["content"] == "Analytical player"
        assert doc.value["version"] == 3

    def test_create_from_global_memory(self):
        record = GlobalMemoryRecord(
            record_id="global_hint_strategy",
            summary_type=MemorySummaryType.HINT_STRATEGY,
            content="General hint strategy",
        )

        doc = MemoryDocument.from_global_memory(record)

        assert doc.namespace == "global"
        assert doc.key == "global_hint_strategy"
        assert doc.value["content"] == "General hint strategy"


class TestEventTag:
    def test_event_tag_values(self):
        assert EventTag.QUESTION.value == "question"
        assert EventTag.ANSWER.value == "answer"
        assert EventTag.HINT.value == "hint"
        assert EventTag.HYPOTHESIS.value == "hypothesis"
        assert EventTag.FINAL_VERDICT.value == "final_verdict"


class TestMemorySummaryType:
    def test_summary_type_values(self):
        assert MemorySummaryType.STYLE_PROFILE.value == "style_profile"
        assert MemorySummaryType.PERFORMANCE_SUMMARY.value == "performance_summary"
        assert MemorySummaryType.SESSION_SUMMARY.value == "session_summary"
        assert MemorySummaryType.PUZZLE_STATS.value == "puzzle_stats"
