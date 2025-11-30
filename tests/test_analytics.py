"""Tests for Analytics Service."""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from game.memory.analytics import (
    AnalyticsService,
    PuzzleAnalytics,
    PlayerAnalytics,
    GlobalAnalytics,
    SessionEventAggregator,
)
from game.memory.entities import EventTag, SessionEventRecord
from game.memory.manager import MemoryManager
from game.memory.file_store import FileMemoryStore


@pytest.fixture
def temp_dir():
    dir_path = Path(tempfile.mkdtemp())
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture
def analytics_service(temp_dir):
    return AnalyticsService(export_dir=temp_dir / "analytics")


@pytest.fixture
def memory_manager(temp_dir):
    store = FileMemoryStore(base_dir=temp_dir / "memory")
    return MemoryManager(store=store)


class TestPuzzleAnalytics:
    def test_to_dict(self):
        analytics = PuzzleAnalytics(
            puzzle_id="puzzle1",
            total_sessions=10,
            total_questions=50,
            success_count=7,
            avg_questions_per_session=5.0,
            success_rate=0.7,
        )

        data = analytics.to_dict()

        assert data["puzzle_id"] == "puzzle1"
        assert data["total_sessions"] == 10
        assert data["success_rate"] == 0.7

    def test_default_values(self):
        analytics = PuzzleAnalytics(puzzle_id="p1")

        assert analytics.total_sessions == 0
        assert analytics.success_rate == 0.0


class TestPlayerAnalytics:
    def test_to_dict(self):
        analytics = PlayerAnalytics(
            player_id="player1",
            total_sessions=5,
            puzzles_solved=3,
            success_rate=0.6,
        )

        data = analytics.to_dict()

        assert data["player_id"] == "player1"
        assert data["success_rate"] == 0.6


class TestGlobalAnalytics:
    def test_to_dict(self):
        analytics = GlobalAnalytics(
            total_sessions=100,
            total_players=20,
            overall_success_rate=0.65,
        )

        data = analytics.to_dict()

        assert data["total_sessions"] == 100
        assert data["total_players"] == 20
        assert "generated_at" in data


class TestAnalyticsService:
    def test_categorize_session_length(self, analytics_service):
        assert analytics_service._categorize_session_length(3) == "short"
        assert analytics_service._categorize_session_length(10) == "medium"
        assert analytics_service._categorize_session_length(20) == "long"
        assert analytics_service._categorize_session_length(50) == "very_long"

    def test_analyze_puzzle_no_data(self, analytics_service):
        result = analytics_service.analyze_puzzle("puzzle1")

        assert result.puzzle_id == "puzzle1"
        assert result.total_sessions == 0

    def test_analyze_puzzle_with_data(self, analytics_service):
        sessions_data = [
            {"question_count": 5, "hint_count": 1, "success": True},
            {"question_count": 8, "hint_count": 2, "success": True},
            {"question_count": 12, "hint_count": 0, "success": False},
        ]

        result = analytics_service.analyze_puzzle("puzzle1", sessions_data)

        assert result.total_sessions == 3
        assert result.total_questions == 25
        assert result.success_count == 2
        assert result.success_rate == pytest.approx(0.667, rel=0.01)

    def test_analyze_puzzle_session_lengths(self, analytics_service):
        sessions_data = [
            {"question_count": 3, "success": True},
            {"question_count": 4, "success": True},
            {"question_count": 10, "success": False},
        ]

        result = analytics_service.analyze_puzzle("p1", sessions_data)

        assert result.common_session_lengths["short"] == 2
        assert result.common_session_lengths["medium"] == 1

    def test_analyze_player_no_data(self, analytics_service):
        result = analytics_service.analyze_player("player1")

        assert result.player_id == "player1"
        assert result.total_sessions == 0

    def test_analyze_player_with_data(self, analytics_service):
        sessions_data = [
            {"puzzle_id": "p1", "question_count": 5, "hint_count": 1, "success": True},
            {"puzzle_id": "p2", "question_count": 8, "hint_count": 0, "success": True},
            {"puzzle_id": "p1", "question_count": 6, "hint_count": 2, "success": False},
        ]

        result = analytics_service.analyze_player("player1", sessions_data)

        assert result.total_sessions == 3
        assert result.total_puzzles_attempted == 2
        assert result.puzzles_solved == 2
        assert result.total_questions_asked == 19

    def test_generate_global_analytics(self, analytics_service):
        puzzle_analytics = [
            PuzzleAnalytics(
                puzzle_id="p1",
                total_sessions=10,
                total_questions=50,
                success_count=7,
                avg_hints_used=1.5,
            ),
            PuzzleAnalytics(
                puzzle_id="p2",
                total_sessions=5,
                total_questions=20,
                success_count=3,
                avg_hints_used=2.0,
            ),
        ]
        player_analytics = [
            PlayerAnalytics(player_id="player1"),
            PlayerAnalytics(player_id="player2"),
        ]

        result = analytics_service.generate_global_analytics(
            puzzle_analytics=puzzle_analytics,
            player_analytics=player_analytics,
        )

        assert result.total_sessions == 15
        assert result.total_players == 2
        assert result.total_puzzles == 2
        assert result.overall_success_rate == pytest.approx(0.667, rel=0.01)

    def test_export_to_json(self, analytics_service, temp_dir):
        data = {"test": "value", "count": 42}

        filepath = analytics_service.export_to_json(data, "test_export")

        assert filepath.exists()
        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded["test"] == "value"

    def test_export_to_csv(self, analytics_service, temp_dir):
        data = [
            {"id": 1, "name": "test1", "value": 10},
            {"id": 2, "name": "test2", "value": 20},
        ]

        filepath = analytics_service.export_to_csv(data, "test_export")

        assert filepath.exists()
        content = filepath.read_text()
        assert "id,name,value" in content
        assert "test1" in content

    def test_export_to_csv_empty(self, analytics_service, temp_dir):
        filepath = analytics_service.export_to_csv([], "empty_export")

        assert filepath.exists()
        assert filepath.read_text() == ""

    def test_generate_full_report_json(self, analytics_service, temp_dir):
        files = analytics_service.generate_full_report(
            puzzle_ids=["p1", "p2"],
            player_ids=["player1"],
            export_format="json",
        )

        assert "report" in files
        assert files["report"].exists()

    def test_generate_full_report_csv(self, analytics_service, temp_dir):
        files = analytics_service.generate_full_report(
            puzzle_ids=["p1"],
            player_ids=["player1"],
            export_format="csv",
        )

        assert "global" in files
        assert "puzzles" in files
        assert "players" in files


class TestSessionEventAggregator:
    def test_aggregate_session_events_no_manager(self):
        aggregator = SessionEventAggregator()
        result = aggregator.aggregate_session_events("session1")

        assert result == {}

    def test_aggregate_session_events(self, memory_manager):
        events = [
            SessionEventRecord(
                session_id="s1",
                turn_index=0,
                role="player",
                message="Question 1",
                tags=["question"],
            ),
            SessionEventRecord(
                session_id="s1",
                turn_index=1,
                role="dm",
                message="Yes",
                tags=["answer"],
            ),
            SessionEventRecord(
                session_id="s1",
                turn_index=2,
                role="player",
                message="Question 2",
                tags=["question"],
            ),
            SessionEventRecord(
                session_id="s1",
                turn_index=3,
                role="dm",
                message="Correct!",
                tags=["final_verdict"],
            ),
        ]

        for event in events:
            memory_manager.append_session_event("s1", event)

        aggregator = SessionEventAggregator(memory_manager=memory_manager)
        result = aggregator.aggregate_session_events("s1")

        assert result["session_id"] == "s1"
        assert result["event_count"] == 4
        assert result["question_count"] == 2
        assert result["success"] is True

    def test_aggregate_from_event_logs(self, temp_dir):
        events_dir = temp_dir / "events"
        events_dir.mkdir()

        events = [
            {"role": "player", "message": "Q1", "tags": ["question"]},
            {"role": "dm", "message": "Yes", "tags": ["answer"]},
            {"role": "player", "message": "Solution", "tags": ["hypothesis", "final_verdict"]},
        ]

        with open(events_dir / "session1.jsonl", "w") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")

        aggregator = SessionEventAggregator()
        results = aggregator.aggregate_from_event_logs(events_dir)

        assert len(results) == 1
        assert results[0]["session_id"] == "session1"
        assert results[0]["question_count"] == 1
        assert results[0]["hypothesis_count"] == 1

    def test_aggregate_from_event_logs_empty_dir(self, temp_dir):
        events_dir = temp_dir / "empty_events"
        events_dir.mkdir()

        aggregator = SessionEventAggregator()
        results = aggregator.aggregate_from_event_logs(events_dir)

        assert results == []

    def test_aggregate_from_event_logs_nonexistent(self, temp_dir):
        aggregator = SessionEventAggregator()
        results = aggregator.aggregate_from_event_logs(temp_dir / "nonexistent")

        assert results == []
