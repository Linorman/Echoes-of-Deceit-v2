"""Tests for CLI commands and application."""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from game.cli.app import GameCLIApp
from game.cli.commands import (
    CommandResult,
    list_puzzles,
    start_session,
    get_session_status,
    list_sessions,
    play_session,
    resume_session,
)
from game.cli.formatters import TextFormatter, JsonFormatter, get_formatter
from game.domain.entities import (
    GameSession,
    GameState,
    Puzzle,
    PuzzleSummary,
    PuzzleConstraints,
    SessionConfig,
)
from game.session_runner import GameResponse


@pytest.fixture
def sample_puzzle():
    return Puzzle(
        id="test_puzzle",
        title="Test Puzzle",
        description="A test puzzle",
        puzzle_statement="A man walks into a bar...",
        answer="He had hiccups.",
        hints=["Think about why", "The gun wasn't harmful"],
        constraints=PuzzleConstraints(max_hints=3),
        tags=["classic", "test"],
    )


@pytest.fixture
def sample_puzzle_summary():
    return PuzzleSummary(
        id="test_puzzle",
        title="Test Puzzle",
        description="A test puzzle",
        difficulty="easy",
        tags=["classic", "test"],
        language="en",
    )


@pytest.fixture
def sample_session(sample_puzzle):
    return GameSession(
        session_id="test-session-123",
        puzzle_id=sample_puzzle.id,
        player_ids=["test_player"],
        kb_id="game_test_puzzle",
        config=SessionConfig(),
        state=GameState.IN_PROGRESS,
    )


@pytest.fixture
def mock_app(sample_puzzle, sample_puzzle_summary, sample_session):
    app = MagicMock(spec=GameCLIApp)
    app.list_puzzles.return_value = [sample_puzzle_summary]
    app.create_session = AsyncMock(return_value=sample_session)
    app.get_session.return_value = sample_session
    app.list_sessions.return_value = [sample_session]
    app.get_session_status.return_value = {
        "session_id": sample_session.session_id,
        "puzzle_id": sample_session.puzzle_id,
        "puzzle_title": "Test Puzzle",
        "state": sample_session.state.value,
        "turn_count": 0,
        "question_count": 0,
        "hint_count": 0,
        "max_hints": 3,
        "score": None,
        "created_at": sample_session.created_at.isoformat(),
        "updated_at": sample_session.updated_at.isoformat(),
        "completed_at": None,
    }

    mock_runner = MagicMock()
    mock_runner.session = sample_session
    mock_runner.is_active = False
    mock_runner.start_game.return_value = GameResponse(message="Game started!")
    app.create_runner.return_value = mock_runner

    return app


class TestListPuzzlesCommand:
    def test_list_puzzles_success(self, mock_app, sample_puzzle_summary):
        result = list_puzzles(mock_app)

        assert result.success is True
        assert "Found 1 puzzle(s)" in result.message
        assert len(result.data["puzzles"]) == 1
        assert result.data["puzzles"][0]["id"] == sample_puzzle_summary.id

    def test_list_puzzles_empty(self, mock_app):
        mock_app.list_puzzles.return_value = []
        result = list_puzzles(mock_app)

        assert result.success is True
        assert "No puzzles found" in result.message
        assert result.data["puzzles"] == []

    def test_list_puzzles_error(self, mock_app):
        mock_app.list_puzzles.side_effect = Exception("Database error")
        result = list_puzzles(mock_app)

        assert result.success is False
        assert "Failed to list puzzles" in result.message
        assert result.error == "Database error"


class TestStartSessionCommand:
    @pytest.mark.asyncio
    async def test_start_session_success(self, mock_app, sample_session):
        result = await start_session(mock_app, "test_puzzle", "test_player")

        assert result.success is True
        assert sample_session.session_id in result.message
        assert result.data["puzzle_id"] == "test_puzzle"
        assert result.data["player"] == "test_player"
        mock_app.create_session.assert_called_once_with("test_puzzle", "test_player", None)

    @pytest.mark.asyncio
    async def test_start_session_puzzle_not_found(self, mock_app):
        mock_app.create_session.side_effect = ValueError("Puzzle not found: invalid_puzzle")
        result = await start_session(mock_app, "invalid_puzzle", "test_player")

        assert result.success is False
        assert "Puzzle not found" in result.error

    @pytest.mark.asyncio
    async def test_start_session_with_options(self, mock_app, sample_session):
        options = {"difficulty": "hard"}
        result = await start_session(mock_app, "test_puzzle", "test_player", options)

        assert result.success is True
        mock_app.create_session.assert_called_once_with("test_puzzle", "test_player", options)


class TestGetSessionStatusCommand:
    def test_get_status_success(self, mock_app):
        result = get_session_status(mock_app, "test-session-123")

        assert result.success is True
        assert result.data["session_id"] == "test-session-123"
        assert result.data["puzzle_title"] == "Test Puzzle"
        assert result.data["state"] == "in_progress"

    def test_get_status_not_found(self, mock_app):
        mock_app.get_session_status.side_effect = ValueError("Session not found")
        result = get_session_status(mock_app, "invalid-session")

        assert result.success is False
        assert "Session not found" in result.message


class TestListSessionsCommand:
    def test_list_sessions_success(self, mock_app, sample_session):
        result = list_sessions(mock_app)

        assert result.success is True
        assert "Found 1 session(s)" in result.message
        assert len(result.data["sessions"]) == 1

    def test_list_sessions_with_state_filter(self, mock_app):
        result = list_sessions(mock_app, state_filter="in_progress")

        assert result.success is True
        mock_app.list_sessions.assert_called_once()
        call_args = mock_app.list_sessions.call_args
        assert call_args.kwargs["state_filter"] == GameState.IN_PROGRESS

    def test_list_sessions_invalid_state(self, mock_app):
        result = list_sessions(mock_app, state_filter="invalid_state")

        assert result.success is False
        assert "Invalid state" in result.message

    def test_list_sessions_empty(self, mock_app):
        mock_app.list_sessions.return_value = []
        result = list_sessions(mock_app)

        assert result.success is True
        assert result.data["sessions"] == []


class TestPlaySessionCommand:
    @pytest.mark.asyncio
    async def test_play_new_session(self, mock_app, sample_session):
        sample_session.state = GameState.LOBBY
        inputs = iter([""])
        outputs: List[str] = []

        mock_runner = mock_app.create_runner.return_value
        mock_runner.is_active = False
        mock_runner.session = sample_session
        mock_runner.start_game.return_value = GameResponse(message="Welcome!", game_over=False)

        result = await play_session(
            mock_app,
            sample_session,
            lambda: next(inputs),
            lambda msg: outputs.append(msg),
        )

        assert result.success is True
        mock_runner.start_game.assert_called_once()

    @pytest.mark.asyncio
    async def test_play_session_with_interrupt(self, mock_app, sample_session):
        sample_session.state = GameState.IN_PROGRESS
        mock_runner = mock_app.create_runner.return_value
        mock_runner.is_active = True

        call_count = 0
        def raise_interrupt():
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise KeyboardInterrupt()
            return ""

        result = await play_session(
            mock_app,
            sample_session,
            raise_interrupt,
            lambda msg: None,
        )

        assert result.success is True
        assert result.data is not None
        assert result.data.get("interrupted") is True


class TestResumeSessionCommand:
    @pytest.mark.asyncio
    async def test_resume_success(self, mock_app, sample_session):
        sample_session.state = GameState.IN_PROGRESS
        mock_runner = mock_app.create_runner.return_value
        mock_runner.is_active = False

        outputs: List[str] = []

        result = await resume_session(
            mock_app,
            sample_session.session_id,
            lambda: "",
            lambda msg: outputs.append(msg),
        )

        assert result.success is True
        assert any("resumed" in o.lower() for o in outputs)

    @pytest.mark.asyncio
    async def test_resume_not_found(self, mock_app):
        mock_app.get_session.side_effect = ValueError("Not found")

        result = await resume_session(
            mock_app,
            "invalid-id",
            lambda: "",
            lambda msg: None,
        )

        assert result.success is False
        assert "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_resume_completed_session(self, mock_app, sample_session):
        sample_session.state = GameState.COMPLETED
        mock_app.get_session.return_value = sample_session

        result = await resume_session(
            mock_app,
            sample_session.session_id,
            lambda: "",
            lambda msg: None,
        )

        assert result.success is False
        assert "not active" in result.message.lower()


class TestTextFormatter:
    def test_format_puzzles(self, sample_puzzle_summary):
        formatter = TextFormatter()
        puzzles = [{
            "id": sample_puzzle_summary.id,
            "title": sample_puzzle_summary.title,
            "description": sample_puzzle_summary.description,
            "difficulty": sample_puzzle_summary.difficulty,
            "tags": sample_puzzle_summary.tags,
        }]

        output = formatter.format_puzzles(puzzles)

        assert "Available Puzzles" in output
        assert sample_puzzle_summary.id in output
        assert sample_puzzle_summary.title in output

    def test_format_puzzles_empty(self):
        formatter = TextFormatter()
        output = formatter.format_puzzles([])

        assert "No puzzles found" in output

    def test_format_sessions(self, sample_session):
        formatter = TextFormatter()
        sessions = [{
            "session_id": sample_session.session_id,
            "puzzle_id": sample_session.puzzle_id,
            "state": sample_session.state.value,
            "turn_count": 5,
            "hint_count": 1,
            "created_at": sample_session.created_at.isoformat(),
        }]

        output = formatter.format_sessions(sessions)

        assert "Game Sessions" in output
        assert sample_session.session_id in output
        assert sample_session.puzzle_id in output

    def test_format_status(self):
        formatter = TextFormatter()
        status = {
            "session_id": "sess-123",
            "puzzle_id": "puzzle-1",
            "puzzle_title": "Test Puzzle",
            "state": "in_progress",
            "question_count": 10,
            "hint_count": 2,
            "max_hints": 5,
            "score": None,
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T01:00:00",
            "completed_at": None,
        }

        output = formatter.format_status(status)

        assert "Session Status" in output
        assert "sess-123" in output
        assert "Test Puzzle" in output
        assert "Questions asked: 10" in output
        assert "Hints used: 2/5" in output

    def test_format_error(self):
        formatter = TextFormatter()
        output = formatter.format_error("Something went wrong", "Details here")

        assert "Error:" in output
        assert "Something went wrong" in output
        assert "Details here" in output


class TestJsonFormatter:
    def test_format_puzzles(self, sample_puzzle_summary):
        formatter = JsonFormatter()
        puzzles = [{
            "id": sample_puzzle_summary.id,
            "title": sample_puzzle_summary.title,
        }]

        output = formatter.format_puzzles(puzzles)

        import json
        data = json.loads(output)
        assert "puzzles" in data
        assert len(data["puzzles"]) == 1
        assert data["puzzles"][0]["id"] == sample_puzzle_summary.id

    def test_format_result(self):
        formatter = JsonFormatter()
        result = CommandResult(
            success=True,
            message="Operation completed",
            data={"key": "value"},
        )

        output = formatter.format_result(result)

        import json
        data = json.loads(output)
        assert data["success"] is True
        assert data["message"] == "Operation completed"
        assert data["data"]["key"] == "value"


class TestGetFormatter:
    def test_get_text_formatter(self):
        formatter = get_formatter(json_mode=False)
        assert isinstance(formatter, TextFormatter)

    def test_get_json_formatter(self):
        formatter = get_formatter(json_mode=True)
        assert isinstance(formatter, JsonFormatter)


class TestCommandResult:
    def test_success_result(self):
        result = CommandResult(
            success=True,
            message="Operation succeeded",
            data={"key": "value"},
        )

        assert result.success is True
        assert result.message == "Operation succeeded"
        assert result.data == {"key": "value"}
        assert result.error is None

    def test_failure_result(self):
        result = CommandResult(
            success=False,
            message="Operation failed",
            error="Some error",
        )

        assert result.success is False
        assert result.error == "Some error"
