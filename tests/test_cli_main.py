"""Integration tests for CLI main entry point."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import argparse

import pytest

from game.cli.main import create_parser, InteractiveCLI, async_main
from game.cli.app import GameCLIApp
from game.cli.formatters import TextFormatter
from game.domain.entities import GameSession, GameState, PuzzleSummary, SessionConfig


@pytest.fixture
def sample_puzzle_summary():
    return PuzzleSummary(
        id="test_puzzle",
        title="Test Puzzle",
        description="A test puzzle",
        difficulty="easy",
        tags=["classic"],
        language="en",
    )


@pytest.fixture
def sample_session():
    return GameSession(
        session_id="test-session-123",
        puzzle_id="test_puzzle",
        player_ids=["test_player"],
        kb_id="game_test_puzzle",
        config=SessionConfig(),
        state=GameState.IN_PROGRESS,
    )


class TestParser:
    def test_create_parser(self):
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "turtle-soup"

    def test_list_puzzles_command(self):
        parser = create_parser()
        args = parser.parse_args(["list-puzzles"])
        assert args.command == "list-puzzles"

    def test_start_session_command(self):
        parser = create_parser()
        args = parser.parse_args(["start-session", "--puzzle", "puzzle1", "--player", "alice"])
        assert args.command == "start-session"
        assert args.puzzle == "puzzle1"
        assert args.player == "alice"

    def test_play_command_with_puzzle(self):
        parser = create_parser()
        args = parser.parse_args(["play", "--puzzle", "puzzle1"])
        assert args.command == "play"
        assert args.puzzle == "puzzle1"
        assert args.session is None

    def test_play_command_with_session(self):
        parser = create_parser()
        args = parser.parse_args(["play", "--session", "sess-123"])
        assert args.command == "play"
        assert args.session == "sess-123"
        assert args.puzzle is None

    def test_status_command(self):
        parser = create_parser()
        args = parser.parse_args(["status", "--session", "sess-123"])
        assert args.command == "status"
        assert args.session == "sess-123"

    def test_sessions_command(self):
        parser = create_parser()
        args = parser.parse_args(["sessions", "--state", "in_progress"])
        assert args.command == "sessions"
        assert args.state == "in_progress"

    def test_json_flag(self):
        parser = create_parser()
        args = parser.parse_args(["--json", "list-puzzles"])
        assert args.json is True

    def test_verbose_flag(self):
        parser = create_parser()
        args = parser.parse_args(["-v", "list-puzzles"])
        assert args.verbose is True


class TestInteractiveCLI:
    @pytest.fixture
    def mock_app(self, sample_puzzle_summary, sample_session):
        app = MagicMock(spec=GameCLIApp)
        app.list_puzzles.return_value = [sample_puzzle_summary]
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
        app.create_session = AsyncMock(return_value=sample_session)
        app.get_session.return_value = sample_session
        return app

    @pytest.fixture
    def cli(self, mock_app):
        formatter = TextFormatter()
        return InteractiveCLI(mock_app, formatter)

    @pytest.mark.asyncio
    async def test_run_list_puzzles(self, cli, mock_app):
        result = await cli.run_list_puzzles()
        assert result == 0
        mock_app.list_puzzles.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_start_session(self, cli, mock_app, sample_session):
        result = await cli.run_start_session("test_puzzle", "test_player")
        assert result == 0
        mock_app.create_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_status(self, cli, mock_app):
        result = await cli.run_status("test-session-123")
        assert result == 0
        mock_app.get_session_status.assert_called_once_with("test-session-123")

    @pytest.mark.asyncio
    async def test_run_sessions(self, cli, mock_app):
        result = await cli.run_sessions()
        assert result == 0
        mock_app.list_sessions.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_sessions_with_filter(self, cli, mock_app):
        result = await cli.run_sessions(state="in_progress")
        assert result == 0


class TestAsyncMain:
    @pytest.mark.asyncio
    async def test_async_main_list_puzzles(self, sample_puzzle_summary):
        args = argparse.Namespace(
            command="list-puzzles",
            json=False,
            verbose=False,
        )

        with patch("game.cli.main.GameCLIApp") as MockApp:
            mock_app = MockApp.return_value
            mock_app.list_puzzles.return_value = [sample_puzzle_summary]
            mock_app.close = AsyncMock()

            result = await async_main(args)

            assert result == 0
            mock_app.list_puzzles.assert_called_once()
            mock_app.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_main_start_session(self, sample_session):
        args = argparse.Namespace(
            command="start-session",
            puzzle="test_puzzle",
            player="test_player",
            json=False,
            verbose=False,
        )

        with patch("game.cli.main.GameCLIApp") as MockApp:
            mock_app = MockApp.return_value
            mock_app.create_session = AsyncMock(return_value=sample_session)
            mock_app.close = AsyncMock()

            result = await async_main(args)

            assert result == 0
            mock_app.create_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_main_status(self, sample_session):
        args = argparse.Namespace(
            command="status",
            session="test-session-123",
            json=False,
            verbose=False,
        )

        with patch("game.cli.main.GameCLIApp") as MockApp:
            mock_app = MockApp.return_value
            mock_app.get_session_status.return_value = {
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
            mock_app.close = AsyncMock()

            result = await async_main(args)

            assert result == 0

    @pytest.mark.asyncio
    async def test_async_main_sessions(self, sample_session):
        args = argparse.Namespace(
            command="sessions",
            state=None,
            puzzle=None,
            player=None,
            json=False,
            verbose=False,
        )

        with patch("game.cli.main.GameCLIApp") as MockApp:
            mock_app = MockApp.return_value
            mock_app.list_sessions.return_value = [sample_session]
            mock_app.close = AsyncMock()

            result = await async_main(args)

            assert result == 0

    @pytest.mark.asyncio
    async def test_async_main_unknown_command(self):
        args = argparse.Namespace(
            command="unknown",
            json=False,
            verbose=False,
        )

        with patch("game.cli.main.GameCLIApp") as MockApp:
            mock_app = MockApp.return_value
            mock_app.close = AsyncMock()

            result = await async_main(args)

            assert result == 1
