"""Tests for domain entities."""

import uuid
from datetime import datetime

import pytest

from game.domain.entities import (
    AgentRole,
    Game,
    GameSession,
    GameState,
    PlayerProfile,
    Puzzle,
    PuzzleConstraints,
    PuzzleSummary,
    SessionConfig,
    SessionEvent,
)


class TestAgentRole:
    def test_agent_roles(self):
        assert AgentRole.DM.value == "dm"
        assert AgentRole.PLAYER.value == "player"
        assert AgentRole.OBSERVER.value == "observer"
        assert AgentRole.HINT_MASTER.value == "hint_master"


class TestGameState:
    def test_game_states(self):
        assert GameState.LOBBY.value == "lobby"
        assert GameState.IN_PROGRESS.value == "in_progress"
        assert GameState.COMPLETED.value == "completed"
        assert GameState.ABORTED.value == "aborted"


class TestPuzzleConstraints:
    def test_default_constraints(self):
        constraints = PuzzleConstraints()
        assert constraints.max_questions is None
        assert constraints.max_hints == 5
        assert "yes_no" in constraints.allowed_question_types
        assert constraints.time_limit_minutes is None

    def test_custom_constraints(self):
        constraints = PuzzleConstraints(
            max_questions=50,
            max_hints=3,
            allowed_question_types=["yes_no"],
            time_limit_minutes=30,
        )
        assert constraints.max_questions == 50
        assert constraints.max_hints == 3
        assert constraints.time_limit_minutes == 30


class TestPuzzle:
    def test_puzzle_creation(self):
        puzzle = Puzzle(
            id="test_puzzle",
            title="Test Puzzle",
            puzzle_statement="A man walks into a bar...",
            answer="He was a ghost",
        )
        assert puzzle.id == "test_puzzle"
        assert puzzle.title == "Test Puzzle"
        assert puzzle.puzzle_statement == "A man walks into a bar..."
        assert puzzle.answer == "He was a ghost"
        assert puzzle.language == "en"

    def test_puzzle_with_hints(self):
        puzzle = Puzzle(
            id="test",
            puzzle_statement="Test",
            answer="Answer",
            hints=["Hint 1", "Hint 2"],
        )
        assert puzzle.has_hints
        assert len(puzzle.hints) == 2

    def test_puzzle_without_hints(self):
        puzzle = Puzzle(
            id="test",
            puzzle_statement="Test",
            answer="Answer",
        )
        assert not puzzle.has_hints

    def test_puzzle_serialization(self):
        puzzle = Puzzle(
            id="test",
            title="Test",
            puzzle_statement="Statement",
            answer="Answer",
            tags=["tag1", "tag2"],
        )
        data = puzzle.model_dump()
        assert data["id"] == "test"
        assert data["tags"] == ["tag1", "tag2"]

        restored = Puzzle.model_validate(data)
        assert restored.id == puzzle.id
        assert restored.tags == puzzle.tags


class TestPuzzleSummary:
    def test_puzzle_summary(self):
        summary = PuzzleSummary(
            id="test",
            title="Test Puzzle",
            difficulty="medium",
            tags=["mystery"],
        )
        assert summary.id == "test"
        assert summary.title == "Test Puzzle"
        assert summary.difficulty == "medium"


class TestGame:
    def test_game_creation(self):
        game = Game(
            id="game_1",
            name="Test Game",
            puzzle_ids=["puzzle1", "puzzle2"],
        )
        assert game.id == "game_1"
        assert len(game.puzzle_ids) == 2
        assert game.game_type == "situation_puzzle"


class TestPlayerProfile:
    def test_profile_creation(self):
        profile = PlayerProfile(
            player_id="player_1",
            display_name="Test Player",
        )
        assert profile.player_id == "player_1"
        assert profile.display_name == "Test Player"
        assert profile.preferences == {}

    def test_profile_preferences(self):
        profile = PlayerProfile(
            player_id="player_1",
            display_name="Test Player",
            preferences={"difficulty": "hard", "language": "en"},
        )
        assert profile.preferences["difficulty"] == "hard"

    def test_update_preference(self):
        profile = PlayerProfile(
            player_id="player_1",
            display_name="Test Player",
        )
        original_updated = profile.updated_at
        profile.update_preference("theme", "dark")
        assert profile.preferences["theme"] == "dark"
        assert profile.updated_at >= original_updated

    def test_profile_serialization(self):
        profile = PlayerProfile(
            player_id="player_1",
            display_name="Test Player",
            preferences={"key": "value"},
        )
        data = profile.model_dump(mode="json")
        assert data["player_id"] == "player_1"

        restored = PlayerProfile.model_validate(data)
        assert restored.player_id == profile.player_id


class TestSessionEvent:
    def test_event_creation(self):
        event = SessionEvent(
            session_id="session_1",
            turn_index=0,
            role=AgentRole.PLAYER,
            message="Is it a person?",
            tags=["question"],
        )
        assert event.session_id == "session_1"
        assert event.turn_index == 0
        assert event.role == AgentRole.PLAYER
        assert "question" in event.tags

    def test_event_serialization(self):
        event = SessionEvent(
            session_id="session_1",
            turn_index=0,
            role=AgentRole.DM,
            message="Welcome to the game!",
        )
        data = event.model_dump(mode="json")
        assert data["role"] == "dm"

        restored = SessionEvent.model_validate(data)
        assert restored.role == AgentRole.DM


class TestSessionConfig:
    def test_default_config(self):
        config = SessionConfig()
        assert config.llm_provider == "ollama"
        assert config.rag_provider == "lightrag"
        assert config.max_turns == 100
        assert config.hint_limit == 5

    def test_custom_config(self):
        config = SessionConfig(
            llm_provider="api",
            llm_model="gpt-4",
            max_turns=50,
        )
        assert config.llm_provider == "api"
        assert config.llm_model == "gpt-4"
        assert config.max_turns == 50


class TestGameSession:
    def test_session_creation(self):
        session = GameSession(puzzle_id="puzzle_1")
        assert session.puzzle_id == "puzzle_1"
        assert session.state == GameState.LOBBY
        assert session.turn_count == 0
        assert session.session_id is not None

    def test_session_with_players(self):
        session = GameSession(
            puzzle_id="puzzle_1",
            player_ids=["player_1", "player_2"],
        )
        assert len(session.player_ids) == 2

    def test_session_start(self):
        session = GameSession(puzzle_id="puzzle_1")
        assert session.state == GameState.LOBBY

        session.start()
        assert session.state == GameState.IN_PROGRESS
        assert session.is_active

    def test_session_start_invalid_state(self):
        session = GameSession(puzzle_id="puzzle_1")
        session.start()

        with pytest.raises(ValueError):
            session.start()

    def test_session_complete(self):
        session = GameSession(puzzle_id="puzzle_1")
        session.start()
        session.complete(score=100)

        assert session.state == GameState.COMPLETED
        assert session.is_completed
        assert session.score == 100
        assert session.completed_at is not None

    def test_session_complete_invalid_state(self):
        session = GameSession(puzzle_id="puzzle_1")

        with pytest.raises(ValueError):
            session.complete()

    def test_session_abort(self):
        session = GameSession(puzzle_id="puzzle_1")
        session.start()
        session.abort()

        assert session.state == GameState.ABORTED

    def test_session_abort_from_lobby(self):
        session = GameSession(puzzle_id="puzzle_1")
        session.abort()
        assert session.state == GameState.ABORTED

    def test_session_abort_invalid_state(self):
        session = GameSession(puzzle_id="puzzle_1")
        session.start()
        session.complete()

        with pytest.raises(ValueError):
            session.abort()

    def test_add_event(self):
        session = GameSession(puzzle_id="puzzle_1")
        event = session.add_event(
            role=AgentRole.PLAYER,
            message="Is it a person?",
            tags=["question"],
        )

        assert session.turn_count == 1
        assert event.turn_index == 0
        assert event.session_id == session.session_id

    def test_question_count(self):
        """Test that question_count only counts events with 'question' tag."""
        session = GameSession(puzzle_id="puzzle_1")
        
        # Add a question
        session.add_event(
            role=AgentRole.PLAYER,
            message="Is it a person?",
            tags=["question"],
        )
        
        # Add a DM answer (not a question)
        session.add_event(
            role=AgentRole.DM,
            message="Yes, it is a person.",
            tags=["answer"],
        )
        
        # Add another question
        session.add_event(
            role=AgentRole.PLAYER,
            message="Is the person alive?",
            tags=["question"],
        )
        
        # Add hypothesis (not a question)
        session.add_event(
            role=AgentRole.PLAYER,
            message="I think it's the butler.",
            tags=["hypothesis"],
        )
        
        # turn_count counts all events
        assert session.turn_count == 4
        # question_count only counts questions
        assert session.question_count == 2

    def test_get_recent_events(self):
        session = GameSession(puzzle_id="puzzle_1")

        for i in range(15):
            session.add_event(
                role=AgentRole.PLAYER,
                message=f"Question {i}",
            )

        recent = session.get_recent_events(limit=10)
        assert len(recent) == 10
        assert recent[-1].message == "Question 14"

    def test_use_hint(self):
        session = GameSession(puzzle_id="puzzle_1")
        session.config.hint_limit = 3

        assert session.use_hint()
        assert session.hint_count == 1

        assert session.use_hint()
        assert session.use_hint()

        assert not session.use_hint()
        assert session.hint_count == 3

    def test_session_serialization(self):
        session = GameSession(
            puzzle_id="puzzle_1",
            player_ids=["player_1"],
        )
        session.start()
        session.add_event(AgentRole.DM, "Welcome!")

        data = session.model_dump(mode="json")
        assert data["state"] == "in_progress"
        assert len(data["turn_history"]) == 1

        restored = GameSession.model_validate(data)
        assert restored.session_id == session.session_id
        assert restored.state == GameState.IN_PROGRESS
        assert restored.turn_count == 1
