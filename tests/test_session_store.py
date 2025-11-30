"""Tests for GameSessionStore and PlayerProfileStore."""

import json
import tempfile
from pathlib import Path
from datetime import datetime

import pytest

from game.storage.session_store import (
    GameSessionStore,
    PlayerProfileStore,
    DateTimeEncoder,
    datetime_decoder,
)
from game.domain.entities import (
    AgentRole,
    GameSession,
    GameState,
    PlayerProfile,
    SessionConfig,
    SessionEvent,
)
from config.models import GameConfig, DirectoriesConfig


@pytest.fixture
def temp_storage_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        game_storage = tmpdir / "game_storage"
        game_storage.mkdir(parents=True)
        yield {"base": tmpdir, "storage": game_storage}


@pytest.fixture
def session_store(temp_storage_dir):
    config = GameConfig(
        directories=DirectoriesConfig(
            game_storage_dir="game_storage",
        )
    )
    return GameSessionStore(config=config, base_dir=temp_storage_dir["base"])


@pytest.fixture
def profile_store(temp_storage_dir):
    config = GameConfig(
        directories=DirectoriesConfig(
            game_storage_dir="game_storage",
        )
    )
    return PlayerProfileStore(config=config, base_dir=temp_storage_dir["base"])


class TestDateTimeHandling:
    def test_datetime_encoder(self):
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = json.dumps({"time": dt}, cls=DateTimeEncoder)
        assert "2024-01-15T10:30:00" in result

    def test_datetime_decoder(self):
        data = {"time": "2024-01-15T10:30:00"}
        result = datetime_decoder(data)
        assert isinstance(result["time"], datetime)
        assert result["time"].year == 2024

    def test_datetime_decoder_with_timezone(self):
        data = {"time": "2024-01-15T10:30:00+00:00"}
        result = datetime_decoder(data)
        assert isinstance(result["time"], datetime)


class TestGameSessionStore:
    def test_create_session(self, session_store):
        session = session_store.create_session(
            puzzle_id="puzzle_1",
            player_ids=["player_1"],
        )

        assert session.puzzle_id == "puzzle_1"
        assert session.player_ids == ["player_1"]
        assert session.state == GameState.LOBBY
        assert session_store.session_exists(session.session_id)

    def test_create_session_with_kb_id(self, session_store):
        session = session_store.create_session(
            puzzle_id="puzzle_1",
            kb_id="game_puzzle_1",
        )

        assert session.kb_id == "game_puzzle_1"

    def test_save_and_load_session(self, session_store):
        session = session_store.create_session(puzzle_id="puzzle_1")
        session.start()
        session.add_event(AgentRole.DM, "Welcome to the game!")
        session_store.save_session(session)

        loaded = session_store.load_session(session.session_id)

        assert loaded.session_id == session.session_id
        assert loaded.state == GameState.IN_PROGRESS
        assert loaded.turn_count == 1

    def test_load_session_not_found(self, session_store):
        with pytest.raises(ValueError, match="not found"):
            session_store.load_session("nonexistent_id")

    def test_session_exists(self, session_store):
        session = session_store.create_session(puzzle_id="puzzle_1")

        assert session_store.session_exists(session.session_id)
        assert not session_store.session_exists("nonexistent")

    def test_delete_session(self, session_store):
        session = session_store.create_session(puzzle_id="puzzle_1")
        session_id = session.session_id

        assert session_store.session_exists(session_id)

        result = session_store.delete_session(session_id)

        assert result is True
        assert not session_store.session_exists(session_id)

    def test_delete_session_not_found(self, session_store):
        result = session_store.delete_session("nonexistent")
        assert result is False

    def test_list_sessions(self, session_store):
        session1 = session_store.create_session(puzzle_id="puzzle_1")
        session2 = session_store.create_session(puzzle_id="puzzle_2")
        session2.start()
        session_store.save_session(session2)

        sessions = session_store.list_sessions()

        assert len(sessions) == 2

    def test_list_sessions_with_state_filter(self, session_store):
        session1 = session_store.create_session(puzzle_id="puzzle_1")
        session2 = session_store.create_session(puzzle_id="puzzle_2")
        session2.start()
        session_store.save_session(session2)

        lobby_sessions = session_store.list_sessions(state_filter=GameState.LOBBY)
        active_sessions = session_store.list_sessions(state_filter=GameState.IN_PROGRESS)

        assert len(lobby_sessions) == 1
        assert len(active_sessions) == 1

    def test_list_sessions_with_puzzle_filter(self, session_store):
        session_store.create_session(puzzle_id="puzzle_1")
        session_store.create_session(puzzle_id="puzzle_2")

        sessions = session_store.list_sessions(puzzle_id="puzzle_1")

        assert len(sessions) == 1
        assert sessions[0].puzzle_id == "puzzle_1"

    def test_list_sessions_with_player_filter(self, session_store):
        session_store.create_session(puzzle_id="puzzle_1", player_ids=["player_1"])
        session_store.create_session(puzzle_id="puzzle_2", player_ids=["player_2"])

        sessions = session_store.list_sessions(player_id="player_1")

        assert len(sessions) == 1
        assert "player_1" in sessions[0].player_ids

    def test_append_and_get_events(self, session_store):
        session = session_store.create_session(puzzle_id="puzzle_1")

        event1 = SessionEvent(
            session_id=session.session_id,
            turn_index=0,
            role=AgentRole.DM,
            message="Welcome!",
        )
        event2 = SessionEvent(
            session_id=session.session_id,
            turn_index=1,
            role=AgentRole.PLAYER,
            message="Is it a person?",
            tags=["question"],
        )

        session_store.append_event(session.session_id, event1)
        session_store.append_event(session.session_id, event2)

        events = session_store.get_events(session.session_id)

        assert len(events) == 2
        assert events[0].message == "Welcome!"
        assert events[1].tags == ["question"]

    def test_get_events_no_file(self, session_store):
        events = session_store.get_events("nonexistent")
        assert events == []

    def test_session_with_full_lifecycle(self, session_store):
        session = session_store.create_session(
            puzzle_id="puzzle_1",
            player_ids=["player_1"],
        )

        session.start()
        session.add_event(AgentRole.DM, "Welcome!")
        session.add_event(AgentRole.PLAYER, "Is it a person?", ["question"])
        session.add_event(AgentRole.DM, "Yes", ["answer"])
        session.use_hint()
        session.complete(score=85)

        session_store.save_session(session)

        loaded = session_store.load_session(session.session_id)

        assert loaded.state == GameState.COMPLETED
        assert loaded.score == 85
        assert loaded.hint_count == 1
        assert loaded.turn_count == 3
        assert loaded.completed_at is not None


class TestPlayerProfileStore:
    def test_create_profile(self, profile_store):
        profile = profile_store.create_profile(
            player_id="player_1",
            display_name="Test Player",
        )

        assert profile.player_id == "player_1"
        assert profile.display_name == "Test Player"
        assert profile_store.profile_exists("player_1")

    def test_create_profile_with_preferences(self, profile_store):
        profile = profile_store.create_profile(
            player_id="player_1",
            display_name="Test Player",
            preferences={"difficulty": "hard", "language": "en"},
        )

        assert profile.preferences["difficulty"] == "hard"

    def test_create_profile_duplicate(self, profile_store):
        profile_store.create_profile("player_1", "Test Player")

        with pytest.raises(ValueError, match="already exists"):
            profile_store.create_profile("player_1", "Another Name")

    def test_save_and_load_profile(self, profile_store):
        profile = profile_store.create_profile(
            player_id="player_1",
            display_name="Test Player",
        )
        profile.update_preference("theme", "dark")
        profile_store.save_profile(profile)

        loaded = profile_store.load_profile("player_1")

        assert loaded.player_id == "player_1"
        assert loaded.preferences["theme"] == "dark"

    def test_load_profile_not_found(self, profile_store):
        with pytest.raises(ValueError, match="not found"):
            profile_store.load_profile("nonexistent")

    def test_profile_exists(self, profile_store):
        profile_store.create_profile("player_1", "Test Player")

        assert profile_store.profile_exists("player_1")
        assert not profile_store.profile_exists("nonexistent")

    def test_delete_profile(self, profile_store):
        profile_store.create_profile("player_1", "Test Player")

        assert profile_store.profile_exists("player_1")

        result = profile_store.delete_profile("player_1")

        assert result is True
        assert not profile_store.profile_exists("player_1")

    def test_delete_profile_not_found(self, profile_store):
        result = profile_store.delete_profile("nonexistent")
        assert result is False

    def test_get_or_create_profile_new(self, profile_store):
        profile = profile_store.get_or_create_profile(
            player_id="player_1",
            display_name="New Player",
        )

        assert profile.player_id == "player_1"
        assert profile.display_name == "New Player"

    def test_get_or_create_profile_existing(self, profile_store):
        profile_store.create_profile("player_1", "Original Name")

        profile = profile_store.get_or_create_profile(
            player_id="player_1",
            display_name="Different Name",
        )

        assert profile.display_name == "Original Name"

    def test_list_profiles(self, profile_store):
        profile_store.create_profile("player_1", "Player 1")
        profile_store.create_profile("player_2", "Player 2")

        profiles = profile_store.list_profiles()

        assert len(profiles) == 2
        player_ids = [p.player_id for p in profiles]
        assert "player_1" in player_ids
        assert "player_2" in player_ids

    def test_update_profile(self, profile_store):
        profile_store.create_profile("player_1", "Test Player")

        updated = profile_store.update_profile(
            player_id="player_1",
            display_name="New Name",
            preferences={"theme": "light"},
        )

        assert updated.display_name == "New Name"
        assert updated.preferences["theme"] == "light"

        reloaded = profile_store.load_profile("player_1")
        assert reloaded.display_name == "New Name"

    def test_update_profile_partial(self, profile_store):
        profile_store.create_profile(
            "player_1",
            "Test Player",
            preferences={"existing": "value"},
        )

        updated = profile_store.update_profile(
            player_id="player_1",
            preferences={"new": "value"},
        )

        assert updated.display_name == "Test Player"
        assert updated.preferences["existing"] == "value"
        assert updated.preferences["new"] == "value"


class TestStorageDirectoryStructure:
    def test_directories_created(self, session_store, profile_store):
        assert session_store._sessions_dir.exists()
        assert session_store._events_dir.exists()
        assert profile_store._profiles_dir.exists()

    def test_session_file_location(self, session_store):
        session = session_store.create_session(puzzle_id="test")
        session_file = session_store._sessions_dir / f"{session.session_id}.json"

        assert session_file.exists()

    def test_events_file_location(self, session_store):
        session = session_store.create_session(puzzle_id="test")
        event = SessionEvent(
            session_id=session.session_id,
            turn_index=0,
            role=AgentRole.DM,
            message="Test",
        )
        session_store.append_event(session.session_id, event)

        events_file = session_store._events_dir / f"{session.session_id}.jsonl"
        assert events_file.exists()

    def test_profile_file_location(self, profile_store):
        profile_store.create_profile("test_player", "Test")
        profile_file = profile_store._profiles_dir / "test_player.json"

        assert profile_file.exists()
