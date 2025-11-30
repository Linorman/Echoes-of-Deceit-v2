"""Tests for GameEngine."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from game.engine import GameEngine
from game.domain.entities import GameState
from config.models import GameConfig, DirectoriesConfig, ModelsConfig


@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        
        config_dir = base / "config"
        config_dir.mkdir()
        
        (config_dir / "game.yaml").write_text("""
rag:
  default_provider: lightrag
directories:
  data_base_dir: data/situation_puzzles
  rag_storage_dir: rag_storage
  game_storage_dir: game_storage
game:
  default_language: en
  max_turn_count: 100
puzzle:
  kb_id_prefix: game_
""")
        
        (config_dir / "models.yaml").write_text("""
provider: ollama
ollama:
  base_url: http://localhost:11434
  llm_model_name: qwen2.5:7b
  embedding_model_name: nomic-embed-text
""")
        
        (config_dir / "agents.yaml").write_text("""
dm:
  persona:
    name: Narrator
    tone: mysterious
hint:
  strategy:
    initial_vagueness: high
judge:
  strictness: moderate
""")
        
        data_dir = base / "data" / "situation_puzzles" / "test_puzzle"
        data_dir.mkdir(parents=True)
        
        (data_dir / "puzzle.json").write_text("""{
    "title": "Test Puzzle",
    "description": "A test puzzle",
    "puzzle": "A man walks into a bar. Why?",
    "answer": "He is a bartender.",
    "hints": ["Think about occupation", "He works there"],
    "tags": ["test"]
}""")
        
        (base / "rag_storage").mkdir()
        (base / "game_storage").mkdir()
        
        yield {"base": base, "config": config_dir}


class TestGameEngineInit:
    def test_engine_initialization(self, temp_workspace):
        engine = GameEngine(
            config_dir=temp_workspace["config"],
            base_dir=temp_workspace["base"],
        )
        
        assert engine.game_config is not None
        assert engine.models_config is not None
        assert engine.agents_config is not None

    def test_engine_properties(self, temp_workspace):
        engine = GameEngine(
            config_dir=temp_workspace["config"],
            base_dir=temp_workspace["base"],
        )
        
        assert engine.model_registry is not None
        assert engine.kb_manager is not None
        assert engine.puzzle_repository is not None
        assert engine.session_store is not None
        assert engine.profile_store is not None
        assert engine.memory_manager is not None


class TestGameEnginePuzzles:
    def test_list_puzzles(self, temp_workspace):
        engine = GameEngine(
            config_dir=temp_workspace["config"],
            base_dir=temp_workspace["base"],
        )
        
        puzzles = engine.list_puzzles()
        
        assert len(puzzles) == 1
        assert puzzles[0].id == "test_puzzle"
        assert puzzles[0].title == "Test Puzzle"

    def test_get_puzzle(self, temp_workspace):
        engine = GameEngine(
            config_dir=temp_workspace["config"],
            base_dir=temp_workspace["base"],
        )
        
        puzzle = engine.get_puzzle("test_puzzle")
        
        assert puzzle.id == "test_puzzle"
        assert puzzle.title == "Test Puzzle"
        assert puzzle.puzzle_statement == "A man walks into a bar. Why?"
        assert puzzle.answer == "He is a bartender."

    def test_get_puzzle_not_found(self, temp_workspace):
        engine = GameEngine(
            config_dir=temp_workspace["config"],
            base_dir=temp_workspace["base"],
        )
        
        with pytest.raises(ValueError):
            engine.get_puzzle("nonexistent_puzzle")


class TestGameEngineSessions:
    @pytest.mark.asyncio
    async def test_create_session(self, temp_workspace):
        engine = GameEngine(
            config_dir=temp_workspace["config"],
            base_dir=temp_workspace["base"],
        )
        
        with patch.object(engine._kb_manager, 'ensure_puzzle_kb', new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = "game_test_puzzle"
            
            session = await engine.create_session("test_puzzle", "player1")
            
            assert session.puzzle_id == "test_puzzle"
            assert "player1" in session.player_ids
            assert session.kb_id == "game_test_puzzle"
            assert session.state == GameState.LOBBY

    @pytest.mark.asyncio
    async def test_create_session_puzzle_not_found(self, temp_workspace):
        engine = GameEngine(
            config_dir=temp_workspace["config"],
            base_dir=temp_workspace["base"],
        )
        
        with pytest.raises(ValueError):
            await engine.create_session("nonexistent", "player1")

    def test_get_session(self, temp_workspace):
        engine = GameEngine(
            config_dir=temp_workspace["config"],
            base_dir=temp_workspace["base"],
        )
        
        session = engine.session_store.create_session(
            puzzle_id="test_puzzle",
            player_ids=["player1"],
        )
        
        loaded = engine.get_session(session.session_id)
        
        assert loaded.session_id == session.session_id
        assert loaded.puzzle_id == "test_puzzle"

    def test_list_sessions(self, temp_workspace):
        engine = GameEngine(
            config_dir=temp_workspace["config"],
            base_dir=temp_workspace["base"],
        )
        
        session1 = engine.session_store.create_session(puzzle_id="test_puzzle")
        session2 = engine.session_store.create_session(puzzle_id="test_puzzle")
        
        sessions = engine.list_sessions()
        
        assert len(sessions) >= 2
        session_ids = [s.session_id for s in sessions]
        assert session1.session_id in session_ids
        assert session2.session_id in session_ids

    def test_list_sessions_with_filter(self, temp_workspace):
        engine = GameEngine(
            config_dir=temp_workspace["config"],
            base_dir=temp_workspace["base"],
        )
        
        session = engine.session_store.create_session(puzzle_id="test_puzzle")
        session.start()
        engine.save_session(session)
        
        in_progress = engine.list_sessions(state_filter=GameState.IN_PROGRESS)
        lobby = engine.list_sessions(state_filter=GameState.LOBBY)
        
        in_progress_ids = [s.session_id for s in in_progress]
        assert session.session_id in in_progress_ids


class TestGameEnginePlayerProfile:
    def test_get_player_profile_creates_new(self, temp_workspace):
        engine = GameEngine(
            config_dir=temp_workspace["config"],
            base_dir=temp_workspace["base"],
        )
        
        profile = engine.get_player_profile("new_player")
        
        assert profile.player_id == "new_player"
        assert profile.display_name == "new_player"

    def test_get_player_profile_returns_existing(self, temp_workspace):
        engine = GameEngine(
            config_dir=temp_workspace["config"],
            base_dir=temp_workspace["base"],
        )
        
        engine.profile_store.create_profile("existing", "Existing Player")
        
        profile = engine.get_player_profile("existing")
        
        assert profile.player_id == "existing"
        assert profile.display_name == "Existing Player"


class TestGameEngineClose:
    @pytest.mark.asyncio
    async def test_close(self, temp_workspace):
        engine = GameEngine(
            config_dir=temp_workspace["config"],
            base_dir=temp_workspace["base"],
        )
        
        with patch.object(engine._kb_manager, 'close', new_callable=AsyncMock) as mock_close:
            await engine.close()
            mock_close.assert_called_once()
