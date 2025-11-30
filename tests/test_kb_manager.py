"""Tests for KnowledgeBaseManager."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from game.kb_manager import (
    DOCUMENT_TYPE_ALL,
    DOCUMENT_TYPE_HINT,
    DOCUMENT_TYPE_PUBLIC,
    DOCUMENT_TYPE_SECRET,
    KnowledgeBaseManager,
    PuzzleInfo,
)
from config.models import GameConfig, DirectoriesConfig, RagConfig, PuzzleConfig
from rag.base_provider import QueryResult, ProviderStatus


@pytest.fixture
def temp_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_dir = tmpdir / "data" / "situation_puzzles"
        rag_dir = tmpdir / "rag_storage"
        game_dir = tmpdir / "game_storage"
        
        data_dir.mkdir(parents=True)
        rag_dir.mkdir(parents=True)
        game_dir.mkdir(parents=True)
        
        yield {
            "base": tmpdir,
            "data": data_dir,
            "rag": rag_dir,
            "game": game_dir,
        }


@pytest.fixture
def sample_puzzle(temp_dirs):
    puzzle_dir = temp_dirs["data"] / "puzzle1"
    puzzle_dir.mkdir()
    
    puzzle_data = {
        "type": "situation_puzzle",
        "title": "Test Puzzle",
        "description": "A test puzzle",
        "puzzle": "A man walks into a bar...",
        "answer": "He was a ghost",
        "additional_info": [{"hint": "Think about what's unusual"}],
    }
    
    puzzle_file = puzzle_dir / "puzzle.json"
    puzzle_file.write_text(json.dumps(puzzle_data), encoding="utf-8")
    
    return puzzle_dir


@pytest.fixture
def game_config(temp_dirs):
    return GameConfig(
        rag=RagConfig(default_provider="lightrag"),
        directories=DirectoriesConfig(
            data_base_dir="data/situation_puzzles",
            rag_storage_dir="rag_storage",
            game_storage_dir="game_storage",
        ),
        puzzle=PuzzleConfig(kb_id_prefix="game_"),
    )


@pytest.fixture
def base_dir(temp_dirs):
    data_dir = temp_dirs["base"] / "data" / "situation_puzzles"
    data_dir.mkdir(parents=True, exist_ok=True)
    rag_dir = temp_dirs["base"] / "rag_storage"
    rag_dir.mkdir(parents=True, exist_ok=True)
    return temp_dirs["base"]


class TestDocumentTypeConstants:
    def test_public_types(self):
        assert "puzzle_statement" in DOCUMENT_TYPE_PUBLIC
        assert "public_fact" in DOCUMENT_TYPE_PUBLIC
        assert "puzzle_answer" not in DOCUMENT_TYPE_PUBLIC

    def test_hint_types(self):
        assert "hint" in DOCUMENT_TYPE_HINT

    def test_secret_types(self):
        assert "puzzle_answer" in DOCUMENT_TYPE_SECRET
        assert "additional_info" in DOCUMENT_TYPE_SECRET

    def test_all_types(self):
        assert DOCUMENT_TYPE_ALL == DOCUMENT_TYPE_PUBLIC | DOCUMENT_TYPE_HINT | DOCUMENT_TYPE_SECRET


class TestPuzzleInfo:
    def test_dataclass_creation(self):
        info = PuzzleInfo(
            puzzle_id="puzzle1",
            kb_id="game_puzzle1",
            title="Test Puzzle",
            description="A test",
            game_type="situation_puzzle",
            source_files=["file1.json"],
            document_count=3,
        )
        assert info.puzzle_id == "puzzle1"
        assert info.kb_id == "game_puzzle1"
        assert info.document_count == 3


class TestKnowledgeBaseManager:
    def test_puzzle_to_kb_id(self, game_config, base_dir):
        with patch("game.kb_manager.KnowledgeBase"):
            manager = KnowledgeBaseManager(config=game_config, base_dir=base_dir)
            kb_id = manager._puzzle_to_kb_id("puzzle1")
            assert kb_id == "game_puzzle1"

    def test_kb_id_to_puzzle(self, game_config, base_dir):
        with patch("game.kb_manager.KnowledgeBase"):
            manager = KnowledgeBaseManager(config=game_config, base_dir=base_dir)
            puzzle_id = manager._kb_id_to_puzzle("game_puzzle1")
            assert puzzle_id == "puzzle1"

    def test_kb_id_to_puzzle_no_prefix(self, game_config, base_dir):
        with patch("game.kb_manager.KnowledgeBase"):
            manager = KnowledgeBaseManager(config=game_config, base_dir=base_dir)
            puzzle_id = manager._kb_id_to_puzzle("other_puzzle")
            assert puzzle_id == "other_puzzle"

    def test_get_puzzle_kb_id(self, game_config, base_dir):
        with patch("game.kb_manager.KnowledgeBase"):
            manager = KnowledgeBaseManager(config=game_config, base_dir=base_dir)
            kb_id = manager.get_puzzle_kb_id("test_puzzle")
            assert kb_id == "game_test_puzzle"

    def test_discover_puzzles(self, game_config, sample_puzzle, base_dir):
        with patch("game.kb_manager.KnowledgeBase"):
            manager = KnowledgeBaseManager(config=game_config, base_dir=base_dir)
            puzzles = manager.discover_puzzles()
            
            assert len(puzzles) == 1
            puzzle_id, puzzle_dir = puzzles[0]
            assert puzzle_id == "puzzle1"
            assert puzzle_dir == sample_puzzle

    def test_discover_puzzles_empty_dir(self, game_config, base_dir):
        with patch("game.kb_manager.KnowledgeBase"):
            manager = KnowledgeBaseManager(config=game_config, base_dir=base_dir)
            puzzles = manager.discover_puzzles()
            assert len(puzzles) == 0

    def test_kb_exists_true(self, game_config, base_dir):
        mock_kb = MagicMock()
        mock_kb.get_knowledge_base.return_value = MagicMock()
        
        with patch("game.kb_manager.KnowledgeBase", return_value=mock_kb):
            manager = KnowledgeBaseManager(config=game_config, base_dir=base_dir)
            assert manager.kb_exists("puzzle1") is True
            mock_kb.get_knowledge_base.assert_called_with("game_puzzle1")

    def test_kb_exists_false(self, game_config, base_dir):
        mock_kb = MagicMock()
        mock_kb.get_knowledge_base.return_value = None
        
        with patch("game.kb_manager.KnowledgeBase", return_value=mock_kb):
            manager = KnowledgeBaseManager(config=game_config, base_dir=base_dir)
            assert manager.kb_exists("puzzle1") is False

    @pytest.mark.asyncio
    async def test_ensure_puzzle_kb_existing(self, game_config, sample_puzzle, base_dir):
        mock_kb = MagicMock()
        mock_kb.get_knowledge_base.return_value = MagicMock()
        
        with patch("game.kb_manager.KnowledgeBase", return_value=mock_kb):
            manager = KnowledgeBaseManager(config=game_config, base_dir=base_dir)
            kb_id = await manager.ensure_puzzle_kb("puzzle1", sample_puzzle)
            
            assert kb_id == "game_puzzle1"
            mock_kb.create_knowledge_base.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_puzzle_kb_new(self, game_config, sample_puzzle, base_dir):
        mock_kb = MagicMock()
        mock_kb.get_knowledge_base.return_value = None
        mock_kb.create_knowledge_base = MagicMock()
        mock_kb.insert_documents = AsyncMock()
        
        with patch("game.kb_manager.KnowledgeBase", return_value=mock_kb):
            manager = KnowledgeBaseManager(config=game_config, base_dir=base_dir)
            kb_id = await manager.ensure_puzzle_kb("puzzle1", sample_puzzle)
            
            assert kb_id == "game_puzzle1"
            mock_kb.create_knowledge_base.assert_called_once()
            mock_kb.insert_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_public_filters_correctly(self, game_config, base_dir):
        mock_kb = MagicMock()
        mock_kb.query = AsyncMock(return_value=QueryResult(
            answer="Test answer",
            sources=[
                {"metadata": {"type": "puzzle_statement"}},
                {"metadata": {"type": "puzzle_answer"}},
                {"metadata": {"type": "public_fact"}},
            ],
        ))
        
        with patch("game.kb_manager.KnowledgeBase", return_value=mock_kb):
            manager = KnowledgeBaseManager(config=game_config, base_dir=base_dir)
            result = await manager.query_public("game_puzzle1", "test query")
            
            assert len(list(result.sources)) == 2
            for source in result.sources:
                assert source["metadata"]["type"] in DOCUMENT_TYPE_PUBLIC

    @pytest.mark.asyncio
    async def test_query_with_hints_includes_hints(self, game_config, base_dir):
        mock_kb = MagicMock()
        mock_kb.query = AsyncMock(return_value=QueryResult(
            answer="Test answer",
            sources=[
                {"metadata": {"type": "puzzle_statement"}},
                {"metadata": {"type": "hint"}},
                {"metadata": {"type": "puzzle_answer"}},
            ],
        ))
        
        with patch("game.kb_manager.KnowledgeBase", return_value=mock_kb):
            manager = KnowledgeBaseManager(config=game_config, base_dir=base_dir)
            result = await manager.query_with_hints("game_puzzle1", "test query")
            
            assert len(list(result.sources)) == 2
            types = {s["metadata"]["type"] for s in result.sources}
            assert "puzzle_statement" in types
            assert "hint" in types
            assert "puzzle_answer" not in types

    @pytest.mark.asyncio
    async def test_query_full_includes_all(self, game_config, base_dir):
        mock_kb = MagicMock()
        mock_kb.query = AsyncMock(return_value=QueryResult(
            answer="Test answer",
            sources=[
                {"metadata": {"type": "puzzle_statement"}},
                {"metadata": {"type": "hint"}},
                {"metadata": {"type": "puzzle_answer"}},
                {"metadata": {"type": "additional_info"}},
            ],
        ))
        
        with patch("game.kb_manager.KnowledgeBase", return_value=mock_kb):
            manager = KnowledgeBaseManager(config=game_config, base_dir=base_dir)
            result = await manager.query_full("game_puzzle1", "test query")
            
            assert len(list(result.sources)) == 4

    @pytest.mark.asyncio
    async def test_health_check(self, game_config, base_dir):
        mock_kb = MagicMock()
        mock_kb.health_check = AsyncMock(return_value=ProviderStatus(
            is_ready=True,
            details={"status": "ok"},
        ))
        
        with patch("game.kb_manager.KnowledgeBase", return_value=mock_kb):
            manager = KnowledgeBaseManager(config=game_config, base_dir=base_dir)
            status = await manager.health_check("puzzle1")
            
            assert status.is_ready is True
            mock_kb.health_check.assert_called_with("game_puzzle1")

    def test_list_puzzle_kbs(self, game_config, base_dir):
        from rag.knowledge_base import KnowledgeBaseConfig

        mock_kb = MagicMock()
        mock_kb.list_knowledge_bases.return_value = [
            KnowledgeBaseConfig(
                kb_id="game_puzzle1",
                name="Test Puzzle 1",
                description="First puzzle",
                working_dir="/tmp/game_puzzle1",
                provider_type="lightrag",
                created_at="2024-01-01",
                updated_at="2024-01-01",
                metadata={
                    "game_type": "situation_puzzle",
                    "source_files": ["file1.json"],
                    "document_count": 3,
                },
            ),
            KnowledgeBaseConfig(
                kb_id="other_kb",
                name="Other KB",
                description="Not a puzzle",
                working_dir="/tmp/other_kb",
                provider_type="lightrag",
                created_at="2024-01-01",
                updated_at="2024-01-01",
            ),
        ]
        
        with patch("game.kb_manager.KnowledgeBase", return_value=mock_kb):
            manager = KnowledgeBaseManager(config=game_config, base_dir=base_dir)
            puzzles = manager.list_puzzle_kbs()
            
            assert len(puzzles) == 1
            assert puzzles[0].puzzle_id == "puzzle1"
            assert puzzles[0].kb_id == "game_puzzle1"
            assert puzzles[0].document_count == 3

    @pytest.mark.asyncio
    async def test_close(self, game_config, base_dir):
        mock_kb = MagicMock()
        mock_kb.close_all = AsyncMock()
        
        with patch("game.kb_manager.KnowledgeBase", return_value=mock_kb):
            manager = KnowledgeBaseManager(config=game_config, base_dir=base_dir)
            await manager.close()
            
            mock_kb.close_all.assert_called_once()
