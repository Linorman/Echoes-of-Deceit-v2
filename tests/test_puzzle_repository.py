"""Tests for PuzzleRepository."""

import json
import tempfile
from pathlib import Path

import pytest

from game.repository.puzzle_repository import PuzzleRepository
from game.domain.entities import Puzzle, PuzzleSummary
from config.models import GameConfig, DirectoriesConfig


@pytest.fixture
def temp_data_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_dir = tmpdir / "data" / "situation_puzzles"
        data_dir.mkdir(parents=True)
        yield {"base": tmpdir, "data": data_dir}


@pytest.fixture
def sample_puzzles(temp_data_dir):
    puzzle1_dir = temp_data_dir["data"] / "puzzle1"
    puzzle1_dir.mkdir()
    puzzle1_data = {
        "title": "Test Puzzle 1",
        "description": "A test puzzle",
        "puzzle": "A man walks into a bar...",
        "answer": "He was a ghost",
        "hints": ["Think about the obvious"],
        "tags": ["mystery", "easy"],
        "difficulty": "easy",
    }
    (puzzle1_dir / "puzzle_1_en.json").write_text(
        json.dumps(puzzle1_data), encoding="utf-8"
    )

    puzzle2_dir = temp_data_dir["data"] / "puzzle2"
    puzzle2_dir.mkdir()
    puzzle2_data = {
        "puzzle": "时间旅行者的故事...",
        "answer": "他是自己的祖父",
        "additional_info": [{"hint": "想想时间悖论"}],
    }
    (puzzle2_dir / "puzzle2_zh.json").write_text(
        json.dumps(puzzle2_data), encoding="utf-8"
    )

    template_dir = temp_data_dir["data"] / "template"
    template_dir.mkdir()
    (template_dir / "template.json").write_text("{}", encoding="utf-8")

    return temp_data_dir


@pytest.fixture
def puzzle_repo(sample_puzzles):
    config = GameConfig(
        directories=DirectoriesConfig(
            data_base_dir="data/situation_puzzles",
        )
    )
    return PuzzleRepository(config=config, base_dir=sample_puzzles["base"])


class TestPuzzleRepository:
    def test_discover_puzzles(self, puzzle_repo):
        puzzles = puzzle_repo.discover_puzzles()
        puzzle_ids = [p[0] for p in puzzles]

        assert len(puzzles) == 2
        assert "puzzle1" in puzzle_ids
        assert "puzzle2" in puzzle_ids
        assert "template" not in puzzle_ids

    def test_list_puzzles(self, puzzle_repo):
        summaries = puzzle_repo.list_puzzles()

        assert len(summaries) == 2
        assert all(isinstance(s, PuzzleSummary) for s in summaries)

        puzzle1 = next((s for s in summaries if s.id == "puzzle1"), None)
        assert puzzle1 is not None
        assert puzzle1.title == "Test Puzzle 1"
        assert puzzle1.difficulty == "easy"

    def test_get_puzzle(self, puzzle_repo):
        puzzle = puzzle_repo.get_puzzle("puzzle1")

        assert isinstance(puzzle, Puzzle)
        assert puzzle.id == "puzzle1"
        assert puzzle.title == "Test Puzzle 1"
        assert puzzle.puzzle_statement == "A man walks into a bar..."
        assert puzzle.answer == "He was a ghost"
        assert "Think about the obvious" in puzzle.hints

    def test_get_puzzle_caching(self, puzzle_repo):
        puzzle1 = puzzle_repo.get_puzzle("puzzle1")
        puzzle2 = puzzle_repo.get_puzzle("puzzle1")

        assert puzzle1 is puzzle2

    def test_get_puzzle_not_found(self, puzzle_repo):
        with pytest.raises(ValueError, match="not found"):
            puzzle_repo.get_puzzle("nonexistent")

    def test_extract_hints_from_additional_info(self, puzzle_repo):
        puzzle = puzzle_repo.get_puzzle("puzzle2")

        assert len(puzzle.hints) == 1
        assert "时间悖论" in puzzle.hints[0]

    def test_language_detection_from_filename(self, puzzle_repo):
        puzzle1 = puzzle_repo.get_puzzle("puzzle1")
        puzzle2 = puzzle_repo.get_puzzle("puzzle2")

        assert puzzle1.language == "en"
        assert puzzle2.language == "zh"

    def test_get_puzzle_dir(self, puzzle_repo, sample_puzzles):
        puzzle_dir = puzzle_repo.get_puzzle_dir("puzzle1")

        assert puzzle_dir.exists()
        assert puzzle_dir.is_dir()
        assert puzzle_dir == sample_puzzles["data"] / "puzzle1"

    def test_get_puzzle_dir_not_found(self, puzzle_repo):
        with pytest.raises(ValueError, match="not found"):
            puzzle_repo.get_puzzle_dir("nonexistent")

    def test_clear_cache(self, puzzle_repo):
        puzzle_repo.get_puzzle("puzzle1")
        assert len(puzzle_repo._puzzle_cache) == 1

        puzzle_repo.clear_cache()
        assert len(puzzle_repo._puzzle_cache) == 0

    def test_find_random_puzzle(self, puzzle_repo):
        puzzle = puzzle_repo.find_random_puzzle()

        assert isinstance(puzzle, Puzzle)
        assert puzzle.id in ("puzzle1", "puzzle2")

    def test_find_random_puzzle_with_language_filter(self, puzzle_repo):
        puzzle = puzzle_repo.find_random_puzzle(language="en")

        assert puzzle.id == "puzzle1"
        assert puzzle.language == "en"

    def test_find_random_puzzle_no_match(self, puzzle_repo):
        with pytest.raises(ValueError, match="No puzzles match"):
            puzzle_repo.find_random_puzzle(language="fr")


class TestPuzzleRepositoryEdgeCases:
    def test_empty_data_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            data_dir = tmpdir / "data" / "situation_puzzles"
            data_dir.mkdir(parents=True)

            config = GameConfig(
                directories=DirectoriesConfig(
                    data_base_dir="data/situation_puzzles",
                )
            )
            repo = PuzzleRepository(config=config, base_dir=tmpdir)

            assert repo.discover_puzzles() == []
            assert repo.list_puzzles() == []

    def test_missing_data_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GameConfig(
                directories=DirectoriesConfig(
                    data_base_dir="nonexistent/path",
                )
            )
            repo = PuzzleRepository(config=config, base_dir=Path(tmpdir))

            assert repo.discover_puzzles() == []

    def test_puzzle_with_constraints(self, temp_data_dir):
        puzzle_dir = temp_data_dir["data"] / "puzzle_constrained"
        puzzle_dir.mkdir()
        puzzle_data = {
            "puzzle": "Test puzzle",
            "answer": "Answer",
            "constraints": {
                "max_questions": 50,
                "max_hints": 3,
                "time_limit_minutes": 30,
            },
        }
        (puzzle_dir / "puzzle.json").write_text(
            json.dumps(puzzle_data), encoding="utf-8"
        )

        config = GameConfig(
            directories=DirectoriesConfig(
                data_base_dir="data/situation_puzzles",
            )
        )
        repo = PuzzleRepository(config=config, base_dir=temp_data_dir["base"])
        puzzle = repo.get_puzzle("puzzle_constrained")

        assert puzzle.constraints.max_questions == 50
        assert puzzle.constraints.max_hints == 3
        assert puzzle.constraints.time_limit_minutes == 30
