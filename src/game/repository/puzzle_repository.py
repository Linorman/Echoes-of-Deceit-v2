"""Repository for puzzle discovery and loading."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import ConfigLoader, GameConfig
from game.domain.entities import Puzzle, PuzzleConstraints, PuzzleSummary

logger = logging.getLogger(__name__)


class PuzzleRepository:
    def __init__(
        self,
        config: Optional[GameConfig] = None,
        base_dir: Optional[Path] = None,
    ):
        if config is None:
            loader = ConfigLoader()
            config = loader.load_game_config()

        self._config = config

        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent.parent

        self._data_base_dir = base_dir / config.directories.data_base_dir
        self._puzzle_cache: Dict[str, Puzzle] = {}

    @property
    def data_dir(self) -> Path:
        return self._data_base_dir

    def discover_puzzles(self) -> List[tuple[str, Path]]:
        if not self._data_base_dir.exists() or not self._data_base_dir.is_dir():
            logger.warning("Data base directory not found: %s", self._data_base_dir)
            return []

        puzzle_dirs = []
        for item in self._data_base_dir.iterdir():
            if item.is_dir() and not item.name.startswith((".", "_", "template")):
                json_files = list(item.glob("*.json"))
                if json_files:
                    puzzle_id = item.name
                    puzzle_dirs.append((puzzle_id, item))

        logger.info("Discovered %d puzzle directories in %s", len(puzzle_dirs), self._data_base_dir)
        return puzzle_dirs

    def list_puzzles(self) -> List[PuzzleSummary]:
        puzzle_dirs = self.discover_puzzles()
        summaries = []

        for puzzle_id, puzzle_dir in puzzle_dirs:
            try:
                puzzle = self.get_puzzle(puzzle_id)
                summaries.append(
                    PuzzleSummary(
                        id=puzzle.id,
                        title=puzzle.title,
                        description=puzzle.description,
                        difficulty=puzzle.difficulty,
                        tags=puzzle.tags,
                        language=puzzle.language,
                    )
                )
            except Exception as exc:
                logger.error("Failed to load puzzle %s: %s", puzzle_id, exc)

        return summaries

    def get_puzzle(self, puzzle_id: str) -> Puzzle:
        if puzzle_id in self._puzzle_cache:
            return self._puzzle_cache[puzzle_id]

        puzzle_dir = self._data_base_dir / puzzle_id
        if not puzzle_dir.exists() or not puzzle_dir.is_dir():
            raise ValueError(f"Puzzle directory not found: {puzzle_id}")

        json_files = list(puzzle_dir.glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found for puzzle: {puzzle_id}")

        puzzle_file = json_files[0]
        puzzle = self._load_puzzle_from_file(puzzle_id, puzzle_file)
        self._puzzle_cache[puzzle_id] = puzzle
        return puzzle

    def _load_puzzle_from_file(self, puzzle_id: str, puzzle_file: Path) -> Puzzle:
        with open(puzzle_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        hints = self._extract_hints(data)
        language = self._detect_language(puzzle_file, data)

        return Puzzle(
            id=puzzle_id,
            title=data.get("title", puzzle_id),
            description=data.get("description", ""),
            puzzle_statement=data.get("puzzle", ""),
            answer=data.get("answer", ""),
            hints=hints,
            additional_info=data.get("additional_info", []),
            constraints=self._load_constraints(data),
            tags=data.get("tags", []),
            language=language,
            difficulty=data.get("difficulty"),
        )

    def _extract_hints(self, data: Dict[str, Any]) -> List[str]:
        hints = []

        if "hints" in data and isinstance(data["hints"], list):
            hints.extend(str(h) for h in data["hints"] if h)

        additional_info = data.get("additional_info", [])
        if isinstance(additional_info, list):
            for info in additional_info:
                if isinstance(info, dict):
                    hint = info.get("hint") or info.get("Hint")
                    if hint:
                        hints.append(str(hint))

        return hints

    def _load_constraints(self, data: Dict[str, Any]) -> PuzzleConstraints:
        constraints_data = data.get("constraints", {})
        if isinstance(constraints_data, dict):
            return PuzzleConstraints(**constraints_data)
        return PuzzleConstraints()

    def _detect_language(self, puzzle_file: Path, data: Dict[str, Any]) -> str:
        if "language" in data:
            return data["language"]

        filename = puzzle_file.name.lower()
        if "_en" in filename or "english" in filename:
            return "en"
        if "_zh" in filename or "chinese" in filename:
            return "zh"

        return self._config.game.default_language

    def find_random_puzzle(
        self,
        language: Optional[str] = None,
        difficulty: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Puzzle:
        puzzles = self.list_puzzles()

        if language:
            puzzles = [p for p in puzzles if p.language == language]

        if difficulty:
            puzzles = [p for p in puzzles if p.difficulty == difficulty]

        if tags:
            puzzles = [p for p in puzzles if any(t in p.tags for t in tags)]

        if not puzzles:
            raise ValueError("No puzzles match the specified filters")

        selected = random.choice(puzzles)
        return self.get_puzzle(selected.id)

    def get_puzzle_dir(self, puzzle_id: str) -> Path:
        puzzle_dir = self._data_base_dir / puzzle_id
        if not puzzle_dir.exists():
            raise ValueError(f"Puzzle directory not found: {puzzle_id}")
        return puzzle_dir

    def clear_cache(self) -> None:
        self._puzzle_cache.clear()
