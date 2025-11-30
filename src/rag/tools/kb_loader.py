"""Game data loader for RAG system."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..base_provider import RAGDocument

logger = logging.getLogger(__name__)


class GameDataLoader:
    """Loads game data from various formats into RAG documents."""

    @staticmethod
    def load_puzzle_json(puzzle_file: Path) -> Tuple[Dict[str, Any], List[RAGDocument]]:
        """Load puzzle from JSON file and convert to documents."""
        if not puzzle_file.exists():
            raise FileNotFoundError(f"Puzzle file not found: {puzzle_file}")

        with open(puzzle_file, "r", encoding="utf-8") as f:
            puzzle_data = json.load(f)

        game_info = {
            "game_type": puzzle_data.get("type", "situation_puzzle"),
            "title": puzzle_data.get("title", puzzle_file.stem),
            "description": puzzle_data.get("description", ""),
            "source_file": str(puzzle_file),
        }

        documents = GameDataLoader._create_puzzle_documents(
            puzzle_data, puzzle_file.stem
        )

        return game_info, documents

    @staticmethod
    def _create_puzzle_documents(
        puzzle_data: Dict[str, Any],
        base_id: str,
    ) -> List[RAGDocument]:
        """Convert puzzle data to RAG documents."""
        documents = []

        puzzle_text = puzzle_data.get("puzzle", "")
        if puzzle_text:
            documents.append(
                RAGDocument(
                    content=f"Puzzle statement: {puzzle_text}",
                    doc_id=f"{base_id}_puzzle",
                    metadata={
                        "type": "puzzle_statement",
                        "base_id": base_id,
                    },
                )
            )

        answer_text = puzzle_data.get("answer", "")
        if answer_text:
            documents.append(
                RAGDocument(
                    content=f"Puzzle answer: {answer_text}",
                    doc_id=f"{base_id}_answer",
                    metadata={
                        "type": "puzzle_answer",
                        "base_id": base_id,
                    },
                )
            )

        additional_info = puzzle_data.get("additional_info", [])
        if isinstance(additional_info, list):
            for idx, info in enumerate(additional_info):
                if isinstance(info, dict):
                    for key, value in info.items():
                        if key and value:
                            documents.append(
                                RAGDocument(
                                    content=f"{key}: {value}",
                                    doc_id=f"{base_id}_info_{idx}_{key}",
                                    metadata={
                                        "type": "additional_info",
                                        "base_id": base_id,
                                        "info_key": key,
                                    },
                                )
                            )

        if not documents:
            raise ValueError(f"No valid content in puzzle: {base_id}")

        return documents

    @staticmethod
    def load_game_directory(
        game_dir: Path,
        game_type: str = "situation_puzzle",
    ) -> Tuple[Dict[str, Any], List[Path], List[RAGDocument]]:
        """Load all files from a game directory."""
        if not game_dir.exists() or not game_dir.is_dir():
            raise ValueError(f"Invalid game directory: {game_dir}")

        json_files = list(game_dir.glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in {game_dir}")

        all_documents = []
        file_paths = []
        game_info = None

        for json_file in json_files:
            try:
                info, docs = GameDataLoader.load_puzzle_json(json_file)
                if game_info is None:
                    game_info = info
                    game_info["game_type"] = game_type
                all_documents.extend(docs)
                file_paths.append(json_file)
                logger.info("Loaded %d documents from %s", len(docs), json_file.name)
            except Exception as exc:
                logger.error("Failed to load %s: %s", json_file, exc)

        if not game_info:
            game_info = {
                "game_type": game_type,
                "title": game_dir.name,
                "description": f"Game from {game_dir.name}",
            }

        return game_info, file_paths, all_documents

    @staticmethod
    def discover_games(
        base_dir: Path,
        game_type: str = "situation_puzzle",
    ) -> List[Tuple[str, Path]]:
        """Discover all game directories in base directory."""
        if not base_dir.exists() or not base_dir.is_dir():
            raise ValueError(f"Invalid base directory: {base_dir}")

        game_dirs = []
        for item in base_dir.iterdir():
            if item.is_dir() and not item.name.startswith((".", "_", "template")):
                json_files = list(item.glob("*.json"))
                if json_files:
                    game_id = item.name
                    game_dirs.append((game_id, item))

        logger.info("Discovered %d game directories in %s", len(game_dirs), base_dir)
        return game_dirs

