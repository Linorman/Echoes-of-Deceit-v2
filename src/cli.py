"""CLI tool for game system initialization and health checks."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ConfigLoader
from models import ModelProviderRegistry
from game import KnowledgeBaseManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class GameCLI:
    def __init__(self):
        self._config_loader = ConfigLoader()
        self._model_registry: ModelProviderRegistry | None = None
        self._kb_manager: KnowledgeBaseManager | None = None

    def _ensure_initialized(self) -> None:
        if self._model_registry is None:
            self._model_registry = ModelProviderRegistry(
                self._config_loader.load_models_config()
            )
        if self._kb_manager is None:
            self._kb_manager = KnowledgeBaseManager(
                config=self._config_loader.load_game_config(),
                model_registry=self._model_registry,
            )

    def show_config(self) -> None:
        game_config = self._config_loader.load_game_config()
        models_config = self._config_loader.load_models_config()
        agents_config = self._config_loader.load_agents_config()

        print("\n=== Game Configuration ===")
        print(f"RAG Provider: {game_config.rag.default_provider}")
        print(f"Data Directory: {game_config.directories.data_base_dir}")
        print(f"RAG Storage: {game_config.directories.rag_storage_dir}")
        print(f"Game Storage: {game_config.directories.game_storage_dir}")
        print(f"Max Turns: {game_config.game.max_turn_count}")
        print(f"Hint Limit: {game_config.game.default_hint_limit}")

        print("\n=== Models Configuration ===")
        print(f"Provider: {models_config.provider}")
        active_config = models_config.get_active_config()
        print(f"LLM Model: {active_config.llm_model_name}")
        print(f"Embedding Model: {active_config.embedding_model_name}")
        print(f"Embedding Dim: {active_config.embedding_dim}")

        print("\n=== Agents Configuration ===")
        print(f"DM Persona: {agents_config.dm.persona.name} ({agents_config.dm.persona.tone})")
        print(f"Judge Strictness: {agents_config.judge.strictness}")
        print(f"Hint Strategy: {agents_config.hint.strategy.initial_vagueness} vagueness")

    def list_puzzles(self) -> None:
        self._ensure_initialized()
        assert self._kb_manager is not None

        print("\n=== Discovering Puzzles ===")
        puzzles = self._kb_manager.discover_puzzles()

        if not puzzles:
            print("No puzzles found in data directory.")
            return

        print(f"Found {len(puzzles)} puzzle(s):\n")
        for puzzle_id, puzzle_dir in puzzles:
            kb_exists = self._kb_manager.kb_exists(puzzle_id)
            status = "[KB EXISTS]" if kb_exists else "[NO KB]"
            print(f"  - {puzzle_id}: {puzzle_dir} {status}")

    def list_kbs(self) -> None:
        self._ensure_initialized()
        assert self._kb_manager is not None

        print("\n=== Existing Knowledge Bases ===")
        puzzle_kbs = self._kb_manager.list_puzzle_kbs()

        if not puzzle_kbs:
            print("No puzzle knowledge bases found.")
            return

        print(f"Found {len(puzzle_kbs)} knowledge base(s):\n")
        for info in puzzle_kbs:
            print(f"  - {info.puzzle_id}")
            print(f"      KB ID: {info.kb_id}")
            print(f"      Title: {info.title}")
            print(f"      Type: {info.game_type}")
            print(f"      Documents: {info.document_count}")
            print()

    async def ensure_kbs(self, puzzle_ids: list[str] | None = None) -> None:
        self._ensure_initialized()
        assert self._kb_manager is not None

        print("\n=== Ensuring Knowledge Bases ===")
        puzzles = self._kb_manager.discover_puzzles()

        if puzzle_ids:
            puzzles = [(pid, pdir) for pid, pdir in puzzles if pid in puzzle_ids]

        if not puzzles:
            print("No puzzles to process.")
            return

        for puzzle_id, puzzle_dir in puzzles:
            try:
                kb_id = await self._kb_manager.ensure_puzzle_kb(puzzle_id, puzzle_dir)
                print(f"  [OK] {puzzle_id} -> {kb_id}")
            except Exception as e:
                print(f"  [ERROR] {puzzle_id}: {e}")
                logger.exception("Failed to create KB for %s", puzzle_id)

    async def health_check(self, puzzle_ids: list[str] | None = None) -> None:
        self._ensure_initialized()
        assert self._kb_manager is not None

        print("\n=== Health Check ===")
        puzzle_kbs = self._kb_manager.list_puzzle_kbs()

        if puzzle_ids:
            puzzle_kbs = [p for p in puzzle_kbs if p.puzzle_id in puzzle_ids]

        if not puzzle_kbs:
            print("No knowledge bases to check.")
            return

        for info in puzzle_kbs:
            try:
                status = await self._kb_manager.health_check(info.puzzle_id)
                status_str = "READY" if status.is_ready else "NOT READY"
                print(f"  [{status_str}] {info.puzzle_id} ({info.kb_id})")
                if status.details:
                    for key, value in status.details.items():
                        print(f"      {key}: {value}")
            except Exception as e:
                print(f"  [ERROR] {info.puzzle_id}: {e}")

    async def close(self) -> None:
        if self._kb_manager is not None:
            await self._kb_manager.close()


def main():
    parser = argparse.ArgumentParser(
        description="Game System CLI - Phase 1 utilities"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("config", help="Show current configuration")
    subparsers.add_parser("list-puzzles", help="List discovered puzzles")
    subparsers.add_parser("list-kbs", help="List existing knowledge bases")

    ensure_parser = subparsers.add_parser(
        "ensure-kbs", help="Ensure knowledge bases exist for puzzles"
    )
    ensure_parser.add_argument(
        "--puzzles",
        nargs="*",
        help="Specific puzzle IDs to process (default: all)",
    )

    health_parser = subparsers.add_parser(
        "health-check", help="Run health checks on knowledge bases"
    )
    health_parser.add_argument(
        "--puzzles",
        nargs="*",
        help="Specific puzzle IDs to check (default: all)",
    )

    subparsers.add_parser("init", help="Initialize all puzzles and run health checks")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    cli = GameCLI()

    try:
        if args.command == "config":
            cli.show_config()
        elif args.command == "list-puzzles":
            cli.list_puzzles()
        elif args.command == "list-kbs":
            cli.list_kbs()
        elif args.command == "ensure-kbs":
            asyncio.run(cli.ensure_kbs(args.puzzles))
        elif args.command == "health-check":
            asyncio.run(cli.health_check(args.puzzles))
        elif args.command == "init":
            async def init_all():
                await cli.ensure_kbs()
                await cli.health_check()
            asyncio.run(init_all())
    finally:
        asyncio.run(cli.close())


if __name__ == "__main__":
    main()
