"""Main entry point for the Turtle Soup Game CLI.

Usage:
    python -m game.cli.main list-puzzles
    python -m game.cli.main start-session --puzzle <id> --player <name>
    python -m game.cli.main play --session <id>
    python -m game.cli.main status --session <id>
    python -m game.cli.main sessions [--state <state>]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from game.cli.app import GameCLIApp
from game.cli.commands import (
    list_puzzles,
    start_session,
    play_session,
    resume_session,
    get_session_status,
    list_sessions,
)
from game.cli.formatters import get_formatter, TextFormatter, JsonFormatter
from game.domain.entities import GameState


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="turtle-soup",
        description="Turtle Soup Puzzle Game CLI - Play situation puzzles with AI",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser(
        "list-puzzles",
        help="List all available puzzles",
    )

    start_parser = subparsers.add_parser(
        "start-session",
        help="Start a new game session",
    )
    start_parser.add_argument(
        "--puzzle", "-p",
        required=True,
        help="Puzzle ID to play",
    )
    start_parser.add_argument(
        "--player", "-n",
        default="player",
        help="Player name (default: player)",
    )

    play_parser = subparsers.add_parser(
        "play",
        help="Play an interactive game session",
    )
    play_parser.add_argument(
        "--puzzle", "-p",
        help="Puzzle ID for a new game",
    )
    play_parser.add_argument(
        "--session", "-s",
        help="Session ID to resume",
    )
    play_parser.add_argument(
        "--player", "-n",
        default="player",
        help="Player name for new games (default: player)",
    )

    status_parser = subparsers.add_parser(
        "status",
        help="Get session status",
    )
    status_parser.add_argument(
        "--session", "-s",
        required=True,
        help="Session ID to check",
    )

    sessions_parser = subparsers.add_parser(
        "sessions",
        help="List game sessions",
    )
    sessions_parser.add_argument(
        "--state",
        choices=["lobby", "in_progress", "completed", "aborted"],
        help="Filter by state",
    )
    sessions_parser.add_argument(
        "--puzzle", "-p",
        help="Filter by puzzle ID",
    )
    sessions_parser.add_argument(
        "--player", "-n",
        help="Filter by player ID",
    )

    return parser


class InteractiveCLI:
    def __init__(self, app: GameCLIApp, formatter: TextFormatter | JsonFormatter):
        self._app = app
        self._formatter = formatter

    def get_input(self) -> str:
        return input("\nYou: ").strip()

    def show_output(self, message: str) -> None:
        print(f"\n{message}")

    async def run_list_puzzles(self) -> int:
        result = list_puzzles(self._app)
        if result.success and result.data:
            print(self._formatter.format_puzzles(result.data.get("puzzles", [])))
        else:
            print(self._formatter.format_error(result.message, result.error))
        return 0 if result.success else 1

    async def run_start_session(self, puzzle_id: str, player_name: str) -> int:
        result = await start_session(self._app, puzzle_id, player_name)
        print(self._formatter.format_result(result))
        return 0 if result.success else 1

    async def run_play(
        self,
        puzzle_id: str | None = None,
        session_id: str | None = None,
        player_name: str = "player",
    ) -> int:
        if session_id:
            result = await resume_session(
                self._app,
                session_id,
                self.get_input,
                self.show_output,
            )
        elif puzzle_id:
            session_result = await start_session(self._app, puzzle_id, player_name)
            if not session_result.success:
                print(self._formatter.format_error(
                    session_result.message,
                    session_result.error
                ))
                return 1

            session = self._app.get_session(session_result.data["session_id"])
            result = await play_session(
                self._app,
                session,
                self.get_input,
                self.show_output,
            )
        else:
            print(self._formatter.format_error(
                "Must specify either --puzzle or --session"
            ))
            return 1

        if result.success:
            self.show_output("\nThanks for playing!")
        return 0 if result.success else 1

    async def run_status(self, session_id: str) -> int:
        result = get_session_status(self._app, session_id)
        if result.success and result.data:
            print(self._formatter.format_status(result.data))
        else:
            print(self._formatter.format_error(result.message, result.error))
        return 0 if result.success else 1

    async def run_sessions(
        self,
        state: str | None = None,
        puzzle_id: str | None = None,
        player_id: str | None = None,
    ) -> int:
        result = list_sessions(self._app, state, puzzle_id, player_id)
        if result.success and result.data:
            print(self._formatter.format_sessions(result.data.get("sessions", [])))
        else:
            print(self._formatter.format_error(result.message, result.error))
        return 0 if result.success else 1


async def async_main(args: argparse.Namespace) -> int:
    formatter = get_formatter(args.json)
    app = GameCLIApp()
    cli = InteractiveCLI(app, formatter)

    try:
        if args.command == "list-puzzles":
            return await cli.run_list_puzzles()

        elif args.command == "start-session":
            return await cli.run_start_session(args.puzzle, args.player)

        elif args.command == "play":
            return await cli.run_play(
                puzzle_id=args.puzzle,
                session_id=args.session,
                player_name=args.player,
            )

        elif args.command == "status":
            return await cli.run_status(args.session)

        elif args.command == "sessions":
            return await cli.run_sessions(
                state=args.state,
                puzzle_id=args.puzzle,
                player_id=args.player,
            )

        else:
            return 1

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        await app.close()


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    setup_logging(args.verbose)
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())
