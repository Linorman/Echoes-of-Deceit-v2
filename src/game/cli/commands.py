"""CLI command handlers for the Turtle Soup game system.

This module provides individual command implementations that can be used
by the CLI entry point or tested independently.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from game.cli.app import GameCLIApp
from game.session_runner import GameSessionRunner, GameResponse
from game.domain.entities import GameSession, GameState, PuzzleSummary


@dataclass
class CommandResult:
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def list_puzzles(app: GameCLIApp) -> CommandResult:
    try:
        puzzles = app.list_puzzles()
        if not puzzles:
            return CommandResult(
                success=True,
                message="No puzzles found.",
                data={"puzzles": []},
            )

        puzzle_list = []
        for p in puzzles:
            puzzle_list.append({
                "id": p.id,
                "title": p.title,
                "description": p.description,
                "difficulty": p.difficulty,
                "tags": p.tags,
                "language": p.language,
            })

        return CommandResult(
            success=True,
            message=f"Found {len(puzzles)} puzzle(s).",
            data={"puzzles": puzzle_list},
        )
    except Exception as e:
        return CommandResult(
            success=False,
            message="Failed to list puzzles.",
            error=str(e),
        )


async def start_session(
    app: GameCLIApp,
    puzzle_id: str,
    player_name: str,
    options: Optional[Dict[str, Any]] = None,
) -> CommandResult:
    try:
        session = await app.create_session(puzzle_id, player_name, options)
        return CommandResult(
            success=True,
            message=f"Session created: {session.session_id}",
            data={
                "session_id": session.session_id,
                "puzzle_id": session.puzzle_id,
                "player": player_name,
                "state": session.state.value,
            },
        )
    except ValueError as e:
        return CommandResult(
            success=False,
            message=f"Failed to create session: {e}",
            error=str(e),
        )
    except Exception as e:
        return CommandResult(
            success=False,
            message="Failed to create session.",
            error=str(e),
        )


def get_session_status(app: GameCLIApp, session_id: str) -> CommandResult:
    try:
        status = app.get_session_status(session_id)
        return CommandResult(
            success=True,
            message="Session status retrieved.",
            data=status,
        )
    except ValueError as e:
        return CommandResult(
            success=False,
            message=f"Session not found: {session_id}",
            error=str(e),
        )
    except Exception as e:
        return CommandResult(
            success=False,
            message="Failed to get session status.",
            error=str(e),
        )


def list_sessions(
    app: GameCLIApp,
    state_filter: Optional[str] = None,
    puzzle_id: Optional[str] = None,
    player_id: Optional[str] = None,
) -> CommandResult:
    try:
        state = None
        if state_filter:
            try:
                state = GameState(state_filter)
            except ValueError:
                return CommandResult(
                    success=False,
                    message=f"Invalid state: {state_filter}",
                    error=f"Valid states: {', '.join(s.value for s in GameState)}",
                )

        sessions = app.list_sessions(
            state_filter=state,
            puzzle_id=puzzle_id,
            player_id=player_id,
        )

        session_list = []
        for s in sessions:
            session_list.append({
                "session_id": s.session_id,
                "puzzle_id": s.puzzle_id,
                "state": s.state.value,
                "turn_count": s.turn_count,
                "hint_count": s.hint_count,
                "created_at": s.created_at.isoformat(),
            })

        return CommandResult(
            success=True,
            message=f"Found {len(sessions)} session(s).",
            data={"sessions": session_list},
        )
    except Exception as e:
        return CommandResult(
            success=False,
            message="Failed to list sessions.",
            error=str(e),
        )


async def play_session(
    app: GameCLIApp,
    session: GameSession,
    input_handler: Callable[[], str],
    output_handler: Callable[[str], None],
) -> CommandResult:
    try:
        runner = app.create_runner(session)

        if session.state == GameState.LOBBY:
            response = runner.start_game()
            output_handler(response.message)
            if response.game_over:
                return CommandResult(
                    success=True,
                    message="Game ended.",
                    data={"final_response": response.message},
                )

        while runner.is_active:
            try:
                user_input = input_handler()
            except (EOFError, KeyboardInterrupt):
                return CommandResult(
                    success=True,
                    message="Game interrupted by user.",
                    data={"interrupted": True},
                )

            if not user_input:
                continue

            response = await runner.process_player_input(user_input)
            output_handler(response.message)

            if response.game_over:
                return CommandResult(
                    success=True,
                    message="Game completed.",
                    data={
                        "final_response": response.message,
                        "score": runner.session.score,
                    },
                )

        return CommandResult(
            success=True,
            message="Game session ended.",
            data={"state": runner.session.state.value},
        )

    except Exception as e:
        return CommandResult(
            success=False,
            message="Error during gameplay.",
            error=str(e),
        )


async def resume_session(
    app: GameCLIApp,
    session_id: str,
    input_handler: Callable[[], str],
    output_handler: Callable[[str], None],
) -> CommandResult:
    try:
        session = app.get_session(session_id)
    except ValueError as e:
        return CommandResult(
            success=False,
            message=f"Session not found: {session_id}",
            error=str(e),
        )

    if session.state != GameState.IN_PROGRESS:
        return CommandResult(
            success=False,
            message=f"Session is not active (state: {session.state.value})",
            error="Can only resume sessions in 'in_progress' state.",
        )

    output_handler("Game resumed. Type /help for commands.")
    return await play_session(app, session, input_handler, output_handler)
