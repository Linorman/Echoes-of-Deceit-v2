"""CLI output formatters for the Turtle Soup game system.

This module provides consistent formatting for CLI output,
supporting both plain text and JSON output modes.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Protocol

from game.cli.commands import CommandResult
from game.domain.entities import GameSession, PuzzleSummary


class OutputFormatter(Protocol):
    def format_result(self, result: CommandResult) -> str:
        ...

    def format_puzzles(self, puzzles: List[Dict[str, Any]]) -> str:
        ...

    def format_sessions(self, sessions: List[Dict[str, Any]]) -> str:
        ...

    def format_status(self, status: Dict[str, Any]) -> str:
        ...

    def format_error(self, message: str, error: Optional[str] = None) -> str:
        ...


class TextFormatter:
    def format_result(self, result: CommandResult) -> str:
        if not result.success:
            return self.format_error(result.message, result.error)
        return result.message

    def format_puzzles(self, puzzles: List[Dict[str, Any]]) -> str:
        if not puzzles:
            return "No puzzles found."

        lines = [f"\n{'='*50}", "Available Puzzles", f"{'='*50}\n"]

        for i, p in enumerate(puzzles, 1):
            lines.append(f"{i}. {p['id']}")
            lines.append(f"   Title: {p['title']}")
            if p.get('description'):
                lines.append(f"   Description: {p['description'][:60]}...")
            lines.append(f"   Difficulty: {p.get('difficulty') or 'unspecified'}")
            tags = ", ".join(p.get('tags', [])) or "none"
            lines.append(f"   Tags: {tags}")
            lines.append("")

        return "\n".join(lines)

    def format_sessions(self, sessions: List[Dict[str, Any]]) -> str:
        if not sessions:
            return "No sessions found."

        lines = [f"\n{'='*50}", "Game Sessions", f"{'='*50}\n"]

        for s in sessions:
            lines.append(f"Session: {s['session_id']}")
            lines.append(f"  Puzzle: {s['puzzle_id']}")
            lines.append(f"  State: {s['state']}")
            lines.append(f"  Turns: {s['turn_count']}")
            lines.append(f"  Hints: {s['hint_count']}")
            lines.append(f"  Created: {s['created_at']}")
            lines.append("")

        return "\n".join(lines)

    def format_status(self, status: Dict[str, Any]) -> str:
        lines = [
            f"\n{'='*50}",
            "Session Status",
            f"{'='*50}\n",
            f"Session ID: {status['session_id']}",
            f"Puzzle: {status['puzzle_title']} ({status['puzzle_id']})",
            f"State: {status['state']}",
            f"Questions asked: {status['question_count']}",
            f"Hints used: {status['hint_count']}/{status['max_hints']}",
        ]

        if status.get('score') is not None:
            lines.append(f"Score: {status['score']}")

        lines.extend([
            f"Created: {status['created_at']}",
            f"Updated: {status['updated_at']}",
        ])

        if status.get('completed_at'):
            lines.append(f"Completed: {status['completed_at']}")

        return "\n".join(lines)

    def format_error(self, message: str, error: Optional[str] = None) -> str:
        lines = [f"\nError: {message}"]
        if error:
            lines.append(f"Details: {error}")
        return "\n".join(lines)


class JsonFormatter:
    def format_result(self, result: CommandResult) -> str:
        return json.dumps({
            "success": result.success,
            "message": result.message,
            "data": result.data,
            "error": result.error,
        }, indent=2, default=str)

    def format_puzzles(self, puzzles: List[Dict[str, Any]]) -> str:
        return json.dumps({"puzzles": puzzles}, indent=2, default=str)

    def format_sessions(self, sessions: List[Dict[str, Any]]) -> str:
        return json.dumps({"sessions": sessions}, indent=2, default=str)

    def format_status(self, status: Dict[str, Any]) -> str:
        return json.dumps({"status": status}, indent=2, default=str)

    def format_error(self, message: str, error: Optional[str] = None) -> str:
        return json.dumps({
            "success": False,
            "message": message,
            "error": error,
        }, indent=2)


def get_formatter(json_mode: bool = False) -> OutputFormatter:
    return JsonFormatter() if json_mode else TextFormatter()
