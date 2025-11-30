"""Analytics service for game session analysis and reporting.

This module provides analytics aggregation, export functionality, and
reporting capabilities for puzzle and player statistics.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from game.memory.entities import (
    EventTag,
    MemorySummaryType,
    SessionEventRecord,
)

if TYPE_CHECKING:
    from game.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


@dataclass
class PuzzleAnalytics:
    puzzle_id: str
    total_sessions: int = 0
    total_questions: int = 0
    success_count: int = 0
    avg_questions_per_session: float = 0.0
    success_rate: float = 0.0
    avg_session_duration_seconds: Optional[float] = None
    avg_hints_used: float = 0.0
    common_session_lengths: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "puzzle_id": self.puzzle_id,
            "total_sessions": self.total_sessions,
            "total_questions": self.total_questions,
            "success_count": self.success_count,
            "avg_questions_per_session": round(self.avg_questions_per_session, 2),
            "success_rate": round(self.success_rate, 4),
            "avg_session_duration_seconds": self.avg_session_duration_seconds,
            "avg_hints_used": round(self.avg_hints_used, 2),
            "common_session_lengths": self.common_session_lengths,
        }


@dataclass
class PlayerAnalytics:
    player_id: str
    total_sessions: int = 0
    total_puzzles_attempted: int = 0
    puzzles_solved: int = 0
    total_questions_asked: int = 0
    total_hints_used: int = 0
    avg_questions_per_session: float = 0.0
    success_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "total_sessions": self.total_sessions,
            "total_puzzles_attempted": self.total_puzzles_attempted,
            "puzzles_solved": self.puzzles_solved,
            "total_questions_asked": self.total_questions_asked,
            "total_hints_used": self.total_hints_used,
            "avg_questions_per_session": round(self.avg_questions_per_session, 2),
            "success_rate": round(self.success_rate, 4),
        }


@dataclass
class GlobalAnalytics:
    total_sessions: int = 0
    total_players: int = 0
    total_puzzles: int = 0
    overall_success_rate: float = 0.0
    avg_questions_per_session: float = 0.0
    avg_hints_per_session: float = 0.0
    most_popular_puzzles: List[Dict[str, Any]] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_sessions": self.total_sessions,
            "total_players": self.total_players,
            "total_puzzles": self.total_puzzles,
            "overall_success_rate": round(self.overall_success_rate, 4),
            "avg_questions_per_session": round(self.avg_questions_per_session, 2),
            "avg_hints_per_session": round(self.avg_hints_per_session, 2),
            "most_popular_puzzles": self.most_popular_puzzles,
            "generated_at": self.generated_at.isoformat(),
        }


class AnalyticsService:
    def __init__(
        self,
        memory_manager: Optional[MemoryManager] = None,
        export_dir: Optional[Path] = None,
    ):
        self._memory_manager = memory_manager
        self._export_dir = export_dir or Path("game_storage/analytics")
        self._export_dir.mkdir(parents=True, exist_ok=True)

    def _categorize_session_length(self, question_count: int) -> str:
        if question_count <= 5:
            return "short"
        elif question_count <= 15:
            return "medium"
        elif question_count <= 30:
            return "long"
        return "very_long"

    def analyze_puzzle(
        self,
        puzzle_id: str,
        sessions_data: Optional[List[Dict[str, Any]]] = None,
    ) -> PuzzleAnalytics:
        if self._memory_manager:
            stats = self._memory_manager.get_puzzle_stats(puzzle_id)
            if stats:
                return PuzzleAnalytics(
                    puzzle_id=puzzle_id,
                    total_sessions=stats.get("total_sessions", 0),
                    total_questions=stats.get("total_questions", 0),
                    success_count=stats.get("success_count", 0),
                    avg_questions_per_session=stats.get("avg_questions", 0.0),
                    success_rate=stats.get("success_rate", 0.0),
                )

        if not sessions_data:
            return PuzzleAnalytics(puzzle_id=puzzle_id)

        total_sessions = len(sessions_data)
        total_questions = sum(s.get("question_count", 0) for s in sessions_data)
        success_count = sum(1 for s in sessions_data if s.get("success", False))
        total_hints = sum(s.get("hint_count", 0) for s in sessions_data)

        session_lengths: Dict[str, int] = {}
        durations = []

        for session in sessions_data:
            q_count = session.get("question_count", 0)
            length_cat = self._categorize_session_length(q_count)
            session_lengths[length_cat] = session_lengths.get(length_cat, 0) + 1

            if session.get("duration_seconds"):
                durations.append(session["duration_seconds"])

        avg_duration = sum(durations) / len(durations) if durations else None

        return PuzzleAnalytics(
            puzzle_id=puzzle_id,
            total_sessions=total_sessions,
            total_questions=total_questions,
            success_count=success_count,
            avg_questions_per_session=total_questions / total_sessions if total_sessions > 0 else 0,
            success_rate=success_count / total_sessions if total_sessions > 0 else 0,
            avg_session_duration_seconds=avg_duration,
            avg_hints_used=total_hints / total_sessions if total_sessions > 0 else 0,
            common_session_lengths=session_lengths,
        )

    def analyze_player(
        self,
        player_id: str,
        sessions_data: Optional[List[Dict[str, Any]]] = None,
    ) -> PlayerAnalytics:
        if not sessions_data:
            sessions_data = []

        if self._memory_manager and not sessions_data:
            results = self._memory_manager.retrieve_player_profile(player_id, limit=100)
            for result in results:
                if result.value.get("summary_type") == MemorySummaryType.SESSION_SUMMARY.value:
                    sessions_data.append(result.value)

        total_sessions = len(sessions_data)
        puzzles = set()
        solved = 0
        total_questions = 0
        total_hints = 0

        for session in sessions_data:
            puzzle_id = session.get("puzzle_id")
            if puzzle_id:
                puzzles.add(puzzle_id)

            if session.get("success", False):
                solved += 1

            total_questions += session.get("question_count", 0)
            total_hints += session.get("hint_count", 0)

        return PlayerAnalytics(
            player_id=player_id,
            total_sessions=total_sessions,
            total_puzzles_attempted=len(puzzles),
            puzzles_solved=solved,
            total_questions_asked=total_questions,
            total_hints_used=total_hints,
            avg_questions_per_session=total_questions / total_sessions if total_sessions > 0 else 0,
            success_rate=solved / total_sessions if total_sessions > 0 else 0,
        )

    def generate_global_analytics(
        self,
        puzzle_analytics: Optional[List[PuzzleAnalytics]] = None,
        player_analytics: Optional[List[PlayerAnalytics]] = None,
    ) -> GlobalAnalytics:
        puzzle_analytics = puzzle_analytics or []
        player_analytics = player_analytics or []

        total_sessions = sum(p.total_sessions for p in puzzle_analytics)
        total_success = sum(p.success_count for p in puzzle_analytics)
        total_questions = sum(p.total_questions for p in puzzle_analytics)
        total_hints = sum(
            p.avg_hints_used * p.total_sessions
            for p in puzzle_analytics
        )

        most_popular = sorted(
            puzzle_analytics,
            key=lambda p: p.total_sessions,
            reverse=True,
        )[:5]

        return GlobalAnalytics(
            total_sessions=total_sessions,
            total_players=len(player_analytics),
            total_puzzles=len(puzzle_analytics),
            overall_success_rate=total_success / total_sessions if total_sessions > 0 else 0,
            avg_questions_per_session=total_questions / total_sessions if total_sessions > 0 else 0,
            avg_hints_per_session=total_hints / total_sessions if total_sessions > 0 else 0,
            most_popular_puzzles=[
                {"puzzle_id": p.puzzle_id, "sessions": p.total_sessions}
                for p in most_popular
            ],
        )

    def export_to_json(
        self,
        data: Dict[str, Any],
        filename: str,
    ) -> Path:
        filepath = self._export_dir / f"{filename}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Exported analytics to %s", filepath)
        return filepath

    def export_to_csv(
        self,
        data: List[Dict[str, Any]],
        filename: str,
    ) -> Path:
        if not data:
            filepath = self._export_dir / f"{filename}.csv"
            filepath.write_text("")
            return filepath

        filepath = self._export_dir / f"{filename}.csv"
        fieldnames = list(data[0].keys())

        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                flat_row = {}
                for key, value in row.items():
                    if isinstance(value, (list, dict)):
                        flat_row[key] = json.dumps(value)
                    else:
                        flat_row[key] = value
                writer.writerow(flat_row)

        logger.info("Exported analytics to %s", filepath)
        return filepath

    def generate_full_report(
        self,
        puzzle_ids: Optional[List[str]] = None,
        player_ids: Optional[List[str]] = None,
        export_format: str = "json",
    ) -> Dict[str, Path]:
        puzzle_analytics = []
        if puzzle_ids:
            for pid in puzzle_ids:
                analytics = self.analyze_puzzle(pid)
                puzzle_analytics.append(analytics)

        player_analytics = []
        if player_ids:
            for pid in player_ids:
                analytics = self.analyze_player(pid)
                player_analytics.append(analytics)

        global_analytics = self.generate_global_analytics(
            puzzle_analytics=puzzle_analytics,
            player_analytics=player_analytics,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exported_files: Dict[str, Path] = {}

        report_data = {
            "global": global_analytics.to_dict(),
            "puzzles": [p.to_dict() for p in puzzle_analytics],
            "players": [p.to_dict() for p in player_analytics],
        }

        if export_format == "json":
            exported_files["report"] = self.export_to_json(
                report_data,
                f"full_report_{timestamp}",
            )
        else:
            exported_files["global"] = self.export_to_json(
                report_data["global"],
                f"global_analytics_{timestamp}",
            )
            if puzzle_analytics:
                exported_files["puzzles"] = self.export_to_csv(
                    [p.to_dict() for p in puzzle_analytics],
                    f"puzzle_analytics_{timestamp}",
                )
            if player_analytics:
                exported_files["players"] = self.export_to_csv(
                    [p.to_dict() for p in player_analytics],
                    f"player_analytics_{timestamp}",
                )

        return exported_files


class SessionEventAggregator:
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        self._memory_manager = memory_manager

    def aggregate_session_events(
        self,
        session_id: str,
    ) -> Dict[str, Any]:
        if not self._memory_manager:
            return {}

        events = self._memory_manager.get_session_history(session_id)
        if not events:
            return {"session_id": session_id, "event_count": 0}

        question_count = sum(1 for e in events if e.has_tag(EventTag.QUESTION))
        hint_count = sum(1 for e in events if e.has_tag(EventTag.HINT))
        hypothesis_count = sum(1 for e in events if e.has_tag(EventTag.HYPOTHESIS))

        has_success = any(
            e.has_tag(EventTag.FINAL_VERDICT) and "correct" in e.message.lower()
            for e in events
        )

        first_event = events[0] if events else None
        last_event = events[-1] if events else None
        duration_seconds = None
        if first_event and last_event:
            duration_seconds = (last_event.timestamp - first_event.timestamp).total_seconds()

        verdict_distribution: Dict[str, int] = {}
        for event in events:
            if event.has_tag(EventTag.ANSWER):
                verdict = event.metadata.get("verdict", "unknown")
                verdict_distribution[verdict] = verdict_distribution.get(verdict, 0) + 1

        return {
            "session_id": session_id,
            "event_count": len(events),
            "question_count": question_count,
            "hint_count": hint_count,
            "hypothesis_count": hypothesis_count,
            "success": has_success,
            "duration_seconds": duration_seconds,
            "verdict_distribution": verdict_distribution,
            "start_time": first_event.timestamp.isoformat() if first_event else None,
            "end_time": last_event.timestamp.isoformat() if last_event else None,
        }

    def aggregate_from_event_logs(
        self,
        events_dir: Path,
    ) -> List[Dict[str, Any]]:
        results = []
        if not events_dir.exists():
            return results

        for event_file in events_dir.glob("*.jsonl"):
            session_id = event_file.stem
            events = []

            try:
                with open(event_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            event_data = json.loads(line)
                            events.append(event_data)
            except Exception as e:
                logger.warning("Failed to read event log %s: %s", event_file, e)
                continue

            if not events:
                continue

            question_count = sum(
                1 for e in events
                if "question" in e.get("tags", [])
            )
            hint_count = sum(
                1 for e in events
                if "hint" in e.get("tags", [])
            )
            hypothesis_count = sum(
                1 for e in events
                if "hypothesis" in e.get("tags", [])
            )
            has_success = any(
                "final_verdict" in e.get("tags", [])
                and "correct" in e.get("message", "").lower()
                for e in events
            )

            results.append({
                "session_id": session_id,
                "event_count": len(events),
                "question_count": question_count,
                "hint_count": hint_count,
                "hypothesis_count": hypothesis_count,
                "success": has_success,
            })

        return results
