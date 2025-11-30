"""LLM-driven session summarization for player memory enhancement.

This module provides session summarization services that use LLM to generate
rich summaries of player sessions, including reasoning style, common mistakes,
and notable strengths.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from game.memory.entities import (
    EventTag,
    MemorySummaryType,
    SessionEventRecord,
)

if TYPE_CHECKING:
    from config import SummarizationConfig
    from models.base import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class SessionSummaryResult:
    session_id: str
    player_id: Optional[str]
    puzzle_id: Optional[str]
    summary: str
    reasoning_style: Optional[str] = None
    common_mistakes: Optional[List[str]] = None
    notable_strengths: Optional[List[str]] = None
    statistics: Optional[Dict[str, Any]] = None
    generated_at: Optional[datetime] = None

    def __post_init__(self):
        if self.generated_at is None:
            self.generated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "player_id": self.player_id,
            "puzzle_id": self.puzzle_id,
            "summary": self.summary,
            "reasoning_style": self.reasoning_style,
            "common_mistakes": self.common_mistakes,
            "notable_strengths": self.notable_strengths,
            "statistics": self.statistics,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
        }


class SessionSummarizer:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config: Optional[SummarizationConfig] = None,
    ):
        self._llm_client = llm_client
        self._config = config

    def _extract_statistics(
        self,
        events: List[SessionEventRecord],
    ) -> Dict[str, Any]:
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

        return {
            "total_events": len(events),
            "question_count": question_count,
            "hint_count": hint_count,
            "hypothesis_count": hypothesis_count,
            "success": has_success,
            "duration_seconds": duration_seconds,
        }

    def _build_conversation_text(
        self,
        events: List[SessionEventRecord],
    ) -> str:
        lines = []
        for event in events:
            role = event.role.upper()
            tags = ", ".join(event.tags) if event.tags else ""
            tag_str = f" [{tags}]" if tags else ""
            lines.append(f"{role}{tag_str}: {event.message}")
        return "\n".join(lines)

    def _build_summary_prompt(
        self,
        events: List[SessionEventRecord],
        stats: Dict[str, Any],
    ) -> str:
        conversation = self._build_conversation_text(events)
        use_llm = self._config.use_llm if self._config else True
        include_reasoning = self._config.include_reasoning_style if self._config else True
        include_mistakes = self._config.include_common_mistakes if self._config else True
        include_strengths = self._config.include_notable_strengths if self._config else True
        max_length = self._config.max_summary_length if self._config else 500

        sections = []
        if include_reasoning:
            sections.append("- Reasoning style (e.g., methodical, intuitive, random)")
        if include_mistakes:
            sections.append("- Common mistakes or ineffective strategies")
        if include_strengths:
            sections.append("- Notable strengths or good approaches")

        sections_text = "\n".join(sections)

        return f"""Analyze this Turtle Soup puzzle game session and provide a structured summary.

SESSION STATISTICS:
- Total events: {stats['total_events']}
- Questions asked: {stats['question_count']}
- Hints used: {stats['hint_count']}
- Hypotheses proposed: {stats['hypothesis_count']}
- Outcome: {"SUCCESS" if stats['success'] else "DID NOT COMPLETE"}

CONVERSATION:
{conversation}

Please provide:
1. A brief overall summary (2-3 sentences)
2. Analysis of:
{sections_text}

FORMAT YOUR RESPONSE AS:
SUMMARY: [Your summary here]
REASONING_STYLE: [One-line description]
MISTAKES: [Comma-separated list or "None observed"]
STRENGTHS: [Comma-separated list or "None observed"]

Keep the total response under {max_length} characters."""

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        result = {
            "summary": "",
            "reasoning_style": None,
            "common_mistakes": None,
            "notable_strengths": None,
        }

        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.upper().startswith("SUMMARY:"):
                result["summary"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("REASONING_STYLE:"):
                result["reasoning_style"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("MISTAKES:"):
                mistakes_text = line.split(":", 1)[1].strip()
                if mistakes_text.lower() not in ("none observed", "none", "n/a"):
                    result["common_mistakes"] = [
                        m.strip() for m in mistakes_text.split(",") if m.strip()
                    ]
            elif line.upper().startswith("STRENGTHS:"):
                strengths_text = line.split(":", 1)[1].strip()
                if strengths_text.lower() not in ("none observed", "none", "n/a"):
                    result["notable_strengths"] = [
                        s.strip() for s in strengths_text.split(",") if s.strip()
                    ]

        if not result["summary"]:
            result["summary"] = response[:500] if len(response) > 500 else response

        return result

    def _generate_fallback_summary(
        self,
        events: List[SessionEventRecord],
        stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        summary_parts = [
            f"Session with {stats['total_events']} events.",
            f"Asked {stats['question_count']} questions.",
        ]

        if stats['hint_count'] > 0:
            summary_parts.append(f"Used {stats['hint_count']} hints.")

        if stats['success']:
            summary_parts.append("Successfully solved the puzzle.")
        else:
            summary_parts.append("Did not complete the puzzle.")

        reasoning_style = "Unknown"
        if stats['question_count'] > 10:
            reasoning_style = "Methodical - asks many questions"
        elif stats['question_count'] < 3 and stats['hypothesis_count'] > 0:
            reasoning_style = "Intuitive - jumps to conclusions quickly"
        elif stats['question_count'] >= 3:
            reasoning_style = "Balanced approach"

        return {
            "summary": " ".join(summary_parts),
            "reasoning_style": reasoning_style,
            "common_mistakes": None,
            "notable_strengths": None,
        }

    async def summarize_session(
        self,
        session_id: str,
        events: List[SessionEventRecord],
        player_id: Optional[str] = None,
        puzzle_id: Optional[str] = None,
    ) -> SessionSummaryResult:
        if not events:
            return SessionSummaryResult(
                session_id=session_id,
                player_id=player_id,
                puzzle_id=puzzle_id,
                summary="No events recorded in this session.",
                statistics={"total_events": 0},
            )

        stats = self._extract_statistics(events)

        use_llm = self._config.use_llm if self._config else True
        if use_llm and self._llm_client:
            try:
                prompt = self._build_summary_prompt(events, stats)
                response = await self._llm_client.agenerate(prompt)
                parsed = self._parse_llm_response(response)
                logger.info("Generated LLM summary for session %s", session_id)
            except Exception as e:
                logger.warning("LLM summarization failed, using fallback: %s", e)
                parsed = self._generate_fallback_summary(events, stats)
        else:
            parsed = self._generate_fallback_summary(events, stats)

        return SessionSummaryResult(
            session_id=session_id,
            player_id=player_id,
            puzzle_id=puzzle_id,
            summary=parsed["summary"],
            reasoning_style=parsed.get("reasoning_style"),
            common_mistakes=parsed.get("common_mistakes"),
            notable_strengths=parsed.get("notable_strengths"),
            statistics=stats,
        )

    def summarize_session_sync(
        self,
        session_id: str,
        events: List[SessionEventRecord],
        player_id: Optional[str] = None,
        puzzle_id: Optional[str] = None,
    ) -> SessionSummaryResult:
        if not events:
            return SessionSummaryResult(
                session_id=session_id,
                player_id=player_id,
                puzzle_id=puzzle_id,
                summary="No events recorded in this session.",
                statistics={"total_events": 0},
            )

        stats = self._extract_statistics(events)

        use_llm = self._config.use_llm if self._config else True
        if use_llm and self._llm_client:
            try:
                prompt = self._build_summary_prompt(events, stats)
                response = self._llm_client.generate(prompt)
                parsed = self._parse_llm_response(response)
                logger.info("Generated LLM summary for session %s", session_id)
            except Exception as e:
                logger.warning("LLM summarization failed, using fallback: %s", e)
                parsed = self._generate_fallback_summary(events, stats)
        else:
            parsed = self._generate_fallback_summary(events, stats)

        return SessionSummaryResult(
            session_id=session_id,
            player_id=player_id,
            puzzle_id=puzzle_id,
            summary=parsed["summary"],
            reasoning_style=parsed.get("reasoning_style"),
            common_mistakes=parsed.get("common_mistakes"),
            notable_strengths=parsed.get("notable_strengths"),
            statistics=stats,
        )


class PlayerProfileGenerator:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config: Optional[SummarizationConfig] = None,
    ):
        self._llm_client = llm_client
        self._config = config

    def _build_profile_prompt(
        self,
        session_summaries: List[SessionSummaryResult],
        existing_profile: Optional[str] = None,
    ) -> str:
        summaries_text = []
        for s in session_summaries[-5:]:
            parts = [f"Session {s.session_id}:"]
            parts.append(f"  Summary: {s.summary}")
            if s.reasoning_style:
                parts.append(f"  Reasoning: {s.reasoning_style}")
            if s.statistics:
                parts.append(f"  Questions: {s.statistics.get('question_count', 0)}")
                parts.append(f"  Outcome: {'Success' if s.statistics.get('success') else 'Incomplete'}")
            summaries_text.append("\n".join(parts))

        summaries_block = "\n\n".join(summaries_text)

        existing_section = ""
        if existing_profile:
            existing_section = f"\nEXISTING PROFILE:\n{existing_profile}\n"

        return f"""Generate an updated player profile based on their game session history.
{existing_section}
RECENT SESSION SUMMARIES:
{summaries_block}

Create a comprehensive player profile that includes:
1. Overall skill level and experience
2. Dominant reasoning style
3. Areas for improvement
4. Recommended hint strategy

FORMAT:
SKILL_LEVEL: [Beginner/Intermediate/Advanced]
REASONING_STYLE: [Description]
IMPROVEMENT_AREAS: [Comma-separated list]
HINT_STRATEGY: [Direct/Moderate/Subtle]
PROFILE_SUMMARY: [2-3 sentence summary]"""

    def _parse_profile_response(self, response: str) -> Dict[str, Any]:
        result = {
            "skill_level": "Intermediate",
            "reasoning_style": "Unknown",
            "improvement_areas": [],
            "hint_strategy": "Moderate",
            "profile_summary": "",
        }

        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.upper().startswith("SKILL_LEVEL:"):
                result["skill_level"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("REASONING_STYLE:"):
                result["reasoning_style"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("IMPROVEMENT_AREAS:"):
                areas = line.split(":", 1)[1].strip()
                result["improvement_areas"] = [a.strip() for a in areas.split(",") if a.strip()]
            elif line.upper().startswith("HINT_STRATEGY:"):
                result["hint_strategy"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("PROFILE_SUMMARY:"):
                result["profile_summary"] = line.split(":", 1)[1].strip()

        if not result["profile_summary"]:
            result["profile_summary"] = response[:300]

        return result

    async def generate_profile(
        self,
        player_id: str,
        session_summaries: List[SessionSummaryResult],
        existing_profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not session_summaries:
            return {
                "player_id": player_id,
                "skill_level": "Unknown",
                "reasoning_style": "Unknown",
                "improvement_areas": [],
                "hint_strategy": "Moderate",
                "profile_summary": "No session data available.",
            }

        if self._llm_client:
            try:
                prompt = self._build_profile_prompt(session_summaries, existing_profile)
                response = await self._llm_client.agenerate(prompt)
                result = self._parse_profile_response(response)
                result["player_id"] = player_id
                return result
            except Exception as e:
                logger.warning("LLM profile generation failed: %s", e)

        total_sessions = len(session_summaries)
        success_count = sum(
            1 for s in session_summaries
            if s.statistics and s.statistics.get("success")
        )
        avg_questions = sum(
            s.statistics.get("question_count", 0)
            for s in session_summaries if s.statistics
        ) / max(total_sessions, 1)

        skill_level = "Beginner"
        if success_count > total_sessions * 0.7:
            skill_level = "Advanced"
        elif success_count > total_sessions * 0.3:
            skill_level = "Intermediate"

        return {
            "player_id": player_id,
            "skill_level": skill_level,
            "reasoning_style": session_summaries[-1].reasoning_style if session_summaries else "Unknown",
            "improvement_areas": [],
            "hint_strategy": "Moderate",
            "profile_summary": f"Player with {total_sessions} sessions, {success_count} successes, avg {avg_questions:.1f} questions per session.",
        }
