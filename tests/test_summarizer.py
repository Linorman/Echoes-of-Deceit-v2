"""Tests for Session Summarizer and Player Profile Generator."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from game.memory.summarizer import (
    SessionSummarizer,
    SessionSummaryResult,
    PlayerProfileGenerator,
)
from game.memory.entities import (
    EventTag,
    SessionEventRecord,
)


class MockLLMClient:
    def __init__(self, response: str = ""):
        self._response = response

    async def agenerate(self, prompt: str) -> str:
        return self._response

    def generate(self, prompt: str) -> str:
        return self._response


class MockSummarizationConfig:
    use_llm = True
    include_reasoning_style = True
    include_common_mistakes = True
    include_notable_strengths = True
    max_summary_length = 500


@pytest.fixture
def sample_events():
    return [
        SessionEventRecord(
            session_id="session1",
            turn_index=0,
            role="dm",
            message="Welcome to the puzzle!",
            tags=["intro"],
        ),
        SessionEventRecord(
            session_id="session1",
            turn_index=1,
            role="player",
            message="Is the man dead?",
            tags=["question"],
        ),
        SessionEventRecord(
            session_id="session1",
            turn_index=2,
            role="dm",
            message="Yes",
            tags=["answer"],
        ),
        SessionEventRecord(
            session_id="session1",
            turn_index=3,
            role="player",
            message="Did he die of natural causes?",
            tags=["question"],
        ),
        SessionEventRecord(
            session_id="session1",
            turn_index=4,
            role="dm",
            message="No",
            tags=["answer"],
        ),
        SessionEventRecord(
            session_id="session1",
            turn_index=5,
            role="dm",
            message="Here's a hint",
            tags=["hint"],
        ),
        SessionEventRecord(
            session_id="session1",
            turn_index=6,
            role="player",
            message="I think the man was murdered",
            tags=["hypothesis", "final_verdict"],
        ),
    ]


class TestSessionSummaryResult:
    def test_to_dict(self):
        result = SessionSummaryResult(
            session_id="s1",
            player_id="p1",
            puzzle_id="puzzle1",
            summary="Test summary",
            reasoning_style="Methodical",
            common_mistakes=["Too broad questions"],
            notable_strengths=["Good deduction"],
            statistics={"question_count": 5},
        )

        data = result.to_dict()

        assert data["session_id"] == "s1"
        assert data["player_id"] == "p1"
        assert data["summary"] == "Test summary"
        assert data["reasoning_style"] == "Methodical"
        assert len(data["common_mistakes"]) == 1
        assert len(data["notable_strengths"]) == 1

    def test_generated_at_auto_set(self):
        result = SessionSummaryResult(
            session_id="s1",
            player_id=None,
            puzzle_id=None,
            summary="Test",
        )

        assert result.generated_at is not None


class TestSessionSummarizer:
    def test_extract_statistics(self, sample_events):
        summarizer = SessionSummarizer()
        stats = summarizer._extract_statistics(sample_events)

        assert stats["total_events"] == 7
        assert stats["question_count"] == 2
        assert stats["hint_count"] == 1
        assert stats["hypothesis_count"] == 1

    def test_extract_statistics_empty_events(self):
        summarizer = SessionSummarizer()
        stats = summarizer._extract_statistics([])

        assert stats["total_events"] == 0
        assert stats["question_count"] == 0

    def test_build_conversation_text(self, sample_events):
        summarizer = SessionSummarizer()
        text = summarizer._build_conversation_text(sample_events)

        assert "PLAYER" in text
        assert "DM" in text
        assert "Is the man dead?" in text

    def test_parse_llm_response(self):
        summarizer = SessionSummarizer()
        response = """SUMMARY: The player used a methodical approach.
REASONING_STYLE: Analytical and systematic
MISTAKES: Too many broad questions, missed obvious clues
STRENGTHS: Good final deduction, persistent"""

        result = summarizer._parse_llm_response(response)

        assert "methodical" in result["summary"]
        assert "Analytical" in result["reasoning_style"]
        assert len(result["common_mistakes"]) == 2
        assert len(result["notable_strengths"]) == 2

    def test_parse_llm_response_no_mistakes(self):
        summarizer = SessionSummarizer()
        response = """SUMMARY: Good session.
REASONING_STYLE: Intuitive
MISTAKES: None observed
STRENGTHS: Quick thinking"""

        result = summarizer._parse_llm_response(response)

        assert result["common_mistakes"] is None
        assert len(result["notable_strengths"]) == 1

    def test_generate_fallback_summary(self, sample_events):
        summarizer = SessionSummarizer()
        stats = summarizer._extract_statistics(sample_events)
        result = summarizer._generate_fallback_summary(sample_events, stats)

        assert "7 events" in result["summary"]
        assert "2 questions" in result["summary"]
        assert result["reasoning_style"] is not None

    @pytest.mark.asyncio
    async def test_summarize_session_empty(self):
        summarizer = SessionSummarizer()
        result = await summarizer.summarize_session("s1", [])

        assert "No events recorded" in result.summary
        assert result.statistics["total_events"] == 0

    @pytest.mark.asyncio
    async def test_summarize_session_with_llm(self, sample_events):
        mock_llm = MockLLMClient("""SUMMARY: Player used analytical approach.
REASONING_STYLE: Methodical
MISTAKES: None observed
STRENGTHS: Good deduction""")

        config = MockSummarizationConfig()
        summarizer = SessionSummarizer(llm_client=mock_llm, config=config)

        result = await summarizer.summarize_session(
            "session1",
            sample_events,
            player_id="player1",
            puzzle_id="puzzle1",
        )

        assert "analytical" in result.summary.lower()
        assert result.reasoning_style == "Methodical"

    @pytest.mark.asyncio
    async def test_summarize_session_fallback_on_error(self, sample_events):
        class FailingLLM:
            async def agenerate(self, prompt: str) -> str:
                raise Exception("LLM error")

        config = MockSummarizationConfig()
        summarizer = SessionSummarizer(llm_client=FailingLLM(), config=config)

        result = await summarizer.summarize_session("session1", sample_events)

        assert result.summary is not None
        assert result.statistics["total_events"] == 7

    def test_summarize_session_sync(self, sample_events):
        summarizer = SessionSummarizer()
        result = summarizer.summarize_session_sync("session1", sample_events)

        assert result.session_id == "session1"
        assert result.summary is not None


class TestPlayerProfileGenerator:
    @pytest.fixture
    def sample_summaries(self):
        return [
            SessionSummaryResult(
                session_id="s1",
                player_id="p1",
                puzzle_id="puzzle1",
                summary="Good analytical session",
                reasoning_style="Methodical",
                statistics={"question_count": 5, "success": True},
            ),
            SessionSummaryResult(
                session_id="s2",
                player_id="p1",
                puzzle_id="puzzle2",
                summary="Quick intuitive solving",
                reasoning_style="Intuitive",
                statistics={"question_count": 3, "success": True},
            ),
        ]

    @pytest.mark.asyncio
    async def test_generate_profile_no_summaries(self):
        generator = PlayerProfileGenerator()
        result = await generator.generate_profile("p1", [])

        assert result["player_id"] == "p1"
        assert result["skill_level"] == "Unknown"

    @pytest.mark.asyncio
    async def test_generate_profile_with_llm(self, sample_summaries):
        mock_llm = MockLLMClient("""SKILL_LEVEL: Advanced
REASONING_STYLE: Methodical and intuitive mix
IMPROVEMENT_AREAS: Try broader questions
HINT_STRATEGY: Subtle
PROFILE_SUMMARY: Experienced player with good deduction skills.""")

        generator = PlayerProfileGenerator(llm_client=mock_llm)
        result = await generator.generate_profile("p1", sample_summaries)

        assert result["skill_level"] == "Advanced"
        assert "Methodical" in result["reasoning_style"]
        assert result["hint_strategy"] == "Subtle"

    @pytest.mark.asyncio
    async def test_generate_profile_fallback(self, sample_summaries):
        generator = PlayerProfileGenerator()
        result = await generator.generate_profile("p1", sample_summaries)

        assert result["player_id"] == "p1"
        assert result["skill_level"] in ["Beginner", "Intermediate", "Advanced"]
        assert "2 sessions" in result["profile_summary"]
