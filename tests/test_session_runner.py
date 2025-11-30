"""Tests for GameSessionRunner."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from game.session_runner import (
    GameSessionRunner,
    GameResponse,
    MessageType,
    DMVerdict,
    HypothesisVerdict,
)
from game.domain.entities import (
    AgentRole,
    GameSession,
    GameState,
    Puzzle,
    PuzzleConstraints,
    SessionConfig,
)
from game.memory.entities import EventTag
from config.models import (
    AgentsConfig,
    DMConfig,
    DMPersonaConfig,
    DMBehaviorConfig,
    JudgeConfig,
    QuestionResponseFormatConfig,
    HintConfig,
    HintStrategyConfig,
    HintTimingConfig,
)


@pytest.fixture
def sample_puzzle():
    return Puzzle(
        id="test_puzzle",
        title="Test Puzzle",
        description="A test puzzle for unit tests",
        puzzle_statement="A man walks into a bar and asks for water. The bartender points a gun at him. The man says 'thank you' and leaves. Why?",
        answer="The man had hiccups. The bartender scared them away by pointing a gun at him.",
        hints=[
            "Think about why someone might want water",
            "The gun wasn't meant to hurt him",
            "The man's problem was solved by being startled",
        ],
        constraints=PuzzleConstraints(max_hints=3),
        tags=["classic", "test"],
    )


@pytest.fixture
def sample_session(sample_puzzle):
    return GameSession(
        puzzle_id=sample_puzzle.id,
        player_ids=["test_player"],
        kb_id="game_test_puzzle",
        config=SessionConfig(),
    )


@pytest.fixture
def agents_config():
    return AgentsConfig(
        dm=DMConfig(
            persona=DMPersonaConfig(name="Narrator", tone="mysterious"),
            behavior=DMBehaviorConfig(encourage_player=True),
        ),
        judge=JudgeConfig(
            strictness="moderate",
            response_format=QuestionResponseFormatConfig(
                include_explanation=True,
                max_explanation_length=100,
            ),
        ),
        hint=HintConfig(
            strategy=HintStrategyConfig(initial_vagueness="high"),
            timing=HintTimingConfig(),
        ),
    )


@pytest.fixture
def mock_kb_manager():
    manager = MagicMock()
    manager.query_full = AsyncMock(return_value=MagicMock(answer="Context info"))
    return manager


@pytest.fixture
def mock_memory_manager():
    manager = MagicMock()
    manager.append_session_event = MagicMock()
    manager.summarize_session = MagicMock()
    return manager


@pytest.fixture
def mock_session_store():
    store = MagicMock()
    store.save_session = MagicMock()
    return store


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.agenerate = AsyncMock(return_value="VERDICT: YES\nEXPLANATION: That's correct!")
    return client


@pytest.fixture
def runner(sample_session, sample_puzzle, mock_kb_manager, mock_memory_manager, mock_session_store, mock_llm_client, agents_config):
    return GameSessionRunner(
        session=sample_session,
        puzzle=sample_puzzle,
        kb_manager=mock_kb_manager,
        memory_manager=mock_memory_manager,
        session_store=mock_session_store,
        llm_client=mock_llm_client,
        agents_config=agents_config,
    )


class TestMessageClassification:
    def test_classify_command(self, runner):
        assert runner._classify_input("/hint") == MessageType.COMMAND
        assert runner._classify_input("!status") == MessageType.COMMAND
        assert runner._classify_input("/help") == MessageType.COMMAND

    def test_classify_hypothesis(self, runner):
        assert runner._classify_input("I think the answer is...") == MessageType.HYPOTHESIS
        assert runner._classify_input("My guess is he was scared") == MessageType.HYPOTHESIS
        assert runner._classify_input("The solution is hiccups") == MessageType.HYPOTHESIS
        assert runner._classify_input("我认为答案是...") == MessageType.HYPOTHESIS

    def test_classify_question(self, runner):
        assert runner._classify_input("Was the man sick?") == MessageType.QUESTION
        assert runner._classify_input("Did he have hiccups?") == MessageType.QUESTION
        assert runner._classify_input("Is the bartender a friend?") == MessageType.QUESTION


class TestGameStart:
    def test_start_game(self, runner):
        response = runner.start_game()
        
        assert runner.session.state == GameState.IN_PROGRESS
        assert "Test Puzzle" in response.message
        assert runner.session.turn_count == 1

    def test_start_already_started(self, runner):
        runner.start_game()
        response = runner.start_game()
        
        assert "already started" in response.message.lower()


class TestCommands:
    def test_hint_command(self, runner):
        runner.start_game()
        
        response = runner._handle_hint_command()
        
        assert runner.session.hint_count == 1
        assert "nudge" in response.message.lower() or "hint" in response.message.lower()
        assert response.metadata["hints_used"] == 1

    def test_hint_exhausted(self, runner):
        runner.start_game()
        runner._session.hint_count = 3
        
        response = runner._handle_hint_command()
        
        assert "all" in response.message.lower() and "hints" in response.message.lower()

    def test_status_command(self, runner):
        runner.start_game()
        
        response = runner._handle_status_command()
        
        assert "Test Puzzle" in response.message
        assert "in_progress" in response.message.lower()

    def test_help_command(self, runner):
        response = runner._handle_help_command()
        
        assert "/hint" in response.message
        assert "/status" in response.message
        assert "/quit" in response.message

    def test_quit_command(self, runner):
        runner.start_game()
        
        response = runner._handle_quit_command()
        
        assert runner.session.state == GameState.ABORTED
        assert response.game_over is True


class TestQuestionHandling:
    @pytest.mark.asyncio
    async def test_handle_question(self, runner, mock_llm_client):
        runner.start_game()
        mock_llm_client.agenerate.return_value = "VERDICT: YES\nEXPLANATION: That's on the right track!"
        
        response = await runner._handle_question("Did he have hiccups?")
        
        assert response.verdict == "YES"
        assert "YES" in response.message

    @pytest.mark.asyncio
    async def test_handle_question_no(self, runner, mock_llm_client):
        runner.start_game()
        mock_llm_client.agenerate.return_value = "VERDICT: NO\nEXPLANATION: That's not it."
        
        response = await runner._handle_question("Was he injured?")
        
        assert response.verdict == "NO"
        assert "NO" in response.message

    @pytest.mark.asyncio
    async def test_handle_question_irrelevant(self, runner, mock_llm_client):
        runner.start_game()
        mock_llm_client.agenerate.return_value = "VERDICT: IRRELEVANT\nEXPLANATION: That doesn't relate to the puzzle."
        
        response = await runner._handle_question("What color was the bar?")
        
        assert response.verdict == "IRRELEVANT"


class TestHypothesisHandling:
    @pytest.mark.asyncio
    async def test_correct_hypothesis(self, runner, mock_llm_client):
        runner.start_game()
        mock_llm_client.agenerate.return_value = "VERDICT: CORRECT\nEXPLANATION: You got it!"
        
        response = await runner._handle_hypothesis("I think he had hiccups and the gun scared them away")
        
        assert response.verdict == "correct"
        assert response.game_over is True
        assert runner.session.state == GameState.COMPLETED

    @pytest.mark.asyncio
    async def test_incorrect_hypothesis(self, runner, mock_llm_client):
        runner.start_game()
        mock_llm_client.agenerate.return_value = "VERDICT: INCORRECT\nEXPLANATION: That's not the solution."
        
        response = await runner._handle_hypothesis("I think he was a robber")
        
        assert response.verdict == "incorrect"
        assert response.game_over is False
        assert runner.session.state == GameState.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_partial_hypothesis(self, runner, mock_llm_client):
        runner.start_game()
        mock_llm_client.agenerate.return_value = "VERDICT: PARTIAL\nEXPLANATION: You're close but missing something."
        
        response = await runner._handle_hypothesis("I think the man had a problem")
        
        assert response.verdict == "partial"
        assert response.game_over is False


class TestProcessPlayerInput:
    @pytest.mark.asyncio
    async def test_process_not_active_lobby(self, runner):
        response = await runner.process_player_input("Hello")
        
        assert runner.session.state == GameState.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_process_command(self, runner):
        runner.start_game()
        
        response = await runner.process_player_input("/status")
        
        assert "Test Puzzle" in response.message

    @pytest.mark.asyncio
    async def test_process_question(self, runner, mock_llm_client):
        runner.start_game()
        mock_llm_client.agenerate.return_value = "VERDICT: NO\nEXPLANATION: No."
        
        response = await runner.process_player_input("Was he tall?")
        
        assert response.verdict == "NO"

    @pytest.mark.asyncio
    async def test_process_hypothesis(self, runner, mock_llm_client):
        runner.start_game()
        mock_llm_client.agenerate.return_value = "VERDICT: INCORRECT\nEXPLANATION: No."
        
        response = await runner.process_player_input("I think he was a spy")
        
        assert response.verdict == "incorrect"


class TestVerdictParsing:
    def test_parse_yes_verdict(self, runner):
        response = "VERDICT: YES\nEXPLANATION: Correct thinking!"
        verdict, explanation = runner._parse_verdict_response(response)
        
        assert verdict == DMVerdict.YES
        assert "Correct" in explanation

    def test_parse_no_verdict(self, runner):
        response = "VERDICT: NO\nEXPLANATION: Not quite."
        verdict, explanation = runner._parse_verdict_response(response)
        
        assert verdict == DMVerdict.NO

    def test_parse_yes_and_no_verdict(self, runner):
        response = "VERDICT: YES_AND_NO\nEXPLANATION: Partially correct."
        verdict, explanation = runner._parse_verdict_response(response)
        
        assert verdict == DMVerdict.YES_AND_NO

    def test_parse_irrelevant_verdict(self, runner):
        response = "VERDICT: IRRELEVANT\nEXPLANATION: Not related."
        verdict, explanation = runner._parse_verdict_response(response)
        
        assert verdict == DMVerdict.IRRELEVANT

    def test_parse_hypothesis_correct(self, runner):
        response = "VERDICT: CORRECT\nEXPLANATION: Well done!"
        verdict, explanation = runner._parse_hypothesis_response(response)
        
        assert verdict == HypothesisVerdict.CORRECT

    def test_parse_hypothesis_partial(self, runner):
        response = "VERDICT: PARTIAL\nEXPLANATION: Almost there."
        verdict, explanation = runner._parse_hypothesis_response(response)
        
        assert verdict == HypothesisVerdict.PARTIAL


class TestScoring:
    def test_calculate_score_perfect(self, runner):
        runner.start_game()
        
        score = runner._calculate_score()
        
        assert score == 1000

    def test_calculate_score_with_hints(self, runner):
        runner.start_game()
        runner._session.hint_count = 2
        
        score = runner._calculate_score()
        
        assert score == 1000 - 100  # 50 penalty per hint


class TestEventTracking:
    def test_count_questions(self, runner, mock_memory_manager):
        runner.start_game()
        runner._append_event(AgentRole.PLAYER, "Q1?", [EventTag.QUESTION])
        runner._append_event(AgentRole.DM, "A1", [EventTag.ANSWER])
        runner._append_event(AgentRole.PLAYER, "Q2?", [EventTag.QUESTION])
        runner._append_event(AgentRole.DM, "A2", [EventTag.ANSWER])
        
        count = runner._count_questions()
        
        assert count == 2

    def test_get_recent_qa_pairs(self, runner, mock_memory_manager):
        runner.start_game()
        runner._append_event(AgentRole.PLAYER, "Question 1?", [EventTag.QUESTION])
        runner._append_event(AgentRole.DM, "Answer 1", [EventTag.ANSWER])
        runner._append_event(AgentRole.PLAYER, "Question 2?", [EventTag.QUESTION])
        runner._append_event(AgentRole.DM, "Answer 2", [EventTag.ANSWER])
        
        pairs = runner._get_recent_qa_pairs(limit=5)
        
        assert len(pairs) == 2
        assert pairs[0] == ("Question 1?", "Answer 1")
        assert pairs[1] == ("Question 2?", "Answer 2")
