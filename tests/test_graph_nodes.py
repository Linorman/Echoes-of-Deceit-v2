"""Tests for LangGraph Nodes."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from game.graph.state import (
    GameGraphState,
    GamePhase,
    MessageType,
    DMVerdict,
    HypothesisVerdict,
    TurnEvent,
)
from game.graph.nodes import (
    PlayerMessageNode,
    IntroNode,
    DMQuestionNode,
    DMHypothesisNode,
    CommandHandlerNode,
    MemoryUpdateNode,
    RevealSolutionNode,
    COMMAND_PREFIXES,
    HYPOTHESIS_KEYWORDS,
)
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
def mock_llm_client():
    client = MagicMock()
    client.agenerate = AsyncMock(return_value="VERDICT: YES\nEXPLANATION: That's correct!")
    return client


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
def base_state():
    return GameGraphState(
        session_id="test-session",
        kb_id="game_puzzle1",
        player_id="player1",
        puzzle_id="puzzle1",
        puzzle_statement="A man walks into a bar...",
        puzzle_answer="He had hiccups.",
        puzzle_title="The Bar",
        max_hints=5,
        game_phase=GamePhase.PLAYING,
    )


class TestPlayerMessageNode:
    @pytest.mark.asyncio
    async def test_classify_question(self, base_state):
        node = PlayerMessageNode()
        base_state.last_user_message = "Is the man alive?"
        
        result = await node(base_state)
        
        assert result["message_type"] == MessageType.QUESTION

    @pytest.mark.asyncio
    async def test_classify_command(self, base_state):
        node = PlayerMessageNode()
        base_state.last_user_message = "/hint"
        
        result = await node(base_state)
        
        assert result["message_type"] == MessageType.COMMAND

    @pytest.mark.asyncio
    async def test_classify_hypothesis(self, base_state):
        node = PlayerMessageNode()
        base_state.last_user_message = "I think the man had hiccups"
        
        result = await node(base_state)
        
        assert result["message_type"] == MessageType.HYPOTHESIS

    @pytest.mark.asyncio
    async def test_classify_empty_message(self, base_state):
        node = PlayerMessageNode()
        base_state.last_user_message = ""
        
        result = await node(base_state)
        
        assert result["message_type"] == MessageType.UNKNOWN

    def test_command_prefixes(self):
        assert "/" in COMMAND_PREFIXES
        assert "!" in COMMAND_PREFIXES
        assert "\\" in COMMAND_PREFIXES

    def test_hypothesis_keywords(self):
        assert "i think" in HYPOTHESIS_KEYWORDS
        assert "my guess" in HYPOTHESIS_KEYWORDS
        assert "我认为" in HYPOTHESIS_KEYWORDS


class TestIntroNode:
    @pytest.mark.asyncio
    async def test_intro_mysterious_tone(self, base_state, agents_config):
        node = IntroNode(agents_config=agents_config)
        base_state.game_phase = GamePhase.INTRO
        
        result = await node(base_state)
        
        assert "mystery" in result["last_dm_response"].lower()
        assert base_state.puzzle_title in result["last_dm_response"]
        assert result["game_phase"] == GamePhase.PLAYING
        assert len(result["turn_history"]) == 1
        assert result["turn_index"] == base_state.turn_index + 1

    @pytest.mark.asyncio
    async def test_intro_friendly_tone(self, base_state):
        config = AgentsConfig(
            dm=DMConfig(persona=DMPersonaConfig(tone="friendly")),
        )
        node = IntroNode(agents_config=config)
        base_state.game_phase = GamePhase.INTRO
        
        result = await node(base_state)
        
        assert "welcome" in result["last_dm_response"].lower()


class TestDMQuestionNode:
    @pytest.mark.asyncio
    async def test_evaluate_question_yes(
        self, base_state, agents_config, mock_llm_client, mock_kb_manager
    ):
        mock_llm_client.agenerate = AsyncMock(
            return_value="VERDICT: YES\nEXPLANATION: That's on the right track!"
        )
        
        node = DMQuestionNode(
            llm_client=mock_llm_client,
            kb_manager=mock_kb_manager,
            agents_config=agents_config,
        )
        base_state.last_user_message = "Does it involve water?"
        
        result = await node(base_state)
        
        assert "YES" in result["last_dm_response"]
        assert result["last_verdict"] == "YES"
        assert len(result["turn_history"]) == 2

    @pytest.mark.asyncio
    async def test_evaluate_question_no(
        self, base_state, agents_config, mock_llm_client, mock_kb_manager
    ):
        mock_llm_client.agenerate = AsyncMock(
            return_value="VERDICT: NO\nEXPLANATION: That's not relevant."
        )
        
        node = DMQuestionNode(
            llm_client=mock_llm_client,
            kb_manager=mock_kb_manager,
            agents_config=agents_config,
        )
        base_state.last_user_message = "Is it about money?"
        
        result = await node(base_state)
        
        assert "NO" in result["last_dm_response"]
        assert result["last_verdict"] == "NO"

    @pytest.mark.asyncio
    async def test_evaluate_question_without_llm(self, base_state, agents_config):
        node = DMQuestionNode(agents_config=agents_config)
        base_state.last_user_message = "Is it alive?"
        
        result = await node(base_state)
        
        assert result["last_verdict"] == "IRRELEVANT"


class TestDMHypothesisNode:
    @pytest.mark.asyncio
    async def test_correct_hypothesis(
        self, base_state, agents_config, mock_llm_client, mock_kb_manager
    ):
        mock_llm_client.agenerate = AsyncMock(
            return_value="VERDICT: CORRECT\nEXPLANATION: Well done! You solved it!"
        )
        
        node = DMHypothesisNode(
            llm_client=mock_llm_client,
            kb_manager=mock_kb_manager,
            agents_config=agents_config,
        )
        base_state.last_user_message = "I think the man had hiccups"
        
        result = await node(base_state)
        
        assert "CORRECT" in result["last_dm_response"]
        assert result["last_verdict"] == "correct"
        assert result["game_phase"] == GamePhase.COMPLETED
        assert result["score"] is not None

    @pytest.mark.asyncio
    async def test_incorrect_hypothesis(
        self, base_state, agents_config, mock_llm_client, mock_kb_manager
    ):
        mock_llm_client.agenerate = AsyncMock(
            return_value="VERDICT: INCORRECT\nEXPLANATION: That's not the solution."
        )
        
        node = DMHypothesisNode(
            llm_client=mock_llm_client,
            kb_manager=mock_kb_manager,
            agents_config=agents_config,
        )
        base_state.last_user_message = "I think he was thirsty"
        
        result = await node(base_state)
        
        assert "Not Quite" in result["last_dm_response"]
        assert result["last_verdict"] == "incorrect"
        assert result["game_phase"] == base_state.game_phase

    @pytest.mark.asyncio
    async def test_partial_hypothesis(
        self, base_state, agents_config, mock_llm_client, mock_kb_manager
    ):
        mock_llm_client.agenerate = AsyncMock(
            return_value="VERDICT: PARTIAL\nEXPLANATION: You're getting closer!"
        )
        
        node = DMHypothesisNode(
            llm_client=mock_llm_client,
            kb_manager=mock_kb_manager,
            agents_config=agents_config,
        )
        base_state.last_user_message = "Something startled him"
        
        result = await node(base_state)
        
        assert "Partially Correct" in result["last_dm_response"]
        assert result["last_verdict"] == "partial"


class TestCommandHandlerNode:
    @pytest.mark.asyncio
    async def test_hint_command(self, base_state, agents_config):
        node = CommandHandlerNode(agents_config=agents_config)
        base_state.last_user_message = "/hint"
        
        result = await node(base_state)
        
        assert "hint" in result["last_dm_response"].lower() or "nudge" in result["last_dm_response"].lower()
        assert result["hint_count"] == 1

    @pytest.mark.asyncio
    async def test_hint_command_max_reached(self, base_state, agents_config):
        node = CommandHandlerNode(agents_config=agents_config)
        base_state.last_user_message = "/hint"
        base_state.hint_count = 5
        
        result = await node(base_state)
        
        assert "used all" in result["last_dm_response"].lower()
        assert "hint_count" not in result

    @pytest.mark.asyncio
    async def test_status_command(self, base_state, agents_config):
        node = CommandHandlerNode(agents_config=agents_config)
        base_state.last_user_message = "/status"
        
        result = await node(base_state)
        
        assert "Game Status" in result["last_dm_response"]
        assert base_state.puzzle_title in result["last_dm_response"]

    @pytest.mark.asyncio
    async def test_quit_command(self, base_state, agents_config):
        node = CommandHandlerNode(agents_config=agents_config)
        base_state.last_user_message = "/quit"
        
        result = await node(base_state)
        
        assert "ended" in result["last_dm_response"].lower()
        assert result["game_phase"] == GamePhase.ABORTED

    @pytest.mark.asyncio
    async def test_help_command(self, base_state, agents_config):
        node = CommandHandlerNode(agents_config=agents_config)
        base_state.last_user_message = "/help"
        
        result = await node(base_state)
        
        assert "Available Commands" in result["last_dm_response"]

    @pytest.mark.asyncio
    async def test_unknown_command(self, base_state, agents_config):
        node = CommandHandlerNode(agents_config=agents_config)
        base_state.last_user_message = "/unknown"
        
        result = await node(base_state)
        
        assert "Unknown command" in result["last_dm_response"]


class TestMemoryUpdateNode:
    @pytest.mark.asyncio
    async def test_append_events(self, base_state, mock_memory_manager):
        node = MemoryUpdateNode(memory_manager=mock_memory_manager)
        base_state.turn_history = [
            TurnEvent(turn_index=0, role="player", message="Q1", tags=["question"]),
            TurnEvent(turn_index=1, role="dm", message="YES", tags=["answer"]),
        ]
        
        result = await node(base_state)
        
        assert mock_memory_manager.append_session_event.called

    @pytest.mark.asyncio
    async def test_summarize_on_complete(self, base_state, mock_memory_manager):
        node = MemoryUpdateNode(memory_manager=mock_memory_manager)
        base_state.game_phase = GamePhase.COMPLETED
        base_state.turn_history = [
            TurnEvent(turn_index=0, role="player", message="Q1", tags=["question"]),
        ]
        
        result = await node(base_state)
        
        assert mock_memory_manager.summarize_session.called


class TestRevealSolutionNode:
    @pytest.mark.asyncio
    async def test_reveal_solution(self, base_state):
        node = RevealSolutionNode()
        base_state.game_phase = GamePhase.COMPLETED
        base_state.score = 800
        base_state.turn_history = [
            TurnEvent(turn_index=0, role="player", message="Q1", tags=["question"]),
            TurnEvent(turn_index=1, role="dm", message="YES", tags=["answer"]),
        ]
        
        result = await node(base_state)
        
        assert "Game Complete" in result["last_dm_response"]
        assert base_state.puzzle_answer in result["last_dm_response"]
        assert "800" in result["last_dm_response"] or "N/A" in result["last_dm_response"]
