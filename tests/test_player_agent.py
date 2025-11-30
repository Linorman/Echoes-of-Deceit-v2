"""Tests for Player Agent functionality."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from game.graph.nodes import PlayerAgentNode
from game.graph.state import (
    GameGraphState,
    GamePhase,
    GameMode,
    MessageType,
    DMVerdict,
    TurnEvent,
)


class MockPlayerAgentPersonaConfig:
    def __init__(self):
        self.name = "Detective"
        self.tone = "curious"
        self.style = "analytical"


class MockPlayerAgentBehaviorConfig:
    def __init__(self):
        self.ask_followup_questions = True
        self.form_hypothesis_after_questions = 10
        self.max_questions_before_guess = 20
        self.question_strategies = ["binary_elimination", "detail_probing"]


class MockPlayerAgentRagAccessConfig:
    def __init__(self):
        self.allowed_types = ["puzzle_statement", "public_fact"]
        self.max_context_length = 500


class MockPlayerAgentConfig:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.persona = MockPlayerAgentPersonaConfig()
        self.behavior = MockPlayerAgentBehaviorConfig()
        self.rag_access = MockPlayerAgentRagAccessConfig()


class MockAgentsConfig:
    def __init__(self, player_agent_enabled=True):
        self.player_agent = MockPlayerAgentConfig(enabled=player_agent_enabled)
        self.dm = MagicMock()
        self.judge = MagicMock()
        self.hint = MagicMock()


class MockLLMClient:
    def __init__(self, response="Is the man human?"):
        self._response = response
    
    async def agenerate(self, prompt: str) -> str:
        return self._response


class MockQueryResult:
    def __init__(self, answer="Public context"):
        self.answer = answer


class MockKBManager:
    def __init__(self, result=None):
        self._result = result or MockQueryResult()
    
    async def query_public(self, kb_id: str, query: str):
        return self._result


@pytest.fixture
def base_state():
    return GameGraphState(
        session_id="test_session",
        kb_id="test_kb",
        player_id="test_player",
        puzzle_id="test_puzzle",
        puzzle_title="The Man in the Bar",
        puzzle_statement="A man walks into a bar and asks for a glass of water. The bartender pulls out a gun. The man says thank you and leaves. Why?",
        puzzle_answer="The man had hiccups. The bartender scared him with the gun to cure his hiccups.",
        game_phase=GamePhase.PLAYING,
        turn_index=0,
        player_agent_enabled=True,
        game_mode=GameMode.AI_PLAYER,
    )


@pytest.fixture
def state_with_history():
    events = [
        TurnEvent(
            turn_index=0,
            role="player_agent",
            message="Was the man thirsty?",
            tags=["question", "ai_player"],
            verdict=None,
        ),
        TurnEvent(
            turn_index=1,
            role="dm",
            message="YES",
            tags=["answer"],
            verdict="YES",
        ),
        TurnEvent(
            turn_index=2,
            role="player_agent",
            message="Did he drink the water?",
            tags=["question", "ai_player"],
            verdict=None,
        ),
        TurnEvent(
            turn_index=3,
            role="dm",
            message="NO",
            tags=["answer"],
            verdict="NO",
        ),
    ]
    return GameGraphState(
        session_id="test_session",
        kb_id="test_kb",
        player_id="test_player",
        puzzle_id="test_puzzle",
        puzzle_title="The Man in the Bar",
        puzzle_statement="A man walks into a bar and asks for a glass of water. The bartender pulls out a gun. The man says thank you and leaves. Why?",
        puzzle_answer="The man had hiccups. The bartender scared him with the gun to cure his hiccups.",
        game_phase=GamePhase.PLAYING,
        turn_index=4,
        player_agent_enabled=True,
        game_mode=GameMode.AI_PLAYER,
        turn_history=events,
        player_agent_question_count=2,
    )


class TestPlayerAgentNodeBasic:
    @pytest.mark.asyncio
    async def test_player_agent_disabled_returns_error(self, base_state):
        config = MockAgentsConfig(player_agent_enabled=False)
        llm_client = MockLLMClient()
        
        node = PlayerAgentNode(
            llm_client=llm_client,
            agents_config=config,
        )
        
        result = await node(base_state)
        
        assert "error" in result
        assert "not enabled" in result["error"]

    @pytest.mark.asyncio
    async def test_player_agent_no_llm_returns_error(self, base_state):
        config = MockAgentsConfig(player_agent_enabled=True)
        
        node = PlayerAgentNode(
            llm_client=None,
            agents_config=config,
        )
        
        result = await node(base_state)
        
        assert "error" in result
        assert "LLM client not available" in result["error"]


class TestPlayerAgentQuestionGeneration:
    @pytest.mark.asyncio
    async def test_generates_question(self, base_state):
        config = MockAgentsConfig()
        llm_client = MockLLMClient(response="Was the man afraid of something?")
        kb_manager = MockKBManager()
        
        node = PlayerAgentNode(
            llm_client=llm_client,
            kb_manager=kb_manager,
            agents_config=config,
        )
        
        result = await node(base_state)
        
        assert "last_user_message" in result
        assert "Was the man afraid of something?" in result["last_user_message"]
        assert result["message_type"] == MessageType.QUESTION
        assert result["player_agent_question_count"] == 1
        assert result["awaiting_player_agent"] == False

    @pytest.mark.asyncio
    async def test_adds_question_mark_if_missing(self, base_state):
        config = MockAgentsConfig()
        llm_client = MockLLMClient(response="Was the man afraid")
        
        node = PlayerAgentNode(
            llm_client=llm_client,
            agents_config=config,
        )
        
        result = await node(base_state)
        
        assert result["last_user_message"].endswith("?")

    @pytest.mark.asyncio
    async def test_creates_turn_event(self, base_state):
        config = MockAgentsConfig()
        llm_client = MockLLMClient(response="Is the bartender dangerous?")
        
        node = PlayerAgentNode(
            llm_client=llm_client,
            agents_config=config,
        )
        
        result = await node(base_state)
        
        assert "turn_history" in result
        assert len(result["turn_history"]) == 1
        event = result["turn_history"][0]
        assert event.role == "player_agent"
        assert "question" in event.tags
        assert "ai_player" in event.tags


class TestPlayerAgentHypothesisGeneration:
    @pytest.mark.asyncio
    async def test_generates_hypothesis_after_enough_questions(self, state_with_history):
        state_with_history.player_agent_question_count = 10
        
        yes_events = [
            TurnEvent(turn_index=i, role="dm", message="YES", tags=["answer"], verdict="YES")
            for i in range(0, 10, 2)
        ]
        question_events = [
            TurnEvent(turn_index=i, role="player_agent", message="Question?", tags=["question"], verdict=None)
            for i in range(1, 10, 2)
        ]
        state_with_history.turn_history = []
        for q, a in zip(question_events, yes_events):
            state_with_history.turn_history.extend([q, a])
        
        config = MockAgentsConfig()
        config.player_agent.behavior.form_hypothesis_after_questions = 5
        llm_client = MockLLMClient(response="I think the man had hiccups and the bartender scared him.")
        
        node = PlayerAgentNode(
            llm_client=llm_client,
            agents_config=config,
        )
        
        result = await node(state_with_history)
        
        assert result["message_type"] == MessageType.HYPOTHESIS
        assert "I think" in result["last_user_message"]

    @pytest.mark.asyncio
    async def test_hypothesis_starts_with_i_think(self, state_with_history):
        state_with_history.player_agent_question_count = 15
        
        yes_events = [
            TurnEvent(turn_index=i, role="dm", message="YES", tags=["answer"], verdict="YES")
            for i in range(10)
        ]
        state_with_history.turn_history = yes_events
        
        config = MockAgentsConfig()
        config.player_agent.behavior.form_hypothesis_after_questions = 5
        llm_client = MockLLMClient(response="The man had hiccups.")
        
        node = PlayerAgentNode(
            llm_client=llm_client,
            agents_config=config,
        )
        
        result = await node(state_with_history)
        
        assert result["last_user_message"].lower().startswith("i think")


class TestPlayerAgentFormatting:
    def test_format_recent_qa(self, state_with_history):
        config = MockAgentsConfig()
        node = PlayerAgentNode(agents_config=config)
        
        result = node._format_recent_qa(state_with_history)
        
        assert "Was the man thirsty" in result
        assert "YES" in result
        assert "Did he drink the water" in result
        assert "NO" in result

    def test_format_recent_qa_empty(self, base_state):
        config = MockAgentsConfig()
        node = PlayerAgentNode(agents_config=config)
        
        result = node._format_recent_qa(base_state)
        
        assert "No questions asked yet" in result

    def test_build_question_prompt_contains_puzzle(self, base_state):
        config = MockAgentsConfig()
        node = PlayerAgentNode(agents_config=config)
        
        prompt = node._build_question_prompt(
            puzzle_statement=base_state.puzzle_statement,
            recent_qa="No questions asked yet.",
            rag_context="",
            persona_name="Detective",
            strategies=["binary_elimination"],
        )
        
        assert "man walks into a bar" in prompt
        assert "Detective" in prompt
        assert "binary_elimination" in prompt or "divide possibilities" in prompt


class TestPlayerAgentDecisionLogic:
    def test_should_attempt_hypothesis_high_yes_rate(self, state_with_history):
        state_with_history.turn_history = [
            TurnEvent(turn_index=0, role="dm", message="YES", tags=["answer"], verdict="YES"),
            TurnEvent(turn_index=1, role="dm", message="YES", tags=["answer"], verdict="YES"),
            TurnEvent(turn_index=2, role="dm", message="YES", tags=["answer"], verdict="YES"),
            TurnEvent(turn_index=3, role="dm", message="NO", tags=["answer"], verdict="NO"),
        ]
        state_with_history.player_agent_question_count = 4
        
        config = MockAgentsConfig()
        node = PlayerAgentNode(agents_config=config)
        
        result = node._should_attempt_hypothesis(state_with_history)
        
        assert result is True

    def test_should_not_attempt_hypothesis_low_yes_rate(self, state_with_history):
        state_with_history.turn_history = [
            TurnEvent(turn_index=0, role="dm", message="NO", tags=["answer"], verdict="NO"),
            TurnEvent(turn_index=1, role="dm", message="NO", tags=["answer"], verdict="NO"),
            TurnEvent(turn_index=2, role="dm", message="NO", tags=["answer"], verdict="NO"),
            TurnEvent(turn_index=3, role="dm", message="YES", tags=["answer"], verdict="YES"),
        ]
        state_with_history.player_agent_question_count = 4
        
        config = MockAgentsConfig()
        node = PlayerAgentNode(agents_config=config)
        
        result = node._should_attempt_hypothesis(state_with_history)
        
        assert result is False


class TestPlayerAgentWithRAG:
    @pytest.mark.asyncio
    async def test_uses_rag_context(self, base_state):
        config = MockAgentsConfig()
        llm_client = MockLLMClient(response="Was the gun real?")
        kb_manager = MockKBManager(MockQueryResult("The bartender is known to be friendly."))
        
        node = PlayerAgentNode(
            llm_client=llm_client,
            kb_manager=kb_manager,
            agents_config=config,
        )
        
        result = await node(base_state)
        
        assert "last_user_message" in result

    @pytest.mark.asyncio
    async def test_handles_rag_failure(self, base_state):
        config = MockAgentsConfig()
        llm_client = MockLLMClient(response="Was the man a regular?")
        
        class FailingKBManager:
            async def query_public(self, kb_id, query):
                raise Exception("RAG failed")
        
        node = PlayerAgentNode(
            llm_client=llm_client,
            kb_manager=FailingKBManager(),
            agents_config=config,
        )
        
        result = await node(base_state)
        
        assert "last_user_message" in result
        assert "error" not in result


class TestPlayerAgentErrorHandling:
    @pytest.mark.asyncio
    async def test_handles_llm_failure_for_question(self, base_state):
        config = MockAgentsConfig()
        
        class FailingLLMClient:
            async def agenerate(self, prompt):
                raise Exception("LLM failed")
        
        node = PlayerAgentNode(
            llm_client=FailingLLMClient(),
            agents_config=config,
        )
        
        result = await node(base_state)
        
        assert "error" in result

    @pytest.mark.asyncio
    async def test_handles_llm_failure_for_hypothesis(self, state_with_history):
        state_with_history.player_agent_question_count = 15
        
        yes_events = [
            TurnEvent(turn_index=i, role="dm", message="YES", tags=["answer"], verdict="YES")
            for i in range(10)
        ]
        state_with_history.turn_history = yes_events
        
        config = MockAgentsConfig()
        config.player_agent.behavior.form_hypothesis_after_questions = 5
        
        class FailingLLMClient:
            async def agenerate(self, prompt):
                raise Exception("LLM failed")
        
        node = PlayerAgentNode(
            llm_client=FailingLLMClient(),
            agents_config=config,
        )
        
        result = await node(state_with_history)
        
        assert "error" in result
