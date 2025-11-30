"""Tests for Phase 7 Graph Node enhancements - Profile-Aware DM and Observability."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from game.graph.nodes import (
    BaseNode,
    DMQuestionNode,
    DMHypothesisNode,
    CommandHandlerNode,
    IntroNode,
    MemoryUpdateNode,
    log_game_event,
    GAME_EVENT_SESSION_START,
    GAME_EVENT_SESSION_END,
    GAME_EVENT_HINT_USED,
    GAME_EVENT_HYPOTHESIS_VERDICT,
)
from game.graph.state import (
    GameGraphState,
    GamePhase,
    MessageType,
    DMVerdict,
    HypothesisVerdict,
    TurnEvent,
)


class MockProfileIntegration:
    def __init__(self, enabled=True, profile_weight="medium"):
        self.enabled = enabled
        self.profile_weight = profile_weight
        self.adapt_difficulty = True
        self.adapt_explanations = True
        self.adapt_hint_strength = True


class MockDMConfig:
    def __init__(self):
        self.persona = type("Persona", (), {"name": "TestDM", "tone": "mysterious", "style": "immersive"})()
        self.behavior = type("Behavior", (), {"reveal_answer_early": False, "verbose_explanations": False, "encourage_player": True})()
        self.profile_integration = MockProfileIntegration()


class MockJudgeConfig:
    def __init__(self):
        self.strictness = "moderate"
        self.response_format = type("ResponseFormat", (), {"include_explanation": True, "max_explanation_length": 100})()


class MockHintConfig:
    def __init__(self):
        self.strategy = type("Strategy", (), {"initial_vagueness": "high", "progressive_clarity": True, "max_hints_before_direct": 3})()
        self.timing = type("Timing", (), {"auto_hint_after_questions": 10, "allow_player_request": True})()


class MockAgentsConfig:
    def __init__(self):
        self.dm = MockDMConfig()
        self.judge = MockJudgeConfig()
        self.hint = MockHintConfig()


class MockMemoryManager:
    def __init__(self, profile_results=None):
        self._profile_results = profile_results or []
        self.appended_events = []
        self.summarized_sessions = []
    
    def retrieve_player_profile(self, player_id, limit=3):
        return self._profile_results
    
    def append_session_event(self, session_id, event):
        self.appended_events.append((session_id, event))
    
    def summarize_session(self, session_id, player_id=None, puzzle_id=None):
        self.summarized_sessions.append({
            "session_id": session_id,
            "player_id": player_id,
            "puzzle_id": puzzle_id,
        })


class MockLLMClient:
    def __init__(self, response="VERDICT: YES\nEXPLANATION: Correct."):
        self._response = response
    
    async def agenerate(self, prompt: str) -> str:
        return self._response


class MockProfileSearchResult:
    def __init__(self, summary_type, content):
        self.value = {
            "summary_type": summary_type,
            "content": content,
        }


@pytest.fixture
def base_state():
    return GameGraphState(
        session_id="test_session",
        kb_id="test_kb",
        player_id="test_player",
        puzzle_id="test_puzzle",
        puzzle_title="Test Puzzle",
        puzzle_statement="A man walks into a bar...",
        puzzle_answer="He was a robot.",
        game_phase=GamePhase.PLAYING,
        turn_index=0,
        last_user_message="Is he human?",
    )


class TestLogGameEvent:
    def test_log_game_event(self, caplog):
        with caplog.at_level("INFO"):
            log_game_event(
                "test_event",
                "session1",
                "player1",
                {"key": "value"},
            )
        
        assert "GAME_EVENT" in caplog.text
        assert "test_event" in caplog.text
        assert "session1" in caplog.text


class TestBaseNodeProfileIntegration:
    def test_get_player_profile_context_no_manager(self, base_state):
        node = DMQuestionNode()
        result = node._get_player_profile_context("player1")
        assert result is None
    
    def test_get_player_profile_context_disabled(self, base_state):
        config = MockAgentsConfig()
        config.dm.profile_integration = MockProfileIntegration(enabled=False)
        
        memory_manager = MockMemoryManager([
            MockProfileSearchResult("style_profile", "Analytical player"),
        ])
        
        node = DMQuestionNode(
            memory_manager=memory_manager,
            agents_config=config,
        )
        result = node._get_player_profile_context("player1")
        assert result is None
    
    def test_get_player_profile_context_with_results(self, base_state):
        config = MockAgentsConfig()

        memory_manager = MockMemoryManager([
            MockProfileSearchResult("style_profile", "Analytical player"),
            MockProfileSearchResult("performance_summary", "Good at puzzles"),
        ])
        
        node = DMQuestionNode(
            memory_manager=memory_manager,
            agents_config=config,
        )
        result = node._get_player_profile_context("player1")
        
        assert result is not None
        assert "Analytical player" in result
        assert "Good at puzzles" in result
    
    def test_get_profile_adaptation_hints_high_weight(self, base_state):
        config = MockAgentsConfig()
        config.dm.profile_integration.profile_weight = "high"
        
        node = DMQuestionNode(agents_config=config)
        profile = "Beginner player"
        
        result = node._get_profile_adaptation_hints(profile)
        
        assert "IMPORTANT" in result
        assert "Beginner player" in result
    
    def test_get_profile_adaptation_hints_low_weight(self, base_state):
        config = MockAgentsConfig()
        config.dm.profile_integration.profile_weight = "low"
        
        node = DMQuestionNode(agents_config=config)
        profile = "Advanced player"
        
        result = node._get_profile_adaptation_hints(profile)
        
        assert "Lightly consider" in result


class TestDMQuestionNodeWithProfile:
    @pytest.mark.asyncio
    async def test_build_prompt_with_profile(self, base_state):
        memory_manager = MockMemoryManager([
            MockProfileSearchResult("style_profile", "Beginner player"),
        ])
        llm = MockLLMClient("VERDICT: YES\nEXPLANATION: Good question.")
        
        node = DMQuestionNode(
            llm_client=llm,
            memory_manager=memory_manager,
            agents_config=MockAgentsConfig(),
        )
        
        result = await node(base_state)
        
        assert result["last_verdict"] == "YES"
        assert "YES" in result["last_dm_response"]


class TestDMHypothesisNodeWithLogging:
    @pytest.mark.asyncio
    async def test_hypothesis_correct_logs_event(self, base_state, caplog):
        base_state.last_user_message = "I think he was a robot"
        
        llm = MockLLMClient("VERDICT: CORRECT\nEXPLANATION: You got it!")
        
        node = DMHypothesisNode(
            llm_client=llm,
            agents_config=MockAgentsConfig(),
        )
        
        with caplog.at_level("INFO"):
            result = await node(base_state)
        
        assert result["game_phase"] == GamePhase.COMPLETED
        assert "GAME_EVENT" in caplog.text
        assert "hypothesis_verdict" in caplog.text
    
    @pytest.mark.asyncio
    async def test_hypothesis_incorrect_logs_event(self, base_state, caplog):
        base_state.last_user_message = "I think he was a ghost"
        
        llm = MockLLMClient("VERDICT: INCORRECT\nEXPLANATION: Not quite.")
        
        node = DMHypothesisNode(
            llm_client=llm,
            agents_config=MockAgentsConfig(),
        )
        
        with caplog.at_level("INFO"):
            result = await node(base_state)
        
        assert result["game_phase"] == GamePhase.PLAYING
        assert "GAME_EVENT" in caplog.text


class TestCommandHandlerNodeWithProfile:
    @pytest.mark.asyncio
    async def test_hint_adapts_to_beginner_profile(self, base_state):
        base_state.last_user_message = "/hint"
        
        memory_manager = MockMemoryManager([
            MockProfileSearchResult("style_profile", "Skill Level: Beginner"),
        ])
        
        node = CommandHandlerNode(
            memory_manager=memory_manager,
            agents_config=MockAgentsConfig(),
        )
        
        result = await node(base_state)
        
        assert "hint_count" in result
        assert result["hint_count"] == 1
    
    @pytest.mark.asyncio
    async def test_hint_logs_event(self, base_state, caplog):
        base_state.last_user_message = "/hint"
        
        node = CommandHandlerNode(agents_config=MockAgentsConfig())
        
        with caplog.at_level("INFO"):
            result = await node(base_state)
        
        assert "GAME_EVENT" in caplog.text
        assert "hint_used" in caplog.text


class TestIntroNodeWithLogging:
    @pytest.mark.asyncio
    async def test_intro_logs_session_start(self, base_state, caplog):
        base_state.game_phase = GamePhase.INTRO
        
        node = IntroNode(agents_config=MockAgentsConfig())
        
        with caplog.at_level("INFO"):
            result = await node(base_state)
        
        assert result["game_phase"] == GamePhase.PLAYING
        assert "GAME_EVENT" in caplog.text
        assert "session_start" in caplog.text


class TestMemoryUpdateNodeWithPhaseLogging:
    @pytest.mark.asyncio
    async def test_memory_update_logs_completed_session(self, base_state, caplog):
        base_state.game_phase = GamePhase.COMPLETED
        base_state.score = 850
        base_state.turn_history = [
            TurnEvent(turn_index=0, role="player", message="Q1", tags=["question"]),
            TurnEvent(turn_index=1, role="dm", message="Yes", tags=["answer"]),
        ]
        
        memory_manager = MockMemoryManager()
        node = MemoryUpdateNode(memory_manager=memory_manager)
        
        with caplog.at_level("INFO"):
            result = await node(base_state)
        
        assert "GAME_EVENT" in caplog.text
        assert "session_end" in caplog.text
        assert len(memory_manager.summarized_sessions) == 1
    
    @pytest.mark.asyncio
    async def test_memory_update_logs_aborted_session(self, base_state, caplog):
        base_state.game_phase = GamePhase.ABORTED
        base_state.turn_history = [
            TurnEvent(turn_index=0, role="player", message="Q1", tags=["question"]),
        ]
        
        memory_manager = MockMemoryManager()
        node = MemoryUpdateNode(memory_manager=memory_manager)
        
        with caplog.at_level("INFO"):
            result = await node(base_state)
        
        assert "GAME_EVENT" in caplog.text
        assert "session_end" in caplog.text
        assert "aborted" in caplog.text.lower() or "false" in caplog.text.lower()
    
    @pytest.mark.asyncio
    async def test_memory_update_no_log_during_play(self, base_state, caplog):
        base_state.game_phase = GamePhase.PLAYING
        base_state.turn_history = [
            TurnEvent(turn_index=0, role="player", message="Q1", tags=["question"]),
        ]
        
        memory_manager = MockMemoryManager()
        node = MemoryUpdateNode(memory_manager=memory_manager)
        
        with caplog.at_level("INFO"):
            result = await node(base_state)
        
        assert "session_end" not in caplog.text
