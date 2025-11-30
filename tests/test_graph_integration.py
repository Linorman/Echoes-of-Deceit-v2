"""Integration tests for LangGraph game flow."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from game.graph.state import GameGraphState, GamePhase, MessageType
from game.graph.builder import GameGraphBuilder
from game.graph.runner import GameGraphRunner
from game.domain.entities import GameSession, Puzzle, PuzzleConstraints, SessionConfig
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
        id="hiccups_puzzle",
        title="The Bar Incident",
        description="A classic lateral thinking puzzle",
        puzzle_statement="A man walks into a bar and asks the bartender for a glass of water. The bartender pulls out a gun and points it at the man. The man says 'thank you' and leaves. Why?",
        answer="The man had hiccups. He asked for water to cure them, but when the bartender pointed the gun at him, it scared him so much that his hiccups went away. That's why he thanked the bartender and left.",
        hints=[
            "Think about why someone might suddenly not need water anymore.",
            "The bartender was trying to help, not harm.",
            "What physical condition might cause someone to need water urgently?",
        ],
        constraints=PuzzleConstraints(max_hints=3, max_questions=50),
    )


@pytest.fixture
def sample_session(sample_puzzle):
    session = GameSession(
        puzzle_id=sample_puzzle.id,
        player_ids=["test_player_1"],
        kb_id="game_hiccups_puzzle",
        config=SessionConfig(),
    )
    return session


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
                max_explanation_length=150,
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
    
    async def smart_response(prompt, **params):
        prompt_lower = prompt.lower()
        
        if "hypothesis" in prompt_lower or "player's hypothesis" in prompt_lower:
            if "hiccup" in prompt_lower:
                return "VERDICT: CORRECT\nEXPLANATION: Excellent deduction! The man indeed had hiccups."
            else:
                return "VERDICT: INCORRECT\nEXPLANATION: That's not quite right."
        
        if "gun" in prompt_lower and "scary" in prompt_lower:
            return "VERDICT: YES\nEXPLANATION: You're on the right track!"
        elif "water" in prompt_lower and "drink" in prompt_lower:
            return "VERDICT: NO\nEXPLANATION: He didn't actually need to drink the water."
        elif "hiccup" in prompt_lower:
            return "VERDICT: YES\nEXPLANATION: This is very relevant to the puzzle!"
        elif "bartender" in prompt_lower and "friend" in prompt_lower:
            return "VERDICT: IRRELEVANT\nEXPLANATION: Their relationship isn't key to this puzzle."
        else:
            return "VERDICT: NO\nEXPLANATION: Try a different approach."
    
    client.agenerate = AsyncMock(side_effect=smart_response)
    return client


@pytest.fixture
def mock_memory_manager():
    manager = MagicMock()
    manager.append_session_event = MagicMock()
    manager.summarize_session = MagicMock(return_value="Session summary generated")
    return manager


class TestGameFlowIntegration:
    @pytest.mark.asyncio
    async def test_start_game_produces_intro(
        self, sample_session, sample_puzzle, agents_config, mock_llm_client, mock_memory_manager
    ):
        runner = GameGraphRunner(
            session=sample_session,
            puzzle=sample_puzzle,
            llm_client=mock_llm_client,
            memory_manager=mock_memory_manager,
            agents_config=agents_config,
        )
        
        result = await runner.start_game()
        
        assert result.message
        assert sample_puzzle.title in result.message or "mystery" in result.message.lower()
        assert result.game_phase == GamePhase.PLAYING

    @pytest.mark.asyncio
    async def test_question_flow(
        self, sample_session, sample_puzzle, agents_config, mock_llm_client, mock_memory_manager
    ):
        runner = GameGraphRunner(
            session=sample_session,
            puzzle=sample_puzzle,
            llm_client=mock_llm_client,
            memory_manager=mock_memory_manager,
            agents_config=agents_config,
        )
        
        await runner.start_game()
        
        result = await runner.process_input("Did the man have hiccups?")
        
        assert result.message
        assert "YES" in result.message
        assert result.game_over is False

    @pytest.mark.asyncio
    async def test_command_flow(
        self, sample_session, sample_puzzle, agents_config, mock_llm_client, mock_memory_manager
    ):
        runner = GameGraphRunner(
            session=sample_session,
            puzzle=sample_puzzle,
            llm_client=mock_llm_client,
            memory_manager=mock_memory_manager,
            agents_config=agents_config,
        )
        
        await runner.start_game()
        
        result = await runner.process_input("/status")
        
        assert "Game Status" in result.message
        assert sample_puzzle.title in result.message

    @pytest.mark.asyncio
    async def test_hint_command(
        self, sample_session, sample_puzzle, agents_config, mock_llm_client, mock_memory_manager
    ):
        runner = GameGraphRunner(
            session=sample_session,
            puzzle=sample_puzzle,
            llm_client=mock_llm_client,
            memory_manager=mock_memory_manager,
            agents_config=agents_config,
        )
        
        await runner.start_game()
        
        result = await runner.process_input("/hint")
        
        assert "hint" in result.message.lower() or "nudge" in result.message.lower()

    @pytest.mark.asyncio
    async def test_quit_command(
        self, sample_session, sample_puzzle, agents_config, mock_llm_client, mock_memory_manager
    ):
        runner = GameGraphRunner(
            session=sample_session,
            puzzle=sample_puzzle,
            llm_client=mock_llm_client,
            memory_manager=mock_memory_manager,
            agents_config=agents_config,
        )
        
        await runner.start_game()
        
        result = await runner.process_input("/quit")
        
        assert result.game_over is True
        assert "ended" in result.message.lower()

    @pytest.mark.asyncio
    async def test_correct_hypothesis_ends_game(
        self, sample_session, sample_puzzle, agents_config, mock_llm_client, mock_memory_manager
    ):
        runner = GameGraphRunner(
            session=sample_session,
            puzzle=sample_puzzle,
            llm_client=mock_llm_client,
            memory_manager=mock_memory_manager,
            agents_config=agents_config,
        )
        
        await runner.start_game()
        
        result = await runner.process_input("I think the man had hiccups and the gun scared them away!")
        
        assert result.game_over is True
        assert "Complete" in result.message or "hiccups" in result.message

    @pytest.mark.asyncio
    async def test_full_game_flow(
        self, sample_session, sample_puzzle, agents_config, mock_llm_client, mock_memory_manager
    ):
        runner = GameGraphRunner(
            session=sample_session,
            puzzle=sample_puzzle,
            llm_client=mock_llm_client,
            memory_manager=mock_memory_manager,
            agents_config=agents_config,
        )
        
        intro = await runner.start_game()
        assert intro.game_phase == GamePhase.PLAYING
        assert runner.is_active
        
        q1 = await runner.process_input("Was the bartender a friend of the man?")
        assert q1.message
        assert runner.is_active
        
        q2 = await runner.process_input("Did the man have hiccups?")
        assert "YES" in q2.message or "yes" in q2.message.lower()
        assert runner.is_active
        
        hint = await runner.process_input("/hint")
        assert hint.message
        assert runner.is_active
        
        final = await runner.process_input("I think the man had hiccups and the bartender scared them away with the gun")
        assert "Complete" in final.message or "hiccups" in final.message
        assert final.game_over is True
        assert not runner.is_active


class TestGraphBuilderIntegration:
    def test_graph_compiles_successfully(self, agents_config, mock_llm_client):
        builder = GameGraphBuilder(
            llm_client=mock_llm_client,
            agents_config=agents_config,
        )
        
        compiled = builder.compile()
        
        assert compiled is not None

    def test_graph_with_checkpointer(self, agents_config, mock_llm_client):
        builder = GameGraphBuilder.create_with_memory_saver(
            llm_client=mock_llm_client,
            agents_config=agents_config,
        )
        
        compiled = builder.compile()
        
        assert compiled is not None
        assert builder._checkpointer is not None


class TestStateManagement:
    @pytest.mark.asyncio
    async def test_turn_history_accumulates(
        self, sample_session, sample_puzzle, agents_config, mock_llm_client, mock_memory_manager
    ):
        runner = GameGraphRunner(
            session=sample_session,
            puzzle=sample_puzzle,
            llm_client=mock_llm_client,
            memory_manager=mock_memory_manager,
            agents_config=agents_config,
        )
        
        await runner.start_game()
        await runner.process_input("Question 1")
        await runner.process_input("Question 2")
        
        history = runner.get_turn_history()
        
        assert len(history) >= 4

    @pytest.mark.asyncio
    async def test_hint_count_increments(
        self, sample_session, sample_puzzle, agents_config, mock_llm_client, mock_memory_manager
    ):
        runner = GameGraphRunner(
            session=sample_session,
            puzzle=sample_puzzle,
            llm_client=mock_llm_client,
            memory_manager=mock_memory_manager,
            agents_config=agents_config,
        )
        
        await runner.start_game()
        
        initial_state = runner.get_state()
        initial_hints = initial_state.hint_count if initial_state else 0
        
        await runner.process_input("/hint")
        
        final_state = runner.get_state()
        assert final_state.hint_count == initial_hints + 1
