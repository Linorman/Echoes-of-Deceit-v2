"""Tests for LangGraph Builder and Runner."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from game.graph.state import GameGraphState, GamePhase, MessageType
from game.graph.builder import (
    GameGraphBuilder,
    route_by_phase,
    route_by_message_type,
    route_after_dm,
    route_after_command,
    NODE_PLAYER_MESSAGE,
    NODE_INTRO,
    NODE_DM_QUESTION,
    NODE_DM_HYPOTHESIS,
    NODE_COMMAND_HANDLER,
    NODE_MEMORY_UPDATE,
    NODE_REVEAL_SOLUTION,
)
from game.graph.runner import GameGraphRunner, GameGraphRunnerFactory
from game.domain.entities import GameSession, Puzzle, PuzzleConstraints, SessionConfig
from config.models import AgentsConfig, DMConfig, DMPersonaConfig


@pytest.fixture
def sample_puzzle():
    return Puzzle(
        id="test_puzzle",
        title="Test Puzzle",
        description="A test puzzle",
        puzzle_statement="A man walks into a bar...",
        answer="He had hiccups.",
        hints=["Think about why", "The gun wasn't meant to hurt"],
        constraints=PuzzleConstraints(max_hints=3),
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
        dm=DMConfig(persona=DMPersonaConfig(tone="mysterious")),
    )


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.agenerate = AsyncMock(return_value="VERDICT: YES\nEXPLANATION: Correct!")
    return client


class TestRoutingFunctions:
    def test_route_by_phase_intro(self):
        state = GameGraphState(
            session_id="test",
            kb_id="kb1",
            player_id="p1",
            puzzle_id="pz1",
            game_phase=GamePhase.INTRO,
        )
        assert route_by_phase(state) == NODE_INTRO

    def test_route_by_phase_playing(self):
        state = GameGraphState(
            session_id="test",
            kb_id="kb1",
            player_id="p1",
            puzzle_id="pz1",
            game_phase=GamePhase.PLAYING,
        )
        assert route_by_phase(state) == NODE_PLAYER_MESSAGE

    def test_route_by_phase_completed(self):
        from langgraph.graph import END
        state = GameGraphState(
            session_id="test",
            kb_id="kb1",
            player_id="p1",
            puzzle_id="pz1",
            game_phase=GamePhase.COMPLETED,
        )
        assert route_by_phase(state) == END

    def test_route_by_message_type_question(self):
        state = GameGraphState(
            session_id="test",
            kb_id="kb1",
            player_id="p1",
            puzzle_id="pz1",
            message_type=MessageType.QUESTION,
        )
        assert route_by_message_type(state) == NODE_DM_QUESTION

    def test_route_by_message_type_hypothesis(self):
        state = GameGraphState(
            session_id="test",
            kb_id="kb1",
            player_id="p1",
            puzzle_id="pz1",
            message_type=MessageType.HYPOTHESIS,
        )
        assert route_by_message_type(state) == NODE_DM_HYPOTHESIS

    def test_route_by_message_type_command(self):
        state = GameGraphState(
            session_id="test",
            kb_id="kb1",
            player_id="p1",
            puzzle_id="pz1",
            message_type=MessageType.COMMAND,
        )
        assert route_by_message_type(state) == NODE_COMMAND_HANDLER

    def test_route_after_dm_completed(self):
        state = GameGraphState(
            session_id="test",
            kb_id="kb1",
            player_id="p1",
            puzzle_id="pz1",
            game_phase=GamePhase.COMPLETED,
        )
        assert route_after_dm(state) == NODE_REVEAL_SOLUTION

    def test_route_after_dm_playing(self):
        state = GameGraphState(
            session_id="test",
            kb_id="kb1",
            player_id="p1",
            puzzle_id="pz1",
            game_phase=GamePhase.PLAYING,
        )
        assert route_after_dm(state) == NODE_MEMORY_UPDATE

    def test_route_after_command_aborted(self):
        from langgraph.graph import END
        state = GameGraphState(
            session_id="test",
            kb_id="kb1",
            player_id="p1",
            puzzle_id="pz1",
            game_phase=GamePhase.ABORTED,
        )
        assert route_after_command(state) == END

    def test_route_after_command_playing(self):
        state = GameGraphState(
            session_id="test",
            kb_id="kb1",
            player_id="p1",
            puzzle_id="pz1",
            game_phase=GamePhase.PLAYING,
        )
        assert route_after_command(state) == NODE_MEMORY_UPDATE


class TestGameGraphBuilder:
    def test_builder_creation(self, agents_config, mock_llm_client):
        builder = GameGraphBuilder(
            llm_client=mock_llm_client,
            agents_config=agents_config,
        )
        
        assert builder._llm_client is mock_llm_client
        assert builder._agents_config is agents_config

    def test_builder_creates_nodes(self, agents_config, mock_llm_client):
        builder = GameGraphBuilder(
            llm_client=mock_llm_client,
            agents_config=agents_config,
        )
        
        nodes = builder._nodes
        assert NODE_PLAYER_MESSAGE in nodes
        assert NODE_INTRO in nodes
        assert NODE_DM_QUESTION in nodes
        assert NODE_DM_HYPOTHESIS in nodes
        assert NODE_COMMAND_HANDLER in nodes
        assert NODE_MEMORY_UPDATE in nodes
        assert NODE_REVEAL_SOLUTION in nodes

    def test_builder_build_returns_state_graph(self, agents_config):
        builder = GameGraphBuilder(agents_config=agents_config)
        
        graph = builder.build()
        
        assert graph is not None

    def test_builder_compile(self, agents_config):
        builder = GameGraphBuilder(agents_config=agents_config)
        
        compiled = builder.compile()
        
        assert compiled is not None

    def test_builder_with_memory_saver(self, agents_config, mock_llm_client):
        builder = GameGraphBuilder.create_with_memory_saver(
            llm_client=mock_llm_client,
            agents_config=agents_config,
        )
        
        assert builder._checkpointer is not None


class TestGameGraphRunner:
    def test_runner_creation(self, sample_session, sample_puzzle, agents_config, mock_llm_client):
        runner = GameGraphRunner(
            session=sample_session,
            puzzle=sample_puzzle,
            llm_client=mock_llm_client,
            agents_config=agents_config,
        )
        
        assert runner.session == sample_session
        assert runner.puzzle == sample_puzzle
        assert runner.thread_id == sample_session.session_id

    def test_runner_is_active_initial(self, sample_session, sample_puzzle):
        sample_session.start()
        runner = GameGraphRunner(
            session=sample_session,
            puzzle=sample_puzzle,
        )
        
        assert runner.is_active is True

    def test_runner_get_config(self, sample_session, sample_puzzle):
        runner = GameGraphRunner(
            session=sample_session,
            puzzle=sample_puzzle,
        )
        
        config = runner._get_config()
        
        assert "configurable" in config
        assert config["configurable"]["thread_id"] == sample_session.session_id

    def test_runner_create_initial_state(self, sample_session, sample_puzzle):
        runner = GameGraphRunner(
            session=sample_session,
            puzzle=sample_puzzle,
        )
        
        state = runner._create_initial_state()
        
        assert state["session_id"] == sample_session.session_id
        assert state["puzzle_id"] == sample_puzzle.id
        assert state["puzzle_statement"] == sample_puzzle.puzzle_statement
        assert state["puzzle_answer"] == sample_puzzle.answer
        assert state["game_phase"] == GamePhase.INTRO


class TestGameGraphRunnerFactory:
    def test_factory_creation(self, mock_llm_client, agents_config):
        factory = GameGraphRunnerFactory(
            llm_client=mock_llm_client,
            agents_config=agents_config,
        )
        
        assert factory._llm_client is mock_llm_client
        assert factory._agents_config is agents_config

    def test_factory_create_runner(
        self, sample_session, sample_puzzle, mock_llm_client, agents_config
    ):
        factory = GameGraphRunnerFactory(
            llm_client=mock_llm_client,
            agents_config=agents_config,
        )
        
        runner = factory.create_runner(sample_session, sample_puzzle)
        
        assert runner is not None
        assert runner.session == sample_session

    def test_factory_get_runner(
        self, sample_session, sample_puzzle, mock_llm_client, agents_config
    ):
        factory = GameGraphRunnerFactory(
            llm_client=mock_llm_client,
            agents_config=agents_config,
        )
        
        created_runner = factory.create_runner(sample_session, sample_puzzle)
        retrieved_runner = factory.get_runner(sample_session.session_id)
        
        assert retrieved_runner is created_runner

    def test_factory_get_nonexistent_runner(self, mock_llm_client, agents_config):
        factory = GameGraphRunnerFactory(
            llm_client=mock_llm_client,
            agents_config=agents_config,
        )
        
        runner = factory.get_runner("nonexistent")
        
        assert runner is None

    def test_factory_remove_runner(
        self, sample_session, sample_puzzle, mock_llm_client, agents_config
    ):
        factory = GameGraphRunnerFactory(
            llm_client=mock_llm_client,
            agents_config=agents_config,
        )
        
        factory.create_runner(sample_session, sample_puzzle)
        factory.remove_runner(sample_session.session_id)
        
        assert factory.get_runner(sample_session.session_id) is None
