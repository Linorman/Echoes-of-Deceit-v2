"""Tests for LangGraph State Schema."""

import pytest
from datetime import datetime

from game.graph.state import (
    GameGraphState,
    GameGraphInput,
    GameGraphOutput,
    MessageType,
    GamePhase,
    DMVerdict,
    HypothesisVerdict,
    TurnEvent,
    merge_turn_history,
)


class TestEnums:
    def test_message_type_values(self):
        assert MessageType.QUESTION.value == "question"
        assert MessageType.HYPOTHESIS.value == "hypothesis"
        assert MessageType.COMMAND.value == "command"
        assert MessageType.UNKNOWN.value == "unknown"

    def test_game_phase_values(self):
        assert GamePhase.INTRO.value == "intro"
        assert GamePhase.PLAYING.value == "playing"
        assert GamePhase.AWAITING_FINAL.value == "awaiting_final"
        assert GamePhase.COMPLETED.value == "completed"
        assert GamePhase.ABORTED.value == "aborted"

    def test_dm_verdict_values(self):
        assert DMVerdict.YES.value == "YES"
        assert DMVerdict.NO.value == "NO"
        assert DMVerdict.YES_AND_NO.value == "YES_AND_NO"
        assert DMVerdict.IRRELEVANT.value == "IRRELEVANT"

    def test_hypothesis_verdict_values(self):
        assert HypothesisVerdict.CORRECT.value == "correct"
        assert HypothesisVerdict.INCORRECT.value == "incorrect"
        assert HypothesisVerdict.PARTIAL.value == "partial"


class TestTurnEvent:
    def test_turn_event_creation(self):
        event = TurnEvent(
            turn_index=0,
            role="player",
            message="Is the man alive?",
            tags=["question"],
        )
        assert event.turn_index == 0
        assert event.role == "player"
        assert event.message == "Is the man alive?"
        assert "question" in event.tags
        assert event.verdict is None

    def test_turn_event_with_verdict(self):
        event = TurnEvent(
            turn_index=1,
            role="dm",
            message="YES",
            tags=["answer"],
            verdict="YES",
        )
        assert event.verdict == "YES"

    def test_turn_event_default_timestamp(self):
        before = datetime.now()
        event = TurnEvent(turn_index=0, role="player", message="test")
        after = datetime.now()
        assert before <= event.timestamp <= after


class TestMergeTurnHistory:
    def test_merge_empty_new(self):
        existing = [TurnEvent(turn_index=0, role="player", message="Q1")]
        result = merge_turn_history(existing, [])
        assert result == existing

    def test_merge_empty_existing(self):
        new = [TurnEvent(turn_index=0, role="player", message="Q1")]
        result = merge_turn_history([], new)
        assert result == new

    def test_merge_both_populated(self):
        existing = [TurnEvent(turn_index=0, role="player", message="Q1")]
        new = [TurnEvent(turn_index=1, role="dm", message="YES")]
        result = merge_turn_history(existing, new)
        assert len(result) == 2
        assert result[0].message == "Q1"
        assert result[1].message == "YES"


class TestGameGraphState:
    def test_state_creation_minimal(self):
        state = GameGraphState(
            session_id="test-session",
            kb_id="game_puzzle1",
            player_id="player1",
            puzzle_id="puzzle1",
        )
        assert state.session_id == "test-session"
        assert state.kb_id == "game_puzzle1"
        assert state.player_id == "player1"
        assert state.puzzle_id == "puzzle1"
        assert state.turn_index == 0
        assert state.game_phase == GamePhase.INTRO
        assert state.hint_count == 0
        assert len(state.turn_history) == 0

    def test_state_creation_full(self):
        state = GameGraphState(
            session_id="test-session",
            kb_id="game_puzzle1",
            player_id="player1",
            puzzle_id="puzzle1",
            turn_index=5,
            last_user_message="Is it about water?",
            message_type=MessageType.QUESTION,
            last_dm_response="YES",
            last_verdict="YES",
            game_phase=GamePhase.PLAYING,
            hint_count=2,
            score=800,
            puzzle_statement="A man walks into a bar...",
            puzzle_answer="He had hiccups.",
            puzzle_title="The Bar",
            max_hints=5,
        )
        assert state.turn_index == 5
        assert state.message_type == MessageType.QUESTION
        assert state.game_phase == GamePhase.PLAYING
        assert state.hint_count == 2
        assert state.score == 800

    def test_state_with_turn_history(self):
        events = [
            TurnEvent(turn_index=0, role="dm", message="Welcome!", tags=["intro"]),
            TurnEvent(turn_index=1, role="player", message="Is he alive?", tags=["question"]),
        ]
        state = GameGraphState(
            session_id="test",
            kb_id="kb1",
            player_id="p1",
            puzzle_id="pz1",
            turn_history=events,
        )
        assert len(state.turn_history) == 2


class TestGameGraphInput:
    def test_input_creation(self):
        input_data = GameGraphInput(user_message="Is the man alive?")
        assert input_data.user_message == "Is the man alive?"


class TestGameGraphOutput:
    def test_output_minimal(self):
        output = GameGraphOutput(message="YES")
        assert output.message == "YES"
        assert output.verdict is None
        assert output.game_over is False
        assert output.game_phase == ""
        assert output.turn_index == 0

    def test_output_full(self):
        output = GameGraphOutput(
            message="🎉 CORRECT!",
            verdict="correct",
            game_over=True,
            game_phase="completed",
            turn_index=10,
            metadata={"score": 800, "hint_count": 2},
        )
        assert output.verdict == "correct"
        assert output.game_over is True
        assert output.game_phase == "completed"
        assert output.metadata["score"] == 800
