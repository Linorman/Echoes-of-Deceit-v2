"""LangGraph State Schema for Game Session.

This module defines the state schema used by LangGraph to track
the game session state across graph executions.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Annotated
import operator

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    QUESTION = "question"
    HYPOTHESIS = "hypothesis"
    COMMAND = "command"
    PLAYER_AGENT_TURN = "player_agent_turn"
    UNKNOWN = "unknown"


class GamePhase(str, Enum):
    INTRO = "intro"
    PLAYING = "playing"
    AWAITING_FINAL = "awaiting_final"
    COMPLETED = "completed"
    ABORTED = "aborted"


class GameMode(str, Enum):
    HUMAN_PLAYER = "human_player"
    AI_PLAYER = "ai_player"
    MIXED = "mixed"


class DMVerdict(str, Enum):
    YES = "YES"
    NO = "NO"
    YES_AND_NO = "YES_AND_NO"
    IRRELEVANT = "IRRELEVANT"


class HypothesisVerdict(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIAL = "partial"


class TurnEvent(BaseModel):
    turn_index: int
    timestamp: datetime = Field(default_factory=datetime.now)
    role: str
    message: str
    tags: List[str] = Field(default_factory=list)
    verdict: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


def merge_turn_history(
    existing: List[TurnEvent],
    new: List[TurnEvent],
) -> List[TurnEvent]:
    if not new:
        return existing
    return existing + new


class GameGraphState(BaseModel):
    session_id: str
    kb_id: str
    player_id: str
    puzzle_id: str

    turn_index: int = 0
    last_user_message: str = ""
    message_type: MessageType = MessageType.UNKNOWN
    last_dm_response: str = ""
    last_verdict: Optional[str] = None

    game_phase: GamePhase = GamePhase.INTRO
    hint_count: int = 0
    score: Optional[int] = None

    turn_history: Annotated[List[TurnEvent], merge_turn_history] = Field(
        default_factory=list
    )

    puzzle_statement: str = ""
    puzzle_answer: str = ""
    puzzle_title: str = ""
    max_hints: int = 5

    game_mode: GameMode = GameMode.HUMAN_PLAYER
    player_agent_enabled: bool = False
    player_agent_question_count: int = 0
    awaiting_player_agent: bool = False

    error: Optional[str] = None

    class Config:
        use_enum_values = True


class GameGraphInput(BaseModel):
    user_message: str


class GameGraphOutput(BaseModel):
    message: str
    verdict: Optional[str] = None
    game_over: bool = False
    game_phase: str = ""
    turn_index: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
