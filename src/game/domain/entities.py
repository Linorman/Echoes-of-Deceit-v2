"""Domain entities for the Turtle Soup game system.

This module defines all core domain entities including:
- Puzzle: A situation puzzle with statement, answer, and constraints
- Game: A collection of puzzles (simple mode: 1:1 with puzzle)
- GameSession: A single playthrough session
- PlayerProfile: Player information and preferences
- AgentRole: Roles in the game (DM, Player, Observer, etc.)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    DM = "dm"
    PLAYER = "player"
    OBSERVER = "observer"
    HINT_MASTER = "hint_master"


class GameState(str, Enum):
    LOBBY = "lobby"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABORTED = "aborted"


class PuzzleConstraints(BaseModel):
    max_questions: Optional[int] = None
    max_hints: int = 5
    allowed_question_types: List[str] = Field(
        default_factory=lambda: ["yes_no", "yes_and_no", "irrelevant"]
    )
    time_limit_minutes: Optional[int] = None


class Puzzle(BaseModel):
    id: str
    title: str = ""
    description: str = ""
    puzzle_statement: str
    answer: str
    hints: List[str] = Field(default_factory=list)
    additional_info: List[Dict[str, Any]] = Field(default_factory=list)
    constraints: PuzzleConstraints = Field(default_factory=PuzzleConstraints)
    tags: List[str] = Field(default_factory=list)
    language: str = "en"
    difficulty: Optional[str] = None

    @property
    def has_hints(self) -> bool:
        return len(self.hints) > 0


class PuzzleSummary(BaseModel):
    id: str
    title: str
    description: str = ""
    difficulty: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    language: str = "en"


class Game(BaseModel):
    id: str
    name: str
    description: str = ""
    puzzle_ids: List[str] = Field(default_factory=list)
    game_type: str = "situation_puzzle"
    created_at: datetime = Field(default_factory=datetime.now)


class PlayerProfile(BaseModel):
    player_id: str
    display_name: str
    preferences: Dict[str, Any] = Field(default_factory=dict)
    long_term_memory_ref: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def update_preference(self, key: str, value: Any) -> None:
        self.preferences[key] = value
        self.updated_at = datetime.now()


class SessionEvent(BaseModel):
    session_id: str
    turn_index: int
    timestamp: datetime = Field(default_factory=datetime.now)
    role: AgentRole
    message: str
    tags: List[str] = Field(default_factory=list)
    raw_tool_calls: Optional[Dict[str, Any]] = None
    verdict: Optional[str] = None


class SessionConfig(BaseModel):
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:7b"
    rag_provider: str = "lightrag"
    max_turns: int = 100
    hint_limit: int = 5


class GameSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    puzzle_id: str
    player_ids: List[str] = Field(default_factory=list)
    state: GameState = GameState.LOBBY
    turn_history: List[SessionEvent] = Field(default_factory=list)
    config: SessionConfig = Field(default_factory=SessionConfig)
    hint_count: int = 0
    score: Optional[int] = None
    kb_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    schema_version: str = "1.0"

    @property
    def turn_count(self) -> int:
        return len(self.turn_history)

    @property
    def question_count(self) -> int:
        """Returns the actual number of questions asked (real turns)."""
        return sum(
            1 for event in self.turn_history
            if "question" in event.tags
        )

    @property
    def is_active(self) -> bool:
        return self.state == GameState.IN_PROGRESS

    @property
    def is_completed(self) -> bool:
        return self.state == GameState.COMPLETED

    def start(self) -> None:
        if self.state != GameState.LOBBY:
            raise ValueError(f"Cannot start session in state: {self.state}")
        self.state = GameState.IN_PROGRESS
        self.updated_at = datetime.now()

    def complete(self, score: Optional[int] = None) -> None:
        if self.state != GameState.IN_PROGRESS:
            raise ValueError(f"Cannot complete session in state: {self.state}")
        self.state = GameState.COMPLETED
        self.score = score
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()

    def abort(self) -> None:
        if self.state not in (GameState.LOBBY, GameState.IN_PROGRESS):
            raise ValueError(f"Cannot abort session in state: {self.state}")
        self.state = GameState.ABORTED
        self.updated_at = datetime.now()

    def add_event(self, role: AgentRole, message: str, tags: Optional[List[str]] = None) -> SessionEvent:
        event = SessionEvent(
            session_id=self.session_id,
            turn_index=self.turn_count,
            role=role,
            message=message,
            tags=tags or [],
        )
        self.turn_history.append(event)
        self.updated_at = datetime.now()
        return event

    def get_recent_events(self, limit: int = 10) -> List[SessionEvent]:
        return self.turn_history[-limit:] if self.turn_history else []

    def use_hint(self) -> bool:
        if self.hint_count >= self.config.hint_limit:
            return False
        self.hint_count += 1
        self.updated_at = datetime.now()
        return True
