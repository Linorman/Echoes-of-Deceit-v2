"""Memory entities for session and player memory management.

This module defines:
- SessionEventRecord: Enhanced session event for episodic memory
- PlayerMemoryRecord: Long-term player semantic memory
- GlobalMemoryRecord: System-level global memory
- MemoryDocument: Generic memory document for store abstraction
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EventTag(str, Enum):
    QUESTION = "question"
    ANSWER = "answer"
    HINT = "hint"
    HYPOTHESIS = "hypothesis"
    FINAL_VERDICT = "final_verdict"
    SYSTEM = "system"
    NARRATION = "narration"
    INTRO = "intro"
    RECAP = "recap"


class MemorySummaryType(str, Enum):
    STYLE_PROFILE = "style_profile"
    PERFORMANCE_SUMMARY = "performance_summary"
    SESSION_SUMMARY = "session_summary"
    HINT_STRATEGY = "hint_strategy"
    PUZZLE_STATS = "puzzle_stats"
    BEHAVIOR_PATTERN = "behavior_pattern"


class SessionEventRecord(BaseModel):
    session_id: str
    turn_index: int
    timestamp: datetime = Field(default_factory=datetime.now)
    role: str
    message: str
    event_type: str = "message"
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    raw_tool_calls: Optional[Dict[str, Any]] = None

    def has_tag(self, tag: str | EventTag) -> bool:
        tag_value = tag.value if isinstance(tag, EventTag) else tag
        return tag_value in self.tags

    def add_tag(self, tag: str | EventTag) -> None:
        tag_value = tag.value if isinstance(tag, EventTag) else tag
        if tag_value not in self.tags:
            self.tags.append(tag_value)


class PlayerMemoryRecord(BaseModel):
    player_id: str
    summary_type: MemorySummaryType
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    version: int = 1

    def update_content(self, new_content: str) -> None:
        self.content = new_content
        self.updated_at = datetime.now()
        self.version += 1


class GlobalMemoryRecord(BaseModel):
    record_id: str
    summary_type: MemorySummaryType
    content: str
    puzzle_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def update_content(self, new_content: str) -> None:
        self.content = new_content
        self.updated_at = datetime.now()


class MemoryDocument(BaseModel):
    namespace: str
    key: str
    value: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @classmethod
    def from_session_event(cls, event: SessionEventRecord) -> "MemoryDocument":
        return cls(
            namespace=f"session:{event.session_id}",
            key=f"event_{event.turn_index}",
            value={
                "session_id": event.session_id,
                "turn_index": event.turn_index,
                "role": event.role,
                "message": event.message,
                "event_type": event.event_type,
                "tags": event.tags,
            },
            metadata={
                "timestamp": event.timestamp.isoformat(),
                "tags": event.tags,
                **event.metadata,
            },
            created_at=event.timestamp,
        )

    @classmethod
    def from_player_memory(cls, record: PlayerMemoryRecord) -> "MemoryDocument":
        return cls(
            namespace=f"player:{record.player_id}",
            key=record.summary_type.value,
            value={
                "player_id": record.player_id,
                "summary_type": record.summary_type.value,
                "content": record.content,
                "version": record.version,
            },
            metadata={
                "summary_type": record.summary_type.value,
                **record.metadata,
            },
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    @classmethod
    def from_global_memory(cls, record: GlobalMemoryRecord) -> "MemoryDocument":
        return cls(
            namespace="global",
            key=record.record_id,
            value={
                "record_id": record.record_id,
                "summary_type": record.summary_type.value,
                "content": record.content,
                "puzzle_id": record.puzzle_id,
            },
            metadata={
                "summary_type": record.summary_type.value,
                "puzzle_id": record.puzzle_id,
                **record.metadata,
            },
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
