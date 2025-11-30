"""Memory Manager for session and player memory operations.

This module provides the MemoryManager service that implements all memory-related
operations as defined in the design:
- append_session_event: Add events to session history
- get_session_history: Retrieve recent session events
- summarize_session: Generate session summaries
- update_player_profile: Update player long-term memory
- retrieve_player_profile: Get player profile for DM/Judge prompts
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from config import ConfigLoader, GameConfig
from game.memory.base_store import BaseMemoryStore, MemorySearchResult
from game.memory.entities import (
    EventTag,
    GlobalMemoryRecord,
    MemoryDocument,
    MemorySummaryType,
    PlayerMemoryRecord,
    SessionEventRecord,
)
from game.memory.file_store import FileMemoryStore

logger = logging.getLogger(__name__)

SummarizerFn = Callable[[List[SessionEventRecord]], str]
MergerFn = Callable[[str, str], str]


def default_summarizer(events: List[SessionEventRecord]) -> str:
    if not events:
        return "No events recorded."

    question_count = sum(1 for e in events if e.has_tag(EventTag.QUESTION))
    hint_count = sum(1 for e in events if e.has_tag(EventTag.HINT))
    hypothesis_count = sum(1 for e in events if e.has_tag(EventTag.HYPOTHESIS))

    final_verdict = None
    for e in reversed(events):
        if e.has_tag(EventTag.FINAL_VERDICT):
            final_verdict = e.message
            break

    summary_parts = [
        f"Session contained {len(events)} total events.",
        f"Questions asked: {question_count}",
        f"Hints used: {hint_count}",
        f"Hypotheses proposed: {hypothesis_count}",
    ]

    if final_verdict:
        summary_parts.append(f"Final verdict: {final_verdict}")

    return " ".join(summary_parts)


def default_merger(existing: str, new_info: str) -> str:
    if not existing:
        return new_info
    return f"{existing}\n\nUpdated: {new_info}"


class MemoryManager:
    def __init__(
        self,
        store: Optional[BaseMemoryStore] = None,
        config: Optional[GameConfig] = None,
        base_dir: Optional[Path] = None,
        summarizer: Optional[SummarizerFn] = None,
        merger: Optional[MergerFn] = None,
    ):
        if config is None:
            loader = ConfigLoader()
            config = loader.load_game_config()

        self._config = config

        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent.parent

        self._memory_dir = base_dir / config.directories.game_storage_dir / "memory"

        if store is None:
            store = FileMemoryStore(self._memory_dir)

        self._store = store
        self._summarizer = summarizer or default_summarizer
        self._merger = merger or default_merger

    @property
    def store(self) -> BaseMemoryStore:
        return self._store

    @property
    def memory_dir(self) -> Path:
        return self._memory_dir

    def _session_namespace(self, session_id: str) -> str:
        return f"session:{session_id}"

    def _player_namespace(self, player_id: str) -> str:
        return f"player:{player_id}"

    def _global_namespace(self) -> str:
        return "global"

    def append_session_event(
        self,
        session_id: str,
        event: SessionEventRecord,
    ) -> MemoryDocument:
        namespace = self._session_namespace(session_id)
        key = f"event_{event.turn_index:04d}"

        doc = self._store.put(
            namespace=namespace,
            key=key,
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
                "role": event.role,
                **event.metadata,
            },
        )

        logger.debug(
            "Appended event %d to session %s",
            event.turn_index,
            session_id,
        )
        return doc

    def get_session_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[SessionEventRecord]:
        namespace = self._session_namespace(session_id)
        docs = self._store.get_all_in_namespace(namespace)

        events = []
        for doc in docs:
            if not doc.key.startswith("event_"):
                continue
            try:
                event = SessionEventRecord(
                    session_id=doc.value.get("session_id", session_id),
                    turn_index=doc.value.get("turn_index", 0),
                    role=doc.value.get("role", "system"),
                    message=doc.value.get("message", ""),
                    event_type=doc.value.get("event_type", "message"),
                    tags=doc.value.get("tags", []),
                    timestamp=doc.created_at,
                )
                events.append(event)
            except Exception as e:
                logger.warning("Failed to parse event %s: %s", doc.key, e)

        events.sort(key=lambda e: e.turn_index)

        if limit is not None:
            return events[-limit:]
        return events

    def get_recent_events(
        self,
        session_id: str,
        limit: int = 10,
    ) -> List[SessionEventRecord]:
        return self.get_session_history(session_id, limit=limit)

    def summarize_session(
        self,
        session_id: str,
        player_id: Optional[str] = None,
        puzzle_id: Optional[str] = None,
        custom_summarizer: Optional[SummarizerFn] = None,
    ) -> str:
        events = self.get_session_history(session_id)
        summarizer = custom_summarizer or self._summarizer
        summary = summarizer(events)

        session_ns = self._session_namespace(session_id)
        self._store.put(
            namespace=session_ns,
            key="summary",
            value={
                "session_id": session_id,
                "summary": summary,
                "event_count": len(events),
                "generated_at": datetime.now().isoformat(),
            },
            metadata={
                "summary_type": MemorySummaryType.SESSION_SUMMARY.value,
                "player_id": player_id,
                "puzzle_id": puzzle_id,
            },
        )

        if player_id:
            self._update_player_from_session(player_id, session_id, summary, events)

        if puzzle_id:
            self._update_puzzle_stats(puzzle_id, events)

        logger.info("Generated summary for session %s", session_id)
        return summary

    def _update_player_from_session(
        self,
        player_id: str,
        session_id: str,
        summary: str,
        events: List[SessionEventRecord],
    ) -> None:
        question_count = sum(1 for e in events if e.has_tag(EventTag.QUESTION))
        hint_count = sum(1 for e in events if e.has_tag(EventTag.HINT))

        has_success = any(
            e.has_tag(EventTag.FINAL_VERDICT) and "correct" in e.message.lower()
            for e in events
        )

        session_record = PlayerMemoryRecord(
            player_id=player_id,
            summary_type=MemorySummaryType.SESSION_SUMMARY,
            content=summary,
            metadata={
                "session_id": session_id,
                "question_count": question_count,
                "hint_count": hint_count,
                "success": has_success,
            },
        )

        player_ns = self._player_namespace(player_id)
        self._store.put(
            namespace=player_ns,
            key=f"session_{session_id}",
            value={
                "summary_type": session_record.summary_type.value,
                "content": session_record.content,
                "version": session_record.version,
            },
            metadata=session_record.metadata,
        )

    def _update_puzzle_stats(
        self,
        puzzle_id: str,
        events: List[SessionEventRecord],
    ) -> None:
        global_ns = self._global_namespace()
        stats_key = f"puzzle_stats_{puzzle_id}"

        existing = self._store.get(global_ns, stats_key)
        question_count = sum(1 for e in events if e.has_tag(EventTag.QUESTION))
        has_success = any(
            e.has_tag(EventTag.FINAL_VERDICT) and "correct" in e.message.lower()
            for e in events
        )

        if existing:
            current_stats = existing.value
            total_sessions = current_stats.get("total_sessions", 0) + 1
            total_questions = current_stats.get("total_questions", 0) + question_count
            success_count = current_stats.get("success_count", 0) + (1 if has_success else 0)
        else:
            total_sessions = 1
            total_questions = question_count
            success_count = 1 if has_success else 0

        avg_questions = total_questions / total_sessions if total_sessions > 0 else 0
        success_rate = success_count / total_sessions if total_sessions > 0 else 0

        self._store.put(
            namespace=global_ns,
            key=stats_key,
            value={
                "puzzle_id": puzzle_id,
                "total_sessions": total_sessions,
                "total_questions": total_questions,
                "success_count": success_count,
                "avg_questions": avg_questions,
                "success_rate": success_rate,
            },
            metadata={
                "summary_type": MemorySummaryType.PUZZLE_STATS.value,
                "puzzle_id": puzzle_id,
            },
        )

    def update_player_profile(
        self,
        player_id: str,
        new_info: str,
        summary_type: MemorySummaryType = MemorySummaryType.STYLE_PROFILE,
        custom_merger: Optional[MergerFn] = None,
    ) -> PlayerMemoryRecord:
        player_ns = self._player_namespace(player_id)
        existing = self._store.get(player_ns, summary_type.value)

        merger = custom_merger or self._merger

        if existing:
            existing_content = existing.value.get("content", "")
            merged_content = merger(existing_content, new_info)
            version = existing.value.get("version", 1) + 1
        else:
            merged_content = new_info
            version = 1

        record = PlayerMemoryRecord(
            player_id=player_id,
            summary_type=summary_type,
            content=merged_content,
            version=version,
        )

        self._store.put(
            namespace=player_ns,
            key=summary_type.value,
            value={
                "player_id": record.player_id,
                "summary_type": record.summary_type.value,
                "content": record.content,
                "version": record.version,
            },
            metadata={
                "summary_type": summary_type.value,
            },
        )

        logger.info(
            "Updated player %s profile (%s), version %d",
            player_id,
            summary_type.value,
            version,
        )
        return record

    def retrieve_player_profile(
        self,
        player_id: str,
        limit: int = 3,
        query: Optional[str] = None,
    ) -> List[MemorySearchResult]:
        player_ns = self._player_namespace(player_id)

        results = self._store.search(
            namespace=player_ns,
            query=query,
            limit=limit,
        )

        return results

    def get_player_memory(
        self,
        player_id: str,
        summary_type: MemorySummaryType,
    ) -> Optional[PlayerMemoryRecord]:
        player_ns = self._player_namespace(player_id)
        doc = self._store.get(player_ns, summary_type.value)

        if doc is None:
            return None

        return PlayerMemoryRecord(
            player_id=player_id,
            summary_type=summary_type,
            content=doc.value.get("content", ""),
            version=doc.value.get("version", 1),
            created_at=doc.created_at,
            updated_at=doc.updated_at,
        )

    def store_global_memory(
        self,
        record_id: str,
        content: str,
        summary_type: MemorySummaryType,
        puzzle_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GlobalMemoryRecord:
        record = GlobalMemoryRecord(
            record_id=record_id,
            summary_type=summary_type,
            content=content,
            puzzle_id=puzzle_id,
            metadata=metadata or {},
        )

        global_ns = self._global_namespace()
        self._store.put(
            namespace=global_ns,
            key=record_id,
            value={
                "record_id": record.record_id,
                "summary_type": record.summary_type.value,
                "content": record.content,
                "puzzle_id": record.puzzle_id,
            },
            metadata={
                "summary_type": summary_type.value,
                "puzzle_id": puzzle_id,
                **(metadata or {}),
            },
        )

        logger.info("Stored global memory record: %s", record_id)
        return record

    def get_global_memory(
        self,
        record_id: str,
    ) -> Optional[GlobalMemoryRecord]:
        global_ns = self._global_namespace()
        doc = self._store.get(global_ns, record_id)

        if doc is None:
            return None

        return GlobalMemoryRecord(
            record_id=record_id,
            summary_type=MemorySummaryType(doc.value.get("summary_type", "session_summary")),
            content=doc.value.get("content", ""),
            puzzle_id=doc.value.get("puzzle_id"),
            created_at=doc.created_at,
            updated_at=doc.updated_at,
        )

    def search_global_memory(
        self,
        query: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[MemorySearchResult]:
        global_ns = self._global_namespace()
        return self._store.search(
            namespace=global_ns,
            query=query,
            filter=filter,
            limit=limit,
        )

    def get_puzzle_stats(
        self,
        puzzle_id: str,
    ) -> Optional[Dict[str, Any]]:
        global_ns = self._global_namespace()
        doc = self._store.get(global_ns, f"puzzle_stats_{puzzle_id}")

        if doc is None:
            return None

        return doc.value

    def clear_session_memory(self, session_id: str) -> int:
        namespace = self._session_namespace(session_id)
        return self._store.clear_namespace(namespace)

    def clear_player_memory(self, player_id: str) -> int:
        namespace = self._player_namespace(player_id)
        return self._store.clear_namespace(namespace)

    def create_event(
        self,
        session_id: str,
        turn_index: int,
        role: str,
        message: str,
        tags: Optional[List[str]] = None,
        event_type: str = "message",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SessionEventRecord:
        return SessionEventRecord(
            session_id=session_id,
            turn_index=turn_index,
            role=role,
            message=message,
            event_type=event_type,
            tags=tags or [],
            metadata=metadata or {},
        )
