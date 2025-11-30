"""Storage layer for game sessions and player profiles."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import ConfigLoader, GameConfig
from game.domain.entities import GameSession, GameState, PlayerProfile, SessionEvent

logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def datetime_decoder(dct: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in dct.items():
        if isinstance(value, str):
            try:
                if "T" in value and (value.endswith("Z") or "+" in value or len(value) >= 19):
                    dct[key] = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass
    return dct


class GameSessionStore:
    def __init__(
        self,
        config: Optional[GameConfig] = None,
        base_dir: Optional[Path] = None,
    ):
        if config is None:
            loader = ConfigLoader()
            config = loader.load_game_config()

        self._config = config

        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent.parent

        self._game_storage_dir = base_dir / config.directories.game_storage_dir
        self._sessions_dir = self._game_storage_dir / "sessions"
        self._events_dir = self._game_storage_dir / "events"

        self._ensure_directories()

    def _ensure_directories(self) -> None:
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._events_dir.mkdir(parents=True, exist_ok=True)

    @property
    def storage_dir(self) -> Path:
        return self._game_storage_dir

    def _session_file(self, session_id: str) -> Path:
        return self._sessions_dir / f"{session_id}.json"

    def _events_file(self, session_id: str) -> Path:
        return self._events_dir / f"{session_id}.jsonl"

    def create_session(
        self,
        puzzle_id: str,
        player_ids: Optional[List[str]] = None,
        kb_id: Optional[str] = None,
        **options: Any,
    ) -> GameSession:
        session = GameSession(
            puzzle_id=puzzle_id,
            player_ids=player_ids or [],
            kb_id=kb_id,
        )

        if "config" in options:
            session.config = options["config"]

        self.save_session(session)
        logger.info("Created session %s for puzzle %s", session.session_id, puzzle_id)
        return session

    def save_session(self, session: GameSession) -> None:
        session_file = self._session_file(session.session_id)
        session_data = session.model_dump(mode="json")

        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, cls=DateTimeEncoder, ensure_ascii=False, indent=2)

        logger.debug("Saved session %s", session.session_id)

    def load_session(self, session_id: str) -> GameSession:
        session_file = self._session_file(session_id)

        if not session_file.exists():
            raise ValueError(f"Session not found: {session_id}")

        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f, object_hook=datetime_decoder)

        return GameSession.model_validate(data)

    def session_exists(self, session_id: str) -> bool:
        return self._session_file(session_id).exists()

    def delete_session(self, session_id: str) -> bool:
        session_file = self._session_file(session_id)
        events_file = self._events_file(session_id)

        deleted = False

        if session_file.exists():
            session_file.unlink()
            deleted = True

        if events_file.exists():
            events_file.unlink()
            deleted = True

        if deleted:
            logger.info("Deleted session %s", session_id)

        return deleted

    def list_sessions(
        self,
        state_filter: Optional[GameState] = None,
        puzzle_id: Optional[str] = None,
        player_id: Optional[str] = None,
    ) -> List[GameSession]:
        sessions = []

        for session_file in self._sessions_dir.glob("*.json"):
            try:
                session = self.load_session(session_file.stem)

                if state_filter and session.state != state_filter:
                    continue

                if puzzle_id and session.puzzle_id != puzzle_id:
                    continue

                if player_id and player_id not in session.player_ids:
                    continue

                sessions.append(session)
            except Exception as exc:
                logger.error("Failed to load session %s: %s", session_file.stem, exc)

        return sessions

    def append_event(self, session_id: str, event: SessionEvent) -> None:
        events_file = self._events_file(session_id)
        event_data = event.model_dump(mode="json")

        with open(events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_data, cls=DateTimeEncoder, ensure_ascii=False) + "\n")

    def get_events(self, session_id: str) -> List[SessionEvent]:
        events_file = self._events_file(session_id)

        if not events_file.exists():
            return []

        events = []
        with open(events_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line, object_hook=datetime_decoder)
                    events.append(SessionEvent.model_validate(data))

        return events


class PlayerProfileStore:
    def __init__(
        self,
        config: Optional[GameConfig] = None,
        base_dir: Optional[Path] = None,
    ):
        if config is None:
            loader = ConfigLoader()
            config = loader.load_game_config()

        self._config = config

        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent.parent

        self._game_storage_dir = base_dir / config.directories.game_storage_dir
        self._profiles_dir = self._game_storage_dir / "player_memory"

        self._ensure_directories()

    def _ensure_directories(self) -> None:
        self._profiles_dir.mkdir(parents=True, exist_ok=True)

    @property
    def profiles_dir(self) -> Path:
        return self._profiles_dir

    def _profile_file(self, player_id: str) -> Path:
        return self._profiles_dir / f"{player_id}.json"

    def create_profile(
        self,
        player_id: str,
        display_name: str,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> PlayerProfile:
        if self.profile_exists(player_id):
            raise ValueError(f"Profile already exists: {player_id}")

        profile = PlayerProfile(
            player_id=player_id,
            display_name=display_name,
            preferences=preferences or {},
        )

        self.save_profile(profile)
        logger.info("Created player profile: %s", player_id)
        return profile

    def save_profile(self, profile: PlayerProfile) -> None:
        profile_file = self._profile_file(profile.player_id)
        profile_data = profile.model_dump(mode="json")

        with open(profile_file, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, cls=DateTimeEncoder, ensure_ascii=False, indent=2)

        logger.debug("Saved player profile: %s", profile.player_id)

    def load_profile(self, player_id: str) -> PlayerProfile:
        profile_file = self._profile_file(player_id)

        if not profile_file.exists():
            raise ValueError(f"Profile not found: {player_id}")

        with open(profile_file, "r", encoding="utf-8") as f:
            data = json.load(f, object_hook=datetime_decoder)

        return PlayerProfile.model_validate(data)

    def profile_exists(self, player_id: str) -> bool:
        return self._profile_file(player_id).exists()

    def delete_profile(self, player_id: str) -> bool:
        profile_file = self._profile_file(player_id)

        if profile_file.exists():
            profile_file.unlink()
            logger.info("Deleted player profile: %s", player_id)
            return True

        return False

    def get_or_create_profile(
        self,
        player_id: str,
        display_name: Optional[str] = None,
    ) -> PlayerProfile:
        if self.profile_exists(player_id):
            return self.load_profile(player_id)

        return self.create_profile(
            player_id=player_id,
            display_name=display_name or player_id,
        )

    def list_profiles(self) -> List[PlayerProfile]:
        profiles = []

        for profile_file in self._profiles_dir.glob("*.json"):
            try:
                profile = self.load_profile(profile_file.stem)
                profiles.append(profile)
            except Exception as exc:
                logger.error("Failed to load profile %s: %s", profile_file.stem, exc)

        return profiles

    def update_profile(
        self,
        player_id: str,
        display_name: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> PlayerProfile:
        profile = self.load_profile(player_id)

        if display_name is not None:
            profile.display_name = display_name

        if preferences is not None:
            profile.preferences.update(preferences)

        profile.updated_at = datetime.now()
        self.save_profile(profile)

        return profile
