"""Game module for game engine and session management."""

from game.kb_manager import KnowledgeBaseManager
from game.engine import GameEngine
from game.session_runner import GameSessionRunner, GameResponse, MessageType, DMVerdict, HypothesisVerdict
from game.domain import (
    AgentRole,
    Game,
    GameSession,
    GameState,
    PlayerProfile,
    Puzzle,
    PuzzleConstraints,
    PuzzleSummary,
    SessionConfig,
    SessionEvent,
)
from game.repository import PuzzleRepository
from game.storage import GameSessionStore, PlayerProfileStore
from game.memory import (
    BaseMemoryStore,
    EventTag,
    FileMemoryStore,
    GlobalMemoryRecord,
    MemoryDocument,
    MemoryManager,
    MemorySearchResult,
    MemorySummaryType,
    PlayerMemoryRecord,
    SessionEventRecord,
)
from game.graph import (
    GameGraphState,
    GamePhase,
    GameGraphBuilder,
    GameGraphRunner,
    GameGraphRunnerFactory,
    GameToolkit,
)
from game.cli import (
    GameCLIApp,
    list_puzzles,
    start_session,
    play_session,
    resume_session,
    get_session_status,
    list_sessions,
)

__all__ = [
    "KnowledgeBaseManager",
    "AgentRole",
    "Game",
    "GameSession",
    "GameState",
    "PlayerProfile",
    "Puzzle",
    "PuzzleConstraints",
    "PuzzleSummary",
    "SessionConfig",
    "SessionEvent",
    "PuzzleRepository",
    "GameSessionStore",
    "PlayerProfileStore",
    "BaseMemoryStore",
    "EventTag",
    "FileMemoryStore",
    "GlobalMemoryRecord",
    "MemoryDocument",
    "MemoryManager",
    "MemorySearchResult",
    "MemorySummaryType",
    "PlayerMemoryRecord",
    "SessionEventRecord",
    "GameEngine",
    "GameSessionRunner",
    "GameResponse",
    "MessageType",
    "DMVerdict",
    "HypothesisVerdict",
    "GameGraphState",
    "GamePhase",
    "GameGraphBuilder",
    "GameGraphRunner",
    "GameGraphRunnerFactory",
    "GameToolkit",
    "GameCLIApp",
    "list_puzzles",
    "start_session",
    "play_session",
    "resume_session",
    "get_session_status",
    "list_sessions",
]
