"""WebUI configuration constants and settings."""

from dataclasses import dataclass, field
from typing import Dict, Any, List

APP_NAME = "Echoes of Deceit"
APP_VERSION = "1.0.0"
APP_ICON = "🎭"

DEFAULT_LANGUAGE = "en"
DEFAULT_PLAYER_ID = "player"

PAGE_CONFIG = {
    "page_title": APP_NAME,
    "page_icon": APP_ICON,
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

EMOJI_MAP = {
    "yes": "✅",
    "no": "❌",
    "yes_and_no": "⚖️",
    "irrelevant": "🔄",
    "correct": "🎉",
    "partial": "🤔",
    "incorrect": "💭",
    "hint": "💡",
    "player": "🎮",
    "dm": "🎲",
    "thinking": "🤔",
    "puzzle": "🧩",
    "trophy": "🏆",
    "star": "⭐",
    "warning": "⚠️",
    "error": "❌",
    "success": "✅",
    "info": "ℹ️",
    "question": "❓",
    "home": "🏠",
    "settings": "⚙️",
    "history": "📜",
    "game": "🎮",
}


@dataclass
class UISettings:
    show_thinking: bool = False
    auto_scroll: bool = True
    sound_effects: bool = False
    player_agent_mode: bool = False


@dataclass
class PlayerSettings:
    player_id: str = DEFAULT_PLAYER_ID
    display_name: str = ""
    language: str = DEFAULT_LANGUAGE


@dataclass
class AppState:
    current_page: str = "home"
    current_session_id: str = ""
    current_puzzle_id: str = ""


@dataclass
class ChatMessage:
    role: str
    content: str
    verdict: str = ""
    timestamp: str = ""
    turn_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "verdict": self.verdict,
            "timestamp": self.timestamp,
            "turn_index": self.turn_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        return cls(
            role=data.get("role", ""),
            content=data.get("content", ""),
            verdict=data.get("verdict", ""),
            timestamp=data.get("timestamp", ""),
            turn_index=data.get("turn_index", 0),
        )


CSS_STYLES = """
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    
    .puzzle-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        color: white;
        margin-bottom: 1rem;
    }
    
    .puzzle-card h3 {
        margin: 0 0 0.5rem 0;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .status-lobby {
        background-color: #fef3c7;
        color: #92400e;
    }
    
    .status-in-progress {
        background-color: #dbeafe;
        color: #1e40af;
    }
    
    .status-completed {
        background-color: #d1fae5;
        color: #065f46;
    }
    
    .status-aborted {
        background-color: #fee2e2;
        color: #991b1b;
    }
    
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        background-color: #f8fafc;
        border-radius: 10px;
    }
    
    .game-stats {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .stat-item {
        flex: 1;
        text-align: center;
        padding: 0.75rem;
        background-color: #f1f5f9;
        border-radius: 8px;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #3b82f6;
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
    }
    
    .command-list {
        font-size: 0.875rem;
        color: #64748b;
    }
    
    .command-list code {
        background-color: #e2e8f0;
        padding: 0.125rem 0.25rem;
        border-radius: 4px;
    }
    
    .stChatMessage {
        padding: 0.75rem !important;
    }
    
    .verdict-badge {
        display: inline-block;
        padding: 0.125rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    .verdict-yes {
        background-color: #d1fae5;
        color: #065f46;
    }
    
    .verdict-no {
        background-color: #fee2e2;
        color: #991b1b;
    }
    
    .verdict-irrelevant {
        background-color: #fef3c7;
        color: #92400e;
    }
    
    .session-card {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: box-shadow 0.2s;
    }
    
    .session-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
</style>
"""
