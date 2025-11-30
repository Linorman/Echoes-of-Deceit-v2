"""WebUI package for Echoes of Deceit game."""

from webui.i18n import I18n, get_available_languages
from webui.config import (
    APP_NAME,
    APP_VERSION,
    APP_ICON,
    PAGE_CONFIG,
    EMOJI_MAP,
    UISettings,
    PlayerSettings,
    AppState,
    ChatMessage,
)

__all__ = [
    "I18n",
    "get_available_languages",
    "APP_NAME",
    "APP_VERSION",
    "APP_ICON",
    "PAGE_CONFIG",
    "EMOJI_MAP",
    "UISettings",
    "PlayerSettings",
    "AppState",
    "ChatMessage",
]
