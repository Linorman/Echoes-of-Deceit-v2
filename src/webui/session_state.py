"""Session state management for Streamlit WebUI."""

from typing import Optional, List, Dict, Any
import streamlit as st

from webui.config import (
    UISettings, 
    PlayerSettings, 
    AppState, 
    ChatMessage,
    DEFAULT_LANGUAGE,
    DEFAULT_PLAYER_ID,
)
from webui.i18n import I18n


def init_session_state() -> None:
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.language = DEFAULT_LANGUAGE
        st.session_state.i18n = I18n(DEFAULT_LANGUAGE)
        st.session_state.player_id = DEFAULT_PLAYER_ID
        st.session_state.display_name = ""
        st.session_state.current_page = "home"
        st.session_state.current_session_id = ""
        st.session_state.current_puzzle_id = ""
        st.session_state.messages = []
        st.session_state.game_engine = None
        st.session_state.session_runner = None
        st.session_state.ui_settings = UISettings()
        st.session_state.error_message = ""
        st.session_state.success_message = ""


def get_i18n() -> I18n:
    if "i18n" not in st.session_state:
        st.session_state.i18n = I18n(get_language())
    return st.session_state.i18n


def get_language() -> str:
    return st.session_state.get("language", DEFAULT_LANGUAGE)


def set_language(language: str) -> None:
    st.session_state.language = language
    st.session_state.i18n = I18n(language)


def get_player_id() -> str:
    return st.session_state.get("player_id", DEFAULT_PLAYER_ID)


def set_player_id(player_id: str) -> None:
    st.session_state.player_id = player_id


def get_display_name() -> str:
    return st.session_state.get("display_name", "")


def set_display_name(name: str) -> None:
    st.session_state.display_name = name


def get_current_page() -> str:
    return st.session_state.get("current_page", "home")


def set_current_page(page: str) -> None:
    st.session_state.current_page = page


def get_current_session_id() -> str:
    return st.session_state.get("current_session_id", "")


def set_current_session_id(session_id: str) -> None:
    st.session_state.current_session_id = session_id


def get_current_puzzle_id() -> str:
    return st.session_state.get("current_puzzle_id", "")


def set_current_puzzle_id(puzzle_id: str) -> None:
    st.session_state.current_puzzle_id = puzzle_id


def get_messages() -> List[Dict[str, Any]]:
    return st.session_state.get("messages", [])


def add_message(role: str, content: str, verdict: str = "", turn_index: int = 0, is_agent: bool = False) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    message = {
        "role": role,
        "content": content,
        "verdict": verdict,
        "turn_index": turn_index,
        "is_agent": is_agent,
    }
    st.session_state.messages.append(message)


def clear_messages() -> None:
    st.session_state.messages = []


def get_game_engine():
    return st.session_state.get("game_engine")


def set_game_engine(engine) -> None:
    st.session_state.game_engine = engine


def get_session_runner():
    return st.session_state.get("session_runner")


def set_session_runner(runner) -> None:
    st.session_state.session_runner = runner


def get_ui_settings() -> UISettings:
    if "ui_settings" not in st.session_state:
        st.session_state.ui_settings = UISettings()
    return st.session_state.ui_settings


def set_ui_settings(settings: UISettings) -> None:
    st.session_state.ui_settings = settings


def get_error_message() -> str:
    return st.session_state.get("error_message", "")


def set_error_message(message: str) -> None:
    st.session_state.error_message = message


def clear_error_message() -> None:
    st.session_state.error_message = ""


def get_success_message() -> str:
    return st.session_state.get("success_message", "")


def set_success_message(message: str) -> None:
    st.session_state.success_message = message


def clear_success_message() -> None:
    st.session_state.success_message = ""


def reset_game_state() -> None:
    st.session_state.current_session_id = ""
    st.session_state.session_runner = None
    st.session_state.messages = []
    clear_error_message()
    clear_success_message()
