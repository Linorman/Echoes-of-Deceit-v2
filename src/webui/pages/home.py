"""Home page for puzzle selection and active sessions."""

import asyncio
from typing import Optional, List
import streamlit as st

from webui.i18n import I18n
from webui.config import EMOJI_MAP
from webui.components import (
    render_header,
    render_puzzle_card,
    render_session_card,
    render_empty_state,
    render_error,
    render_status_badge,
)
from webui import session_state as state


def render_home_page(i18n: I18n) -> Optional[str]:
    render_header(i18n)
    
    st.markdown(f"### {i18n('welcome_title')}")
    st.markdown(i18n("welcome_description"))
    
    ui_settings = state.get_ui_settings()
    if ui_settings.player_agent_mode:
        st.info(f"🤖 {i18n('settings_player_agent_mode')} - AI will play automatically")
    
    st.markdown("---")
    
    if "home_action" not in st.session_state:
        st.session_state.home_action = None
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        _render_puzzle_selection(i18n)
    
    with col2:
        _render_active_sessions(i18n)
    
    action = st.session_state.home_action
    if action:
        st.session_state.home_action = None
        return action
    
    return None


def _on_start_puzzle(puzzle_id: str):
    state.set_current_puzzle_id(puzzle_id)
    state.reset_game_state()
    state.set_current_puzzle_id(puzzle_id)
    st.session_state.home_action = "start_game"


def _on_continue_session(session_id: str, puzzle_id: str):
    state.set_current_session_id(session_id)
    state.set_current_puzzle_id(puzzle_id)
    st.session_state.home_action = "continue_game"


def _render_puzzle_selection(i18n: I18n) -> None:
    st.markdown(f"#### {EMOJI_MAP['puzzle']} {i18n('home_select_puzzle')}")
    
    engine = state.get_game_engine()
    if engine is None:
        render_error(i18n("error_init_required"))
        return
    
    try:
        puzzles = engine.list_puzzles()
    except Exception as e:
        render_error(f"{i18n('error_generic')}: {str(e)}")
        return
    
    if not puzzles:
        render_empty_state(i18n("home_no_puzzles"), icon="puzzle")
        return
    
    for puzzle in puzzles:
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(f"**{puzzle.title if puzzle.title else puzzle.id}**")
                if puzzle.description:
                    st.caption(puzzle.description[:100] + "..." if len(puzzle.description) > 100 else puzzle.description)
                
                meta_cols = st.columns(3)
                if puzzle.difficulty:
                    meta_cols[0].caption(f"{i18n('home_puzzle_difficulty')}: {puzzle.difficulty}")
                meta_cols[1].caption(f"{i18n('home_puzzle_language')}: {puzzle.language.upper()}")
                if puzzle.tags:
                    meta_cols[2].caption(f"{i18n('home_puzzle_tags')}: {', '.join(puzzle.tags[:3])}")
            
            with col2:
                st.button(
                    f"{EMOJI_MAP['game']} {i18n('home_start_game')}",
                    key=f"start_puzzle_{puzzle.id}",
                    use_container_width=True,
                    on_click=_on_start_puzzle,
                    args=[puzzle.id],
                )
            
            st.markdown("<hr style='margin: 0.75rem 0; opacity: 0.2;'>", unsafe_allow_html=True)


def _render_active_sessions(i18n: I18n) -> None:
    st.markdown(f"#### {EMOJI_MAP['game']} {i18n('home_active_sessions')}")
    
    engine = state.get_game_engine()
    if engine is None:
        return
    
    try:
        from game.domain.entities import GameState
        all_sessions = engine.list_sessions(player_id=state.get_player_id())
        sessions = [s for s in all_sessions if s.state == GameState.IN_PROGRESS]
    except Exception as e:
        render_error(str(e))
        return
    
    if not sessions:
        render_empty_state(i18n("home_no_active_sessions"), icon="game")
        return
    
    for session in sessions[:5]:
        with st.container():
            try:
                puzzle = engine.get_puzzle(session.puzzle_id)
                puzzle_title = puzzle.title if puzzle.title else session.puzzle_id
            except:
                puzzle_title = session.puzzle_id
            
            st.markdown(f"**{puzzle_title}**")
            
            status_html = render_status_badge(session.state.value, i18n)
            st.markdown(
                f"{status_html} | {i18n('history_turns')}: {session.question_count}",
                unsafe_allow_html=True,
            )
            
            st.button(
                f"{EMOJI_MAP['game']} {i18n('home_continue_game')}",
                key=f"continue_session_{session.session_id}",
                use_container_width=True,
                on_click=_on_continue_session,
                args=[session.session_id, session.puzzle_id],
            )
            
            st.markdown("<hr style='margin: 0.5rem 0; opacity: 0.2;'>", unsafe_allow_html=True)
