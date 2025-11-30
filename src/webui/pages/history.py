"""History page for viewing past game sessions."""

from typing import Optional, List
import streamlit as st

from webui.i18n import I18n
from webui.config import EMOJI_MAP
from webui.components import (
    render_header,
    render_session_card,
    render_empty_state,
    render_error,
    render_status_badge,
)
from webui import session_state as state


def render_history_page(i18n: I18n) -> Optional[str]:
    st.markdown(f"## {EMOJI_MAP['history']} {i18n('history_title')}")
    
    if "history_action" not in st.session_state:
        st.session_state.history_action = None
    
    engine = state.get_game_engine()
    if engine is None:
        render_error(i18n("error_init_required"))
        return None
    
    filter_col1, filter_col2 = st.columns([1, 3])
    
    with filter_col1:
        state_options = [
            i18n("history_filter_all"),
            i18n("history_filter_completed"),
            i18n("history_filter_in_progress"),
            i18n("history_filter_aborted"),
        ]
        selected_state = st.selectbox(
            i18n("history_filter_state"),
            options=state_options,
            index=0,
            key="history_filter_state_select",
        )
    
    state_filter = None
    if selected_state != i18n("history_filter_all"):
        from game.domain.entities import GameState
        state_map = {
            i18n("history_filter_completed"): GameState.COMPLETED,
            i18n("history_filter_in_progress"): GameState.IN_PROGRESS,
            i18n("history_filter_aborted"): GameState.ABORTED,
        }
        state_filter = state_map.get(selected_state)
    
    try:
        sessions = engine.list_sessions(
            state_filter=state_filter,
            player_id=state.get_player_id(),
        )
    except Exception as e:
        render_error(f"{i18n('error_generic')}: {str(e)}")
        return None
    
    if not sessions:
        render_empty_state(i18n("history_no_sessions"), icon="history")
        return None
    
    st.markdown(f"**{len(sessions)}** {i18n('history_title').lower()}")
    st.markdown("---")
    
    for session in sessions:
        _render_session_row(engine, session, i18n)
    
    action = st.session_state.history_action
    if action:
        st.session_state.history_action = None
        return action
    
    return None


def _on_continue_session(session_id: str, puzzle_id: str):
    state.set_current_session_id(session_id)
    state.set_current_puzzle_id(puzzle_id)
    st.session_state.history_action = "continue_game"


def _render_session_row(engine, session, i18n: I18n) -> None:
    try:
        puzzle = engine.get_puzzle(session.puzzle_id)
        puzzle_title = puzzle.title if puzzle.title else session.puzzle_id
    except:
        puzzle_title = session.puzzle_id
    
    with st.container():
        cols = st.columns([3, 2, 1, 1, 2])
        
        with cols[0]:
            st.markdown(f"**{puzzle_title}**")
            st.caption(f"ID: {session.session_id[:8]}...")
        
        with cols[1]:
            status_html = render_status_badge(session.state.value, i18n)
            st.markdown(status_html, unsafe_allow_html=True)
            
            created_str = session.created_at.strftime("%Y-%m-%d %H:%M")
            st.caption(f"{i18n('history_created')}: {created_str}")
        
        with cols[2]:
            st.metric(i18n("history_turns"), session.question_count)
        
        with cols[3]:
            if session.score is not None:
                st.metric(i18n("history_score"), f"{EMOJI_MAP['star']} {session.score}")
            else:
                st.markdown("-")
        
        with cols[4]:
            from game.domain.entities import GameState
            
            if session.state == GameState.IN_PROGRESS:
                st.button(
                    f"{EMOJI_MAP['game']} {i18n('home_continue_game')}",
                    key=f"continue_{session.session_id}",
                    use_container_width=True,
                    on_click=_on_continue_session,
                    args=[session.session_id, session.puzzle_id],
                )
            else:
                if st.button(
                    f"{EMOJI_MAP['info']} {i18n('history_view_details')}",
                    key=f"view_{session.session_id}",
                    use_container_width=True,
                ):
                    _show_session_details(session, puzzle_title, i18n)
    
    st.markdown("<hr style='margin: 0.5rem 0; opacity: 0.2;'>", unsafe_allow_html=True)


def _show_session_details(session, puzzle_title: str, i18n: I18n) -> None:
    with st.expander(f"📋 {puzzle_title} - {i18n('history_view_details')}", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(i18n("history_turns"), session.question_count)
        
        with col2:
            st.metric(i18n("game_hints_used"), session.hint_count)
        
        with col3:
            status_html = render_status_badge(session.state.value, i18n)
            st.markdown(f"**{i18n('history_state')}**")
            st.markdown(status_html, unsafe_allow_html=True)
        
        with col4:
            if session.score is not None:
                st.metric(i18n("history_score"), session.score)
            else:
                st.markdown(f"**{i18n('history_score')}**")
                st.markdown("-")
        
        st.markdown("---")
        st.markdown(f"**{i18n('game_commands_history')}**")
        
        for event in session.turn_history:
            role_emoji = EMOJI_MAP["player"] if event.role.value == "player" else EMOJI_MAP["dm"]
            role_name = i18n("game_you") if event.role.value == "player" else i18n("game_dm")
            
            st.markdown(f"{role_emoji} **{role_name}:** {event.message[:200]}{'...' if len(event.message) > 200 else ''}")
