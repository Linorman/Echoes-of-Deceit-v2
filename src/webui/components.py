"""Reusable UI components for the WebUI."""

from typing import Optional, List
import streamlit as st

from webui.config import EMOJI_MAP, CSS_STYLES
from webui.i18n import I18n


def render_css() -> None:
    st.markdown(CSS_STYLES, unsafe_allow_html=True)


def render_header(i18n: I18n) -> None:
    st.markdown(
        f"""
        <div class="main-header">
            <h1>{EMOJI_MAP['puzzle']} {i18n('app_title')}</h1>
            <p>{i18n('app_subtitle')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_error(message: str) -> None:
    if message:
        st.error(f"{EMOJI_MAP['error']} {message}")


def render_success(message: str) -> None:
    if message:
        st.success(f"{EMOJI_MAP['success']} {message}")


def render_info(message: str) -> None:
    if message:
        st.info(f"{EMOJI_MAP['info']} {message}")


def render_warning(message: str) -> None:
    if message:
        st.warning(f"{EMOJI_MAP['warning']} {message}")


def render_puzzle_card(
    puzzle_id: str,
    title: str,
    description: str = "",
    difficulty: Optional[str] = None,
    tags: Optional[List[str]] = None,
    language: str = "en",
) -> None:
    tags = tags or []
    
    with st.container():
        st.markdown(
            f"""
            <div class="puzzle-card">
                <h3>{EMOJI_MAP['puzzle']} {title}</h3>
                <p>{description if description else puzzle_id}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        cols = st.columns(3)
        if difficulty:
            cols[0].markdown(f"**Difficulty:** {difficulty}")
        cols[1].markdown(f"**Language:** {language.upper()}")
        if tags:
            cols[2].markdown(f"**Tags:** {', '.join(tags)}")


def render_game_stats(
    turn_count: int,
    hints_used: int,
    hints_total: int,
    state: str,
    i18n: I18n,
) -> None:
    cols = st.columns(4)
    
    with cols[0]:
        st.metric(i18n("game_turn"), turn_count)
    
    with cols[1]:
        st.metric(i18n("game_hints_used"), f"{hints_used}/{hints_total}")
    
    with cols[2]:
        state_key = f"state_{state.lower().replace(' ', '_')}"
        state_display = i18n(state_key)
        st.metric(i18n("game_state"), state_display)
    
    with cols[3]:
        hints_remaining = hints_total - hints_used
        st.metric(i18n("game_hints_remaining"), hints_remaining)


def render_status_badge(state: str, i18n: I18n) -> str:
    state_lower = state.lower().replace(" ", "_")
    state_key = f"state_{state_lower}"
    state_display = i18n(state_key)
    
    css_class = f"status-{state_lower.replace('_', '-')}"
    
    return f'<span class="status-badge {css_class}">{state_display}</span>'


def render_chat_message(
    role: str,
    content: str,
    verdict: Optional[str] = None,
    i18n: Optional[I18n] = None,
) -> None:
    if role.lower() in ["player", "user", "you"]:
        avatar = EMOJI_MAP["player"]
        name = i18n("game_you") if i18n else "You"
    else:
        avatar = EMOJI_MAP["dm"]
        name = i18n("game_dm") if i18n else "DM"
    
    with st.chat_message(role.lower() if role.lower() in ["user", "assistant"] else "user" if role.lower() == "player" else "assistant", avatar=avatar):
        if verdict:
            verdict_emoji = EMOJI_MAP.get(verdict.lower(), "")
            st.markdown(f"**{name}:** {content} {verdict_emoji}")
        else:
            st.markdown(f"**{name}:** {content}")


def render_verdict_badge(verdict: str) -> str:
    verdict_lower = verdict.lower()
    emoji = EMOJI_MAP.get(verdict_lower, "")
    css_class = f"verdict-{verdict_lower.replace('_', '-')}"
    
    return f'<span class="verdict-badge {css_class}">{emoji} {verdict.upper()}</span>'


def render_commands_panel(i18n: I18n) -> None:
    with st.expander(f"{EMOJI_MAP['info']} {i18n('game_commands_title')}", expanded=False):
        st.markdown(
            f"""
            <div class="command-list">
                <p><code>/hint</code> - {i18n('game_commands_hint').split(' - ')[1]}</p>
                <p><code>/status</code> - {i18n('game_commands_status').split(' - ')[1]}</p>
                <p><code>/history</code> - {i18n('game_commands_history').split(' - ')[1]}</p>
                <p><code>/quit</code> - {i18n('game_commands_quit').split(' - ')[1]}</p>
                <p><code>/help</code> - {i18n('game_commands_help').split(' - ')[1]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_how_to_play(i18n: I18n) -> None:
    with st.expander(f"{EMOJI_MAP['question']} {i18n('game_how_to_play')}", expanded=False):
        st.markdown(i18n("game_instructions"))


def render_session_card(
    session_id: str,
    puzzle_title: str,
    state: str,
    turn_count: int,
    score: Optional[int],
    created_at: str,
    i18n: I18n,
    on_click_key: Optional[str] = None,
) -> bool:
    clicked = False
    
    with st.container():
        cols = st.columns([3, 1, 1, 1, 2])
        
        cols[0].markdown(f"**{puzzle_title}**")
        cols[1].markdown(render_status_badge(state, i18n), unsafe_allow_html=True)
        cols[2].markdown(f"{i18n('history_turns')}: {turn_count}")
        
        if score is not None:
            cols[3].markdown(f"{EMOJI_MAP['star']} {score}")
        else:
            cols[3].markdown("-")
        
        if on_click_key:
            clicked = cols[4].button(
                i18n("history_view_details"),
                key=on_click_key,
                use_container_width=True,
            )
    
    st.markdown("<hr style='margin: 0.5rem 0; opacity: 0.3;'>", unsafe_allow_html=True)
    
    return clicked


def render_loading(message: str = "Loading...") -> None:
    with st.spinner(message):
        pass


def render_empty_state(message: str, icon: str = "info") -> None:
    st.markdown(
        f"""
        <div style="text-align: center; padding: 3rem; color: #64748b;">
            <p style="font-size: 3rem;">{EMOJI_MAP.get(icon, '📭')}</p>
            <p>{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
