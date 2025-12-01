"""Game page for active game sessions."""

import asyncio
from typing import Optional
import streamlit as st

from webui.i18n import I18n
from webui.config import EMOJI_MAP
from webui.components import (
    render_header,
    render_error,
    render_success,
    render_game_stats,
    render_chat_message,
    render_commands_panel,
    render_how_to_play,
)
from webui import session_state as state
from webui.async_utils import run_async


def render_game_page(i18n: I18n) -> Optional[str]:
    render_header(i18n)
    
    session_id = state.get_current_session_id()
    puzzle_id = state.get_current_puzzle_id()
    
    if not session_id or not puzzle_id:
        render_error(i18n("error_no_active_session"))
        if st.button(f"{EMOJI_MAP['puzzle']} {i18n('nav_home')}", key="back_to_home"):
            return "back_home"
        return None
    
    engine = state.get_game_engine()
    if engine is None:
        render_error(i18n("error_init_required"))
        return None
    
    runner = state.get_session_runner()
    if runner is None:
        try:
            session = engine.get_session(session_id)
            puzzle = engine.get_puzzle(puzzle_id)
            
            ui_settings = state.get_ui_settings()
            player_agent_mode = ui_settings.player_agent_mode
            
            from game.session_runner import GameSessionRunner
            runner = GameSessionRunner(
                session=session,
                puzzle=puzzle,
                kb_manager=engine.kb_manager,
                memory_manager=engine.memory_manager,
                session_store=engine.session_store,
                llm_client=engine.model_registry.get_llm_client(),
                agents_config=engine.agents_config,
                player_agent_mode=player_agent_mode,
                dm_agent_mode=True,
            )
            state.set_session_runner(runner)
            
            if session.turn_count == 0:
                response = runner.start_game()
                state.add_message("assistant", response.message, turn_index=0)
        except Exception as e:
            render_error(f"{i18n('error_generic')}: {str(e)}")
            if st.button(f"{EMOJI_MAP['puzzle']} {i18n('nav_home')}", key="back_to_home_error"):
                state.reset_game_state()
                return "back_home"
            return None
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        puzzle = engine.get_puzzle(puzzle_id)
        st.markdown(f"### {EMOJI_MAP['puzzle']} {puzzle.title if puzzle.title else puzzle_id}")
        if puzzle.puzzle_statement:
            with st.expander(i18n("game_puzzle_story"), expanded=False):
                st.markdown(puzzle.puzzle_statement)
    
    with col2:
        if st.button(f"🏠 {i18n('nav_home')}", key="nav_home_btn", use_container_width=True):
            return "back_home"
    
    st.markdown("---")
    
    session = runner.session
    render_game_stats(
        turn_count=session.turn_count,
        hints_used=session.hint_count,
        hints_total=runner.puzzle.constraints.max_hints,
        state=session.state.value,
        i18n=i18n,
    )
    
    st.markdown("---")
    
    render_commands_panel(i18n)
    render_how_to_play(i18n)
    
    st.markdown("---")
    st.markdown(f"#### {EMOJI_MAP['game']} {i18n('game_chat_history')}")
    
    messages = state.get_messages()
    chat_container = st.container()
    with chat_container:
        for msg in messages:
            render_chat_message(
                role=msg["role"],
                content=msg["content"],
                verdict=msg.get("verdict"),
                i18n=i18n,
            )
    
    from game.domain.entities import GameState
    if session.state == GameState.IN_PROGRESS:
        ui_settings = state.get_ui_settings()
        
        if ui_settings.player_agent_mode:
            _render_agent_mode_controls(runner, i18n)
        else:
            _render_human_mode_controls(runner, i18n)
    else:
        _render_game_over(session, i18n)
    
    return None


def _render_human_mode_controls(runner, i18n: I18n) -> None:
    with st.form(key="player_input_form", clear_on_submit=True):
        user_input = st.text_input(
            i18n("game_your_question"),
            key="player_question_input",
            placeholder=i18n("game_input_placeholder"),
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            submit = st.form_submit_button(
                f"{EMOJI_MAP['question']} {i18n('game_send')}",
                use_container_width=True,
            )
        with col2:
            hint_btn = st.form_submit_button(
                f"{EMOJI_MAP['info']} {i18n('game_hint')}",
                use_container_width=True,
            )
        
        if submit and user_input.strip():
            _process_player_input(runner, user_input.strip(), i18n)
        elif hint_btn:
            _process_player_input(runner, "/hint", i18n)


def _render_agent_mode_controls(runner, i18n: I18n) -> None:
    st.info(f"🤖 {i18n('settings_player_agent_mode')} - AI will ask questions automatically")
    
    if "auto_play_active" not in st.session_state:
        st.session_state.auto_play_active = False
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button(f"▶️ {i18n('game_agent_next_turn')}", key="agent_next_turn", use_container_width=True):
            _run_agent_turn(runner, i18n)
    
    with col2:
        # Use callback to sync checkbox with our state variable
        def on_auto_play_change():
            st.session_state.auto_play_active = st.session_state.auto_play_checkbox
        
        auto_play = st.checkbox(
            i18n("game_agent_auto_play"), 
            key="auto_play_checkbox",
            value=st.session_state.auto_play_active,
            on_change=on_auto_play_change
        )
    
    with col3:
        def stop_auto_play():
            st.session_state.auto_play_active = False
        
        st.button(
            f"⏹️ {i18n('game_agent_stop')}", 
            key="agent_stop", 
            use_container_width=True,
            on_click=stop_auto_play
        )
    
    if st.session_state.auto_play_active and runner.is_active:
        _run_agent_turn(runner, i18n)
        st.rerun()


def _process_player_input(runner, user_input: str, i18n: I18n) -> None:
    state.add_message("user", user_input, turn_index=runner.session.turn_count + 1)
    
    try:
        response = run_async(runner.process_player_input(user_input))
        
        state.add_message(
            "assistant",
            response.message,
            verdict=response.verdict or "",
            turn_index=runner.session.turn_count,
        )
        
        if response.game_over:
            state.set_success_message(i18n("game_over_message"))
        
        st.rerun()
    except Exception as e:
        state.set_error_message(f"{i18n('error_generic')}: {str(e)}")
        st.rerun()


def _run_agent_turn(runner, i18n: I18n) -> None:
    try:
        with st.spinner(i18n("game_agent_thinking")):
            response = run_async(runner.run_player_agent_turn())
        
        player_msg = response.metadata.get('player_message', '')
        if player_msg:
            state.add_message(
                "user",
                player_msg,
                turn_index=runner.session.turn_count,
                is_agent=True,
            )
        
        state.add_message(
            "assistant",
            response.message,
            verdict=response.verdict or "",
            turn_index=runner.session.turn_count,
        )
        
        if response.game_over:
            state.set_success_message(i18n("game_over_message"))
        
        st.rerun()
    except Exception as e:
        state.set_error_message(f"{i18n('error_generic')}: {str(e)}")
        st.rerun()


def _render_game_over(session, i18n: I18n) -> None:
    from game.domain.entities import GameState
    
    if session.state == GameState.COMPLETED:
        st.success(f"🎉 {i18n('game_completed')}")
    elif session.state == GameState.ABORTED:
        st.warning(f"⚠️ {i18n('game_ended')}: {i18n('state_aborted')}")
    else:
        st.info(f"📋 {i18n('game_ended')}: {session.state.value}")
    
    st.markdown(f"**{i18n('game_final_score')}:** {session.score or 'N/A'}")
    st.markdown(f"**{i18n('game_total_turns')}:** {session.turn_count}")
