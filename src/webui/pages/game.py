"""Game page for playing the puzzle game."""

import time
from typing import Optional, List, Dict, Any
import streamlit as st

from webui.i18n import I18n
from webui.config import EMOJI_MAP
from webui.async_utils import run_async
from game.session_runner import GameResponse
from webui.components import (
    render_game_stats,
    render_commands_panel,
    render_how_to_play,
    render_error,
    render_success,
    render_empty_state,
)
from webui import session_state as state


def render_game_page(i18n: I18n) -> Optional[str]:
    session_id = state.get_current_session_id()
    puzzle_id = state.get_current_puzzle_id()
    
    if not session_id and not puzzle_id:
        render_empty_state(i18n("game_no_session"), icon="game")
        
        def go_home():
            st.session_state.game_action = "go_home"
        
        st.button(
            f"{EMOJI_MAP['home']} {i18n('game_back_home')}",
            on_click=go_home,
        )
        
        if st.session_state.get("game_action") == "go_home":
            st.session_state.game_action = None
            return "go_home"
        return None
    
    engine = state.get_game_engine()
    if engine is None:
        render_error(i18n("error_init_required"))
        return None
    
    runner = state.get_session_runner()
    
    if runner is None:
        if session_id:
            action = _load_existing_session(engine, session_id, i18n)
        else:
            action = _create_new_session(engine, puzzle_id, i18n)
        
        if action:
            return action
        
        runner = state.get_session_runner()
        if runner is None:
            return None
    
    return _render_game_interface(runner, i18n)


def _load_existing_session(engine, session_id: str, i18n: I18n) -> Optional[str]:
    try:
        session = engine.get_session(session_id)
        puzzle = engine.get_puzzle(session.puzzle_id)
        
        from game.session_runner import GameSessionRunner
        
        llm_client = engine.model_registry.get_llm_client()
        
        ui_settings = state.get_ui_settings()
        
        runner = GameSessionRunner(
            session=session,
            puzzle=puzzle,
            kb_manager=engine.kb_manager,
            memory_manager=engine.memory_manager,
            session_store=engine.session_store,
            llm_client=llm_client,
            agents_config=engine.agents_config,
            player_agent_mode=ui_settings.player_agent_mode,
        )
        
        state.set_session_runner(runner)
        
        _load_history_from_session(session)
        
        return None
        
    except Exception as e:
        render_error(f"{i18n('error_session_not_found')}: {str(e)}")
        state.set_current_session_id("")
        return "go_home"


def _create_new_session(engine, puzzle_id: str, i18n: I18n) -> Optional[str]:
    try:
        with st.spinner(i18n("loading")):
            session = run_async(
                engine.create_session(
                    puzzle_id=puzzle_id,
                    player_id=state.get_player_id(),
                )
            )
        
        puzzle = engine.get_puzzle(puzzle_id)
        
        from game.session_runner import GameSessionRunner
        
        llm_client = engine.model_registry.get_llm_client()
        
        ui_settings = state.get_ui_settings()
        
        runner = GameSessionRunner(
            session=session,
            puzzle=puzzle,
            kb_manager=engine.kb_manager,
            memory_manager=engine.memory_manager,
            session_store=engine.session_store,
            llm_client=llm_client,
            agents_config=engine.agents_config,
            player_agent_mode=ui_settings.player_agent_mode,
        )
        
        state.set_current_session_id(session.session_id)
        state.set_session_runner(runner)
        
        intro_response = runner.start_game()
        state.add_message("assistant", intro_response.message)
        
        return None
        
    except Exception as e:
        render_error(f"{i18n('error_generic')}: {str(e)}")
        state.set_current_puzzle_id("")
        return "go_home"


def _load_history_from_session(session) -> None:
    state.clear_messages()
    
    for event in session.turn_history:
        role = "user" if event.role.value == "player" else "assistant"
        verdict = event.verdict if hasattr(event, 'verdict') and event.verdict else ""
        state.add_message(role, event.message, verdict=verdict, turn_index=event.turn_index)


def _render_game_interface(runner, i18n: I18n) -> Optional[str]:
    session = runner.session
    puzzle = runner.puzzle
    ui_settings = state.get_ui_settings()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"### {EMOJI_MAP['puzzle']} {puzzle.title if puzzle.title else puzzle.id}")
    
    with col2:
        mode_label = "🤖 Agent Mode" if ui_settings.player_agent_mode else "👤 Human Mode"
        st.caption(mode_label)
    
    with col3:
        def go_home_action():
            state.reset_game_state()
            st.session_state.game_action = "go_home"
        
        st.button(
            f"{EMOJI_MAP['home']} {i18n('game_back_home')}",
            use_container_width=True,
            on_click=go_home_action,
        )
        
        if st.session_state.get("game_action") == "go_home":
            st.session_state.game_action = None
            return "go_home"
    
    with st.expander(f"{EMOJI_MAP['puzzle']} {i18n('game_puzzle_statement')}", expanded=True):
        st.markdown(puzzle.puzzle_statement)
    
    render_game_stats(
        turn_count=runner.question_count,
        hints_used=session.hint_count,
        hints_total=puzzle.constraints.max_hints,
        state=session.state.value,
        i18n=i18n,
    )
    
    if not session.is_active:
        _render_game_over(session, i18n)
        return None
    
    _render_chat_history(i18n)
    
    if ui_settings.player_agent_mode:
        _render_agent_controls(runner, i18n)
    else:
        _render_human_chat_input(runner, i18n)
    
    _render_sidebar_controls(runner, i18n, ui_settings.player_agent_mode)
    
    return None


def _render_game_over(session, i18n: I18n) -> None:
    from game.domain.entities import GameState
    
    if session.state == GameState.COMPLETED:
        st.success(f"{EMOJI_MAP['trophy']} {i18n('game_congratulations')}")
        if session.score is not None:
            st.metric(i18n("history_score"), f"{EMOJI_MAP['star']} {session.score}")
    else:
        st.info(f"{EMOJI_MAP['info']} {i18n('game_session_ended')}")
    
    _render_chat_history(i18n)


def _render_chat_history(i18n: I18n) -> None:
    messages = state.get_messages()
    
    chat_container = st.container()
    
    with chat_container:
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            verdict = msg.get("verdict", "")
            is_agent = msg.get("is_agent", False)
            
            if role == "user":
                avatar = "🤖" if is_agent else EMOJI_MAP["player"]
                with st.chat_message("user", avatar=avatar):
                    st.markdown(content)
            else:
                with st.chat_message("assistant", avatar=EMOJI_MAP["dm"]):
                    if verdict:
                        verdict_emoji = EMOJI_MAP.get(verdict.lower(), "")
                        st.markdown(f"{verdict_emoji} {content}")
                    else:
                        st.markdown(content)


def _render_human_chat_input(runner, i18n: I18n) -> None:
    if prompt := st.chat_input(i18n("game_your_question")):
        state.add_message("user", prompt)
        
        with st.chat_message("user", avatar=EMOJI_MAP["player"]):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar=EMOJI_MAP["dm"]):
            with st.spinner(i18n("game_thinking")):
                response = _process_player_input(runner, prompt)
            
            if response:
                verdict_emoji = ""
                if response.verdict:
                    verdict_emoji = EMOJI_MAP.get(response.verdict.lower(), "")
                
                st.markdown(f"{verdict_emoji} {response.message}")
                
                state.add_message(
                    "assistant", 
                    response.message, 
                    verdict=response.verdict or "",
                    turn_index=runner.session.turn_count,
                )
        
        if response and response.game_over:
            st.rerun()


def _render_agent_controls(runner, i18n: I18n) -> None:
    st.markdown("---")
    st.markdown(f"### 🤖 {i18n('settings_player_agent_mode')}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button(
            "▶️ Run One Turn",
            key="agent_one_turn",
            use_container_width=True,
            help="Let AI player make one move",
        ):
            _run_agent_turn(runner, i18n)
            st.rerun()
    
    with col2:
        auto_play = st.checkbox(
            "🔄 Auto Play",
            key="agent_auto_play",
            value=st.session_state.get("auto_play_enabled", False),
        )
        st.session_state.auto_play_enabled = auto_play
    
    with col3:
        delay = st.slider(
            "Delay (s)",
            min_value=1,
            max_value=10,
            value=3,
            key="agent_delay",
        )
    
    if auto_play and runner.session.is_active:
        with st.spinner(f"🤖 AI is thinking... (next turn in {delay}s)"):
            time.sleep(delay)
            _run_agent_turn(runner, i18n)
            st.rerun()


def _run_agent_turn(runner, i18n: I18n) -> None:
    try:
        response = run_async(runner.run_player_agent_turn())
        
        player_message = response.metadata.get("player_message", "")
        if player_message:
            state.add_message("user", player_message, is_agent=True)
        
        state.add_message(
            "assistant",
            response.message,
            verdict=response.verdict or "",
            turn_index=runner.session.turn_count,
        )
            
    except Exception as e:
        st.error(f"Agent error: {str(e)}")


def _render_sidebar_controls(runner, i18n: I18n, is_agent_mode: bool) -> None:
    with st.sidebar:
        st.markdown("---")
        render_commands_panel(i18n)
        render_how_to_play(i18n)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        def hint_callback():
            _handle_command(runner, "/hint", i18n)
        
        def quit_callback():
            _handle_command(runner, "/quit", i18n)
        
        with col1:
            st.button(
                f"{EMOJI_MAP['hint']} {i18n('game_hint')}",
                use_container_width=True,
                on_click=hint_callback,
                key="sidebar_hint",
            )
        
        with col2:
            st.button(
                f"❌ {i18n('game_quit')}",
                use_container_width=True,
                on_click=quit_callback,
                key="sidebar_quit",
            )
        
        if is_agent_mode:
            st.markdown("---")
            st.info("🤖 AI Player Mode Active\n\nThe AI will automatically generate questions and hypotheses to solve the puzzle.")


def _process_player_input(runner, message: str) -> GameResponse:
    try:
        response = run_async(
            runner.process_player_input(message)
        )
        return response
    except Exception as e:
        return GameResponse(message=f"Error: {str(e)}", verdict=None, game_over=False)


def _handle_command(runner, command: str, i18n: I18n) -> None:
    state.add_message("user", command)
    
    try:
        response = run_async(
            runner.process_player_input(command)
        )
        
        state.add_message(
            "assistant",
            response.message,
            verdict=response.verdict or "",
        )
        
    except Exception as e:
        state.add_message("assistant", f"Error: {str(e)}")
