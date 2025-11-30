"""Main Streamlit application entry point for Echoes of Deceit WebUI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from webui.config import PAGE_CONFIG, APP_NAME, APP_ICON, CSS_STYLES
from webui.i18n import I18n, get_available_languages
from webui import session_state as state
from webui.components import render_css, render_navigation, render_error
from webui.pages.home import render_home_page

_engine_ready_printed = False


def init_game_engine():
    global _engine_ready_printed
    if state.get_game_engine() is None:
        try:
            from game.engine import GameEngine
            engine = GameEngine()
            state.set_game_engine(engine)
            
            if not _engine_ready_printed:
                print("\n" + "=" * 50)
                print("🎮 Game engine initialized successfully!")
                print("🌐 WebUI is ready at: http://localhost:8501")
                print("=" * 50 + "\n")
                _engine_ready_printed = True
            
            return True
        except Exception as e:
            state.set_error_message(f"Failed to initialize game engine: {str(e)}")
            return False
    return True


def render_sidebar(i18n: I18n) -> str:
    with st.sidebar:
        st.markdown(f"# {APP_ICON} {i18n('app_title')}")
        st.markdown(f"*{i18n('app_subtitle')}*")
        
        st.markdown("---")
        
        languages = get_available_languages()
        lang_options = list(languages.keys())
        lang_labels = list(languages.values())
        current_lang = i18n.language
        current_index = lang_options.index(current_lang) if current_lang in lang_options else 0
        
        new_lang_label = st.selectbox(
            i18n("sidebar_language"),
            options=lang_labels,
            index=current_index,
            key="sidebar_lang_select",
        )
        new_lang = lang_options[lang_labels.index(new_lang_label)]
        
        if new_lang != current_lang:
            state.set_language(new_lang)
            st.rerun()
        
        player_id = st.text_input(
            i18n("sidebar_player_id"),
            value=state.get_player_id(),
            key="sidebar_player_input",
            help=i18n("sidebar_player_id_help"),
        )
        
        if player_id != state.get_player_id():
            state.set_player_id(player_id)
        
        ui_settings = state.get_ui_settings()
        player_agent_mode = st.checkbox(
            i18n("settings_player_agent_mode"),
            value=ui_settings.player_agent_mode,
            key="sidebar_agent_mode",
            help=i18n("settings_player_agent_mode_help"),
        )
        if player_agent_mode != ui_settings.player_agent_mode:
            ui_settings.player_agent_mode = player_agent_mode
            state.set_ui_settings(ui_settings)
        
        st.markdown("---")
        
        current_page = state.get_current_page()
        
        # Only show home page navigation
        pages = [
            ("home", "🏠", i18n("nav_home")),
        ]
        
        for page_key, icon, label in pages:
            is_current = current_page == page_key
            btn_type = "primary" if is_current else "secondary"
            
            if st.button(
                f"{icon} {label}",
                key=f"nav_{page_key}",
                use_container_width=True,
                type=btn_type,
            ):
                state.set_current_page(page_key)
                st.rerun()
        
        st.markdown("---")
        
        session_id = state.get_current_session_id()
        if session_id:
            st.caption(f"Session: {session_id[:8]}...")
        
        if ui_settings.player_agent_mode:
            st.info(f"🤖 {i18n('settings_player_agent_mode')}")
    
    return state.get_current_page()


def main():
    st.set_page_config(**PAGE_CONFIG)
    
    render_css()
    
    state.init_session_state()
    
    i18n = state.get_i18n()
    
    if not init_game_engine():
        error_msg = state.get_error_message()
        if error_msg:
            render_error(error_msg)
        st.stop()
    
    current_page = render_sidebar(i18n)
    
    action = None
    
    # Only render home page
    action = render_home_page(i18n)


if __name__ == "__main__":
    main()
