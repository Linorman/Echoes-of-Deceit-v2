"""Settings page for configuring game preferences."""

from typing import Optional
import streamlit as st

from webui.i18n import I18n, get_available_languages
from webui.config import EMOJI_MAP, UISettings
from webui.components import render_success, render_error
from webui import session_state as state


def render_settings_page(i18n: I18n) -> Optional[str]:
    st.markdown(f"## {EMOJI_MAP['settings']} {i18n('settings_title')}")
    
    st.markdown(f"### {i18n('settings_player_section')}")
    
    current_language = state.get_language()
    current_player_id = state.get_player_id()
    current_display_name = state.get_display_name()
    ui_settings = state.get_ui_settings()
    
    col1, col2 = st.columns(2)
    
    with col1:
        languages = get_available_languages()
        lang_options = list(languages.keys())
        lang_labels = list(languages.values())
        current_index = lang_options.index(current_language) if current_language in lang_options else 0
        
        new_language_label = st.selectbox(
            i18n("sidebar_language"),
            options=lang_labels,
            index=current_index,
            key="settings_language",
        )
        new_language = lang_options[lang_labels.index(new_language_label)]
    
    with col2:
        new_player_id = st.text_input(
            i18n("sidebar_player_id"),
            value=current_player_id,
            key="settings_player_id",
            help=i18n("sidebar_player_id_help"),
        )
    
    new_display_name = st.text_input(
        i18n("settings_display_name"),
        value=current_display_name,
        key="settings_display_name",
        help=i18n("settings_display_name_help"),
    )
    
    st.markdown("---")
    st.markdown(f"### {i18n('settings_game_section')}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        show_thinking = st.checkbox(
            i18n("settings_show_thinking"),
            value=ui_settings.show_thinking,
            key="settings_show_thinking",
            help=i18n("settings_show_thinking_help"),
        )
        
        sound_effects = st.checkbox(
            i18n("settings_sound_effects"),
            value=ui_settings.sound_effects,
            key="settings_sound_effects",
            help=i18n("settings_sound_effects_help"),
        )
    
    with col2:
        auto_scroll = st.checkbox(
            i18n("settings_auto_scroll"),
            value=ui_settings.auto_scroll,
            key="settings_auto_scroll",
            help=i18n("settings_auto_scroll_help"),
        )
    
    st.markdown("---")
    st.markdown(f"### {i18n('settings_advanced_section')}")
    
    player_agent_mode = st.checkbox(
        i18n("settings_player_agent_mode"),
        value=ui_settings.player_agent_mode,
        key="settings_player_agent_mode",
        help=i18n("settings_player_agent_mode_help"),
    )
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button(
            f"{EMOJI_MAP['success']} {i18n('settings_save')}",
            use_container_width=True,
            type="primary",
        ):
            if new_language != current_language:
                state.set_language(new_language)
            
            state.set_player_id(new_player_id)
            state.set_display_name(new_display_name)
            
            new_ui_settings = UISettings(
                show_thinking=show_thinking,
                auto_scroll=auto_scroll,
                sound_effects=sound_effects,
                player_agent_mode=player_agent_mode,
            )
            state.set_ui_settings(new_ui_settings)
            
            state.set_success_message(i18n("settings_saved"))
            st.rerun()
    
    with col2:
        if st.button(
            f"🔄 {i18n('settings_reset')}",
            use_container_width=True,
        ):
            state.set_language("en")
            state.set_player_id("player")
            state.set_display_name("")
            state.set_ui_settings(UISettings())
            st.rerun()
    
    success_msg = state.get_success_message()
    if success_msg:
        render_success(success_msg)
        state.clear_success_message()
    
    return None
