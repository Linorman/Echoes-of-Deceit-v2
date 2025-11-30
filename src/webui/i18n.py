"""Internationalization support for WebUI with Chinese and English."""

from typing import Dict, Any

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        "app_title": "Echoes of Deceit",
        "app_subtitle": "Turtle Soup Puzzle Game",
        "welcome_title": "Welcome to Echoes of Deceit",
        "welcome_description": "A mysterious lateral thinking puzzle game where you uncover hidden truths through yes/no questions.",
        
        "nav_home": "Home",
        "nav_game": "Play Game",
        "nav_history": "History",
        "nav_settings": "Settings",
        
        "sidebar_language": "Language",
        "sidebar_player_id": "Player ID",
        "sidebar_player_id_help": "Enter your player ID to track your progress",
        "sidebar_theme": "Theme",
        "sidebar_theme_light": "Light",
        "sidebar_theme_dark": "Dark",
        
        "home_select_puzzle": "Select a Puzzle",
        "home_no_puzzles": "No puzzles available. Please initialize the system first.",
        "home_puzzle_difficulty": "Difficulty",
        "home_puzzle_language": "Language",
        "home_puzzle_tags": "Tags",
        "home_start_game": "Start Game",
        "home_continue_game": "Continue Game",
        "home_active_sessions": "Active Sessions",
        "home_no_active_sessions": "No active game sessions.",
        
        "game_title": "Game Session",
        "game_puzzle_statement": "The Puzzle",
        "game_your_question": "Your question or hypothesis...",
        "game_send": "Send",
        "game_hint": "Get Hint",
        "game_status": "Status",
        "game_quit": "Quit Game",
        "game_help": "Help",
        "game_turn": "Turn",
        "game_hints_used": "Hints Used",
        "game_hints_remaining": "Hints Remaining",
        "game_state": "Game State",
        "game_no_session": "No active game session. Please select a puzzle first.",
        "game_thinking": "Thinking...",
        "game_you": "You",
        "game_dm": "DM",
        "game_over": "Game Over!",
        "game_congratulations": "Congratulations! You solved the puzzle!",
        "game_session_ended": "The game session has ended.",
        "game_back_home": "Back to Home",
        
        "game_commands_title": "Commands",
        "game_commands_hint": "/hint - Get a hint",
        "game_commands_status": "/status - View game status",
        "game_commands_history": "/history - View recent Q&A",
        "game_commands_quit": "/quit - End the game",
        "game_commands_help": "/help - Show help",
        
        "game_how_to_play": "How to Play",
        "game_instructions": """
1. Read the puzzle statement carefully
2. Ask yes/no questions to gather clues
3. When ready, state your hypothesis (start with "I think..." or "My guess is...")
4. Use hints sparingly - they're limited!
""",
        
        "game_agent_run_turn": "Run One Turn",
        "game_agent_auto_play": "Auto Play",
        "game_agent_delay": "Delay (s)",
        "game_agent_thinking": "AI is thinking...",
        "game_human_mode": "Human Mode",
        "game_agent_mode": "Agent Mode",
        
        "history_title": "Game History",
        "history_no_sessions": "No game history found.",
        "history_session_id": "Session ID",
        "history_puzzle": "Puzzle",
        "history_state": "State",
        "history_turns": "Turns",
        "history_score": "Score",
        "history_created": "Created",
        "history_completed": "Completed",
        "history_view_details": "View Details",
        "history_filter_state": "Filter by State",
        "history_filter_all": "All",
        "history_filter_completed": "Completed",
        "history_filter_in_progress": "In Progress",
        "history_filter_aborted": "Aborted",
        
        "settings_title": "Settings",
        "settings_player_section": "Player Settings",
        "settings_display_name": "Display Name",
        "settings_display_name_help": "Your display name in the game",
        "settings_game_section": "Game Settings",
        "settings_show_thinking": "Show Agent Thinking",
        "settings_show_thinking_help": "Display the AI's reasoning process",
        "settings_auto_scroll": "Auto-scroll Chat",
        "settings_auto_scroll_help": "Automatically scroll to the latest message",
        "settings_sound_effects": "Sound Effects",
        "settings_sound_effects_help": "Play sound effects during the game",
        "settings_advanced_section": "Advanced Settings",
        "settings_player_agent_mode": "Player Agent Mode",
        "settings_player_agent_mode_help": "Let AI play as the player (spectator mode)",
        "settings_save": "Save Settings",
        "settings_saved": "Settings saved successfully!",
        "settings_reset": "Reset to Defaults",
        
        "state_lobby": "Lobby",
        "state_in_progress": "In Progress",
        "state_completed": "Completed",
        "state_aborted": "Aborted",
        
        "error_title": "Error",
        "error_generic": "An error occurred. Please try again.",
        "error_session_not_found": "Game session not found.",
        "error_puzzle_not_found": "Puzzle not found.",
        "error_init_required": "System initialization required. Please run 'python cli.py init' first.",
        
        "loading": "Loading...",
        "confirm": "Confirm",
        "cancel": "Cancel",
        "yes": "Yes",
        "no": "No",
        "close": "Close",
        "refresh": "Refresh",
    },
    
    "zh": {
        "app_title": "谎言回响",
        "app_subtitle": "海龟汤推理游戏",
        "welcome_title": "欢迎来到谎言回响",
        "welcome_description": "一款神秘的横向思维推理游戏，通过是/否问题揭开隐藏的真相。",
        
        "nav_home": "首页",
        "nav_game": "开始游戏",
        "nav_history": "历史记录",
        "nav_settings": "设置",
        
        "sidebar_language": "语言",
        "sidebar_player_id": "玩家ID",
        "sidebar_player_id_help": "输入您的玩家ID以追踪游戏进度",
        "sidebar_theme": "主题",
        "sidebar_theme_light": "浅色",
        "sidebar_theme_dark": "深色",
        
        "home_select_puzzle": "选择谜题",
        "home_no_puzzles": "暂无可用谜题，请先初始化系统。",
        "home_puzzle_difficulty": "难度",
        "home_puzzle_language": "语言",
        "home_puzzle_tags": "标签",
        "home_start_game": "开始游戏",
        "home_continue_game": "继续游戏",
        "home_active_sessions": "进行中的游戏",
        "home_no_active_sessions": "暂无进行中的游戏。",
        
        "game_title": "游戏进行中",
        "game_puzzle_statement": "谜题描述",
        "game_your_question": "输入您的问题或猜测...",
        "game_send": "发送",
        "game_hint": "获取提示",
        "game_status": "状态",
        "game_quit": "退出游戏",
        "game_help": "帮助",
        "game_turn": "回合",
        "game_hints_used": "已用提示",
        "game_hints_remaining": "剩余提示",
        "game_state": "游戏状态",
        "game_no_session": "没有活跃的游戏会话，请先选择一个谜题。",
        "game_thinking": "思考中...",
        "game_you": "你",
        "game_dm": "主持人",
        "game_over": "游戏结束！",
        "game_congratulations": "恭喜！你解开了谜题！",
        "game_session_ended": "游戏会话已结束。",
        "game_back_home": "返回首页",
        
        "game_commands_title": "命令",
        "game_commands_hint": "/hint - 获取提示",
        "game_commands_status": "/status - 查看游戏状态",
        "game_commands_history": "/history - 查看问答历史",
        "game_commands_quit": "/quit - 结束游戏",
        "game_commands_help": "/help - 显示帮助",
        
        "game_how_to_play": "游戏玩法",
        "game_instructions": """
1. 仔细阅读谜题描述
2. 通过是/否问题收集线索
3. 准备好后，提出你的假设（以"我认为..."或"我猜..."开头）
4. 谨慎使用提示 - 数量有限！
""",
        
        "game_agent_run_turn": "执行一回合",
        "game_agent_auto_play": "自动游玩",
        "game_agent_delay": "延迟 (秒)",
        "game_agent_thinking": "AI正在思考...",
        "game_human_mode": "人类玩家模式",
        "game_agent_mode": "AI玩家模式",
        
        "history_title": "游戏历史",
        "history_no_sessions": "暂无游戏历史记录。",
        "history_session_id": "会话ID",
        "history_puzzle": "谜题",
        "history_state": "状态",
        "history_turns": "回合数",
        "history_score": "得分",
        "history_created": "创建时间",
        "history_completed": "完成时间",
        "history_view_details": "查看详情",
        "history_filter_state": "按状态筛选",
        "history_filter_all": "全部",
        "history_filter_completed": "已完成",
        "history_filter_in_progress": "进行中",
        "history_filter_aborted": "已中止",
        
        "settings_title": "设置",
        "settings_player_section": "玩家设置",
        "settings_display_name": "显示名称",
        "settings_display_name_help": "您在游戏中的显示名称",
        "settings_game_section": "游戏设置",
        "settings_show_thinking": "显示AI思考过程",
        "settings_show_thinking_help": "显示AI的推理过程",
        "settings_auto_scroll": "自动滚动聊天",
        "settings_auto_scroll_help": "自动滚动到最新消息",
        "settings_sound_effects": "音效",
        "settings_sound_effects_help": "在游戏中播放音效",
        "settings_advanced_section": "高级设置",
        "settings_player_agent_mode": "AI玩家模式",
        "settings_player_agent_mode_help": "让AI作为玩家进行游戏（观战模式）",
        "settings_save": "保存设置",
        "settings_saved": "设置保存成功！",
        "settings_reset": "重置为默认",
        
        "state_lobby": "等待中",
        "state_in_progress": "进行中",
        "state_completed": "已完成",
        "state_aborted": "已中止",
        
        "error_title": "错误",
        "error_generic": "发生错误，请重试。",
        "error_session_not_found": "未找到游戏会话。",
        "error_puzzle_not_found": "未找到谜题。",
        "error_init_required": "需要初始化系统，请先运行 'python cli.py init'。",
        
        "loading": "加载中...",
        "confirm": "确认",
        "cancel": "取消",
        "yes": "是",
        "no": "否",
        "close": "关闭",
        "refresh": "刷新",
    }
}


class I18n:
    def __init__(self, language: str = "en"):
        self._language = language if language in TRANSLATIONS else "en"
    
    @property
    def language(self) -> str:
        return self._language
    
    @language.setter
    def language(self, value: str) -> None:
        if value in TRANSLATIONS:
            self._language = value
    
    def get(self, key: str, **kwargs: Any) -> str:
        translation = TRANSLATIONS.get(self._language, TRANSLATIONS["en"])
        text = translation.get(key, TRANSLATIONS["en"].get(key, key))
        if kwargs:
            try:
                text = text.format(**kwargs)
            except KeyError:
                pass
        return text
    
    def __call__(self, key: str, **kwargs: Any) -> str:
        return self.get(key, **kwargs)


def get_available_languages() -> Dict[str, str]:
    return {
        "en": "English",
        "zh": "中文",
    }
