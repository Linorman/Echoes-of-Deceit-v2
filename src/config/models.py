"""Configuration data models using Pydantic."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


def resolve_env_vars(value: str) -> str:
    pattern = r'\$\{(\w+)(?::([^}]*))?\}'
    
    def replacer(match):
        var_name = match.group(1)
        default_value = match.group(2) or ""
        return os.environ.get(var_name, default_value)
    
    return re.sub(pattern, replacer, value)


class RagConfig(BaseModel):
    default_provider: str = "lightrag"


class DirectoriesConfig(BaseModel):
    data_base_dir: str = "data/situation_puzzles"
    rag_storage_dir: str = "rag_storage"
    game_storage_dir: str = "game_storage"


class GameSettingsConfig(BaseModel):
    default_language: str = "en"
    max_turn_count: int = 100
    default_hint_limit: int = 5
    allowed_question_types: List[str] = Field(
        default_factory=lambda: ["yes_no", "yes_and_no", "irrelevant"]
    )


class PuzzleConfig(BaseModel):
    kb_id_prefix: str = "game_"


class GameConfig(BaseModel):
    rag: RagConfig = Field(default_factory=RagConfig)
    directories: DirectoriesConfig = Field(default_factory=DirectoriesConfig)
    game: GameSettingsConfig = Field(default_factory=GameSettingsConfig)
    puzzle: PuzzleConfig = Field(default_factory=PuzzleConfig)


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    api_key: str = ""  # Optional: for authenticated Ollama deployments
    llm_model_name: str = "qwen2.5:7b"
    embedding_model_name: str = "nomic-embed-text"
    embedding_dim: int = 768
    default_temperature: float = 0.7
    max_tokens: int = 2048

    @field_validator("base_url", "api_key", mode="before")
    @classmethod
    def resolve_env(cls, v: Any) -> str:
        if isinstance(v, str):
            return resolve_env_vars(v)
        return v


class APIConfig(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    llm_model_name: str = "gpt-4o-mini"
    embedding_model_name: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    default_temperature: float = 0.7
    max_tokens: int = 2048

    @field_validator("base_url", "api_key", mode="before")
    @classmethod
    def resolve_env(cls, v: Any) -> str:
        if isinstance(v, str):
            return resolve_env_vars(v)
        return v


class ModelsConfig(BaseModel):
    provider: str = "ollama"
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    def get_active_config(self) -> OllamaConfig | APIConfig:
        if self.provider == "ollama":
            return self.ollama
        return self.api


class DMPersonaConfig(BaseModel):
    name: str = "Narrator"
    tone: str = "mysterious"
    style: str = "immersive"


class DMBehaviorConfig(BaseModel):
    reveal_answer_early: bool = False
    verbose_explanations: bool = False
    encourage_player: bool = True


class ProfileIntegrationConfig(BaseModel):
    enabled: bool = True
    profile_weight: str = "medium"
    adapt_difficulty: bool = True
    adapt_explanations: bool = True
    adapt_hint_strength: bool = True


class DMConfig(BaseModel):
    persona: DMPersonaConfig = Field(default_factory=DMPersonaConfig)
    behavior: DMBehaviorConfig = Field(default_factory=DMBehaviorConfig)
    profile_integration: ProfileIntegrationConfig = Field(default_factory=ProfileIntegrationConfig)


class PlayerAgentPersonaConfig(BaseModel):
    name: str = "Detective"
    tone: str = "curious"
    style: str = "analytical"


class PlayerAgentBehaviorConfig(BaseModel):
    ask_followup_questions: bool = True
    form_hypothesis_after_questions: int = 10
    max_questions_before_guess: int = 20
    question_strategies: List[str] = Field(
        default_factory=lambda: ["binary_elimination", "detail_probing", "scenario_testing"]
    )


class PlayerAgentRagAccessConfig(BaseModel):
    allowed_types: List[str] = Field(
        default_factory=lambda: ["puzzle_statement", "public_fact"]
    )
    max_context_length: int = 500


class PlayerAgentConfig(BaseModel):
    enabled: bool = False
    persona: PlayerAgentPersonaConfig = Field(default_factory=PlayerAgentPersonaConfig)
    behavior: PlayerAgentBehaviorConfig = Field(default_factory=PlayerAgentBehaviorConfig)
    rag_access: PlayerAgentRagAccessConfig = Field(default_factory=PlayerAgentRagAccessConfig)


class QuestionResponseFormatConfig(BaseModel):
    question_response: List[str] = Field(
        default_factory=lambda: ["YES", "NO", "YES_AND_NO", "IRRELEVANT"]
    )
    include_explanation: bool = True
    max_explanation_length: int = 100


class JudgeConfig(BaseModel):
    strictness: str = "moderate"
    response_format: QuestionResponseFormatConfig = Field(
        default_factory=QuestionResponseFormatConfig
    )


class HintStrategyConfig(BaseModel):
    initial_vagueness: str = "high"
    progressive_clarity: bool = True
    max_hints_before_direct: int = 3


class HintTimingConfig(BaseModel):
    auto_hint_after_questions: int = 10
    allow_player_request: bool = True


class HintConfig(BaseModel):
    strategy: HintStrategyConfig = Field(default_factory=HintStrategyConfig)
    timing: HintTimingConfig = Field(default_factory=HintTimingConfig)


class ObservabilityConfig(BaseModel):
    log_session_events: bool = True
    log_level: str = "INFO"
    structured_logging: bool = True


class AnalyticsConfig(BaseModel):
    enabled: bool = True
    export_format: str = "json"
    export_dir: str = "game_storage/analytics"
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)


class SummarizationConfig(BaseModel):
    use_llm: bool = True
    include_reasoning_style: bool = True
    include_common_mistakes: bool = True
    include_notable_strengths: bool = True
    max_summary_length: int = 500


class AgentsConfig(BaseModel):
    dm: DMConfig = Field(default_factory=DMConfig)
    player_agent: PlayerAgentConfig = Field(default_factory=PlayerAgentConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    hint: HintConfig = Field(default_factory=HintConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)
