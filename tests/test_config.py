"""Tests for configuration module."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from config.models import (
    APIConfig,
    AgentsConfig,
    AnalyticsConfig,
    DMConfig,
    DirectoriesConfig,
    GameConfig,
    GameSettingsConfig,
    HintConfig,
    JudgeConfig,
    ModelsConfig,
    ObservabilityConfig,
    OllamaConfig,
    ProfileIntegrationConfig,
    PuzzleConfig,
    RagConfig,
    SummarizationConfig,
    resolve_env_vars,
)
from config.loader import ConfigLoader


class TestResolveEnvVars:
    def test_resolve_simple_env_var(self):
        os.environ["TEST_VAR"] = "test_value"
        result = resolve_env_vars("${TEST_VAR}")
        assert result == "test_value"
        del os.environ["TEST_VAR"]

    def test_resolve_env_var_with_default(self):
        if "NONEXISTENT_VAR" in os.environ:
            del os.environ["NONEXISTENT_VAR"]
        result = resolve_env_vars("${NONEXISTENT_VAR:default_value}")
        assert result == "default_value"

    def test_resolve_env_var_in_string(self):
        os.environ["API_HOST"] = "localhost"
        result = resolve_env_vars("http://${API_HOST}:8080")
        assert result == "http://localhost:8080"
        del os.environ["API_HOST"]

    def test_resolve_empty_default(self):
        if "MISSING_VAR" in os.environ:
            del os.environ["MISSING_VAR"]
        result = resolve_env_vars("${MISSING_VAR:}")
        assert result == ""

    def test_no_env_vars(self):
        result = resolve_env_vars("plain_string")
        assert result == "plain_string"


class TestRagConfig:
    def test_default_values(self):
        config = RagConfig()
        assert config.default_provider == "lightrag"

    def test_custom_values(self):
        config = RagConfig(default_provider="minirag")
        assert config.default_provider == "minirag"


class TestDirectoriesConfig:
    def test_default_values(self):
        config = DirectoriesConfig()
        assert config.data_base_dir == "data/situation_puzzles"
        assert config.rag_storage_dir == "rag_storage"
        assert config.game_storage_dir == "game_storage"


class TestGameSettingsConfig:
    def test_default_values(self):
        config = GameSettingsConfig()
        assert config.default_language == "en"
        assert config.max_turn_count == 100
        assert config.default_hint_limit == 5
        assert "yes_no" in config.allowed_question_types


class TestGameConfig:
    def test_default_construction(self):
        config = GameConfig()
        assert config.rag.default_provider == "lightrag"
        assert config.directories.data_base_dir == "data/situation_puzzles"
        assert config.game.max_turn_count == 100
        assert config.puzzle.kb_id_prefix == "game_"

    def test_from_dict(self):
        data = {
            "rag": {"default_provider": "minirag"},
            "directories": {"data_base_dir": "custom/path"},
            "game": {"max_turn_count": 50},
            "puzzle": {"kb_id_prefix": "puzzle_"},
        }
        config = GameConfig(**data)
        assert config.rag.default_provider == "minirag"
        assert config.directories.data_base_dir == "custom/path"
        assert config.game.max_turn_count == 50
        assert config.puzzle.kb_id_prefix == "puzzle_"


class TestOllamaConfig:
    def test_default_values(self):
        config = OllamaConfig()
        assert config.base_url == "http://localhost:11434"
        assert config.llm_model_name == "qwen2.5:7b"
        assert config.embedding_model_name == "nomic-embed-text"
        assert config.default_temperature == 0.7
        assert config.max_tokens == 2048


class TestAPIConfig:
    def test_default_values(self):
        config = APIConfig()
        assert config.base_url == "https://api.openai.com/v1"
        assert config.llm_model_name == "gpt-4o-mini"

    def test_env_var_resolution(self):
        os.environ["TEST_API_KEY"] = "sk-test-key"
        config = APIConfig(api_key="${TEST_API_KEY}")
        assert config.api_key == "sk-test-key"
        del os.environ["TEST_API_KEY"]


class TestModelsConfig:
    def test_default_provider_ollama(self):
        config = ModelsConfig()
        assert config.provider == "ollama"

    def test_get_active_config_ollama(self):
        config = ModelsConfig(provider="ollama")
        active = config.get_active_config()
        assert isinstance(active, OllamaConfig)

    def test_get_active_config_api(self):
        config = ModelsConfig(provider="api")
        active = config.get_active_config()
        assert isinstance(active, APIConfig)


class TestAgentsConfig:
    def test_default_construction(self):
        config = AgentsConfig()
        assert config.dm.persona.name == "Narrator"
        assert config.dm.persona.tone == "mysterious"
        assert config.judge.strictness == "moderate"
        assert config.hint.strategy.initial_vagueness == "high"

    def test_dm_behavior(self):
        config = AgentsConfig()
        assert config.dm.behavior.reveal_answer_early is False
        assert config.dm.behavior.encourage_player is True

    def test_profile_integration_config(self):
        config = ProfileIntegrationConfig()
        assert config.enabled is True
        assert config.profile_weight == "medium"
        assert config.adapt_difficulty is True
        assert config.adapt_explanations is True
        assert config.adapt_hint_strength is True

    def test_dm_profile_integration(self):
        config = AgentsConfig()
        assert config.dm.profile_integration.enabled is True
        assert config.dm.profile_integration.profile_weight == "medium"

    def test_analytics_config_defaults(self):
        config = AnalyticsConfig()
        assert config.enabled is True
        assert config.export_format == "json"
        assert config.export_dir == "game_storage/analytics"

    def test_observability_config(self):
        config = ObservabilityConfig()
        assert config.log_session_events is True
        assert config.log_level == "INFO"
        assert config.structured_logging is True

    def test_analytics_with_observability(self):
        config = AgentsConfig()
        assert config.analytics.enabled is True
        assert config.analytics.observability.log_session_events is True

    def test_summarization_config_defaults(self):
        config = SummarizationConfig()
        assert config.use_llm is True
        assert config.include_reasoning_style is True
        assert config.include_common_mistakes is True
        assert config.include_notable_strengths is True
        assert config.max_summary_length == 500

    def test_summarization_in_agents_config(self):
        config = AgentsConfig()
        assert config.summarization.use_llm is True
        assert config.summarization.max_summary_length == 500


class TestConfigLoader:
    def test_load_game_config_with_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            (config_dir / "game.yaml").write_text("", encoding="utf-8")
            
            loader = ConfigLoader(config_dir)
            config = loader.load_game_config()
            
            assert isinstance(config, GameConfig)
            assert config.rag.default_provider == "lightrag"

    def test_load_game_config_with_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            game_yaml = {
                "rag": {"default_provider": "minirag"},
                "game": {"max_turn_count": 200},
            }
            (config_dir / "game.yaml").write_text(
                yaml.safe_dump(game_yaml), encoding="utf-8"
            )
            
            loader = ConfigLoader(config_dir)
            config = loader.load_game_config()
            
            assert config.rag.default_provider == "minirag"
            assert config.game.max_turn_count == 200

    def test_load_models_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            models_yaml = {
                "provider": "api",
                "api": {
                    "base_url": "https://custom.api.com/v1",
                    "llm_model_name": "custom-model",
                },
            }
            (config_dir / "models.yaml").write_text(
                yaml.safe_dump(models_yaml), encoding="utf-8"
            )
            
            loader = ConfigLoader(config_dir)
            config = loader.load_models_config()
            
            assert config.provider == "api"
            assert config.api.base_url == "https://custom.api.com/v1"
            assert config.api.llm_model_name == "custom-model"

    def test_load_agents_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            agents_yaml = {
                "dm": {
                    "persona": {"name": "CustomDM", "tone": "friendly"},
                },
                "judge": {"strictness": "strict"},
            }
            (config_dir / "agents.yaml").write_text(
                yaml.safe_dump(agents_yaml), encoding="utf-8"
            )
            
            loader = ConfigLoader(config_dir)
            config = loader.load_agents_config()
            
            assert config.dm.persona.name == "CustomDM"
            assert config.dm.persona.tone == "friendly"
            assert config.judge.strictness == "strict"

    def test_load_all_configs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            (config_dir / "game.yaml").write_text("", encoding="utf-8")
            (config_dir / "models.yaml").write_text("", encoding="utf-8")
            (config_dir / "agents.yaml").write_text("", encoding="utf-8")
            
            loader = ConfigLoader(config_dir)
            game, models, agents = loader.load_all()
            
            assert isinstance(game, GameConfig)
            assert isinstance(models, ModelsConfig)
            assert isinstance(agents, AgentsConfig)

    def test_config_caching(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            (config_dir / "game.yaml").write_text("", encoding="utf-8")
            
            loader = ConfigLoader(config_dir)
            config1 = loader.load_game_config()
            config2 = loader.load_game_config()
            
            assert config1 is config2

    def test_config_force_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            (config_dir / "game.yaml").write_text("", encoding="utf-8")
            
            loader = ConfigLoader(config_dir)
            config1 = loader.load_game_config()
            config2 = loader.load_game_config(force_reload=True)
            
            assert config1 is not config2

    def test_missing_config_file_uses_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            
            loader = ConfigLoader(config_dir)
            config = loader.load_game_config()
            
            assert isinstance(config, GameConfig)
            assert config.rag.default_provider == "lightrag"

    def test_property_accessors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            (config_dir / "game.yaml").write_text("", encoding="utf-8")
            (config_dir / "models.yaml").write_text("", encoding="utf-8")
            (config_dir / "agents.yaml").write_text("", encoding="utf-8")
            
            loader = ConfigLoader(config_dir)
            
            assert isinstance(loader.game, GameConfig)
            assert isinstance(loader.models, ModelsConfig)
            assert isinstance(loader.agents, AgentsConfig)
