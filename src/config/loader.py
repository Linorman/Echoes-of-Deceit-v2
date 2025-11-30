"""Configuration loader for YAML configuration files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

from .models import AgentsConfig, GameConfig, ModelsConfig

logger = logging.getLogger(__name__)


class ConfigLoader:
    def __init__(self, config_dir: Optional[str | Path] = None):
        if config_dir is None:
            self._config_dir = Path(__file__).parent.parent.parent / "config"
        else:
            self._config_dir = Path(config_dir)
        
        self._game_config: Optional[GameConfig] = None
        self._models_config: Optional[ModelsConfig] = None
        self._agents_config: Optional[AgentsConfig] = None

    @property
    def config_dir(self) -> Path:
        return self._config_dir

    def _load_yaml(self, filename: str) -> dict:
        filepath = self._config_dir / filename
        if not filepath.exists():
            logger.warning("Config file not found: %s, using defaults", filepath)
            return {}
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        return data or {}

    def load_game_config(self, force_reload: bool = False) -> GameConfig:
        if self._game_config is not None and not force_reload:
            return self._game_config
        
        data = self._load_yaml("game.yaml")
        self._game_config = GameConfig(**data)
        logger.info("Loaded game config from %s", self._config_dir / "game.yaml")
        return self._game_config

    def load_models_config(self, force_reload: bool = False) -> ModelsConfig:
        if self._models_config is not None and not force_reload:
            return self._models_config
        
        data = self._load_yaml("models.yaml")
        self._models_config = ModelsConfig(**data)
        logger.info("Loaded models config from %s", self._config_dir / "models.yaml")
        return self._models_config

    def load_agents_config(self, force_reload: bool = False) -> AgentsConfig:
        if self._agents_config is not None and not force_reload:
            return self._agents_config
        
        data = self._load_yaml("agents.yaml")
        self._agents_config = AgentsConfig(**data)
        logger.info("Loaded agents config from %s", self._config_dir / "agents.yaml")
        return self._agents_config

    def load_all(self, force_reload: bool = False) -> tuple[GameConfig, ModelsConfig, AgentsConfig]:
        return (
            self.load_game_config(force_reload),
            self.load_models_config(force_reload),
            self.load_agents_config(force_reload),
        )

    @property
    def game(self) -> GameConfig:
        return self.load_game_config()

    @property
    def models(self) -> ModelsConfig:
        return self.load_models_config()

    @property
    def agents(self) -> AgentsConfig:
        return self.load_agents_config()
