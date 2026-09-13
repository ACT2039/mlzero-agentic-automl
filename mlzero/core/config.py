"""
Configuration management using pydantic-settings and yaml.
"""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseModel):
    name: str = "mlzero"
    version: str = "0.1.0"
    debug: bool = False


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/mlzero.log"


class LLMConfig(BaseModel):
    model: str = "gpt-4-turbo"
    temperature: float = 0.2
    max_tokens: int = 4096


class ExecutionConfig(BaseModel):
    timeout_seconds: int = 3600
    sandbox_enabled: bool = True


class LimitsConfig(BaseModel):
    max_iterations: int = 10
    max_errors_per_iteration: int = 3


class StorageConfig(BaseModel):
    local_dir: str = "./data"
    hf_repo_id: str = "placeholder-repo-id"


class Settings(BaseSettings):
    """
    Main settings class. Reads from environment variables (.env) and yaml configs.
    """
    openai_api_key: str = ""
    hf_token: str = ""
    environment: str = "development"

    app: AppConfig = AppConfig()
    logging: LoggingConfig = LoggingConfig()
    llm: LLMConfig = LLMConfig()
    execution: ExecutionConfig = ExecutionConfig()
    limits: LimitsConfig = LimitsConfig()
    storage: StorageConfig = StorageConfig()

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @classmethod
    def load_from_yaml(cls, yaml_path: str | Path) -> "Settings":
        """
        Load configuration from a YAML file and override default settings.
        """
        path = Path(yaml_path)
        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

        # Instantiate settings (will load from .env automatically)
        settings = cls()

        # Update nested models if data is present in YAML
        if "app" in yaml_data:
            settings.app = AppConfig(**yaml_data["app"])
        if "logging" in yaml_data:
            settings.logging = LoggingConfig(**yaml_data["logging"])
        if "llm" in yaml_data:
            settings.llm = LLMConfig(**yaml_data["llm"])
        if "execution" in yaml_data:
            settings.execution = ExecutionConfig(**yaml_data["execution"])
        if "limits" in yaml_data:
            settings.limits = LimitsConfig(**yaml_data["limits"])
        if "storage" in yaml_data:
            settings.storage = StorageConfig(**yaml_data["storage"])

        return settings


# Global settings instance
_config_path = os.environ.get("MLZERO_CONFIG", "configs/config.yaml")
settings = Settings.load_from_yaml(_config_path)
