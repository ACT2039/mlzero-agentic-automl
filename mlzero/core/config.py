"""
Configuration management using pydantic-settings and yaml.
"""

import os
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseModel):
    name: str = "mlzero"
    version: str = "0.1.0"
    debug: bool = False
    api_host: str = Field(default_factory=lambda: os.environ.get("HOST", "127.0.0.1"))
    api_port: int = Field(default_factory=lambda: int(os.environ.get("PORT", "8000")))
    ui_host: str = "127.0.0.1"
    ui_port: int = 7860
    allowed_data_root: str = "."
    artifact_root: str = "outputs"
    max_instruction_length: int = 1000
    run_timeout_seconds: int = 600
    max_concurrent_runs: int = 2


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/mlzero.log"


class MemoryConfig(BaseModel):
    knowledge_root: str = "knowledge"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_document_size_bytes: int = 1024 * 1024  # 1MB
    embedding_backend: str = "tfidf"
    index_path: str = "knowledge/index.json"
    retrieval_top_k: int = 5
    max_retrieved_context_chars: int = 4000
    semantic_memory_enabled: bool = True


class EpisodicConfig(BaseModel):
    storage_dir: str = "memory/episodic"
    max_episodes_in_context: int = 5
    max_context_chars: int = 4000
    max_stdout_stderr_chars: int = 1500
    max_stored_code_chars: int = 10000
    max_stdout_chars: int = 5000
    max_stderr_chars: int = 5000
    max_retrieved_knowledge_chars: int = 4000



class MLConfig(BaseModel):
    training_time_limit: int = 60  # very conservative for testing/demo
    presets: str = "medium_quality"
    output_dir: str = "outputs/models"
    prediction_filename: str = "predictions.csv"
    validation_handling: str = "auto"


class LLMConfig(BaseModel):
    model: str = "gemini-3.8-flash"
    provider: str = "gemini"
    mode: str = "mock"
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout: int = 30
    retry_count: int = 3


class ExecutionConfig(BaseModel):
    timeout_seconds: int = 300
    sandbox_enabled: bool = True
    max_stdout_size_bytes: int = 100 * 1024
    max_stderr_size_bytes: int = 100 * 1024
    workspace_root: str = "outputs/workspaces"
    output_dir_name: str = "out"
    allowed_env_vars: list[str] = Field(default_factory=lambda: ["PATH", "SYSTEMROOT", "USERPROFILE"])
    python_executable: str = sys.executable
    keep_artifacts: bool = True


class LimitsConfig(BaseModel):
    max_iterations: int = 10
    max_errors_per_iteration: int = 3


class StorageConfig(BaseModel):
    local_dir: str = "./data"
    hf_repo_id: str = "placeholder-repo-id"


class PerceptionConfig(BaseModel):
    max_file_size_bytes: int = Field(default=10 * 1024 * 1024, description="10MB max file size to process")
    max_text_characters: int = Field(default=5000, description="Max characters to read from text/document files")
    max_rows_sampled: int = Field(default=5, description="Max rows to sample from CSV/TSV")
    ignored_directories: list[str] = Field(default_factory=lambda: [".git", ".venv", "venv", "env", "__pycache__", "node_modules"])
    allowed_extensions: list[str] = Field(default_factory=lambda: [".csv", ".tsv", ".json", ".jsonl", ".txt", ".md"])


class Settings(BaseSettings):
    """
    Main settings class. Reads from environment variables (.env) and yaml configs.
    """
    openai_api_key: str = ""
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"
    openrouter_api_key: str = ""
    groq_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openrouter/free"
    llm_mode: str = "mock"
    real_llm_provider: str = "groq"
    real_llm_model: str = "openai/gpt-oss-20b"
    groq_fallback_models: list[str] = Field(default_factory=lambda: ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"])
    real_llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    hf_token: str = ""
    environment: str = "development"

    app: AppConfig = AppConfig()
    logging: LoggingConfig = LoggingConfig()
    llm: LLMConfig = LLMConfig()
    execution: ExecutionConfig = ExecutionConfig()
    limits: LimitsConfig = LimitsConfig()
    storage: StorageConfig = StorageConfig()
    perception: PerceptionConfig = PerceptionConfig()
    memory: MemoryConfig = MemoryConfig()
    episodic: EpisodicConfig = EpisodicConfig()
    ml: MLConfig = MLConfig()

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
        if "perception" in yaml_data:
            settings.perception = PerceptionConfig(**yaml_data["perception"])

        return settings


# Global settings instance
_config_path = os.environ.get("MLZERO_CONFIG", "configs/config.yaml")
settings = Settings.load_from_yaml(_config_path)
