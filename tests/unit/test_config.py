"""Tests for configuration module."""

from pathlib import Path

from mlzero.core.config import AppConfig, Settings


def test_default_config() -> None:
    """Test that default configuration loads successfully."""
    settings = Settings()
    assert isinstance(settings.app, AppConfig)
    assert settings.app.name == "mlzero"

def test_load_from_yaml(tmp_path: Path) -> None:
    """Test that configuration can be loaded from YAML."""
    yaml_content = '''
    app:
      name: "test_app"
    logging:
      level: "DEBUG"
    '''
    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(yaml_content)

    settings = Settings.load_from_yaml(yaml_file)
    assert settings.app.name == "test_app"
    assert settings.logging.level == "DEBUG"
