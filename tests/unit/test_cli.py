"""Tests for CLI module."""

from mlzero.cli import main


def test_cli_main_returns_zero(capsys) -> None:
    """Test that CLI main returns 0 and prints correct output."""
    result = main()
    assert result == 0
    captured = capsys.readouterr()
    assert "MLZero-Agentic-AutoML" in captured.out
    assert "Phase 1 foundation initialized." in captured.out

def test_package_import() -> None:
    """Test that the package can be imported."""
    import mlzero
    assert mlzero is not None
