"""Tests for CLI module."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mlzero.cli import main


def test_cli_main_returns_zero(capsys) -> None:
    """Test that CLI main returns 0 and prints correct output for default behavior."""
    with patch.object(sys, "argv", ["mlzero"]):
        result = main()
        assert result == 0
        captured = capsys.readouterr()
        assert "MLZero-Agentic-AutoML" in captured.out
        assert "Phase 1 foundation initialized." in captured.out

def test_package_import() -> None:
    """Test that the package can be imported."""
    import mlzero
    assert mlzero is not None


def test_cli_perceive(capsys) -> None:
    """Test the perceive command."""
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "data.csv").write_text("a,b\n1,2")
        
        with patch.object(sys, "argv", ["mlzero", "perceive", "--input", str(tmp_path), "--mock-llm"]):
            result = main()
            assert result == 0
            captured = capsys.readouterr()
            assert "PERCEPTION SUMMARY" in captured.out
            assert "data.csv" in captured.out
            assert "autogluon.tabular" in captured.out


def test_cli_perceive_json(capsys) -> None:
    """Test the perceive command with JSON output."""
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "data.csv").write_text("a,b\n1,2")
        
        with patch.object(sys, "argv", ["mlzero", "perceive", "--input", str(tmp_path), "--mock-llm", "--json"]):
            result = main()
            assert result == 0
            captured = capsys.readouterr()
            import json
            data = json.loads(captured.out)
            assert "files" in data
            assert len(data["files"]) == 1
            assert data["files"][0]["metadata"]["path"] == "data.csv"

def test_cli_iterate(capsys) -> None:
    """Test the iterate command."""
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "data.csv").write_text("a,b\n1,2")
        
        with patch.object(sys, "argv", ["mlzero", "iterate", "--input", str(tmp_path), "--mock-llm"]):
            # We mock the executor to always fail once then succeed, or just let it fail.
            # Actually, the default mock LLM logic will generate code that requires data.csv in the exec dir.
            # So it will fail, then retry, and fail again because data.csv isn't copied into the exec dir.
            # That's fine, it will hit max iterations or we can patch the executor.
            from mlzero.agents.coder import ExecutorAgent
            from mlzero.schemas.coder import ExecutionResult
            
            def fake_process(self, artifact, input_files=None):
                return ExecutionResult(success=True, return_code=0, stdout="Mock success")
                
            with patch.object(ExecutorAgent, "process", fake_process):
                result = main()
                assert result == 0
                captured = capsys.readouterr()
                assert "RUNNING PERCEPTION" in captured.out
                assert "RUNNING ITERATIVE CODING" in captured.out
                assert "FINAL RESULT" in captured.out
                assert "Success:      True" in captured.out
