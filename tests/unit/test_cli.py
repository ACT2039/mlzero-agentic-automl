"""Tests for CLI module."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mlzero.cli import main


def test_cli_main_returns_zero(capsys) -> None:
    """Test that CLI main returns 0 and prints Phase 8."""
    with patch.object(sys, "argv", ["mlzero"]):
        result = main()
        assert result == 0
        captured = capsys.readouterr()
        assert "Phase 9 production initialized" in captured.out

def test_cli_run(capsys, monkeypatch) -> None:
    """Test the run command."""
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "data.csv").write_text("a,b\n1,2")
        
        # We patch MLZeroService to just return a dummy status
        from mlzero.application.service import MLZeroService
        from mlzero.schemas.application import RunStatusResponse
        
        def mock_run_mlzero(self, dataset_path, user_instruction=None, options=None):
            return RunStatusResponse(run_id="test-run", status="SUCCESS", success=True)
            
        monkeypatch.setattr(MLZeroService, "run_mlzero", mock_run_mlzero)
        
        with patch.object(sys, "argv", ["mlzero", "run", "--input", str(tmp_path), "--mock-llm"]):
            result = main()
            assert result == 0
            captured = capsys.readouterr()
            assert "=== RUN STATUS: SUCCESS ===" in captured.out
            assert "test-run" in captured.out

def test_cli_iterate_missing_file(capsys, monkeypatch) -> None:
    """Test that iterate fails gracefully with missing perception file."""
    with patch.object(sys, "argv", ["mlzero", "iterate", "--perception-file", "nonexistent.json"]):
        result = main()
        assert result == 1

def test_cli_iterate_success(capsys, monkeypatch) -> None:
    """Test that iterate reads the perception file correctly."""
    import json
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        perc_file = tmp_path / "perc.json"
        perc_data = {
            "files": [],
            "task": {"objective": "Test", "task_type": "binary_classification", "target_column": "target"},
            "library": None
        }
        perc_file.write_text(json.dumps(perc_data))
        
        from mlzero.orchestration.iterative import IterativeCodingOrchestrator
        from mlzero.schemas.orchestration import IterativeRunResult
        
        def mock_process(self, perceptual_context, instruction=None, run_id=None):
            assert perceptual_context.task.objective == "Test"
            return IterativeRunResult(
                success=True,
                total_iterations=1,
                total_duration_seconds=1.0,
                run_id="test_run",
                final_error_context=None,
                execution_history=[]
            )
            
        monkeypatch.setattr(IterativeCodingOrchestrator, "process", mock_process)
        
        with patch.object(sys, "argv", ["mlzero", "iterate", "--perception-file", str(perc_file), "--mock-llm", "--instruction", "Custom task"]):
            result = main()
            assert result == 0
            captured = capsys.readouterr()
            assert "=== RUNNING ITERATIVE CODING ===" in captured.out
            assert "test_run" in captured.out
