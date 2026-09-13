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
