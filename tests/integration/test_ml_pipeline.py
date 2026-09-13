"""
Integration tests for ML pipeline.
"""
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from mlzero.tools.tabular import TabularDatasetAdapter


@pytest.fixture
def tiny_dataset():
    data_dir = Path("tests/data/tiny_classification")
    if not data_dir.exists():
        pytest.skip("Tiny dataset not found.")
    return data_dir


def test_dataset_adapter(tiny_dataset):
    with TemporaryDirectory() as workspace:
        adapter = TabularDatasetAdapter(tiny_dataset, workspace)
        result = adapter.prepare_data()
        
        assert "train_path" in result
        assert "test_path" in result
        assert result["format"] == "csv"
        assert Path(result["train_path"]).exists()
        assert Path(result["test_path"]).exists()


@pytest.mark.skipif(not os.environ.get("RUN_ML_INTEGRATION"), reason="Requires autogluon installed and env var RUN_ML_INTEGRATION")
def test_ml_pipeline_run(tiny_dataset, monkeypatch):
    """
    Test the full run via the CLI method handler, forcing deterministic LLM execution.
    The MockLLMClient handles Iteration 1 failure, and Iteration 2 success.
    The generated mock code WILL run AutoGluon!
    """
    import argparse

    from mlzero.cli import handle_run
    
    # We will pass the arguments
    args = argparse.Namespace(input=str(tiny_dataset), mock_llm=True, command="run")
    
    # Execute the run
    ret = handle_run(args)
    
    assert ret == 0, "CLI handle_run should succeed."
    
    # Verify outputs
    out_dir = Path("outputs/models")
    latest_run = max((d for d in out_dir.iterdir() if d.is_dir()), key=lambda x: x.stat().st_mtime)
    
    summary_file = latest_run / "summary.json"
    assert summary_file.exists(), "summary.json should exist"
    
    preds_file = latest_run / "outputs" / "predictions.csv"
    if not preds_file.exists():
        preds_file = latest_run / "predictions.csv"
    assert preds_file.exists(), "predictions.csv should exist"
    
    import json
    data = json.loads(summary_file.read_text())
    assert data["success"] is True
    assert "model_path" in data
    assert "metrics" in data
