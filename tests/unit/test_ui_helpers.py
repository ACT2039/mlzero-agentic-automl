from unittest.mock import MagicMock, patch

import gradio as gr

from mlzero.schemas.application import RunStatusResponse
from mlzero.ui.app import refresh_status, submit_task


def test_submit_task_success():
    """Test submit task behavior and timer activation."""
    with patch("mlzero.ui.app.run_manager.submit_run", return_value="run-123"):
        run_id, msg, timer = submit_task("dataset", "instruction", True)
        assert run_id == "run-123"
        assert msg == "Run run-123 started."
        assert isinstance(timer, gr.Timer)
        # We cannot easily assert timer.active on the instance due to Gradio internals,
        # but we know it's a timer.

def test_refresh_status_no_run_id():
    """Test empty run ID."""
    res = refresh_status("")
    assert res[0] == "No run selected."
    assert res[6] == "No run selected."

def test_refresh_status_not_found():
    """Test invalid run ID."""
    with patch("mlzero.ui.app.run_manager.get_run_status", return_value=None):
        res = refresh_status("invalid-id")
        assert res[0] == "Run not found."
        assert res[6] == "Run not found."

@patch("mlzero.ui.app.EpisodicStore")
@patch("mlzero.ui.app.run_manager.get_run_status")
def test_refresh_status_duration_formatting(mock_get_status, mock_store_cls):
    """Test formatting of the duration string."""
    mock_get_status.return_value = RunStatusResponse(
        run_id="run-1",
        status="SUCCESS",
        execution_duration=5.907960653305054
    )
    
    mock_store_inst = MagicMock()
    mock_store_inst.get_run_history.return_value = None
    mock_store_cls.return_value = mock_store_inst

    res = refresh_status("run-1")
    assert "Duration: 5.91 seconds" in res[0]
    
def test_refresh_status_no_duration():
    """Test missing duration handling."""
    with (
        patch("mlzero.ui.app.run_manager.get_run_status", return_value=RunStatusResponse(
            run_id="run-1",
            status="RUNNING",
            execution_duration=None
        )),
        patch("mlzero.ui.app.EpisodicStore") as mock_store
    ):
            mock_store.return_value.get_run_history.return_value = None
            res = refresh_status("run-1")
            assert "Duration: None" in res[0]
            assert res[6] == "Waiting for iteration results..."

@patch("mlzero.ui.app.EpisodicStore")
@patch("mlzero.ui.app.run_manager.get_run_status")
def test_refresh_status_iteration_history_fail_to_success(mock_get_status, mock_store_cls):
    """Test fetching and formatting of history for FAIL -> SUCCESS."""
    mock_get_status.return_value = RunStatusResponse(
        run_id="run-1",
        status="SUCCESS",
    )
    
    mock_ep1 = MagicMock()
    mock_ep1.iteration = 1
    mock_ep1.status = "FAIL"
    mock_ep1.error_context_summary = {"error_category": "KeyError", "error_message": "Missing column"}
    mock_ep1.suggested_fix = "Fix the key"
    
    mock_ep2 = MagicMock()
    mock_ep2.iteration = 2
    mock_ep2.status = "SUCCESS"
    
    mock_history = MagicMock()
    mock_history.episodes = [mock_ep1, mock_ep2]
    
    mock_store_inst = MagicMock()
    mock_store_inst.get_run_history.return_value = mock_history
    mock_store_cls.return_value = mock_store_inst

    res = refresh_status("run-1")
    history_text = res[6]
    
    # Check Iteration 1
    assert "Iteration 1" in history_text
    assert "Status: FAILED" in history_text
    assert "Error Category: KeyError" in history_text
    assert "Problem: Missing column" in history_text
    assert "Correction: Fix the key" in history_text
    
    # Check Iteration 2
    assert "Iteration 2" in history_text
    assert "Status: SUCCESS" in history_text
    assert "Correction Applied: YES" in history_text
    
    # Check Recovery Summary
    assert "✓ Error detected" in history_text
    assert "✓ Error analyzed" in history_text
    assert "✓ Correction generated" in history_text
    assert "✓ Corrected code executed successfully" in history_text
    assert "✓ Recovery completed in 2 iterations" in history_text

@patch("mlzero.ui.app.EpisodicStore")
@patch("mlzero.ui.app.run_manager.get_run_status")
def test_refresh_status_iteration_history_success_first(mock_get_status, mock_store_cls):
    """Test history for SUCCESS on first iteration."""
    mock_get_status.return_value = RunStatusResponse(
        run_id="run-1",
        status="SUCCESS",
    )
    
    mock_ep1 = MagicMock()
    mock_ep1.iteration = 1
    mock_ep1.status = "SUCCESS"
    
    mock_history = MagicMock()
    mock_history.episodes = [mock_ep1]
    
    mock_store_inst = MagicMock()
    mock_store_inst.get_run_history.return_value = mock_history
    mock_store_cls.return_value = mock_store_inst

    res = refresh_status("run-1")
    history_text = res[6]
    
    # Check Iteration 1
    assert "Iteration 1" in history_text
    assert "Status: SUCCESS" in history_text
    assert "Correction Applied: YES" not in history_text
    
    # Check Recovery Summary
    assert "✓ Execution succeeded on first iteration" in history_text
