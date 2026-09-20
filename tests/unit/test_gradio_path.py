import time

import pytest

from mlzero.application.manager import run_manager
from mlzero.ui.app import submit_task


def test_gradio_path_success():
    """Ensure the Gradio submit_task path works with mock_llm=True.

    The test submits a run, polls the RunManager until the run reaches a
    terminal state (SUCCESS or FAIL), and then asserts that the run succeeded
    without errors.
    """
    run_id, _, _ = submit_task(
        "tests/data/tiny_classification",
        "Train a classification model using the target column.",
        True,
    )
    assert run_id, "Run ID should be returned"

    # Poll for completion (up to ~30 seconds)
    status = None
    for _ in range(30):
        status = run_manager.get_run_status(run_id)
        if status and status.status in ("SUCCESS", "FAIL"):
            break
        time.sleep(1)
    else:
        pytest.fail("Run did not finish within timeout")

    assert status.success, f"Run should succeed, got success={status.success}"
    assert status.final_error is None, f"Unexpected error: {status.final_error}"
