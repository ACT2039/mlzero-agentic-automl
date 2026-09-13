
import pytest

from mlzero.application.manager import RunManager
from mlzero.application.service import MLZeroService


def test_mlzero_service_validation():
    service = MLZeroService(use_mock_llm=True)
    with pytest.raises(ValueError, match="is outside allowed data root"):
        service.run_mlzero("/tmp/outside")

def test_mlzero_service_invalid_dataset():
    service = MLZeroService(use_mock_llm=True)
    with pytest.raises(ValueError, match="Invalid dataset path"):
        service.run_mlzero("tests/data/does_not_exist")

def test_run_manager():
    manager = RunManager()
    run_id = manager.submit_run("tests/data/tiny_classification", options={"mock_llm": True})
    status = manager.get_run_status(run_id)
    assert status is not None
    assert status.status in ["QUEUED", "RUNNING", "SUCCESS", "FAIL"]

def test_run_manager_per_run_mock_llm_isolation(monkeypatch):
    """Test that two runs with different mock_llm options do not leak configuration."""
    manager = RunManager()
    
    # We will mock MLZeroService.__init__ to record what use_mock_llm was passed
    passed_args = {}
    original_init = MLZeroService.__init__
    
    def mock_init(self, use_mock_llm=False):
        # We can identify the run by the thread it runs in, or just record all calls
        # Let's just append to a list
        passed_args.setdefault("calls", []).append(use_mock_llm)
        original_init(self, use_mock_llm=use_mock_llm)
        
    monkeypatch.setattr(MLZeroService, "__init__", mock_init)
    
    # We mock run_mlzero so it doesn't actually do work
    def mock_run(self, dataset_path, user_instruction=None, options=None):
        from mlzero.schemas.application import RunStatusResponse
        return RunStatusResponse(run_id=options.get("run_id"), status="SUCCESS")
        
    monkeypatch.setattr(MLZeroService, "run_mlzero", mock_run)
    
    # Submit first run with mock_llm=True
    manager.submit_run("tests/data/tiny_classification", options={"mock_llm": True})
    
    # Submit second run with mock_llm=False
    manager.submit_run("tests/data/tiny_classification", options={"mock_llm": False})
    
    # Wait for threads to finish
    import time
    time.sleep(0.5)
    
    assert True in passed_args["calls"]
    assert False in passed_args["calls"]
    assert len(passed_args["calls"]) == 2
