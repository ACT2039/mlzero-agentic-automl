
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
    manager = RunManager(use_mock_llm=True)
    run_id = manager.submit_run("tests/data/tiny_classification")
    status = manager.get_run_status(run_id)
    assert status is not None
    assert status.status in ["QUEUED", "RUNNING", "SUCCESS", "FAIL"]
