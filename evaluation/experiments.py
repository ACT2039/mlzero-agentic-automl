"""Experiment runner."""
import time
import uuid
from typing import Any

from evaluation.metrics import extract_metrics
from mlzero.application.service import MLZeroService


def run_experiment(name: str, config: dict[str, Any]) -> dict[str, Any]:
    print(f"Running experiment: {name}")
    service = MLZeroService(use_mock_llm=config.get("mock_llm", True))
    
    # Configure ablations via cast to avoid mypy complaints on Optional assignments
    if config.get("disable_semantic_memory", False):
        service.semantic_memory = None  # type: ignore[assignment]
    if config.get("disable_episodic_memory", False):
        service.episodic_memory = None  # type: ignore[assignment]
        
    dataset: str = config.get("dataset") or ""
    instruction: str | None = config.get("instruction")
    
    start = time.time()
    try:
        status = service.run_mlzero(dataset_path=dataset, user_instruction=instruction)
        metrics = extract_metrics(status)
    except Exception as e:  # noqa: BLE001
        print(f"Experiment {name} failed catastrophically: {e}")
        metrics = {
            "success": False,
            "iterations": 0,
            "execution_time": time.time() - start,
            "error_category": "system_crash",
            "ml_accuracy": None,
            "ml_f1": None,
            "ml_mae": None,
            "ml_rmse": None
        }
        
    return {
        "experiment_id": str(uuid.uuid4()),
        "name": name,
        "config": config,
        "metrics": metrics
    }
