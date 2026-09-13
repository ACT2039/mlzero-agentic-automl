"""Metrics calculation for evaluation."""
from typing import Any


def extract_metrics(run_status: Any) -> dict[str, Any]:
    """Extract standard metrics from a run status."""
    metrics = {
        "success": run_status.success,
        "iterations": run_status.iterations,
        "execution_time": run_status.execution_duration,
        "error_category": run_status.final_error,
        "ml_accuracy": None,
        "ml_f1": None,
        "ml_mae": None,
        "ml_rmse": None
    }
    
    if run_status.final_metrics:
        # Map common metric names
        fm = run_status.final_metrics
        if "accuracy" in fm:
            metrics["ml_accuracy"] = fm["accuracy"]
        if "f1" in fm:
            metrics["ml_f1"] = fm["f1"]
        if "mean_absolute_error" in fm:
            metrics["ml_mae"] = fm["mean_absolute_error"]
        elif "mae" in fm:
            metrics["ml_mae"] = fm["mae"]
        if "root_mean_squared_error" in fm:
            metrics["ml_rmse"] = fm["root_mean_squared_error"]
        elif "rmse" in fm:
            metrics["ml_rmse"] = fm["rmse"]
            
    return metrics
