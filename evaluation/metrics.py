"""
Evaluation metrics calculation module.
Calculates task-aware ML quality metrics, pipeline success rates, recovery rates,
routing accuracy, relative timing, and dataset-level ranks.
"""
import math
from typing import Any

from evaluation.schemas import EvaluationRun


def extract_metrics(run_status: Any, task_type: str | None = None) -> dict[str, float | int | str | None]:
    """
    Extract task-aware metrics from run status.
    Includes both Stage 7 task-aware metrics and legacy phase 9 keys for full compatibility.
    """
    success_val = getattr(run_status, "success", False) if run_status else False
    iters_val = getattr(run_status, "iterations", 0) if run_status else 0
    exec_val = getattr(run_status, "execution_duration", 0.0) if run_status else 0.0
    err_val = getattr(run_status, "final_error", None) if run_status else None

    metrics: dict[str, float | int | str | None] = {
        "success": success_val,
        "iterations": iters_val,
        "execution_time": exec_val,
        "error_category": err_val,
        "accuracy": None,
        "f1": None,
        "balanced_accuracy": None,
        "mcc": None,
        "mae": None,
        "rmse": None,
        "r2": None,
        "recall_at_k": None,
        "precision_at_k": None,
        "mrr": None,
        "ml_accuracy": None,
        "ml_f1": None,
        "ml_mae": None,
        "ml_rmse": None,
    }

    if not run_status or not getattr(run_status, "final_metrics", None):
        return metrics

    fm: dict[str, Any] = run_status.final_metrics or {}

    # Classification metrics
    for k in ("accuracy", "acc"):
        if k in fm and _is_numeric(fm[k]):
            metrics["accuracy"] = float(fm[k])
            metrics["ml_accuracy"] = float(fm[k])
            break
    for k in ("f1", "f1_score", "macro_f1"):
        if k in fm and _is_numeric(fm[k]):
            metrics["f1"] = float(fm[k])
            metrics["ml_f1"] = float(fm[k])
            break
    for k in ("balanced_accuracy", "bal_acc"):
        if k in fm and _is_numeric(fm[k]):
            metrics["balanced_accuracy"] = float(fm[k])
            break
    for k in ("mcc", "matthews_corrcoef"):
        if k in fm and _is_numeric(fm[k]):
            metrics["mcc"] = float(fm[k])
            break

    # Regression / Time series metrics
    for k in ("mean_absolute_error", "mae"):
        if k in fm and _is_numeric(fm[k]):
            metrics["mae"] = float(fm[k])
            metrics["ml_mae"] = float(fm[k])
            break
    for k in ("root_mean_squared_error", "rmse"):
        if k in fm and _is_numeric(fm[k]):
            metrics["rmse"] = float(fm[k])
            metrics["ml_rmse"] = float(fm[k])
            break
    for k in ("r2", "r_squared"):
        if k in fm and _is_numeric(fm[k]):
            metrics["r2"] = float(fm[k])
            break
    for k in ("balanced_accuracy", "bal_acc"):
        if k in fm and _is_numeric(fm[k]):
            metrics["balanced_accuracy"] = float(fm[k])
            break
    for k in ("mcc", "matthews_corrcoef"):
        if k in fm and _is_numeric(fm[k]):
            metrics["mcc"] = float(fm[k])
            break

    # Regression / Time series metrics
    for k in ("mean_absolute_error", "mae"):
        if k in fm and _is_numeric(fm[k]):
            metrics["mae"] = float(fm[k])
            break
    for k in ("root_mean_squared_error", "rmse"):
        if k in fm and _is_numeric(fm[k]):
            metrics["rmse"] = float(fm[k])
            break
    for k in ("r2", "r_squared"):
        if k in fm and _is_numeric(fm[k]):
            metrics["r2"] = float(fm[k])
            break

    # Retrieval metrics
    for k in ("recall_at_k", "recall_5", "recall"):
        if k in fm and _is_numeric(fm[k]):
            metrics["recall_at_k"] = float(fm[k])
            break
    for k in ("precision_at_k", "precision_5", "precision"):
        if k in fm and _is_numeric(fm[k]):
            metrics["precision_at_k"] = float(fm[k])
            break
    for k in ("mrr", "mean_reciprocal_rank"):
        if k in fm and _is_numeric(fm[k]):
            metrics["mrr"] = float(fm[k])
            break

    return metrics


def calculate_success_rate(runs: list[EvaluationRun]) -> float:
    """Compute overall success rate = successful_runs / total_runs."""
    if not runs:
        return 0.0
    successful = sum(1 for r in runs if r.success)
    return round(successful / len(runs), 4)


def calculate_first_attempt_success_rate(runs: list[EvaluationRun]) -> float:
    """Compute proportion of runs that succeeded on first iteration (iterations == 1)."""
    if not runs:
        return 0.0
    first_success = sum(1 for r in runs if r.success and r.iterations == 1)
    return round(first_success / len(runs), 4)


def calculate_recovery_rate(runs: list[EvaluationRun]) -> float | None:
    """
    Compute recovery rate = recovered_runs / initially_failed_runs.
    If there are no initially failed runs (initially_failed_runs == 0), return None.
    """
    if not runs:
        return None

    # A run initially failed if error_recovery_count > 0 or iterations > 1
    initially_failed = [r for r in runs if r.error_recovery_count > 0 or r.iterations > 1 or not r.success]
    if not initially_failed:
        return None

    recovered = sum(1 for r in initially_failed if r.success)
    return round(recovered / len(initially_failed), 4)


def calculate_routing_accuracy(runs: list[EvaluationRun]) -> dict[str, float]:
    """
    Compute Perception routing accuracy metrics:
    - library_selection_accuracy
    - task_perception_accuracy
    - target_detection_accuracy
    """
    if not runs:
        return {
            "library_selection_accuracy": 0.0,
            "task_perception_accuracy": 0.0,
            "target_detection_accuracy": 0.0,
        }

    lib_correct = sum(1 for r in runs if r.library_correct)
    task_correct = sum(1 for r in runs if r.task_type_correct)

    target_evaluable = [r for r in runs if r.target_correct is not None]
    target_correct = sum(1 for r in target_evaluable if r.target_correct is True) if target_evaluable else 0

    return {
        "library_selection_accuracy": round(lib_correct / len(runs), 4),
        "task_perception_accuracy": round(task_correct / len(runs), 4),
        "target_detection_accuracy": round(target_correct / len(target_evaluable), 4) if target_evaluable else 1.0,
    }


def calculate_relative_time(baseline_time: float, configuration_time: float) -> float | None:
    """
    Compute relative time efficiency = baseline_time / configuration_time.
    Returns None if configuration_time <= 0.
    """
    if configuration_time <= 0:
        return None
    return round(baseline_time / configuration_time, 4)


def calculate_rank(
    system_scores: dict[str, float],
    higher_is_better: bool = True,
) -> dict[str, int]:
    """
    Compute dataset-level ranks across systems for comparison.
    system_scores: dict mapping system_id -> score
    Returns dict mapping system_id -> 1-based rank (1 is best).
    """
    if not system_scores:
        return {}

    sorted_systems = sorted(
        system_scores.items(),
        key=lambda item: item[1],
        reverse=higher_is_better,
    )

    ranks: dict[str, int] = {}
    current_rank = 1
    for i, (sys_id, score) in enumerate(sorted_systems):
        if i > 0 and score == sorted_systems[i - 1][1]:
            ranks[sys_id] = ranks[sorted_systems[i - 1][0]]
        else:
            ranks[sys_id] = current_rank
        current_rank += 1

    return ranks


def _is_numeric(val: Any) -> bool:
    if val is None:
        return False
    try:
        f = float(val)
        return not (math.isnan(f) or math.isinf(f))
    except (ValueError, TypeError):
        return False
