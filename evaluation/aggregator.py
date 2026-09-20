"""
Multi-run statistical aggregator for MLZero Evaluation Framework.
Aggregates independent runs per evaluation case into mean ± std summaries.
"""
import statistics

from evaluation.cases import get_case
from evaluation.metrics import (
    calculate_first_attempt_success_rate,
    calculate_recovery_rate,
    calculate_routing_accuracy,
    calculate_success_rate,
)
from evaluation.schemas import CaseSummary, EvaluationRun


def aggregate_case_runs(runs: list[EvaluationRun]) -> CaseSummary | None:
    """
    Aggregate multiple runs (e.g. 3 runs) for a single EvaluationCase.
    Returns CaseSummary with mean and std stats.
    """
    if not runs:
        return None

    first_run = runs[0]
    case_id = first_run.case_id
    case_def = get_case(case_id)

    name = case_def.name if case_def else case_id
    task_type = case_def.task_type if case_def else (first_run.perceived_task_type or "unknown")
    expected_lib = case_def.expected_library if case_def else (first_run.selected_library or "unknown")
    eval_scope = case_def.evaluation_scope if case_def else getattr(first_run, "evaluation_scope", "end_to_end")

    total_runs = len(runs)
    successful_runs = sum(1 for r in runs if r.success)
    success_rate = calculate_success_rate(runs)
    first_attempt_success_rate = calculate_first_attempt_success_rate(runs)
    recovery_rate = calculate_recovery_rate(runs)

    iterations_list = [r.iterations for r in runs]
    avg_iterations = round(statistics.mean(iterations_list), 2) if iterations_list else 0.0

    times = [r.total_time for r in runs]
    mean_time = round(statistics.mean(times), 2) if times else 0.0
    median_time = round(statistics.median(times), 2) if times else 0.0
    min_time = round(min(times), 2) if times else 0.0
    max_time = round(max(times), 2) if times else 0.0

    routing = calculate_routing_accuracy(runs)
    task_perception_accuracy = routing["task_perception_accuracy"]
    library_selection_accuracy = routing["library_selection_accuracy"]

    # Metric aggregation (numeric metrics only)
    metric_keys: set[str] = set()
    for r in runs:
        if r.metrics:
            metric_keys.update(k for k, v in r.metrics.items() if v is not None and isinstance(v, (int, float)) and not isinstance(v, bool))

    metric_means: dict[str, float] = {}
    metric_stds: dict[str, float] = {}

    for k in metric_keys:
        vals: list[float] = []
        for r in runs:
            if r.metrics:
                val = r.metrics.get(k)
                if val is not None and isinstance(val, (int, float)) and not isinstance(val, bool):
                    vals.append(float(val))
        if vals:
            metric_means[k] = round(statistics.mean(vals), 4)
            if len(vals) > 1:
                metric_stds[k] = round(statistics.stdev(vals), 4)
            else:
                metric_stds[k] = 0.0

    return CaseSummary(
        case_id=case_id,
        name=name,
        task_type=task_type,
        expected_library=expected_lib,
        evaluation_scope=eval_scope,
        total_runs=total_runs,
        successful_runs=successful_runs,
        success_rate=success_rate,
        first_attempt_success_rate=first_attempt_success_rate,
        recovery_rate=recovery_rate,
        avg_iterations=avg_iterations,
        mean_execution_time=mean_time,
        median_execution_time=median_time,
        min_execution_time=min_time,
        max_execution_time=max_time,
        task_perception_accuracy=task_perception_accuracy,
        library_selection_accuracy=library_selection_accuracy,
        metric_means=metric_means,
        metric_stds=metric_stds,
    )
