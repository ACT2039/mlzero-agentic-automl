"""
CLI Entry point and execution engine for MLZero Evaluation Framework.
Supports evaluation modes: smoke, local_formal, ablation, robustness, real_llm.
"""
import argparse
import os
import platform
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.ablations import AblationRunner
from evaluation.aggregator import aggregate_case_runs
from evaluation.cases import BENCHMARK_CASES, get_case
from evaluation.metrics import (
    calculate_routing_accuracy,
    calculate_success_rate,
    extract_metrics,
)
from evaluation.report import generate_evaluation_reports
from evaluation.robustness import RobustnessEvaluator
from evaluation.schemas import (
    EvaluationCase,
    EvaluationRun,
    EvaluationSuiteResult,
    ReproducibilityMetadata,
)
from mlzero.application.service import MLZeroService


def execute_case_run(
    case: EvaluationCase,
    run_index: int = 1,
    use_mock_llm: bool = True,
    disable_semantic: bool = False,
    disable_episodic: bool = False,
    disable_judge: bool = False,
    retrieval_k: int = 5,
    max_iterations: int | None = None,
) -> EvaluationRun:
    """Execute a single evaluation run for a given EvaluationCase."""
    start_total = time.time()
    service = MLZeroService(use_mock_llm=use_mock_llm)

    # Configure ablations / switches
    if disable_semantic:
        service.semantic_memory = None  # type: ignore[assignment]
    if disable_episodic:
        service.episodic_memory = None  # type: ignore[assignment]

    dataset_dir = Path(case.dataset_path)

    # Perception phase timing
    start_percept = time.time()
    pctx = None
    try:
        pctx = service.perceive(input_path=dataset_dir)
    except Exception:  # noqa: BLE001, S110
        pass
    perception_time = round(time.time() - start_percept, 3)

    perceived_task = pctx.task_type if pctx else None
    perceived_target = pctx.target_column if pctx else None
    perceived_ts = pctx.timestamp_column if pctx else None
    perceived_id = pctx.id_column if pctx else None
    selected_lib = pctx.selected_library if pctx else None

    task_type_correct = (perceived_task == case.task_type)
    library_correct = (selected_lib == case.expected_library)

    target_correct: bool | None = None
    if case.target_column is not None:
        target_correct = (perceived_target == case.target_column)

    # Full execution phase timing
    start_exec = time.time()
    success = False
    iterations = 0
    final_status = "FAILED"
    error_cat = None
    failure_reason = None
    metrics_dict: dict[str, float | int | str | None] = {}
    artifacts: list[str] = []
    judge_decisions_list: list[dict[str, Any]] = []
    ptrace: dict[str, Any] = {}
    backend_mode = "mock" if use_mock_llm else "real"
    adapter_meta: dict[str, Any] = {}
    req_k = 0 if disable_semantic else retrieval_k
    act_k = 0

    if case.evaluation_scope == "perception_only":
        success = bool(task_type_correct and library_correct)
        iterations = 0
        final_status = "SUCCESS" if success else "FAILED"
        metrics_dict = {"perception_success": 1.0 if success else 0.0}
        ptrace = {
            "perception_completed": True,
            "semantic_memory_completed": False,
            "coder_completed": False,
            "adapter_used": "None",
            "executor_completed": False,
            "judge_decisions": [],
            "episodic_memory_recorded": False,
            "requested_k": 0,
            "actual_retrieved_k": 0,
        }
    else:
        try:
            run_options: dict[str, Any] = {
                "disable_judge": disable_judge,
                "disable_episodic": disable_episodic,
                "disable_semantic": disable_semantic,
                "retrieval_k": req_k,
            }
            if max_iterations is not None:
                run_options["max_iterations"] = max_iterations
            run_status = service.run_mlzero(
                dataset_path=str(dataset_dir),
                options=run_options,
            )
            success = getattr(run_status, "success", False)
            iterations = getattr(run_status, "iterations", 1)
            final_status = "SUCCESS" if success else "FAILED"
            error_cat = getattr(run_status, "final_error", None)

            if getattr(run_status, "prediction_artifact_reference", None):
                artifacts.append(str(run_status.prediction_artifact_reference))
            if getattr(run_status, "model_artifact_reference", None):
                artifacts.append(str(run_status.model_artifact_reference))

            metrics_dict = extract_metrics(run_status, task_type=case.task_type)
            judge_decisions_list = getattr(run_status, "judge_decisions", [])
            ptrace = getattr(run_status, "pipeline_trace", {})
            backend_mode = getattr(run_status, "execution_backend", "mock")
            adapter_meta = getattr(run_status, "adapter_info", {})
            act_k = ptrace.get("actual_retrieved_k", 0)
        except Exception as e:  # noqa: BLE001
            success = False
            final_status = "CRASHED"
            error_cat = "system_crash"
            failure_reason = str(e)

    total_time = round(time.time() - start_total, 3)
    execution_time = round(time.time() - start_exec, 3)

    run_id = f"eval_{uuid.uuid4().hex[:8]}"

    return EvaluationRun(
        evaluation_id=run_id,
        case_id=case.case_id,
        run_index=run_index,
        llm_mode="mock" if use_mock_llm else "real",
        evaluation_scope=case.evaluation_scope,
        selected_library=selected_lib,
        perceived_task_type=perceived_task,
        perceived_target=perceived_target,
        perceived_timestamp_column=perceived_ts,
        perceived_id_column=perceived_id,
        task_type_correct=task_type_correct,
        library_correct=library_correct,
        target_correct=target_correct,
        success=success,
        iterations=iterations,
        perception_time=perception_time,
        retrieval_time=0.0,
        coding_time=0.0,
        execution_time=execution_time,
        total_time=total_time,
        judge_decisions=judge_decisions_list,
        error_recovery_count=max(0, iterations - 1) if (success and case.evaluation_scope == "end_to_end") else 0,
        final_status=final_status,
        output_artifacts=artifacts,
        metrics=metrics_dict,
        error_category=error_cat,
        failure_reason=failure_reason,
        pipeline_trace=ptrace,
        execution_backend=backend_mode,
        adapter_info=adapter_meta,
        requested_k=req_k,
        actual_retrieved_k=act_k,
    )


def run_evaluation_suite(
    mode: str = "smoke",
    runs_per_case: int = 1,
    selected_cases: list[str] | None = None,
    use_mock_llm: bool = True,
    output_dir: str = "reports",
) -> EvaluationSuiteResult | dict[str, Any] | None:
    """Run specified evaluation suite mode."""
    cases_to_run: list[EvaluationCase] = []
    if selected_cases:
        for c_name in selected_cases:
            c_obj = get_case(c_name)
            if c_obj is not None:
                cases_to_run.append(c_obj)
    elif mode == "smoke":
        cases_to_run = [c for c in BENCHMARK_CASES if c.case_id in ("case_01_tiny_cls", "case_03_house_price_faulty")]
    else:
        cases_to_run = list(BENCHMARK_CASES)

    print(f"=== Running MLZero Evaluation Suite [Mode: {mode}, Cases: {len(cases_to_run)}, Runs/Case: {runs_per_case}] ===")

    all_raw_runs: list[EvaluationRun] = []
    case_summaries = []

    for case in cases_to_run:
        print(f" -> Evaluating Case: {case.name} ({case.case_id})")
        case_runs = []
        for r_idx in range(1, runs_per_case + 1):
            run = execute_case_run(case=case, run_index=r_idx, use_mock_llm=use_mock_llm)
            case_runs.append(run)
            all_raw_runs.append(run)

        summary = aggregate_case_runs(case_runs)
        if summary:
            case_summaries.append(summary)

    # Suite-level metrics
    overall_success = calculate_success_rate(all_raw_runs)
    routing = calculate_routing_accuracy(all_raw_runs)

    avg_iters = round(sum(r.iterations for r in all_raw_runs) / max(1, len(all_raw_runs)), 2)
    avg_time = round(sum(r.total_time for r in all_raw_runs) / max(1, len(all_raw_runs)), 2)

    from datetime import timezone
    meta = ReproducibilityMetadata(
        timestamp=datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        python_version=platform.python_version(),
        platform=platform.platform(),
        git_commit=os.environ.get("GIT_COMMIT", None),
        llm_mode="mock" if use_mock_llm else "real",
        runs_per_case=runs_per_case,
    )

    suite_result = EvaluationSuiteResult(
        suite_name=f"MLZero_{mode.title()}_Evaluation",
        metadata=meta,
        cases=case_summaries,
        overall_success_rate=overall_success,
        overall_library_accuracy=routing["library_selection_accuracy"],
        overall_task_perception_accuracy=routing["task_perception_accuracy"],
        overall_avg_iterations=avg_iters,
        overall_avg_time=avg_time,
        raw_runs=all_raw_runs,
    )

    # Optional Ablations / Robustness if in dedicated mode
    ablations_dict = None
    robustness_dict = None

    if mode == "ablation":
        ab_runner = AblationRunner(use_mock_llm=use_mock_llm)
        ablations_dict = ab_runner.run_full_ablation_suite(runs_per_config=runs_per_case)

    if mode == "robustness":
        rob_eval = RobustnessEvaluator(use_mock_llm=use_mock_llm)
        robustness_dict = rob_eval.run_robustness_suite(base_case_id="case_01_tiny_cls")

    if mode == "four_way_ablation":
        from evaluation.four_way_ablation import (
            FourWayAblationRunner,
            print_comparison_table,
        )
        ab4_runner = FourWayAblationRunner(
            use_mock_llm=use_mock_llm,
            max_iterations=3,
            retrieval_k=5,
            output_dir=output_dir,
        )
        selected_case_objs = None
        if selected_cases:
            selected_case_objs = [c for c in (get_case(cn) for cn in selected_cases) if c is not None]
        ab4_result = ab4_runner.run_experiment(cases=selected_case_objs, runs_per_case=runs_per_case)
        print_comparison_table(ab4_result)
        return ab4_result

    csv_p, json_p, md_p = generate_evaluation_reports(
        suite_result=suite_result,
        ablation_results=ablations_dict,
        robustness_results=robustness_dict,
        output_dir=output_dir,
    )

    print("=== Evaluation Complete ===")
    print(f"  CSV Report: {csv_p}")
    print(f"  JSON Summary: {json_p}")
    print(f"  Markdown Report: {md_p}\n")

    return suite_result


def main() -> None:
    parser = argparse.ArgumentParser(description="MLZero Agentic AutoML Evaluation Runner")
    parser.add_argument(
        "--mode",
        choices=["smoke", "local_formal", "ablation", "robustness", "real_llm", "four_way_ablation"],
        default="local_formal",
        help="Evaluation mode to run.",
    )
    parser.add_argument("--runs", type=int, default=None, help="Number of runs per case (default: 3 for local_formal/four_way_ablation, 1 for smoke).")
    parser.add_argument("--cases", type=str, default=None, help="Comma-separated case IDs or names to run.")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory for report artifacts.")
    parser.add_argument("--llm-mode", choices=["mock", "real"], default="mock", help="LLM provider mode.")

    args = parser.parse_args()

    use_mock = (args.llm_mode == "mock")
    runs_cnt = args.runs
    if runs_cnt is None:
        runs_cnt = 3 if args.mode in ("local_formal", "four_way_ablation") else 1

    selected = args.cases.split(",") if args.cases else None

    run_evaluation_suite(
        mode=args.mode,
        runs_per_case=runs_cnt,
        selected_cases=selected,
        use_mock_llm=use_mock,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
