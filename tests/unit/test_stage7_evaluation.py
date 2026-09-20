"""
Stage 7 Unit Tests for MLZero Evaluation Framework.
Covers 23 required points: Schemas, Aggregation, Metrics, Ablations, Robustness,
Routing Accuracy, Reports (CSV/JSON/MD), Reproducibility, and MockLLM runner.
"""

from pathlib import Path

from evaluation.ablations import AblationRunner
from evaluation.aggregator import aggregate_case_runs
from evaluation.cases import BENCHMARK_CASES
from evaluation.metrics import (
    calculate_first_attempt_success_rate,
    calculate_rank,
    calculate_recovery_rate,
    calculate_relative_time,
    calculate_routing_accuracy,
    calculate_success_rate,
    extract_metrics,
)
from evaluation.report import generate_evaluation_reports
from evaluation.robustness import RobustnessEvaluator
from evaluation.runner import run_evaluation_suite
from evaluation.schemas import (
    CaseSummary,
    EvaluationCase,
    EvaluationRun,
    EvaluationSuiteResult,
    ReproducibilityMetadata,
)


def test_01_evaluation_case_schema():
    """1. Test EvaluationCase schema instantiation and validation."""
    case = EvaluationCase(
        case_id="case_test",
        name="Test Case",
        dataset_path="tests/data/tiny_classification",
        task_type="classification",
        expected_library="autogluon.tabular",
        modality="tabular",
        target_column="target",
    )
    assert case.case_id == "case_test"
    assert case.expected_library == "autogluon.tabular"


def test_02_evaluation_run_schema():
    """2. Test EvaluationRun schema instantiation."""
    run = EvaluationRun(
        evaluation_id="eval_101",
        case_id="case_01_tiny_cls",
        run_index=1,
        llm_mode="mock",
        selected_library="autogluon.tabular",
        perceived_task_type="classification",
        success=True,
        iterations=1,
        total_time=1.5,
    )
    assert run.evaluation_id == "eval_101"
    assert run.success is True


def test_03_aggregation_logic():
    """3. Test multi-run aggregation logic."""
    r1 = EvaluationRun(
        evaluation_id="e1", case_id="case_01_tiny_cls", success=True, iterations=1, total_time=2.0, metrics={"accuracy": 0.9}
    )
    r2 = EvaluationRun(
        evaluation_id="e2", case_id="case_01_tiny_cls", success=True, iterations=2, total_time=4.0, metrics={"accuracy": 0.95}
    )
    summary = aggregate_case_runs([r1, r2])
    assert summary is not None
    assert summary.total_runs == 2
    assert summary.success_rate == 1.0
    assert summary.avg_iterations == 1.5
    assert summary.metric_means["accuracy"] == 0.925


def test_04_success_rate_calculation():
    """4. Test success rate calculation."""
    r1 = EvaluationRun(evaluation_id="e1", case_id="c1", success=True)
    r2 = EvaluationRun(evaluation_id="e2", case_id="c1", success=False)
    assert calculate_success_rate([r1, r2]) == 0.5


def test_05_recovery_rate_calculation():
    """5. Test recovery rate calculation."""
    # Case with no initially failed runs -> return None
    r1 = EvaluationRun(evaluation_id="e1", case_id="c1", success=True, iterations=1)
    assert calculate_recovery_rate([r1]) is None

    # Case with initially failed runs
    r2 = EvaluationRun(evaluation_id="e2", case_id="c1", success=True, iterations=2, error_recovery_count=1)
    r3 = EvaluationRun(evaluation_id="e3", case_id="c1", success=False, iterations=3)
    rec = calculate_recovery_rate([r2, r3])
    assert rec == 0.5


def test_06_iteration_statistics():
    """6. Test iteration statistics and first attempt success rate."""
    r1 = EvaluationRun(evaluation_id="e1", case_id="c1", success=True, iterations=1)
    r2 = EvaluationRun(evaluation_id="e2", case_id="c1", success=True, iterations=2)
    assert calculate_first_attempt_success_rate([r1, r2]) == 0.5


def test_07_runtime_statistics():
    """7. Test runtime statistics calculation and relative time."""
    r1 = EvaluationRun(evaluation_id="e1", case_id="c1", total_time=10.0)
    r2 = EvaluationRun(evaluation_id="e2", case_id="c1", total_time=20.0)
    summary = aggregate_case_runs([r1, r2])
    assert summary is not None
    assert summary.min_execution_time == 10.0
    assert summary.max_execution_time == 20.0
    assert summary.mean_execution_time == 15.0

    rel_time = calculate_relative_time(20.0, 10.0)
    assert rel_time == 2.0


def test_08_metric_aggregation():
    """8. Test task-aware metric extraction and aggregation."""
    class DummyRunStatus:
        success = True
        iterations = 1
        execution_duration = 1.0
        final_error = None
        final_metrics = {"accuracy": 0.85, "f1": 0.84, "mae": 0.1}  # noqa: RUF012

    extracted = extract_metrics(DummyRunStatus())
    assert extracted["accuracy"] == 0.85
    assert extracted["f1"] == 0.84
    assert extracted["mae"] == 0.1


def test_09_rank_calculation():
    """9. Test dataset-level rank calculation across configurations."""
    scores = {"sys_a": 0.95, "sys_b": 0.88, "sys_c": 0.95}
    ranks = calculate_rank(scores, higher_is_better=True)
    assert ranks["sys_a"] == 1
    assert ranks["sys_c"] == 1
    assert ranks["sys_b"] == 3


def test_10_three_run_aggregation():
    """10. Test 3-run case summary aggregation."""
    runs = [
        EvaluationRun(evaluation_id=f"e{i}", case_id="case_01_tiny_cls", success=True, iterations=1, total_time=float(i))
        for i in range(1, 4)
    ]
    summary = aggregate_case_runs(runs)
    assert summary is not None
    assert summary.total_runs == 3
    assert summary.success_rate == 1.0


def test_11_semantic_memory_ablation():
    """11. Test Semantic Memory ablation runner."""
    runner = AblationRunner(use_mock_llm=True)
    res = runner.run_semantic_memory_ablation(case_ids=["case_01_tiny_cls"], runs_per_config=1)
    assert "semantic_memory_off" in res
    assert "semantic_memory_on" in res


def test_12_episodic_memory_ablation():
    """12. Test Episodic Memory ablation runner."""
    runner = AblationRunner(use_mock_llm=True)
    res = runner.run_episodic_memory_ablation(case_ids=["case_03_house_price_faulty"], runs_per_config=1)
    assert "episodic_memory_off" in res
    assert "episodic_memory_on" in res


def test_13_execution_judge_ablation():
    """13. Test Execution Judge ablation runner."""
    runner = AblationRunner(use_mock_llm=True)
    res = runner.run_execution_judge_ablation(case_ids=["case_01_tiny_cls"], runs_per_config=1)
    assert "judge_off" in res
    assert "judge_on" in res


def test_14_retrieval_k_ablation():
    """14. Test Retrieval Top-K ablation runner."""
    runner = AblationRunner(use_mock_llm=True)
    res = runner.run_retrieval_k_ablation(k_values=[0, 5], case_ids=["case_01_tiny_cls"])
    assert 0 in res
    assert 5 in res


def test_15_robustness_evaluation(tmp_path: Path):
    """15. Test data noise robustness evaluator."""
    evaluator = RobustnessEvaluator(temp_dir=tmp_path / "rob_test", use_mock_llm=True)
    res = evaluator.run_robustness_suite(base_case_id="case_01_tiny_cls")
    assert "missing_values" in res["summaries"]
    assert "malformed_numerics" in res["summaries"]


def test_16_task_perception_accuracy():
    """16. Test task perception accuracy calculation."""
    runs = [
        EvaluationRun(evaluation_id="e1", case_id="c1", task_type_correct=True),
        EvaluationRun(evaluation_id="e2", case_id="c1", task_type_correct=False),
    ]
    routing = calculate_routing_accuracy(runs)
    assert routing["task_perception_accuracy"] == 0.5


def test_17_library_selection_accuracy():
    """17. Test library selection accuracy calculation."""
    runs = [
        EvaluationRun(evaluation_id="e1", case_id="c1", library_correct=True),
        EvaluationRun(evaluation_id="e2", case_id="c1", library_correct=True),
    ]
    routing = calculate_routing_accuracy(runs)
    assert routing["library_selection_accuracy"] == 1.0


def test_18_report_generation(tmp_path: Path):
    """18. Test report generation outputs."""
    meta = ReproducibilityMetadata(
        timestamp="2026-09-20T00:00:00Z",
        python_version="3.11.9",
        platform="Windows",
        git_commit="test_commit",
        llm_mode="mock",
        runs_per_case=1,
    )
    suite = EvaluationSuiteResult(
        suite_name="Test_Suite",
        metadata=meta,
        cases=[
            CaseSummary(
                case_id="c1",
                name="Test Case",
                task_type="classification",
                expected_library="autogluon.tabular",
                total_runs=1,
                successful_runs=1,
                success_rate=1.0,
                first_attempt_success_rate=1.0,
                recovery_rate=None,
                avg_iterations=1.0,
                mean_execution_time=1.2,
                median_execution_time=1.2,
                min_execution_time=1.2,
                max_execution_time=1.2,
                task_perception_accuracy=1.0,
                library_selection_accuracy=1.0,
            )
        ],
        overall_success_rate=1.0,
        overall_library_accuracy=1.0,
        overall_task_perception_accuracy=1.0,
    )
    csv_p, json_p, md_p = generate_evaluation_reports(suite, output_dir=str(tmp_path))
    assert csv_p.exists()
    assert json_p.exists()
    assert md_p.exists()


def test_19_csv_output(tmp_path: Path):
    """19. Test CSV summary contents."""
    meta = ReproducibilityMetadata(timestamp="2026", python_version="3.11", platform="Win", git_commit=None, llm_mode="mock", runs_per_case=1)
    suite = EvaluationSuiteResult(suite_name="S1", metadata=meta, cases=[])
    csv_p, _, _ = generate_evaluation_reports(suite, output_dir=str(tmp_path))
    content = csv_p.read_text()
    assert "Case ID" in content
    assert "Expected Library" in content


def test_20_json_output(tmp_path: Path):
    """20. Test JSON summary content and serialization."""
    meta = ReproducibilityMetadata(timestamp="2026", python_version="3.11", platform="Win", git_commit=None, llm_mode="mock", runs_per_case=1)
    suite = EvaluationSuiteResult(suite_name="S1", metadata=meta, cases=[])
    _, json_p, _ = generate_evaluation_reports(suite, output_dir=str(tmp_path))
    content = json_p.read_text()
    assert "suite" in content
    assert "reproducibility" in content or "metadata" in content


def test_21_reproducibility_metadata():
    """21. Test reproducibility metadata field population."""
    meta = ReproducibilityMetadata(timestamp="2026-09-20", python_version="3.11", platform="Win", git_commit="abc", llm_mode="mock", runs_per_case=3)
    assert meta.runs_per_case == 3
    assert meta.evaluator_version == "1.0.0"


def test_22_mock_llm_evaluation_suite():
    """22. Test execution of MockLLM smoke evaluation suite."""
    suite_res = run_evaluation_suite(mode="smoke", runs_per_case=1, use_mock_llm=True)
    assert suite_res.overall_success_rate >= 0.0
    assert len(suite_res.cases) >= 1


def test_23_cases_registry():
    """23. Test BENCHMARK_CASES registry contains 10 cases across modalities."""
    cases = BENCHMARK_CASES
    assert len(cases) == 10
    case_ids = {c.case_id for c in cases}
    assert "case_01_tiny_cls" in case_ids
    assert "case_03_house_price_faulty" in case_ids
    assert "case_04_tiny_timeseries" in case_ids
    assert "case_08_tiny_retrieval" in case_ids
