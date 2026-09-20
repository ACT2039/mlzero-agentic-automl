"""
Pydantic schemas for the MLZero Evaluation Framework.
"""
from typing import Any, Literal

from pydantic import BaseModel, Field


class EvaluationCase(BaseModel):
    """Abstraction for a single evaluation benchmark case."""
    case_id: str = Field(..., description="Unique case identifier.")
    name: str = Field(..., description="Human-readable case name.")
    dataset_path: str = Field(..., description="Path to the dataset directory.")
    task_type: str = Field(..., description="Expected task type (classification, regression, time_series_forecasting, multimodal, retrieval, general_ml).")
    expected_library: str = Field(..., description="Expected target library (autogluon.tabular, autogluon.multimodal, autogluon.timeseries, FlagEmbedding, machine learning).")
    modality: str = Field(..., description="Primary modality (tabular, image, text, json, audio, mixed).")
    target_column: str | None = Field(default=None, description="Expected target column name if applicable.")
    description: str = Field(default="", description="Description of the evaluation case.")
    expected_output_type: str = Field(default="predictions_and_metrics", description="Expected output artifact type.")
    difficulty: Literal["easy", "medium", "hard"] = Field(default="easy", description="Difficulty classification.")
    source_type: str = Field(default="synthetic_local", description="Dataset provenance/source type.")
    tags: list[str] = Field(default_factory=list, description="Categorization tags.")
    evaluation_scope: Literal["end_to_end", "perception_only"] = Field(default="end_to_end", description="Evaluation scope (end_to_end or perception_only).")


class EvaluationRun(BaseModel):
    """Detailed record of a single evaluation run."""
    evaluation_id: str = Field(..., description="Unique run identifier.")
    case_id: str = Field(..., description="Associated EvaluationCase ID.")
    run_index: int = Field(default=1, description="1-based run iteration index.")
    llm_mode: Literal["mock", "real"] = Field(default="mock", description="LLM mode used.")
    evaluation_scope: Literal["end_to_end", "perception_only"] = Field(default="end_to_end", description="Evaluation scope.")
    selected_library: str | None = Field(default=None, description="Library selected by Perception.")
    perceived_task_type: str | None = Field(default=None, description="Task type perceived by Perception.")
    perceived_target: str | None = Field(default=None, description="Target column perceived by Perception.")
    perceived_timestamp_column: str | None = Field(default=None, description="Timestamp column perceived by Perception.")
    perceived_id_column: str | None = Field(default=None, description="Item ID column perceived by Perception.")
    task_type_correct: bool = Field(default=False, description="True if perceived task_type matches case.task_type.")
    library_correct: bool = Field(default=False, description="True if selected_library matches case.expected_library.")
    target_correct: bool | None = Field(default=None, description="True if perceived_target matches case.target_column (None if retrieval).")
    success: bool = Field(default=False, description="True if the execution completed successfully.")
    iterations: int = Field(default=0, description="Total code generation iterations attempted.")
    perception_time: float = Field(default=0.0, description="Wall-clock time spent in Perception phase.")
    retrieval_time: float = Field(default=0.0, description="Wall-clock time spent in Semantic Retrieval.")
    coding_time: float = Field(default=0.0, description="Wall-clock time spent in Coder execution.")
    execution_time: float = Field(default=0.0, description="Wall-clock time spent in Code Execution.")
    total_time: float = Field(default=0.0, description="Total wall-clock duration of the run.")
    judge_decisions: list[dict[str, Any]] = Field(default_factory=list, description="Execution judge decisions per iteration.")
    error_recovery_count: int = Field(default=0, description="Count of recovered code errors.")
    final_status: str = Field(default="NOT_STARTED", description="Final run status (SUCCESS, FAILED, CRASHED).")
    output_artifacts: list[str] = Field(default_factory=list, description="List of generated output file paths.")
    model_artifacts: list[str] = Field(default_factory=list, description="List of saved model file paths.")
    prediction_artifacts: list[str] = Field(default_factory=list, description="List of prediction file paths.")
    metrics: dict[str, float | int | str | None] = Field(default_factory=dict, description="Task-aware ML solution metrics.")
    tokens_input: int | None = Field(default=None, description="Total input tokens consumed if available.")
    tokens_output: int | None = Field(default=None, description="Total output tokens generated if available.")
    estimated_cost: float | None = Field(default=None, description="Estimated API cost in USD if available.")
    error_category: str | None = Field(default=None, description="Categorized final error if run failed.")
    failure_reason: str | None = Field(default=None, description="Detailed failure explanation.")
    pipeline_trace: dict[str, Any] = Field(default_factory=dict, description="Pipeline step execution trace.")
    execution_backend: str = Field(default="mock", description="Execution backend type (real or mock).")
    adapter_info: dict[str, Any] = Field(default_factory=dict, description="Adapter metadata and validation result.")
    requested_k: int = Field(default=5, description="Requested retrieval top_k.")
    actual_retrieved_k: int = Field(default=0, description="Actual retrieved chunk count.")


class CaseSummary(BaseModel):
    """Statistical summary across multiple runs for a single case."""
    case_id: str
    name: str
    task_type: str
    expected_library: str
    evaluation_scope: str = Field(default="end_to_end", description="Evaluation scope.")
    total_runs: int
    successful_runs: int
    success_rate: float
    first_attempt_success_rate: float
    recovery_rate: float | None = Field(default=None, description="Recovered runs / initially failed runs (None if initially failed runs == 0).")
    avg_iterations: float
    mean_execution_time: float
    median_execution_time: float
    min_execution_time: float
    max_execution_time: float
    task_perception_accuracy: float
    library_selection_accuracy: float
    metric_means: dict[str, float] = Field(default_factory=dict)
    metric_stds: dict[str, float] = Field(default_factory=dict)


class ReproducibilityMetadata(BaseModel):
    """Environment and config metadata for reproducible evaluation."""
    timestamp: str
    python_version: str
    platform: str
    git_commit: str | None
    llm_mode: str
    runs_per_case: int
    evaluator_version: str = "1.0.0"


class EvaluationSuiteResult(BaseModel):
    """Overall evaluation suite results container."""
    suite_name: str
    metadata: ReproducibilityMetadata
    cases: list[CaseSummary] = Field(default_factory=list)
    overall_success_rate: float = 0.0
    overall_library_accuracy: float = 0.0
    overall_task_perception_accuracy: float = 0.0
    overall_avg_iterations: float = 0.0
    overall_avg_time: float = 0.0
    raw_runs: list[EvaluationRun] = Field(default_factory=list)
