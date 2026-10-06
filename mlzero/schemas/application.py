from typing import Any

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    dataset_path: str = Field(..., description="Path to the dataset directory.")
    user_instruction: str | None = Field(default=None, description="Optional instruction for the ML task.")
    options: dict[str, Any] = Field(default_factory=dict, description="Additional options for the run.")


class RunResponse(BaseModel):
    run_id: str = Field(..., description="The unique identifier for the run.")
    status: str = Field(..., description="Current status of the run (e.g. QUEUED, RUNNING, SUCCESS, FAIL).")


class RunStatusResponse(BaseModel):
    run_id: str = Field(..., description="The unique identifier for the run.")
    status: str = Field(..., description="Current status of the run.")
    success: bool | None = Field(default=None, description="Whether the run succeeded, if completed.")
    task_summary: dict[str, Any] | None = Field(default=None, description="Summary of the perceived task.")
    dataset_name: str | None = Field(default=None, description="Intelligent display name of the dataset.")
    selected_library: str | None = Field(default=None, description="The ML library selected.")
    iterations: int | None = Field(default=None, description="Total number of iterations taken.")
    final_metrics: dict[str, Any] | None = Field(default=None, description="Metrics from the final ML evaluation.")
    raw_metrics: dict[str, Any] | None = Field(default=None, description="Original un-normalized metrics as returned by the internal framework.")
    prediction_artifact_reference: str | None = Field(default=None, description="Path or reference to predictions.csv.")
    model_artifact_reference: str | None = Field(default=None, description="Path or reference to the trained model.")
    execution_duration: float | None = Field(default=None, description="Duration in seconds.")
    final_error: str | None = Field(default=None, description="Final error context if unsuccessful.")
    judge_decisions: list[dict[str, Any]] = Field(default_factory=list, description="Judge decisions per iteration.")
    pipeline_trace: dict[str, Any] = Field(default_factory=dict, description="Pipeline execution step trace.")
    execution_backend: str = Field(default="mock", description="Backend execution mode (mock or real).")
    adapter_info: dict[str, Any] = Field(default_factory=dict, description="Adapter validation and execution details.")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message.")
    details: str | None = Field(default=None, description="Additional error details.")
