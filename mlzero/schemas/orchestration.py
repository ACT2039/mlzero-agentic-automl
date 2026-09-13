"""
Schemas for orchestration (iteration and workflow).
"""
from pydantic import BaseModel, Field

from mlzero.schemas.coder import CodeArtifact, ErrorContext, ExecutionResult


class IterationRecord(BaseModel):
    """Record of a single coding and execution iteration."""
    iteration_number: int = Field(..., description="Iteration index (1-based).")
    code_artifact: CodeArtifact | None = Field(default=None, description="The generated code artifact.")
    execution_result: ExecutionResult | None = Field(default=None, description="The execution result.")
    error_context: ErrorContext | None = Field(default=None, description="Error analysis if execution failed.")
    duration_seconds: float = Field(default=0.0, description="Total duration of this iteration.")


class IterativeRunResult(BaseModel):
    """Final result of the iterative pipeline."""
    success: bool = Field(..., description="Whether the overall pipeline succeeded.")
    total_iterations: int = Field(..., description="Total number of iterations run.")
    final_code_artifact: CodeArtifact | None = Field(default=None, description="The last generated code artifact.")
    final_execution_result: ExecutionResult | None = Field(default=None, description="The last execution result.")
    iteration_history: list[IterationRecord] = Field(default_factory=list, description="History of all iterations.")
    final_error_context: ErrorContext | None = Field(default=None, description="Final error context if pipeline failed.")
    total_duration_seconds: float = Field(default=0.0, description="Total duration of the pipeline.")
