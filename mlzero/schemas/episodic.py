"""
Schemas for Episodic Memory.
"""
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Episode(BaseModel):
    """A single iteration episode in an ML automation run."""
    episode_id: str = Field(..., description="Unique identifier for the episode.")
    run_id: str = Field(..., description="ID of the ML automation run.")
    iteration: int = Field(..., description="Iteration number.")
    timestamp: str = Field(..., description="ISO 8601 timestamp.")
    status: str = Field(..., description="Outcome status (e.g., SUCCESS, FAIL).")
    
    # Context & inputs for this iteration
    perceptual_context: dict[str, Any] | None = Field(default=None, description="Perceptual context at this iteration.")
    selected_library: str | None = Field(default=None, description="Selected ML library.")
    user_instruction: str | None = Field(default=None, description="User instruction provided.")
    error_context_before: dict[str, Any] | None = Field(default=None, description="Error context that guided this iteration.")
    
    # Code generation
    generated_code: str | None = Field(default=None, description="Actual code generated in this iteration.")
    code_content: str | None = Field(default=None, description="Backward-compatible alias for generated_code.")
    code_artifact_reference: str | None = Field(default=None, description="Summary or reference of generated code.")
    bash_script: str | None = Field(default=None, description="Optional setup or execution shell script.")
    
    # Execution logs & outcome
    execution_status: str | None = Field(default=None, description="Status: SUCCESS, FAILURE, TIMEOUT, INVALID_OUTPUT.")
    execution_result: dict[str, Any] | None = Field(default=None, description="Full execution result details.")
    execution_result_summary: dict[str, Any] | None = Field(default=None, description="Backward-compatible execution summary.")
    stdout: str | None = Field(default=None, description="Standard output from execution.")
    stderr: str | None = Field(default=None, description="Standard error from execution.")
    exit_code: int | None = Field(default=None, description="Process exit code.")
    execution_duration: float | None = Field(default=None, description="Execution duration in seconds.")
    missing_expected_files: list[str] = Field(default_factory=list, description="Expected files that were not produced.")
    
    # Semantic knowledge retrieved for this iteration
    retrieval_query: str | None = Field(default=None, description="Query used for semantic retrieval.")
    retrieved_knowledge: list[dict[str, Any]] | None = Field(default=None, description="Retrieved knowledge chunks.")
    retrieved_knowledge_references: list[str] = Field(default_factory=list, description="Sources of semantic knowledge used.")
    
    # Judgment & Error diagnosis
    judge_decision: str | None = Field(default=None, description="Execution judge decision: FINISH or FIX.")
    judge_reason: str | None = Field(default=None, description="Rationale for judge decision.")
    error_category: str | None = Field(default=None, description="Category of error.")
    error_summary: str | None = Field(default=None, description="Concise error summary.")
    suggested_fix: str | None = Field(default=None, description="Suggested fix from Error Analyzer.")
    error_context_summary: dict[str, Any] | None = Field(default=None, description="Backward-compatible error summary.")
    
    summary: str | None = Field(default=None, description="Brief summary of the iteration.")

    @model_validator(mode="before")
    @classmethod
    def sync_backward_compatible_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Sync code fields
            if "generated_code" in data and not data.get("code_content"):
                data["code_content"] = data["generated_code"]
            elif "code_content" in data and not data.get("generated_code"):
                data["generated_code"] = data["code_content"]

            # Sync execution result fields
            if "execution_result" in data and not data.get("execution_result_summary"):
                data["execution_result_summary"] = data["execution_result"]
            elif "execution_result_summary" in data and not data.get("execution_result"):
                data["execution_result"] = data["execution_result_summary"]

            # Sync error context fields
            if "error_context_summary" in data and isinstance(data["error_context_summary"], dict):
                if not data.get("error_category"):
                    data["error_category"] = data["error_context_summary"].get("error_category")
                if not data.get("error_summary"):
                    data["error_summary"] = data["error_context_summary"].get("error_summary")
                if not data.get("suggested_fix"):
                    data["suggested_fix"] = data["error_context_summary"].get("suggested_fix")
        return data


class RunHistory(BaseModel):
    """The complete history of a single ML automation run."""
    run_id: str = Field(..., description="Unique identifier for the run.")
    start_time: str = Field(..., description="ISO 8601 start timestamp.")
    end_time: str | None = Field(default=None, description="ISO 8601 end timestamp.")
    
    perceptual_context_summary: dict[str, Any] | None = Field(default=None, description="Summary of the initial perception.")
    selected_library: str | None = Field(default=None, description="The ML library selected.")
    user_instruction: str | None = Field(default=None, description="User instruction if any.")
    
    episodes: list[Episode] = Field(default_factory=list, description="Chronological episodes.")
    final_result: str | None = Field(default=None, description="Final outcome of the run.")

