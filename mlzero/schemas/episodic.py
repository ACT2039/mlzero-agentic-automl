"""
Schemas for Episodic Memory.
"""
from typing import Any

from pydantic import BaseModel, Field


class Episode(BaseModel):
    """A single iteration episode in an ML automation run."""
    episode_id: str = Field(..., description="Unique identifier for the episode.")
    run_id: str = Field(..., description="ID of the ML automation run.")
    iteration: int = Field(..., description="Iteration number.")
    timestamp: str = Field(..., description="ISO 8601 timestamp.")
    status: str = Field(..., description="Outcome status (e.g., SUCCESS, FAIL).")
    
    code_artifact_reference: str | None = Field(default=None, description="Metadata/Summary of the generated code.")
    code_content: str | None = Field(default=None, description="The actual code generated.")
    
    execution_result_summary: dict[str, Any] | None = Field(default=None, description="Bounded execution result info.")
    error_context_summary: dict[str, Any] | None = Field(default=None, description="Bounded error context info.")
    
    retrieved_knowledge_references: list[str] = Field(default_factory=list, description="Sources of semantic knowledge used.")
    
    summary: str | None = Field(default=None, description="Brief summary of the iteration.")
    suggested_fix: str | None = Field(default=None, description="Suggested fix from Error Analyzer.")


class RunHistory(BaseModel):
    """The complete history of a single ML automation run."""
    run_id: str = Field(..., description="Unique identifier for the run.")
    start_time: str = Field(..., description="ISO 8601 start timestamp.")
    end_time: str | None = Field(default=None, description="ISO 8601 end timestamp.")
    
    perceptual_context_summary: dict[str, Any] | None = Field(default=None, description="Summary of the initial perception.")
    selected_library: str | None = Field(default=None, description="The ML library selected.")
    
    episodes: list[Episode] = Field(default_factory=list, description="Chronological episodes.")
    final_result: str | None = Field(default=None, description="Final outcome of the run.")
