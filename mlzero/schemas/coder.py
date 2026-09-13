"""
Schemas for Coder and Executor.
"""
from typing import Any

from pydantic import BaseModel, Field


class CodeGenerationRequest(BaseModel):
    """Request for generating code."""
    perceptual_context_json: str = Field(..., description="Serialized PerceptualContext.")
    user_instruction: str | None = Field(default=None, description="Optional user instruction.")
    coding_guidance: str | None = Field(default=None, description="Optional coding guidance.")
    
    # Retry / Episodic context
    previous_code: str | None = Field(default=None, description="Previously generated code that failed.")
    error_context_json: str | None = Field(default=None, description="Serialized ErrorContext explaining the failure.")
    episodic_context_json: str | None = Field(default=None, description="Serialized bounded Episodic Context.")
    
    # Semantic memory
    retrieved_knowledge_json: str | None = Field(default=None, description="Serialized RetrievedKnowledge.")


class CodeArtifact(BaseModel):
    """Artifact produced by the Coder."""
    code: str = Field(..., description="Executable Python code.")
    language: str = Field(default="python", description="Language of the code.")
    entry_point: str | None = Field(default=None, description="Entry point file name if applicable.")
    dependencies: list[str] = Field(default_factory=list, description="Requested dependencies.")
    generation_metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata from LLM generation.")
    status: str = Field(default="generated", description="Status of the artifact.")


class ExecutionResult(BaseModel):
    """Result of running generated code."""
    success: bool = Field(..., description="Whether execution was successful.")
    return_code: int | None = Field(default=None, description="Return code of the process.")
    stdout: str = Field(default="", description="Captured standard output.")
    stderr: str = Field(default="", description="Captured standard error.")
    duration_seconds: float = Field(default=0.0, description="Execution duration in seconds.")
    output_files: list[str] = Field(default_factory=list, description="Paths of files produced by execution.")
    error_info: str | None = Field(default=None, description="Error information if applicable.")
    workspace_dir: str | None = Field(default=None, description="Path to the permanent output directory if persisted.")


class ErrorContext(BaseModel):
    """Context for a failed execution."""
    iteration: int = Field(..., description="The iteration number when the error occurred.")
    error_category: str = Field(..., description="Classification of the error (e.g., syntax, runtime, dependency).")
    error_message: str = Field(..., description="The core error message.")
    stderr_excerpt: str = Field(..., description="Relevant excerpt from standard error.")
    stdout_excerpt: str | None = Field(default=None, description="Relevant excerpt from standard output.")
    suggested_fix: str = Field(..., description="Actionable suggestion to fix the error.")
