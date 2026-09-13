"""
Perception schemas.
"""
from typing import Any

from pydantic import BaseModel, Field


class FileMetadata(BaseModel):
    path: str = Field(..., description="Relative path from dataset root.")
    absolute_path: str = Field(..., description="Absolute path on disk.")
    extension: str = Field(..., description="File extension in lowercase.")
    size_bytes: int = Field(..., description="File size in bytes.")
    file_type: str = Field(..., description="Determined type (e.g., 'csv', 'document', 'binary').")


class FileContext(BaseModel):
    metadata: FileMetadata
    columns: list[str] | None = Field(default=None, description="Column names if tabular.")
    row_count: int | None = Field(default=None, description="Total rows if practically determinable.")
    sample_rows: list[dict[str, Any]] | None = Field(default=None, description="A small sample of data rows.")
    inferred_types: dict[str, str] | None = Field(default=None, description="Inferred basic data types for columns.")
    extracted_text: str | None = Field(default=None, description="Text extracted from document files.")
    error: str | None = Field(default=None, description="Error encountered during inspection.")


class TaskContext(BaseModel):
    objective: str | None = Field(default=None, description="Main objective of the user instruction.")
    task_type: str | None = Field(default=None, description="Type of ML task (e.g., 'classification', 'regression').")
    target_column: str | None = Field(default=None, description="Explicitly identified target label column.")
    input_data_files: list[str] | None = Field(default=None, description="List of primary input file paths.")
    output_requirements: str | None = Field(default=None, description="Requirements for the output.")
    evaluation_metric: str | None = Field(default=None, description="Evaluation metric if explicitly stated.")
    constraints: list[str] | None = Field(default=None, description="Any explicit constraints.")
    relevant_description_files: list[str] | None = Field(default=None, description="Files containing relevant descriptions.")
    explanation: str | None = Field(default=None, description="Explanation for context extraction.")


class LibrarySelection(BaseModel):
    selected_library: str = Field(..., description="The name of the selected library.")
    confidence: str | None = Field(default=None, description="Confidence in the selection.")
    explanation: str = Field(..., description="Explanation of why this library was chosen.")


class PerceptualContext(BaseModel):
    files: list[FileContext] = Field(default_factory=list, description="Context for each inspected file.")
    task: TaskContext | None = Field(default=None, description="Inferred task context.")
    library: LibrarySelection | None = Field(default=None, description="Selected library.")
    errors: list[str] = Field(default_factory=list, description="Any top-level perception errors.")
