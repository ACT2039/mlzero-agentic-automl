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


class ImageMetadataSummary(BaseModel):
    width: int | None = Field(default=None, description="Image width in pixels.")
    height: int | None = Field(default=None, description="Image height in pixels.")
    channels: int | None = Field(default=None, description="Number of channels (e.g., 1 for grayscale, 3 for RGB, 4 for RGBA).")
    mode: str | None = Field(default=None, description="Image color mode (RGB, L, RGBA, etc.).")
    format: str | None = Field(default=None, description="Detected image format (PNG, JPEG, etc.).")


class AudioMetadataSummary(BaseModel):
    sample_rate: int | None = Field(default=None, description="Audio sample rate in Hz.")
    channels: int | None = Field(default=None, description="Number of audio channels.")
    duration_sec: float | None = Field(default=None, description="Duration in seconds if lightweight inspection available.")
    format: str | None = Field(default=None, description="Audio format (WAV, MP3, etc.).")


class JSONStructureSummary(BaseModel):
    keys: list[str] = Field(default_factory=list, description="Top-level object keys if dict.")
    is_list: bool = Field(default=False, description="True if top-level structure is a list.")
    nesting_depth: int = Field(default=1, description="Approximate nesting depth.")
    item_count: int | None = Field(default=None, description="Number of items if top-level list.")
    sample_item: Any | None = Field(default=None, description="Bounded sample item.")


class TextSummary(BaseModel):
    line_count: int | None = Field(default=None, description="Total line count if practically determinable.")
    encoding: str = Field(default="utf-8", description="Detected text encoding.")
    first_lines: list[str] = Field(default_factory=list, description="Bounded initial lines of text.")


class FileGroup(BaseModel):
    group_id: str = Field(..., description="Unique identifier for the file group.")
    files: list[str] = Field(default_factory=list, description="Relative paths of files in this group.")
    modality: str = Field(..., description="Primary modality (tabular, image, text, audio, json, mixed, etc.).")
    relationship: str = Field(..., description="Relationship (e.g. train_test_pair, image_label_mapping, document_labels, query_corpus, multi_table, etc.).")
    purpose: str | None = Field(default=None, description="Inferred purpose of this file group.")
    confidence: float = Field(default=1.0, description="Confidence score.")


class FileContext(BaseModel):
    metadata: FileMetadata
    columns: list[str] | None = Field(default=None, description="Column names if tabular.")
    row_count: int | None = Field(default=None, description="Total rows if practically determinable.")
    sample_rows: list[dict[str, Any]] | None = Field(default=None, description="A small sample of data rows.")
    inferred_types: dict[str, str] | None = Field(default=None, description="Inferred basic data types for columns.")
    extracted_text: str | None = Field(default=None, description="Text extracted from document files.")
    text_summary: TextSummary | None = Field(default=None, description="Text structural summary.")
    json_summary: JSONStructureSummary | None = Field(default=None, description="JSON structural summary.")
    image_summary: ImageMetadataSummary | None = Field(default=None, description="Image metadata summary.")
    audio_summary: AudioMetadataSummary | None = Field(default=None, description="Audio metadata summary.")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal warnings recorded during file inspection.")
    error: str | None = Field(default=None, description="Error encountered during inspection.")


class MissingValueInfo(BaseModel):
    column: str
    missing_count: int
    missing_pct: float


class InvalidNumericInfo(BaseModel):
    column: str
    invalid_values: list[str]
    invalid_count: int


class DataQualityReport(BaseModel):
    """Deterministic data quality summary from profiling a tabular dataset."""
    target_column: str | None = Field(default=None, description="Inferred target column name.")
    timestamp_column: str | None = Field(default=None, description="Inferred timestamp/date column name.")
    id_column: str | None = Field(default=None, description="Inferred item/entity ID column name.")
    task_type: str | None = Field(default=None, description="Inferred task type from data evidence.")
    row_count: int | None = Field(default=None, description="Number of rows in training file.")
    column_names: list[str] = Field(default_factory=list, description="Column names from training file.")
    missing_values: list[MissingValueInfo] = Field(default_factory=list, description="Columns with missing values.")
    invalid_numeric_values: list[InvalidNumericInfo] = Field(default_factory=list, description="Non-numeric values found in otherwise numeric columns.")
    malformed_target_values: list[str] = Field(default_factory=list, description="Non-parseable values in the target column.")
    train_test_schema_mismatches: list[str] = Field(default_factory=list, description="Schema differences between train and test CSVs.")
    constant_columns: list[str] = Field(default_factory=list, description="Columns with only one unique value.")
    duplicate_columns: list[str] = Field(default_factory=list, description="Suspected duplicate column names.")
    candidate_targets: list[str] = Field(default_factory=list, description="Candidate target columns identified by heuristics.")


class TaskContext(BaseModel):
    objective: str | None = Field(default=None, description="Main objective of the user instruction.")
    task_type: str | None = Field(default=None, description="Type of ML task (e.g., 'classification', 'regression', 'time_series_forecasting', 'retrieval').")
    target_column: str | None = Field(default=None, description="Explicitly identified target label column.")
    timestamp_column: str | None = Field(default=None, description="Timestamp/date column for time-series forecasting.")
    id_column: str | None = Field(default=None, description="Item/entity ID column for time-series forecasting.")
    input_data_files: list[str] | None = Field(default=None, description="List of primary input file paths.")
    output_requirements: str | None = Field(default=None, description="Requirements for the output.")
    evaluation_metric: str | None = Field(default=None, description="Evaluation metric if explicitly stated.")
    constraints: list[str] | None = Field(default=None, description="Any explicit constraints.")
    relevant_description_files: list[str] | None = Field(default=None, description="Files containing relevant descriptions.")
    explanation: str | None = Field(default=None, description="Explanation for context extraction.")
    data_quality: DataQualityReport | None = Field(default=None, description="Deterministic data quality profiling results.")


class LibrarySelection(BaseModel):
    selected_library: str = Field(..., description="The name of the selected library.")
    confidence: str | None = Field(default=None, description="Confidence in the selection.")
    explanation: str = Field(default="Selected library", description="Explanation of why this library was chosen.")


class PerceptualContext(BaseModel):
    files: list[FileContext] = Field(default_factory=list, description="Context for each inspected file.")
    file_groups: list[FileGroup] = Field(default_factory=list, description="Structured file groupings.")
    modalities: list[str] = Field(default_factory=list, description="List of detected data modalities.")
    train_files: list[str] = Field(default_factory=list, description="Files identified as training data.")
    test_files: list[str] = Field(default_factory=list, description="Files identified as test/validation data.")
    label_files: list[str] = Field(default_factory=list, description="Files identified as label/annotation files.")
    metadata_files: list[str] = Field(default_factory=list, description="Files identified as metadata/description files.")
    task: TaskContext | None = Field(default=None, description="Inferred task context.")
    task_type: str | None = Field(default=None, description="Top-level inferred task type.")
    target_column: str | None = Field(default=None, description="Selected target column name.")
    timestamp_column: str | None = Field(default=None, description="Timestamp column for time-series forecasting.")
    id_column: str | None = Field(default=None, description="Item/entity ID column for time-series forecasting.")
    candidate_targets: list[str] = Field(default_factory=list, description="Candidate target column names.")
    library: LibrarySelection | None = Field(default=None, description="Selected library object.")
    selected_library: str | None = Field(default=None, description="Selected ML library identifier.")
    library_reason: str | None = Field(default=None, description="Reason for library selection.")
    data_quality: DataQualityReport | None = Field(default=None, description="Deterministic data quality profiling results.")
    quality_issues: list[str] = Field(default_factory=list, description="Summary of data quality findings/issues.")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal perception warnings.")
    confidence: float = Field(default=1.0, description="Overall perception confidence score.")
    errors: list[str] = Field(default_factory=list, description="Any top-level perception errors.")
