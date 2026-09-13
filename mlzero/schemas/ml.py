"""
Machine Learning specific schemas and types.
"""
from typing import Any

from pydantic import BaseModel, Field


class MLRunResult(BaseModel):
    """Structured result of an ML execution pipeline."""
    success: bool = Field(..., description="Whether the ML execution ultimately succeeded.")
    task_type: str = Field(..., description="E.g., classification or regression.")
    selected_library: str = Field(..., description="Library used, e.g., autogluon.tabular.")
    
    model_path: str | None = Field(default=None, description="Path or reference to the trained model artifact.")
    prediction_path: str | None = Field(default=None, description="Path to the generated predictions file.")
    
    training_duration: float | None = Field(default=None, description="Time taken for training in seconds.")
    prediction_duration: float | None = Field(default=None, description="Time taken for prediction in seconds.")
    
    metrics: dict[str, Any] | None = Field(default=None, description="Collected evaluation metrics.")
    summary: str | None = Field(default=None, description="Brief summary of the run.")
    
    error_information: str | None = Field(default=None, description="Error details if the run failed.")
