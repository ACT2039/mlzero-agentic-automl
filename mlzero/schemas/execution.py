"""
Schemas for Execution and Judgment.
"""
from typing import Literal

from pydantic import BaseModel, Field


class ExecutionDecision(BaseModel):
    """Structured decision produced by the ExecutionJudgeAgent."""

    decision: Literal["FINISH", "FIX"] = Field(
        ...,
        description="Whether the automation loop should finish successfully or fix an error.",
    )
    reason: str = Field(
        ...,
        description="Detailed rationale for the decision.",
    )
    confidence: float | None = Field(
        default=None,
        description="Optional confidence score between 0.0 and 1.0.",
    )
    issue_summary: str | None = Field(
        default=None,
        description="Optional summary of the issue if the decision is FIX.",
    )

    @property
    def needs_fix(self) -> bool:
        """Helper property to check if a fix is required."""
        return self.decision == "FIX"
