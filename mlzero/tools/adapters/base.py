import abc
from pathlib import Path
from typing import Any

from mlzero.schemas.perception import PerceptualContext


class ExecutionResultMetadata:
    """Wrapper around adapter-specific validation/results."""


class MLLibraryAdapter(abc.ABC):
    """
    Common provider/adapter abstraction for ML execution.
    Architecture: Library Registry -> Selected Library -> Adapter -> Coder/Executor
    """

    @abc.abstractmethod
    def supports(self, perceptual_context: PerceptualContext) -> bool:
        """Check if this adapter supports the given perceptual context."""

    @abc.abstractmethod
    def validate_inputs(self, raw_input_dir: Path, perceptual_context: PerceptualContext) -> bool:
        """Validate that the necessary data/files are present."""

    @abc.abstractmethod
    def prepare_data(self, raw_input_dir: Path, workspace_dir: Path, perceptual_context: PerceptualContext) -> dict[str, Any]:
        """Prepare the workspace (e.g., copy files, format data)."""

    @abc.abstractmethod
    def build_code_context(self, perceptual_context: PerceptualContext, prepared_data: dict[str, Any]) -> str:
        """Provide library-specific hints/rules to inject into the Coder's prompt."""

    @abc.abstractmethod
    def expected_outputs(self) -> list[str]:
        """Return a list of logical outputs/artifacts expected from this adapter."""

    @abc.abstractmethod
    def validate_result(self, workspace_dir: Path) -> dict[str, Any]:
        """Validate that the execution produced the expected artifacts."""
