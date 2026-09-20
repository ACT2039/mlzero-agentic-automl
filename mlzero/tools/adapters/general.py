import shutil
from pathlib import Path
from typing import Any

from mlzero.core.logger import setup_logger
from mlzero.schemas.perception import PerceptualContext
from mlzero.tools.adapters.base import MLLibraryAdapter

logger = setup_logger(__name__)


class GeneralMLAdapter(MLLibraryAdapter):
    """Execution adapter for generic machine learning algorithms (e.g., scikit-learn)."""

    def supports(self, perceptual_context: PerceptualContext) -> bool:
        if perceptual_context.library and perceptual_context.library.selected_library:
            return perceptual_context.library.selected_library in ("machine learning", "general_ml")
        return True

    def validate_inputs(self, raw_input_dir: Path, perceptual_context: PerceptualContext) -> bool:
        return raw_input_dir.exists()

    def prepare_data(self, raw_input_dir: Path, workspace_dir: Path, perceptual_context: PerceptualContext) -> dict[str, Any]:
        if raw_input_dir != workspace_dir:
            shutil.copytree(raw_input_dir, workspace_dir, dirs_exist_ok=True)
        return {"data_path": str(workspace_dir.absolute())}

    def build_code_context(self, perceptual_context: PerceptualContext, prepared_data: dict[str, Any]) -> str:
        ctx = "Library Context (General ML):\n"
        ctx += "- Use standard libraries like scikit-learn, pandas, or numpy.\n"
        ctx += "- Build a generic ML pipeline depending on the task type (classification/regression/etc).\n"
        return ctx

    def expected_outputs(self) -> list[str]:
        return ["out/predictions.csv", "out/summary.json"]

    def validate_result(self, workspace_dir: Path) -> dict[str, Any]:
        return {
            "prediction_exists": (workspace_dir / "predictions.csv").exists()
        }
