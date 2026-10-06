import shutil
from pathlib import Path
from typing import Any

from mlzero.core.logger import setup_logger
from mlzero.schemas.perception import PerceptualContext
from mlzero.tools.adapters.base import MLLibraryAdapter
from mlzero.utils.metrics import canonicalize_metrics

logger = setup_logger(__name__)


class MultiModalAdapter(MLLibraryAdapter):
    """Execution adapter for autogluon.multimodal."""

    def supports(self, perceptual_context: PerceptualContext) -> bool:
        if perceptual_context.library and perceptual_context.library.selected_library:
            return perceptual_context.library.selected_library == "autogluon.multimodal"
        return bool(perceptual_context.task and perceptual_context.task.task_type in ("multimodal", "image_classification", "text_classification"))

    def validate_inputs(self, raw_input_dir: Path, perceptual_context: PerceptualContext) -> bool:
        return raw_input_dir.exists()

    def prepare_data(self, raw_input_dir: Path, workspace_dir: Path, perceptual_context: PerceptualContext) -> dict[str, Any]:
        # For multimodal, copy the whole directory structure up to a certain depth if needed,
        # or just point back to raw_input_dir for images to avoid duplicating large files.
        # But we don't support huge files right now, so we copy everything.
        if raw_input_dir != workspace_dir:
            shutil.copytree(raw_input_dir, workspace_dir, dirs_exist_ok=True)
        return {"data_path": str(workspace_dir.absolute())}

    def build_code_context(self, perceptual_context: PerceptualContext, prepared_data: dict[str, Any]) -> str:
        ctx = "Library Context (autogluon.multimodal):\n"
        ctx += "- Use `from autogluon.multimodal import MultiModalPredictor`.\n"
        ctx += "- Instantiate with: `predictor = MultiModalPredictor(label=target_col)`\n"
        ctx += "- Keep unit tests small. Only generate a tiny model, e.g. using lightweight presets if asked.\n"
        return ctx

    def expected_outputs(self) -> list[str]:
        return ["out/models", "out/predictions.csv", "out/summary.json"]

    def validate_result(self, workspace_dir: Path) -> dict[str, Any]:
        metrics = None
        raw_metrics = None
        summary_file = workspace_dir / "summary.json"
        
        if summary_file.exists():
            import json
            try:
                with open(summary_file) as f:
                    summary_data = json.load(f)
                    raw_metrics = summary_data.get("raw_metrics") or summary_data.get("metrics")
                    metrics = summary_data.get("metrics")
                    if isinstance(metrics, dict):
                        metrics = canonicalize_metrics(metrics)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Could not load summary.json: {e}")
                
        return {
            "prediction_exists": (workspace_dir / "predictions.csv").exists(),
            "model_exists": (workspace_dir / "models").exists(),
            "metrics": metrics,
            "raw_metrics": raw_metrics,
        }
