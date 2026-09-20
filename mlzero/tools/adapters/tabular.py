import shutil
from pathlib import Path
from typing import Any

from mlzero.core.logger import setup_logger
from mlzero.schemas.perception import PerceptualContext
from mlzero.tools.adapters.base import MLLibraryAdapter

logger = setup_logger(__name__)


class TabularAdapter(MLLibraryAdapter):
    """Execution adapter for autogluon.tabular."""

    def supports(self, perceptual_context: PerceptualContext) -> bool:
        if perceptual_context.library and perceptual_context.library.selected_library:
            return perceptual_context.library.selected_library == "autogluon.tabular"
        return bool(perceptual_context.task and perceptual_context.task.task_type in ("classification", "regression"))

    def _find_file(self, directory: Path, stem_candidates: list[str]) -> Path | None:
        """Find a file like train.csv, train.tsv, etc."""
        if not directory.exists():
            return None
        for file in directory.iterdir():
            if file.is_file() and file.stem.lower() in stem_candidates and file.suffix.lower() in ['.csv', '.tsv']:
                return file
        return None

    def validate_inputs(self, raw_input_dir: Path, perceptual_context: PerceptualContext) -> bool:
        train_file = self._find_file(raw_input_dir, ['train', 'data', 'dataset'])
        return train_file is not None

    def prepare_data(self, raw_input_dir: Path, workspace_dir: Path, perceptual_context: PerceptualContext) -> dict[str, Any]:
        train_file = self._find_file(raw_input_dir, ['train', 'data', 'dataset'])
        test_file = self._find_file(raw_input_dir, ['test', 'validation', 'val'])
        
        if not train_file:
            raise ValueError(f"No valid tabular training data found in {raw_input_dir}")
            
        workspace_train = workspace_dir / train_file.name
        if train_file.absolute() != workspace_train.absolute():
            shutil.copy2(train_file, workspace_train)
        
        result = {
            "train_path": str(workspace_train.absolute()),
            "test_path": None,
            "format": train_file.suffix.lower().lstrip('.')
        }
        
        if test_file:
            workspace_test = workspace_dir / test_file.name
            if test_file.absolute() != workspace_test.absolute():
                shutil.copy2(test_file, workspace_test)
            result["test_path"] = str(workspace_test.absolute())
            
        return result

    def build_code_context(self, perceptual_context: PerceptualContext, prepared_data: dict[str, Any]) -> str:
        ctx = "Library Context (autogluon.tabular):\n"
        ctx += "- Use `from autogluon.tabular import TabularPredictor`.\n"
        ctx += "- Clean invalid/malformed numeric values before fitting if mentioned in perceptual context.\n"
        ctx += "- Make sure to specify the exact label correctly.\n"
        return ctx

    def expected_outputs(self) -> list[str]:
        return ["out/models", "out/predictions.csv", "out/summary.json"]

    def validate_result(self, workspace_dir: Path) -> dict[str, Any]:
        metrics = None
        summary_file = workspace_dir / "summary.json"
        
        if summary_file.exists():
            import json
            try:
                with open(summary_file) as f:
                    summary_data = json.load(f)
                    metrics = summary_data.get("metrics")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Could not load summary.json: {e}")
                
        return {
            "prediction_exists": (workspace_dir / "predictions.csv").exists(),
            "model_exists": (workspace_dir / "models").exists(),
            "metrics": metrics
        }
