import shutil
from pathlib import Path
from typing import Any

from mlzero.core.logger import setup_logger
from mlzero.schemas.perception import PerceptualContext
from mlzero.tools.adapters.base import MLLibraryAdapter
from mlzero.utils.metrics import canonicalize_metrics

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
        # 1. Exact match
        for file in directory.iterdir():
            if file.is_file() and file.stem.lower() in stem_candidates and file.suffix.lower() in ['.csv', '.tsv']:
                return file
        # 2. Substring match
        for file in directory.iterdir():
            if file.is_file() and any(c in file.stem.lower() for c in stem_candidates) and file.suffix.lower() in ['.csv', '.tsv']:
                return file
        return None

    def validate_inputs(self, raw_input_dir: Path, perceptual_context: PerceptualContext) -> bool:
        train_file = self._find_file(raw_input_dir, ['train', 'data', 'dataset'])
        if train_file is not None:
            return True
        if raw_input_dir.exists() and raw_input_dir.is_dir():
            candidates = [
                f for f in raw_input_dir.iterdir()
                if f.is_file() and f.suffix.lower() in ['.csv', '.tsv']
            ]
            non_test = [f for f in candidates if not any(k in f.stem.lower() for k in ('test', 'val', 'validation'))]
            return len(non_test) > 0 or len(candidates) > 0
        return False

    def prepare_data(self, raw_input_dir: Path, workspace_dir: Path, perceptual_context: PerceptualContext) -> dict[str, Any]:
        train_file = self._find_file(raw_input_dir, ['train', 'data', 'dataset'])
        test_file = self._find_file(raw_input_dir, ['test', 'validation', 'val'])

        if not train_file and raw_input_dir.exists() and raw_input_dir.is_dir():
            candidates = [
                f for f in raw_input_dir.iterdir()
                if f.is_file() and f.suffix.lower() in ['.csv', '.tsv'] and f != test_file
            ]
            non_test = [f for f in candidates if not any(k in f.stem.lower() for k in ('test', 'val', 'validation'))]
            if non_test:
                train_file = non_test[0]
            elif candidates:
                train_file = candidates[0]
        
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
        ctx += "- Clean invalid/malformed numeric values and impute missing values (mean/median for numeric, mode for categorical) before fitting.\n"
        ctx += "- CRITICAL: Avoid pandas chained assignment warnings; use `df[col] = df[col].fillna(val)` instead of `inplace=True`.\n"
        ctx += "- Make sure to specify the exact label correctly.\n"
        ctx += "- CRITICAL: Set save path in `TabularPredictor(label=label, path='out/models')`. Do NOT call `predictor.save('out/models')`!\n"
        ctx += "- CRITICAL: If specifying 'eval_metric', pass it ONLY to TabularPredictor constructor: `TabularPredictor(label=label, eval_metric=..., path='out/models')`. NEVER pass 'eval_metric' to `predictor.fit(...)` as it will raise ValueError!\n"
        ctx += "- Fit with: `predictor.fit(train_data, presets='medium_quality', time_limit=60, excluded_model_types=['NN_TORCH', 'FASTAI'])`\n"
        ctx += "- CRITICAL LOW-MEMORY REQUIREMENT: To prevent Out-Of-Memory (OOM) crashes in cloud environments (Render 512MB RAM), always exclude heavy neural networks by passing `excluded_model_types=['NN_TORCH', 'FASTAI']` into `predictor.fit(...)`. If `len(train_data) > 2500`, sample: `train_data = train_data.sample(n=2500, random_state=42)`.\n"
        ctx += "- CRITICAL: Always use forward slashes (/) for all file paths (e.g. 'out/models', 'out/predictions.csv'). Never use Windows backslashes.\n"
        ctx += "- Always output predictions to `out/predictions.csv`.\n"
        ctx += "- Always save comprehensive evaluation metrics to `out/summary.json` (e.g. for classification include `accuracy`, `balanced_accuracy`, `f1`, `precision`, `recall`, `mcc`; for regression include `rmse`, `mae`, `r2`, `mse`).\n"
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
