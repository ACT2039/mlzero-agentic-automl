"""
Tabular dataset adapter and validation tool.
"""
import shutil
from pathlib import Path
from typing import Any

from mlzero.core.logger import setup_logger

logger = setup_logger(__name__)


class TabularDatasetAdapter:
    """Validates and prepares tabular data for ML execution."""
    
    def __init__(self, raw_input_dir: str | Path, workspace_dir: str | Path) -> None:
        self.raw_input_dir = Path(raw_input_dir)
        self.workspace_dir = Path(workspace_dir)
        
    def _find_file(self, stem_candidates: list[str]) -> Path | None:
        """Find a file like train.csv, train.tsv, etc."""
        if not self.raw_input_dir.exists():
            return None
            
        for file in self.raw_input_dir.iterdir():
            if file.is_file() and file.stem.lower() in stem_candidates and file.suffix.lower() in ['.csv', '.tsv']:
                return file
        return None
        
    def prepare_data(self) -> dict[str, Any]:
        """
        Identify and copy raw train/test files into the controlled workspace.
        Return paths and validation info.
        """
        train_file = self._find_file(['train', 'data', 'dataset'])
        test_file = self._find_file(['test', 'validation', 'val'])
        
        if not train_file:
            raise ValueError(f"No valid tabular training data found in {self.raw_input_dir}")
            
        # Copy to workspace
        workspace_train = self.workspace_dir / train_file.name
        shutil.copy2(train_file, workspace_train)
        
        result = {
            "train_path": str(workspace_train.absolute()),
            "test_path": None,
            "format": train_file.suffix.lower().lstrip('.')
        }
        
        if test_file:
            workspace_test = self.workspace_dir / test_file.name
            shutil.copy2(test_file, workspace_test)
            result["test_path"] = str(workspace_test.absolute())
            
        return result
