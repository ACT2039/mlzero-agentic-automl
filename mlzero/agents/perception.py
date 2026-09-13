"""
Perception Agents for MLZero.
Includes interfaces and implementations for File Perception, Task Perception, and ML Library Selection.
"""

import csv
import json
import os
from pathlib import Path
from typing import Any, ClassVar

from mlzero.core.config import settings
from mlzero.core.llm import LLMClient
from mlzero.core.logger import setup_logger
from mlzero.schemas.perception import (
    FileContext,
    FileMetadata,
    LibrarySelection,
    TaskContext,
)

logger = setup_logger(__name__)


class FilePerceptionAgent:
    """Agent responsible for perceiving and understanding files/datasets."""

    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir)

    def process(self) -> list[FileContext]:
        """Analyze the dataset directory and return file contexts."""
        if not self.dataset_dir.exists() or not self.dataset_dir.is_dir():
            raise ValueError(f"Input directory does not exist or is not a directory: {self.dataset_dir}")

        file_contexts = []
        ignored_dirs = set(settings.perception.ignored_directories)
        max_size = settings.perception.max_file_size_bytes
        max_chars = settings.perception.max_text_characters
        max_rows = settings.perception.max_rows_sampled

        for root, dirs, files in os.walk(self.dataset_dir):
            # Prune ignored directories
            dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith(".")]

            for file_name in files:
                file_path = Path(root) / file_name
                rel_path = file_path.relative_to(self.dataset_dir).as_posix()
                ext = file_path.suffix.lower()

                try:
                    size = file_path.stat().st_size
                except OSError as e:
                    logger.warning(f"Could not stat {file_path}: {e}")
                    continue

                if ext in [".csv", ".tsv"]:
                    file_type = "tabular"
                elif ext in [".txt", ".md", ".json", ".jsonl"]:
                    file_type = "document"
                else:
                    file_type = "binary_or_unsupported"

                metadata = FileMetadata(
                    path=rel_path,
                    absolute_path=str(file_path.absolute()),
                    extension=ext,
                    size_bytes=size,
                    file_type=file_type,
                )

                if size > max_size:
                    file_contexts.append(
                        FileContext(
                            metadata=metadata,
                            error=f"File exceeds max size limit of {max_size} bytes.",
                        )
                    )
                    continue

                if file_type == "tabular":
                    ctx = self._inspect_tabular(file_path, metadata, max_rows)
                elif file_type == "document":
                    ctx = self._inspect_document(file_path, metadata, max_chars)
                else:
                    ctx = FileContext(metadata=metadata)

                file_contexts.append(ctx)

        return file_contexts

    def _inspect_tabular(self, file_path: Path, metadata: FileMetadata, max_rows: int) -> FileContext:
        """Safely inspect a tabular file."""
        delimiter = "\t" if metadata.extension == ".tsv" else ","
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=delimiter)
                headers = next(reader, None)
                if not headers:
                    return FileContext(metadata=metadata, error="Empty tabular file.")

                sample_rows: list[dict[str, Any]] = []
                row_count = 1  # We read the header
                for row in reader:
                    row_count += 1
                    if len(sample_rows) < max_rows:
                        # Zip headers with row
                        row_dict = {
                            headers[i] if i < len(headers) else f"col_{i}": val
                            for i, val in enumerate(row)
                        }
                        sample_rows.append(row_dict)
                
                # In a real system, infer types based on sample
                inferred = {col: "unknown" for col in headers}

                return FileContext(
                    metadata=metadata,
                    columns=headers,
                    row_count=row_count,
                    sample_rows=sample_rows,
                    inferred_types=inferred,
                )
        except (OSError, ValueError, UnicodeDecodeError, csv.Error) as e:
            return FileContext(metadata=metadata, error=f"Error reading tabular file: {e!s}")

    def _inspect_document(self, file_path: Path, metadata: FileMetadata, max_chars: int) -> FileContext:
        """Safely inspect a document file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read(max_chars)
                if len(content) == max_chars:
                    content += "\n...[TRUNCATED]"
            return FileContext(metadata=metadata, extracted_text=content)
        except (OSError, ValueError, UnicodeDecodeError) as e:
            return FileContext(metadata=metadata, error=f"Error reading document file: {e!s}")


class TaskPerceptionAgent:
    """Agent responsible for understanding the overall user task."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def process(self, file_contexts: list[FileContext], user_instruction: str | None = None) -> TaskContext:
        """Analyze the task description and file contexts using LLM."""
        
        prompt = "Extract task information.\n"
        if user_instruction:
            prompt += f"Instruction: {user_instruction}\n"
        prompt += f"Files available: {len(file_contexts)}\n"
        # In a real system, serialize the file contexts into the prompt

        return self.llm_client.generate_structured(prompt, TaskContext)


class LibrarySelectorAgent:
    """Agent responsible for selecting appropriate ML libraries."""

    REGISTRY: ClassVar[list[dict[str, Any]]] = [
        {
            "name": "autogluon.tabular",
            "version": "placeholder",
            "description": "AutoML for tabular data (CSV, Parquet) classification and regression.",
            "modalities": ["tabular"],
            "task_types": ["classification", "regression"],
            "limitations": "Requires tabular data formats."
        },
        {
            "name": "autogluon.multimodal",
            "version": "placeholder",
            "description": "AutoML for images, text, and mixed multimodal data.",
            "modalities": ["image", "text", "multimodal"],
            "task_types": ["classification", "regression", "object_detection"],
            "limitations": "Heavy memory usage, requires GPU for optimal performance."
        },
        {
            "name": "autogluon.timeseries",
            "version": "placeholder",
            "description": "AutoML for time series forecasting.",
            "modalities": ["timeseries"],
            "task_types": ["forecasting"],
            "limitations": "Requires timestamps and unique item IDs."
        },
        {
            "name": "general_ml",
            "version": "placeholder",
            "description": "General machine learning using standard libraries (scikit-learn, etc.).",
            "modalities": ["any"],
            "task_types": ["any"],
            "limitations": "Requires manual pipeline building."
        }
    ]

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def process(self, task_context: TaskContext, file_contexts: list[FileContext]) -> LibrarySelection:
        """Select ML library based on the perceived task and data."""
        
        registry_str = json.dumps(self.REGISTRY, indent=2)
        prompt = f"Select library from registry:\n{registry_str}\n"
        prompt += f"Task type: {task_context.task_type}\n"
        
        return self.llm_client.generate_structured(prompt, LibrarySelection)
