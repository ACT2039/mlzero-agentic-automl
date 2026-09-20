import shutil
from pathlib import Path
from typing import Any

from mlzero.core.logger import setup_logger
from mlzero.schemas.perception import PerceptualContext
from mlzero.tools.adapters.base import MLLibraryAdapter

logger = setup_logger(__name__)


class RetrievalAdapter(MLLibraryAdapter):
    """Execution adapter for FlagEmbedding."""

    def supports(self, perceptual_context: PerceptualContext) -> bool:
        if perceptual_context.library and perceptual_context.library.selected_library:
            return perceptual_context.library.selected_library == "FlagEmbedding"
        return bool(perceptual_context.task and perceptual_context.task.task_type in ("retrieval", "reranking", "embedding"))

    def validate_inputs(self, raw_input_dir: Path, perceptual_context: PerceptualContext) -> bool:
        return raw_input_dir.exists()

    def prepare_data(self, raw_input_dir: Path, workspace_dir: Path, perceptual_context: PerceptualContext) -> dict[str, Any]:
        if raw_input_dir != workspace_dir:
            shutil.copytree(raw_input_dir, workspace_dir, dirs_exist_ok=True)
        return {"data_path": str(workspace_dir.absolute())}

    def build_code_context(self, perceptual_context: PerceptualContext, prepared_data: dict[str, Any]) -> str:
        ctx = "Library Context (FlagEmbedding):\n"
        ctx += "- Use `from FlagEmbedding import FlagModel`.\n"
        ctx += "- Encode queries and passages, then compute similarities (e.g. dot product or bi-encoder style).\n"
        ctx += "- Only use lightweight/mocked models if running locally (e.g., small dimension embeddings).\n"
        ctx += "- Save results to out/retrieval_results.json.\n"
        return ctx

    def expected_outputs(self) -> list[str]:
        return ["out/retrieval_results.json", "out/summary.json"]

    def validate_result(self, workspace_dir: Path) -> dict[str, Any]:
        return {
            "results_exist": (workspace_dir / "retrieval_results.json").exists()
        }
