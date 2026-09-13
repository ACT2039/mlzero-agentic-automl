import json
import logging
from pathlib import Path
from typing import Any

from mlzero.agents.coder import CoderAgent, ErrorAnalyzerAgent, ExecutorAgent
from mlzero.agents.perception import (
    FilePerceptionAgent,
    LibrarySelectorAgent,
    TaskPerceptionAgent,
)
from mlzero.core.config import settings
from mlzero.core.llm import get_llm_client
from mlzero.memory.episodic import EpisodicMemory
from mlzero.memory.semantic import SemanticMemory
from mlzero.orchestration.iterative import IterativeCodingOrchestrator
from mlzero.schemas.application import RunStatusResponse
from mlzero.schemas.perception import PerceptualContext
from mlzero.tools.tabular import TabularDatasetAdapter

logger = logging.getLogger(__name__)

class MLZeroService:
    def __init__(self, use_mock_llm: bool = False):
        self.use_mock_llm = use_mock_llm
        self.llm_client = get_llm_client(use_mock=use_mock_llm)
        self.semantic_memory = SemanticMemory(llm_client=self.llm_client)
        self.episodic_memory = EpisodicMemory()

    def run_mlzero(self, dataset_path: str, user_instruction: str | None = None, options: dict[str, Any] | None = None) -> RunStatusResponse:
        """
        Main application service boundary for MLZero execution.
        """
        options = options or {}
        run_id = options.get("run_id", "unknown-run")
        input_path = Path(dataset_path).resolve()
        
        # 1. Validation
        allowed_root = Path(settings.app.allowed_data_root).resolve()
        try:
            input_path.relative_to(allowed_root)
        except ValueError:
            raise ValueError(f"Dataset path {input_path} is outside allowed data root {allowed_root}")

        if not input_path.exists() or not input_path.is_dir():
            raise ValueError(f"Invalid dataset path: {input_path}")

        # 2. Perception & Adapter
        try:
            data_workspace = Path(settings.ml.output_dir) / "data"
            data_workspace.mkdir(parents=True, exist_ok=True)
            adapter = TabularDatasetAdapter(input_path, data_workspace)
            data_info = adapter.prepare_data()
            
            file_agent = FilePerceptionAgent(dataset_dir=adapter.workspace_dir)
            task_agent = TaskPerceptionAgent(llm_client=self.llm_client)
            library_agent = LibrarySelectorAgent(llm_client=self.llm_client)
            
            fctx = file_agent.process()
            tctx = task_agent.process(fctx, user_instruction=user_instruction)
            lib_ctx = library_agent.process(tctx, fctx)
            
            pctx = PerceptualContext(files=fctx, task=tctx, library=lib_ctx)
            if pctx.task:
                pctx.task.input_data_files = [data_info["train_path"]]
                if data_info.get("test_path"):
                    pctx.task.input_data_files.append(data_info["test_path"])
        except Exception as e:
            logger.error(f"Perception phase failed: {e}")
            raise

        # 3. Execution (Iterative Retry Loop)
        coder_agent = CoderAgent(llm_client=self.llm_client)
        executor_agent = ExecutorAgent()
        error_analyzer = ErrorAnalyzerAgent(llm_client=self.llm_client)
        
        orchestrator = IterativeCodingOrchestrator(
            coder=coder_agent,
            executor=executor_agent,
            error_analyzer=error_analyzer,
            semantic_memory=self.semantic_memory,
            episodic_memory=self.episodic_memory
        )
        
        result = orchestrator.process(
            perceptual_context=pctx,
            user_instruction=user_instruction,
            run_id=run_id
        )
        
        # 4. Result Parsing
        metrics = None
        prediction_ref = None
        model_ref = None
        
        if result.success and result.final_execution_result and result.final_execution_result.workspace_dir:
            workspace = Path(result.final_execution_result.workspace_dir)
            summary_file = workspace / "summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file) as f:
                        summary_data = json.load(f)
                        metrics = summary_data.get("metrics")
                except (OSError, json.JSONDecodeError):
                    pass
                    
            model_dir = workspace / "models"
            if model_dir.exists():
                model_ref = str(model_dir.relative_to(Path(settings.ml.output_dir)))
                
            pred_file = workspace / settings.ml.prediction_filename
            if pred_file.exists():
                prediction_ref = str(pred_file.relative_to(Path(settings.ml.output_dir)))

        status = "SUCCESS" if result.success else "FAIL"
        
        task_summary = pctx.task.model_dump() if pctx.task else None
        lib_sel = pctx.library.selected_library if pctx.library else None
        
        final_err = None
        if not result.success and result.final_error_context:
            final_err = result.final_error_context.error_category
            
        dur = result.final_execution_result.duration_seconds if result.final_execution_result else None
        
        return RunStatusResponse(
            run_id=run_id,
            status=status,
            success=result.success,
            task_summary=task_summary,
            selected_library=lib_sel,
            iterations=result.total_iterations,
            final_metrics=metrics,
            prediction_artifact_reference=prediction_ref,
            model_artifact_reference=model_ref,
            execution_duration=dur,
            final_error=final_err
        )
