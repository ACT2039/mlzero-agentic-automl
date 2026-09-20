import json
import logging
from pathlib import Path
from typing import Any

from mlzero.agents.coder import (
    CoderAgent,
    ErrorAnalyzerAgent,
    ExecutionJudgeAgent,
    ExecutorAgent,
)
from mlzero.agents.perception import (
    DataProfiler,
    FileGroupingAgent,
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
from mlzero.tools.adapters.registry import AdapterRegistry

logger = logging.getLogger(__name__)

class MLZeroService:
    def __init__(self, use_mock_llm: bool | None = None, llm_mode: str | None = None):
        if llm_mode is not None:
            self.llm_mode = llm_mode.lower()
            self.use_mock_llm = (self.llm_mode == "mock")
        elif use_mock_llm is not None:
            self.use_mock_llm = use_mock_llm
            self.llm_mode = "mock" if use_mock_llm else "real"
        else:
            self.llm_mode = getattr(settings, "llm_mode", "mock").lower()
            self.use_mock_llm = (self.llm_mode == "mock")

        self.llm_client = get_llm_client(mode=self.llm_mode)
        self.semantic_memory = SemanticMemory(llm_client=self.llm_client)
        self.episodic_memory = EpisodicMemory()

    def perceive(self, input_path: Path | str, user_instruction: str | None = None) -> PerceptualContext:
        """Run perception phase and generate PerceptualContext."""
        path_obj = Path(input_path).resolve()
        profiler = DataProfiler(path_obj)
        data_quality = profiler.profile(user_instruction=user_instruction)

        file_agent = FilePerceptionAgent(dataset_dir=path_obj)
        fctx = file_agent.process()

        grouping_agent = FileGroupingAgent(dataset_dir=path_obj)
        fgroups = grouping_agent.process(fctx)

        task_agent = TaskPerceptionAgent(llm_client=self.llm_client)
        library_agent = LibrarySelectorAgent(llm_client=self.llm_client)

        tctx = task_agent.process(fctx, user_instruction=user_instruction, data_quality=data_quality, file_groups=fgroups)
        lib_ctx = library_agent.process(tctx, fctx)

        detected_modalities = list(dict.fromkeys(fc.metadata.file_type for fc in fctx))
        train_files = [fc.metadata.path for fc in fctx if "train" in Path(fc.metadata.path).stem.lower()]
        test_files = [fc.metadata.path for fc in fctx if any(k in Path(fc.metadata.path).stem.lower() for k in ("test", "val"))]

        return PerceptualContext(
            files=fctx,
            file_groups=fgroups,
            modalities=detected_modalities,
            train_files=train_files,
            test_files=test_files,
            task=tctx,
            task_type=tctx.task_type if tctx else None,
            target_column=tctx.target_column if tctx else None,
            timestamp_column=tctx.timestamp_column if tctx else None,
            id_column=tctx.id_column if tctx else None,
            candidate_targets=data_quality.candidate_targets if data_quality else [],
            library=lib_ctx,
            selected_library=lib_ctx.selected_library if lib_ctx else None,
            library_reason=lib_ctx.explanation if lib_ctx else None,
            data_quality=data_quality,
        )

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
            
            pctx = self.perceive(input_path, user_instruction=user_instruction)
            
            # Retrieve Adapter
            adapter = AdapterRegistry.get_adapter(pctx)
            if not adapter:
                lib_name = pctx.library.selected_library if pctx.library else "Unknown"
                raise ValueError(f"No adapter found for the selected library: {lib_name}")
                
            if not adapter.validate_inputs(input_path, pctx):
                raise ValueError("Input validation failed for the selected adapter.")
                
            data_info = adapter.prepare_data(input_path, data_workspace, pctx)
            
            if pctx.task:
                pctx.task.input_data_files = [str(v) for k, v in data_info.items() if v and isinstance(v, (str, Path))]

        except Exception as e:
            logger.error(f"Perception phase failed: {e}")
            raise

        # 3. Execution (Iterative Retry Loop)
        coder_agent = CoderAgent(llm_client=self.llm_client)
        executor_agent = ExecutorAgent()
        error_analyzer = ErrorAnalyzerAgent(llm_client=self.llm_client)
        judge_agent = ExecutionJudgeAgent(llm_client=self.llm_client)
        
        disable_j = options.get("disable_judge", False) if options else False
        ret_k = options.get("retrieval_k", 5) if options else 5

        orchestrator = IterativeCodingOrchestrator(
            coder=coder_agent,
            executor=executor_agent,
            error_analyzer=error_analyzer,
            judge=judge_agent,
            semantic_memory=self.semantic_memory,
            episodic_memory=self.episodic_memory,
            disable_judge=disable_j,
            retrieval_k=ret_k,
        )
        
        result = orchestrator.process(
            perceptual_context=pctx,
            user_instruction=user_instruction,
            run_id=run_id,
            adapter=adapter
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
        
        j_decisions = getattr(result, "judge_decisions", [])
        p_trace = getattr(result, "pipeline_trace", {})
        backend_type = "mock" if self.use_mock_llm else "real"
        adapter_meta = {
            "selected_adapter": adapter.__class__.__name__,
            "validation_passed": True,
            "selected_library": lib_sel,
            "execution_backend": backend_type,
        }

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
            final_error=final_err,
            judge_decisions=j_decisions,
            pipeline_trace=p_trace,
            execution_backend=backend_type,
            adapter_info=adapter_meta,
        )
