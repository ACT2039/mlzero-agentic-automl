"""
Iterative Orchestrator for MLZero.
Manages the retry loop between Coder, Executor, and Error Analyzer.
"""
import time
from typing import TYPE_CHECKING

from mlzero.agents.coder import CoderAgent, ErrorAnalyzerAgent, ExecutorAgent
from mlzero.core.config import settings
from mlzero.core.logger import setup_logger
from mlzero.memory.semantic import SemanticMemory
from mlzero.schemas.coder import CodeGenerationRequest
from mlzero.schemas.orchestration import IterationRecord, IterativeRunResult
from mlzero.schemas.perception import PerceptualContext

if TYPE_CHECKING:
    from mlzero.memory.episodic import EpisodicMemory
logger = setup_logger(__name__)


class IterativeCodingOrchestrator:
    """Orchestrates the iterative coding and execution loop."""

    def __init__(
        self,
        coder: CoderAgent,
        executor: ExecutorAgent,
        error_analyzer: ErrorAnalyzerAgent,
        max_iterations: int | None = None,
        semantic_memory: "SemanticMemory | None" = None,
        episodic_memory: "EpisodicMemory | None" = None
    ):
        self.coder = coder
        self.executor = executor
        self.error_analyzer = error_analyzer
        self.max_iterations = max_iterations or settings.limits.max_iterations
        self.semantic_memory = semantic_memory
        self.episodic_memory = episodic_memory

    def process(self, perceptual_context: PerceptualContext, user_instruction: str | None = None, run_id: str | None = None) -> IterativeRunResult:
        """Run the iterative loop."""
        import uuid
        run_id = run_id or str(uuid.uuid4())
        
        if self.episodic_memory:
            self.episodic_memory.start_run(run_id, perceptual_context)
            
        start_time = time.time()
        iteration_history = []
        
        perceptual_context_json = perceptual_context.model_dump_json()
        
        # Initial knowledge retrieval
        retrieved_knowledge_json = None
        if self.semantic_memory:
            try:
                knowledge = self.semantic_memory.retrieve(perceptual_context)
                retrieved_knowledge_json = knowledge.model_dump_json() if knowledge.chunks else None
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Initial semantic memory retrieval failed: {e}")

        # Initial request
        current_request = CodeGenerationRequest(
            perceptual_context_json=perceptual_context_json,
            user_instruction=user_instruction,
            retrieved_knowledge_json=retrieved_knowledge_json
        )
        
        success = False
        final_artifact = None
        final_result = None
        final_error = None
        
        for i in range(1, self.max_iterations + 1):
            logger.info(f"--- Starting Iteration {i}/{self.max_iterations} ---")
            iteration_start = time.time()
            
            # 1. Generate code
            try:
                artifact = self.coder.process(current_request)
            except Exception as e:  # noqa: BLE001  # noqa: BLE001
                logger.error(f"Coder failed at iteration {i}: {e}")
                # We can't continue if coder crashes
                break

            final_artifact = artifact

            # 2. Execute code
            try:
                result = self.executor.process(artifact)
            except Exception as e:  # noqa: BLE001  # noqa: BLE001
                logger.error(f"Executor crashed at iteration {i}: {e}")
                break
                
            final_result = result
            
            # 3. Analyze success/failure
            if result.success:
                logger.info(f"Iteration {i} succeeded!")
                duration = time.time() - iteration_start
                iteration_history.append(
                    IterationRecord(
                        iteration_number=i,
                        code_artifact=artifact,
                        execution_result=result,
                        duration_seconds=duration
                    )
                )
                
                if self.episodic_memory:
                    self.episodic_memory.record_iteration(
                        run_id=run_id,
                        iteration=i,
                        status="SUCCESS",
                        artifact=artifact,
                        result=result,
                        retrieved_knowledge=knowledge if 'knowledge' in locals() else None,
                        summary="Execution succeeded."
                    )
                
                success = True
                break
            else:
                logger.warning(f"Iteration {i} failed. stderr: {result.stderr[:200] if result else 'none'}")
                
                # Analyze error
                try:
                    error_ctx = self.error_analyzer.process(artifact, result, iteration=i)
                except Exception as e:  # noqa: BLE001  # noqa: BLE001
                    logger.error(f"ErrorAnalyzer crashed at iteration {i}: {e}")
                    break
                    
                final_error = error_ctx
                
                duration = time.time() - iteration_start
                iteration_history.append(
                    IterationRecord(
                        iteration_number=i,
                        code_artifact=artifact,
                        execution_result=result,
                        error_context=error_ctx,
                        duration_seconds=duration
                    )
                )
                
                # Error-aware knowledge retrieval
                if self.semantic_memory:
                    try:
                        knowledge = self.semantic_memory.retrieve(perceptual_context, error_ctx)
                        retrieved_knowledge_json = knowledge.model_dump_json() if knowledge.chunks else None
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"Error-aware semantic memory retrieval failed: {e}")
                
                if self.episodic_memory:
                    self.episodic_memory.record_iteration(
                        run_id=run_id,
                        iteration=i,
                        status="FAIL",
                        artifact=artifact,
                        result=result,
                        error_ctx=error_ctx,
                        retrieved_knowledge=knowledge if 'knowledge' in locals() else None,
                        summary="Execution failed."
                    )
                    
                    try:
                        import json
                        episodic_context = self.episodic_memory.get_bounded_context(run_id)
                        episodic_context_json = json.dumps(episodic_context)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"Failed to get episodic context: {e}")
                        episodic_context_json = None
                else:
                    episodic_context_json = None
                
                # Prepare for next iteration
                current_request = CodeGenerationRequest(
                    perceptual_context_json=perceptual_context_json,
                    user_instruction=user_instruction,
                    previous_code=artifact.code,
                    error_context_json=error_ctx.model_dump_json(),
                    retrieved_knowledge_json=retrieved_knowledge_json,
                    episodic_context_json=episodic_context_json
                )
                
        total_duration = time.time() - start_time
        if self.episodic_memory:
            self.episodic_memory.end_run(run_id, "SUCCESS" if success else "FAIL")
            
        return IterativeRunResult(
            run_id=run_id,
            success=success,
            total_iterations=len(iteration_history),
            final_code_artifact=final_artifact,
            final_execution_result=final_result,
            iteration_history=iteration_history,
            final_error_context=final_error,
            total_duration_seconds=total_duration
        )
