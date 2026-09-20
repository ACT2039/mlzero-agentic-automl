"""
Iterative Orchestrator for MLZero.
Manages the retry loop between Coder, Executor, and Error Analyzer.
"""
import time
from typing import TYPE_CHECKING, Any

from mlzero.agents.coder import CoderAgent, ErrorAnalyzerAgent, ExecutorAgent
from mlzero.agents.judge import ExecutionJudgeAgent
from mlzero.core.config import settings
from mlzero.core.logger import setup_logger
from mlzero.memory.semantic import SemanticMemory
from mlzero.schemas.coder import CodeGenerationRequest
from mlzero.schemas.orchestration import IterationRecord, IterativeRunResult
from mlzero.schemas.perception import PerceptualContext
from mlzero.tools.adapters.base import MLLibraryAdapter

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
        episodic_memory: "EpisodicMemory | None" = None,
        judge: ExecutionJudgeAgent | None = None,
        disable_judge: bool = False,
        retrieval_k: int = 5,
    ):
        self.coder = coder
        self.executor = executor
        self.error_analyzer = error_analyzer
        self.max_iterations = max_iterations or settings.limits.max_iterations
        self.semantic_memory = semantic_memory
        self.episodic_memory = episodic_memory
        self.judge = judge or ExecutionJudgeAgent(llm_client=getattr(coder, "llm_client", None))
        self.disable_judge = disable_judge
        self.retrieval_k = retrieval_k

    def process(
        self,
        perceptual_context: PerceptualContext,
        user_instruction: str | None = None,
        run_id: str | None = None,
        adapter: "MLLibraryAdapter | None" = None,
    ) -> IterativeRunResult:
        """Run the iterative loop."""
        import uuid
        run_id = run_id or str(uuid.uuid4())
        
        if self.episodic_memory:
            self.episodic_memory.start_run(
                run_id, perceptual_context, user_instruction=user_instruction
            )
            
        start_time = time.time()
        iteration_history = []
        
        perceptual_context_json = perceptual_context.model_dump_json()
        selected_lib = (
            perceptual_context.library.selected_library
            if perceptual_context.library
            else None
        )
        
        # Initial knowledge retrieval (Iteration 1)
        knowledge = None
        retrieved_knowledge_json = None
        actual_retrieved_k = 0
        if self.semantic_memory and self.retrieval_k > 0:
            try:
                knowledge = self.semantic_memory.retrieve(
                    perceptual_context=perceptual_context,
                    error_context=None,
                    user_instruction=user_instruction,
                    iteration=1,
                    top_k=self.retrieval_k,
                )
                retrieved_knowledge_json = knowledge.model_dump_json() if knowledge.chunks else None
                actual_retrieved_k = len(knowledge.chunks) if knowledge and knowledge.chunks else 0
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Initial semantic memory retrieval failed: {e}")

        # Get adapter context if available
        coding_guidance = None
        if adapter:
            try:
                coding_guidance = adapter.build_code_context(perceptual_context, {})
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to build adapter code context: {e}")

        # Initial request
        current_request = CodeGenerationRequest(
            perceptual_context_json=perceptual_context_json,
            user_instruction=user_instruction,
            retrieved_knowledge_json=retrieved_knowledge_json,
            coding_guidance=coding_guidance
        )
        
        success = False
        final_artifact = None
        final_result = None
        final_error = None
        current_error_context_before = None
        judge_decisions_list: list[dict[str, Any]] = []
        
        for i in range(1, self.max_iterations + 1):
            logger.info(f"--- Starting Iteration {i}/{self.max_iterations} ---")
            iteration_start = time.time()
            
            # 1. Generate code
            try:
                artifact = self.coder.process(current_request)
            except Exception as e:  # noqa: BLE001
                logger.error(f"Coder failed at iteration {i}: {e}")
                # We can't continue if coder crashes
                break

            final_artifact = artifact

            # 2. Execute code
            try:
                result = self.executor.process(artifact)
            except Exception as e:  # noqa: BLE001
                logger.error(f"Executor crashed at iteration {i}: {e}")
                break
                
            final_result = result
            
            # 3. Execution Judge evaluation (FINISH vs FIX)
            if self.disable_judge:
                from mlzero.schemas.execution import ExecutionDecision
                decision = ExecutionDecision(
                    decision="FINISH" if result.success else "FIX",
                    reason="Judge disabled (raw exit code status)",
                    confidence=1.0,
                )
            else:
                try:
                    decision = self.judge.process(
                        artifact=artifact,
                        result=result,
                        perceptual_context=perceptual_context,
                        iteration=i,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Judge evaluation failed at iteration {i}: {e}, falling back to result.success")
                    from mlzero.schemas.execution import ExecutionDecision
                    decision = ExecutionDecision(
                        decision="FINISH" if result.success else "FIX",
                        reason=f"Fallback due to judge exception: {e}",
                        confidence=0.5,
                    )

            logger.info(f"Iteration {i} Judge decision: {decision.decision} (reason: {decision.reason})")
            judge_decisions_list.append({
                "iteration": i,
                "decision": decision.decision,
                "reason": decision.reason,
            })

            # 4. Handle Judge Decision
            if decision.decision == "FINISH":
                logger.info(f"Iteration {i} finished successfully (Judge: FINISH)!")
                duration = time.time() - iteration_start
                iteration_history.append(
                    IterationRecord(
                        iteration_number=i,
                        code_artifact=artifact,
                        execution_result=result,
                        judge_decision=decision.decision,
                        judge_reason=decision.reason,
                        duration_seconds=duration,
                    )
                )
                
                if self.episodic_memory:
                    self.episodic_memory.record_iteration(
                        run_id=run_id,
                        iteration=i,
                        status="SUCCESS",
                        artifact=artifact,
                        result=result,
                        retrieved_knowledge=knowledge,
                        retrieval_query=knowledge.query_used if knowledge else None,
                        error_context_before=current_error_context_before,
                        perceptual_context=perceptual_context,
                        selected_library=selected_lib,
                        user_instruction=user_instruction,
                        summary=f"Execution finished successfully: {decision.reason}",
                        judge_decision=decision.decision,
                        judge_reason=decision.reason,
                        execution_status=result.status,
                        bash_script=artifact.bash_script,
                        missing_expected_files=result.missing_expected_files,
                    )
                
                success = True
                break
            else:
                logger.warning(f"Iteration {i} requires FIX (Judge: FIX, reason: {decision.reason}). stderr: {result.stderr[:200] if result else 'none'}")
                
                # Analyze error (ONLY called when Judge decides FIX)
                try:
                    error_ctx = self.error_analyzer.process(
                        artifact,
                        result,
                        iteration=i,
                        perceptual_context=perceptual_context,
                        selected_library=selected_lib,
                        retrieved_knowledge=knowledge,
                    )
                except Exception as e:  # noqa: BLE001
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
                        judge_decision=decision.decision,
                        judge_reason=decision.reason,
                        duration_seconds=duration,
                    )
                )
                
                if self.episodic_memory:
                    self.episodic_memory.record_iteration(
                        run_id=run_id,
                        iteration=i,
                        status="FAIL",
                        artifact=artifact,
                        result=result,
                        error_ctx=error_ctx,
                        retrieved_knowledge=knowledge,
                        retrieval_query=knowledge.query_used if knowledge else None,
                        error_context_before=current_error_context_before,
                        perceptual_context=perceptual_context,
                        selected_library=selected_lib,
                        user_instruction=user_instruction,
                        summary=f"Execution requires fix: {decision.reason}",
                        suggested_fix=error_ctx.suggested_fix,
                        judge_decision=decision.decision,
                        judge_reason=decision.reason,
                        execution_status=result.status,
                        bash_script=artifact.bash_script,
                        missing_expected_files=result.missing_expected_files,
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
                
                # Update current error context before next iteration
                current_error_context_before = error_ctx
                
                # Error-aware knowledge retrieval (Iteration i+1)
                if self.semantic_memory and self.retrieval_k > 0:
                    try:
                        knowledge = self.semantic_memory.retrieve(
                            perceptual_context=perceptual_context,
                            error_context=error_ctx,
                            user_instruction=user_instruction,
                            iteration=i + 1,
                            top_k=self.retrieval_k,
                        )
                        retrieved_knowledge_json = knowledge.model_dump_json() if knowledge.chunks else None
                        actual_retrieved_k = len(knowledge.chunks) if knowledge and knowledge.chunks else 0
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"Error-aware semantic memory retrieval failed: {e}")
                
                # Prepare for next iteration
                current_request = CodeGenerationRequest(
                    perceptual_context_json=perceptual_context_json,
                    user_instruction=user_instruction,
                    previous_code=artifact.code,
                    error_context_json=error_ctx.model_dump_json(),
                    retrieved_knowledge_json=retrieved_knowledge_json,
                    episodic_context_json=episodic_context_json,
                    coding_guidance=coding_guidance
                )
                
        total_duration = time.time() - start_time
        if self.episodic_memory:
            self.episodic_memory.end_run(run_id, "SUCCESS" if success else "FAIL")
            
        pipeline_trace = {
            "perception_completed": True,
            "semantic_memory_completed": bool(self.semantic_memory and self.retrieval_k > 0),
            "coder_completed": bool(final_artifact is not None),
            "adapter_used": adapter.__class__.__name__ if adapter else "None",
            "executor_completed": bool(final_result is not None),
            "judge_decisions": judge_decisions_list,
            "episodic_memory_recorded": bool(self.episodic_memory is not None),
            "requested_k": self.retrieval_k,
            "actual_retrieved_k": actual_retrieved_k,
        }

        return IterativeRunResult(
            run_id=run_id,
            success=success,
            total_iterations=len(iteration_history),
            final_code_artifact=final_artifact,
            final_execution_result=final_result,
            iteration_history=iteration_history,
            final_error_context=final_error,
            total_duration_seconds=total_duration,
            judge_decisions=judge_decisions_list,
            pipeline_trace=pipeline_trace,
        )
