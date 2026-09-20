"""
Execution Judge Agent for MLZero.
Determines whether an execution should FINISH or FIX based on execution logs,
exit codes, exceptions, timeouts, and expected artifact presence.
"""
from __future__ import annotations

from typing import Any

from mlzero.core.llm import LLMClient
from mlzero.core.logger import setup_logger
from mlzero.schemas.coder import CodeArtifact, ExecutionResult
from mlzero.schemas.execution import ExecutionDecision
from mlzero.schemas.perception import PerceptualContext

logger = setup_logger(__name__)


class ExecutionJudgeAgent:
    """Agent responsible for evaluating execution results to decide FINISH or FIX."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def _build_judge_prompt(
        self,
        artifact: CodeArtifact,
        result: ExecutionResult,
        perceptual_context: PerceptualContext | None = None,
        selected_library: str | None = None,
        retrieved_knowledge: Any | None = None,
        iteration: int = 1,
    ) -> str:
        stderr_sample = result.stderr[-2000:] if result.stderr else ""
        stdout_sample = result.stdout[-2000:] if result.stdout else ""

        prompt = (
            f"Evaluate the execution result for Iteration {iteration}.\n\n"
            f"Exit Code: {result.return_code}\n"
            f"Success Flag: {result.success}\n"
            f"Status: {result.status}\n"
            f"Output Files: {result.output_files}\n"
        )
        if artifact.expected_output_files:
            prompt += f"Expected Output Files: {artifact.expected_output_files}\n"
        if result.missing_expected_files:
            prompt += f"Missing Expected Files: {result.missing_expected_files}\n"
        if artifact.bash_script:
            prompt += f"Bash Script:\n{artifact.bash_script}\n\n"
        prompt += (
            f"Stdout:\n{stdout_sample}\n\n"
            f"Stderr:\n{stderr_sample}\n\n"
            f"Error Info: {result.error_info}\n\n"
            f"Code:\n{artifact.code}\n\n"
        )
        if perceptual_context:
            prompt += f"Perceptual Context:\n{perceptual_context.model_dump_json()}\n\n"
        if selected_library:
            prompt += f"Selected ML Library: {selected_library}\n\n"
        if retrieved_knowledge and hasattr(retrieved_knowledge, "sources"):
            prompt += f"Retrieved Knowledge Sources: {retrieved_knowledge.sources}\n\n"

        prompt += (
            "Determine whether this execution is complete and successful ('FINISH') "
            "or requires correction / another iteration ('FIX'). "
            "Provide the decision, reason, confidence, and issue_summary if FIX."
        )
        return prompt

    def process(
        self,
        artifact: CodeArtifact,
        result: ExecutionResult,
        iteration: int = 1,
        perceptual_context: PerceptualContext | None = None,
        selected_library: str | None = None,
        retrieved_knowledge: Any | None = None,
    ) -> ExecutionDecision:
        """
        Evaluate code execution and return a structured ExecutionDecision (FINISH or FIX).
        """
        logger.info(f"ExecutionJudgeAgent evaluating Iteration {iteration}")

        # Fast path 1: Explicit timeout
        if result.timed_out or result.status == "TIMEOUT":
            logger.info(f"ExecutionJudgeAgent: Iteration {iteration} timed out -> FIX")
            return ExecutionDecision(
                decision="FIX",
                reason=result.error_info or f"Execution timed out during iteration {iteration}.",
                confidence=1.0,
                issue_summary="Execution timed out.",
            )

        # Fast path 2: Missing expected artifacts
        if result.missing_expected_files or result.status == "INVALID_OUTPUT":
            missing_str = ", ".join(result.missing_expected_files)
            logger.info(
                f"ExecutionJudgeAgent: Iteration {iteration} missing expected output files ({missing_str}) -> FIX"
            )
            return ExecutionDecision(
                decision="FIX",
                reason=f"Execution completed but required output files were missing: {missing_str}",
                confidence=1.0,
                issue_summary=f"Missing expected output files: {missing_str}",
            )

        # If no LLM client configured, use deterministic heuristic
        if self.llm_client is None:
            if result.success and (result.return_code == 0 or result.return_code is None):
                return ExecutionDecision(
                    decision="FINISH",
                    reason="Execution completed with exit code 0 and no detected failure.",
                    confidence=1.0,
                )
            return ExecutionDecision(
                decision="FIX",
                reason=f"Execution failed or raised error: {result.error_info or result.stderr[:200]}",
                confidence=0.8,
                issue_summary=result.error_info or "Execution failure",
            )

        # Build prompt for LLM judgment
        prompt = self._build_judge_prompt(
            artifact=artifact,
            result=result,
            perceptual_context=perceptual_context,
            selected_library=selected_library,
            retrieved_knowledge=retrieved_knowledge,
            iteration=iteration,
        )

        try:
            decision = self.llm_client.generate_structured(prompt, ExecutionDecision)
            logger.info(f"ExecutionJudgeAgent decided: {decision.decision} (reason: {decision.reason})")
            return decision
        except Exception as e:  # noqa: BLE001
            logger.error(f"ExecutionJudgeAgent failed to generate structured decision: {e}")
            # Safe heuristic fallback
            if result.success and (result.return_code == 0 or result.return_code is None):
                return ExecutionDecision(
                    decision="FINISH",
                    reason="Execution completed with exit code 0 and no detected failure.",
                    confidence=0.8,
                )
            return ExecutionDecision(
                decision="FIX",
                reason=f"Execution failed or raised error: {result.error_info or result.stderr[:200]}",
                confidence=0.8,
                issue_summary=result.error_info or "Execution failure",
            )
