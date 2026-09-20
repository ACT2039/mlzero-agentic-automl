import json
from typing import Any

from mlzero.agents.judge import ExecutionJudgeAgent
from mlzero.core.llm import LLMClient
from mlzero.core.logger import setup_logger
from mlzero.schemas.coder import (
    CodeArtifact,
    CodeGenerationRequest,
    ErrorContext,
    ExecutionResult,
)
from mlzero.schemas.perception import PerceptualContext
from mlzero.tools.python_runner import PythonRunner

__all__ = ["CoderAgent", "ErrorAnalyzerAgent", "ExecutionJudgeAgent", "ExecutorAgent"]

logger = setup_logger(__name__)


class CoderAgent:
    """Agent responsible for writing code based on task and memory."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def process(self, request: CodeGenerationRequest) -> CodeArtifact:
        """
        Generate Python code based on the perceptual context and instructions.
        """
        logger.info("CoderAgent processing generation request.")
        prompt = (
            "Generate complete and executable Python code for the following task context.\n"
            "Do not use interactive input. Output any files to the 'out' directory.\n\n"
            "=== 1. PERCEPTUAL CONTEXT (DATASET & TASK) ===\n"
            f"Context: {request.perceptual_context_json}\n"
        )

        # Add explicit task specifications if perceptual context is present
        try:
            p_data = json.loads(request.perceptual_context_json)
            task_info = p_data.get("task") or {}
            target_col = task_info.get("target_column")
            task_type = task_info.get("task_type")
            dq_info = p_data.get("data_quality") or task_info.get("data_quality")

            if target_col or task_type or dq_info:
                prompt += "\n--- Task Specifications ---\n"
                if target_col:
                    prompt += f"Target Column: {target_col}\n"
                if task_type:
                    prompt += f"Task Type: {task_type}\n"
                if dq_info:
                    prompt += f"Data Quality Findings: {dq_info}\n"
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Could not extract task specifications from perceptual context: {e}")

        # 2. User Instruction
        prompt += "\n=== 2. USER INSTRUCTION ===\n"
        if request.user_instruction:
            prompt += f"User Instruction: {request.user_instruction}\n"
        if request.coding_guidance:
            prompt += f"Guidance: {request.coding_guidance}\n"
        if not request.user_instruction and not request.coding_guidance:
            prompt += "(No additional user instruction provided)\n"

        # 3. Semantic Memory (Retrieved Knowledge)
        prompt += "\n=== 3. SEMANTIC MEMORY / EXTERNAL KNOWLEDGE (RETRIEVED KNOWLEDGE - TOP-5 CONDENSED DOCS) ===\n"
        if request.retrieved_knowledge_json:
            try:
                k_data = json.loads(request.retrieved_knowledge_json)
                chunks = k_data.get("chunks", [])
                if chunks:
                    for idx, chunk in enumerate(chunks, 1):
                        source = chunk.get("source") or chunk.get("title") or f"Doc {idx}"
                        text = chunk.get("condensed_guidance") or chunk.get("content") or ""
                        prompt += f"--- Document {idx}: {source} ---\n{text}\n\n"
                else:
                    prompt += f"{request.retrieved_knowledge_json}\n"
            except Exception:  # noqa: BLE001
                prompt += f"{request.retrieved_knowledge_json}\n"
            prompt += "Use this retrieved knowledge to guide your implementation. Ensure you use the correct APIs.\n"
        else:
            prompt += "(None retrieved)\n"

        # 4. Episodic / Error Recovery Context
        prompt += "\n=== 4. EPISODIC & ERROR RECOVERY CONTEXT ===\n"
        has_episodic = False
        if request.previous_code and request.error_context_json:
            has_episodic = True
            prompt += (
                "\n--- PREVIOUS FAILURE ---\n"
                f"Previous Code Attempt:\n```python\n{request.previous_code}\n```\n"
                f"Error Context:\n{request.error_context_json}\n"
            )
        if request.episodic_context_json:
            has_episodic = True
            prompt += (
                f"Episodic Run History:\n{request.episodic_context_json}\n"
                "Use this history to avoid repeating previous mistakes.\n"
            )
        if not has_episodic:
            prompt += "(Iteration 1: No previous failures recorded)\n"

        artifact = self.llm_client.generate_structured(prompt, CodeArtifact)
        return artifact


class ExecutorAgent:
    """Agent responsible for executing generated code."""

    def __init__(self, runner: PythonRunner | None = None):
        self.runner = runner or PythonRunner()

    def process(self, artifact: CodeArtifact, input_files: dict[str, str] | None = None) -> ExecutionResult:
        """
        Execute the provided code artifact.
        """
        logger.info("ExecutorAgent executing code artifact.")
        if not artifact.code.strip():
            return ExecutionResult(
                success=False,
                status="FAILURE",
                error_info="Code artifact is empty."
            )
            
        return self.runner.run_code(
            artifact.code,
            input_files=input_files,
            bash_script=artifact.bash_script,
            expected_output_files=artifact.expected_output_files,
        )


class ErrorAnalyzerAgent:
    """Agent responsible for analyzing errors from execution."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def process(
        self,
        artifact: CodeArtifact,
        result: ExecutionResult,
        iteration: int,
        perceptual_context: PerceptualContext | None = None,
        selected_library: str | None = None,
        retrieved_knowledge: Any | None = None,
    ) -> ErrorContext:
        """Analyze execution failure and produce structured ErrorContext."""
        logger.info("ErrorAnalyzerAgent processing task")

        # Ensure we cap the stderr/stdout to avoid massive logs in prompt
        stderr_sample = result.stderr[-2000:] if result.stderr else "No stderr."
        stdout_sample = result.stdout[-2000:] if result.stdout else "No stdout."
        logger.error(f"Execution Error Stderr: {result.stderr}")

        prompt = (
            f"Analyze the following execution failure for Iteration {iteration}.\n"
            f"Code:\n{artifact.code}\n\n"
            f"Stdout:\n{stdout_sample}\n\n"
            f"Stderr:\n{stderr_sample}\n\n"
            f"Error Info: {result.error_info}\n\n"
        )
        if perceptual_context:
            prompt += f"Perceptual Context:\n{perceptual_context.model_dump_json()}\n\n"
        if selected_library:
            prompt += f"Selected ML Library: {selected_library}\n\n"
        if retrieved_knowledge and hasattr(retrieved_knowledge, "sources"):
            prompt += f"Retrieved Knowledge Sources: {retrieved_knowledge.sources}\n\n"

        prompt += "Produce a concise error summary, categorize the error, and provide an actionable suggested fix."

        try:
            error_context = self.llm_client.generate_structured(prompt, ErrorContext)
            error_context.iteration = iteration
            if not error_context.error_summary:
                error_context.error_summary = f"{error_context.error_category}: {error_context.error_message}"
            return error_context
        except Exception as e:  # noqa: BLE001
            logger.error(f"ErrorAnalyzerAgent failed to generate context: {e}")
            return ErrorContext(
                iteration=iteration,
                error_category="unknown",
                error_summary=f"Unknown error during execution: {e}",
                error_message=f"Failed to analyze error: {e}",
                stderr_excerpt=stderr_sample[:500],
                suggested_fix="Review the logs manually.",
            )

