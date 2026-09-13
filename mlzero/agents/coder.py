"""
Iterative Coding agents for MLZero.
Includes implementations for Coder, Executor, Error Analyzer, and Retry loop logic.
"""

from mlzero.core.llm import LLMClient
from mlzero.core.logger import setup_logger
from mlzero.schemas.coder import (
    CodeArtifact,
    CodeGenerationRequest,
    ErrorContext,
    ExecutionResult,
)
from mlzero.tools.python_runner import PythonRunner

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
            "Do not use interactive input. Output any files to the 'out' directory.\n"
            f"Context: {request.perceptual_context_json}\n"
        )
        if request.user_instruction:
            prompt += f"User Instruction: {request.user_instruction}\n"
        if request.coding_guidance:
            prompt += f"Guidance: {request.coding_guidance}\n"
            
        if request.retrieved_knowledge_json:
            prompt += (
                "\n--- EXTERNAL KNOWLEDGE (SEMANTIC MEMORY) ---\n"
                f"{request.retrieved_knowledge_json}\n"
                "Use this knowledge to guide your implementation. Ensure you use the correct APIs.\n"
            )
            
        if request.previous_code and request.error_context_json:
            prompt += (
                "\n--- PREVIOUS FAILURE ---\n"
                "Your previous attempt failed. Please diagnose the failure, preserve working parts, "
                "apply the suggested correction, and produce a complete corrected program. "
                "Avoid repeating the exact failed approach.\n"
                f"Previous Code:\n{request.previous_code}\n\n"
                f"Error Analysis:\n{request.error_context_json}\n"
                f"Execution Result Summary:\n{request.previous_result_json}\n"
            )
            
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
                error_info="Code artifact is empty."
            )
            
        return self.runner.run_code(artifact.code, input_files=input_files)


class ErrorAnalyzerAgent:
    """Agent responsible for analyzing errors from execution."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def process(self, artifact: CodeArtifact, result: ExecutionResult, iteration: int) -> ErrorContext:
        """Analyze execution failure and produce ErrorContext."""
        logger.info("ErrorAnalyzerAgent processing task")
        
        # In a real system, you'd serialize the inputs cleanly.
        # Ensure we cap the stderr/stdout to avoid massive logs in prompt
        stderr_sample = result.stderr[-2000:] if result.stderr else "No stderr."
        stdout_sample = result.stdout[-2000:] if result.stdout else "No stdout."
        
        prompt = (
            f"Analyze the following execution failure for Iteration {iteration}.\n"
            f"Code:\n{artifact.code}\n\n"
            f"Stdout:\n{stdout_sample}\n\n"
            f"Stderr:\n{stderr_sample}\n\n"
            f"Error Info: {result.error_info}\n\n"
            "Produce a concise error summary, categorize the error, and provide an actionable suggested fix."
        )
        
        try:
            error_context = self.llm_client.generate_structured(prompt, ErrorContext)
            error_context.iteration = iteration
            return error_context
        except Exception as e:  # noqa: BLE001
            logger.error(f"ErrorAnalyzerAgent failed to generate context: {e}")
            return ErrorContext(
                iteration=iteration,
                error_category="unknown",
                error_message=f"Failed to analyze error: {e}",
                stderr_excerpt=stderr_sample[:500],
                suggested_fix="Review the logs manually."
            )
