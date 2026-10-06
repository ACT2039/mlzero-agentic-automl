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
            "Do not use interactive input. Output all artifacts to the 'out' directory.\n\n"
            "REQUIRED OUTPUT FILES:\n"
            "- Predictions: Save test set predictions to 'out/predictions.csv'.\n"
            "- Metrics: Save comprehensive validation/evaluation metrics to 'out/summary.json'. For classification, compute and output: accuracy, balanced_accuracy, f1, precision, recall, mcc. For regression, compute and output: rmse, mae, r2, mse. Format: {\"metrics\": {\"accuracy\": ..., \"balanced_accuracy\": ..., \"f1\": ..., \"precision\": ..., \"recall\": ..., \"mcc\": ...}}.\n"
            "- Model: If using AutoGluon, pass path='out/models' to TabularPredictor: TabularPredictor(label=label, path='out/models').fit(...). Do NOT call predictor.save('out/models') because .save() does not accept a directory path!\n"
            "- AutoGluon API: Pass 'eval_metric', 'label', and 'path' to TabularPredictor(...), NEVER to predictor.fit()! fit() will raise ValueError for unrecognized keyword argument 'eval_metric'.\n"
            "- Path Strings: CRITICAL: Always use forward slashes (/) for all file paths (e.g. 'C:/Users/...', 'train.csv', 'out/models'), NEVER backslashes (\\). Backslashes cause fatal Python SyntaxErrors.\n\n"
            "DATA PREPROCESSING RULES:\n"
            "- Memory & Model Safety: When fitting TabularPredictor, always pass excluded_model_types=['NN_TORCH', 'FASTAI'] into predictor.fit(...) to avoid memory-heavy PyTorch neural networks. If len(train) > 2500, sample down: train = train.sample(n=2500, random_state=42) to keep peak memory well under 400MB and prevent OOM in cloud environments.\n"
            "- CRITICAL TARGET COLUMN RULE: Never drop or remove the target column from 'train'! When defining feature lists like 'train_features = [c for c in train.columns if c != label]', DO NOT do 'train = train[train_features]'. AutoGluon TabularPredictor.fit(train) strictly requires the target column to remain inside 'train'. Only remove the target column from 'test', never from 'train'!\n"
            "- If Data Quality Findings mention missing values (NaNs), explicitly fill/impute them before training. IMPORTANT: Avoid pandas chained assignment FutureWarning by using 'df[col] = df[col].fillna(val)' instead of 'df[col].fillna(val, inplace=True)'.\n"
            "- Always drop NaNs from the target column before training: df = df.dropna(subset=[label]).\n"
            "- If Data Quality Findings mention invalid numeric values or malformed target values, coerce and clean them.\n"
            "- If test set has column mismatches or missing features compared to train, align the test set columns to match the model's training features.\n\n"
            "=== 1. PERCEPTUAL CONTEXT (DATASET & TASK) ===\n"
            f"Context: {request.perceptual_context_json}\n"
        )

        # Add explicit task specifications compactly without blowing token budgets
        try:
            p_data = json.loads(request.perceptual_context_json) if request.perceptual_context_json else {}
            task_info = p_data.get("task") or {}
            target_col = task_info.get("target_column")
            task_type = task_info.get("task_type")
            dq_info = p_data.get("data_quality") or task_info.get("data_quality")

            input_files = task_info.get("input_data_files") or []
            if input_files:
                clean_input_files = [f.replace("\\", "/") for f in input_files]
                prompt += f"Input Data Files (use forward slashes): {clean_input_files}\n"
            if target_col:
                prompt += f"Target Column: {target_col}\n"
            if task_type:
                prompt += f"Task Type: {task_type}\n"
            if dq_info:
                prompt += f"Data Quality Findings: {dq_info}\n"

            # Compact file context summaries
            file_ctxs = p_data.get("file_contexts", [])
            for fc in file_ctxs[:2]:
                fn = fc.get("path") or fc.get("filename") or ""
                cols = fc.get("columns", [])
                if cols:
                    col_names = [c.get("name") for c in cols if isinstance(c, dict) and "name" in c] or cols[:20]
                    prompt += f"File '{fn}' Columns: {col_names}\n"
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Could not extract task specifications from perceptual context: {e}")
            if request.perceptual_context_json:
                prompt += f"Context: {request.perceptual_context_json[:1000]}\n"

        # 2. User Instruction
        prompt += "\n=== 2. USER INSTRUCTION ===\n"
        if request.user_instruction:
            prompt += f"User Instruction: {request.user_instruction}\n"
        if request.coding_guidance:
            prompt += f"Guidance: {request.coding_guidance[:800]}\n"
        if not request.user_instruction and not request.coding_guidance:
            prompt += "(No additional user instruction provided)\n"

        # 3. Semantic Memory (Retrieved Knowledge - bounded to top-3 and max 350 chars each)
        prompt += "\n=== 3. SEMANTIC MEMORY / EXTERNAL KNOWLEDGE (RETRIEVED KNOWLEDGE - TOP-5 CONDENSED DOCS) ===\n"
        if request.retrieved_knowledge_json:
            try:
                k_data = json.loads(request.retrieved_knowledge_json)
                chunks = k_data.get("chunks", [])
                if chunks:
                    for idx, chunk in enumerate(chunks[:3], 1):
                        source = chunk.get("source") or chunk.get("title") or f"Doc {idx}"
                        text = chunk.get("condensed_guidance") or chunk.get("content") or ""
                        if len(text) > 350:
                            text = text[:350] + "..."
                        prompt += f"--- Doc {idx} ({source}): {text}\n"
                else:
                    prompt += f"{request.retrieved_knowledge_json[:500]}\n"
            except Exception:  # noqa: BLE001
                prompt += f"{request.retrieved_knowledge_json[:500]}\n"
            prompt += "Ensure you use the correct APIs from this retrieved knowledge.\n"
        else:
            prompt += "(None retrieved)\n"

        # 4. Episodic / Error Recovery Context (compacted)
        prompt += "\n=== 4. EPISODIC & ERROR RECOVERY CONTEXT ===\n"
        has_episodic = False
        if request.previous_code and request.error_context_json:
            has_episodic = True
            prev_code = request.previous_code
            if len(prev_code) > 2000:
                lines = prev_code.splitlines()
                if len(lines) > 50:
                    prev_code = "\n".join(lines[:25]) + "\n\n# ... [lines omitted for brevity] ...\n\n" + "\n".join(lines[-25:])
                else:
                    prev_code = prev_code[-2000:]
            err_ctx = request.error_context_json
            if len(err_ctx) > 1200:
                err_ctx = err_ctx[:1200] + "..."
            prompt += (
                "\n--- PREVIOUS FAILURE ---\n"
                f"Previous Code Attempt:\n```python\n{prev_code}\n```\n"
                f"Error Context:\n{err_ctx}\n"
            )
        if request.episodic_context_json:
            has_episodic = True
            epi_ctx = request.episodic_context_json
            if len(epi_ctx) > 800:
                epi_ctx = epi_ctx[:800] + "..."
            prompt += (
                f"Episodic Run History:\n{epi_ctx}\n"
                "Use this history to avoid repeating previous mistakes.\n"
            )
        if not has_episodic:
            prompt += "(Iteration 1: No previous failures recorded)\n"

        artifact = self.llm_client.generate_structured(prompt, CodeArtifact, max_tokens=1800)
        if artifact.code and r"\n" in artifact.code and ("\n" not in artifact.code or artifact.code.count(r"\n") > artifact.code.count("\n")):
            artifact.code = artifact.code.replace(r"\r\n", "\n").replace(r"\n", "\n").replace(r"\t", "\t")
        if artifact.code:
            import re
            # Normalize Windows drive backslashes to forward slashes: e.g. "C:\Users\..." or r"C:\..." -> "C:/Users/..."
            artifact.code = re.sub(r'([A-Za-z]:\\[^"\'\n\r]+)', lambda m: m.group(0).replace("\\", "/"), artifact.code)
            # If the LLM passed eval_metric into .fit(), remove it defensively to prevent ValueError
            artifact.code = re.sub(r"(\.fit\([^)]*),\s*eval_metric\s*=\s*['\"][^'\"]+['\"]", r"\1", artifact.code)
            # Defensively prevent accidental dropping of target column from train DataFrame:
            artifact.code = re.sub(
                r"^[ \t]*train\s*=\s*train\[(?:train_)?features\][ \t]*$",
                "# Preserved target column in train for TabularPredictor.fit\npass",
                artifact.code,
                flags=re.MULTILINE,
            )
            # Defensively ensure TabularPredictor.fit() excludes heavy neural networks to avoid OOM
            if "TabularPredictor" in artifact.code and "excluded_model_types" not in artifact.code:
                artifact.code = re.sub(
                    r"(\.fit\s*\([A-Za-z0-9_]+(?:,[^()]*?)?)\)",
                    r"\1, excluded_model_types=['NN_TORCH', 'FASTAI'])",
                    artifact.code,
                )
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
            ttype = perceptual_context.task_type or (perceptual_context.task.task_type if perceptual_context.task else "Unknown")
            tcol = perceptual_context.target_column or (perceptual_context.task.target_column if perceptual_context.task else "Unknown")
            prompt += f"Perceptual Task: {ttype}, Target: {tcol}\n\n"
        if selected_library:
            prompt += f"Selected ML Library: {selected_library}\n\n"
        if retrieved_knowledge and hasattr(retrieved_knowledge, "sources"):
            prompt += f"Retrieved Knowledge Sources: {retrieved_knowledge.sources}\n\n"

        prompt += "Produce a concise error summary, categorize the error, and provide an actionable suggested fix."

        try:
            error_context = self.llm_client.generate_structured(prompt, ErrorContext, max_tokens=1500)
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

