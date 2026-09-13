"""
Safe Python runner.
"""

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from mlzero.core.config import settings
from mlzero.core.logger import setup_logger
from mlzero.schemas.coder import ExecutionResult

logger = setup_logger(__name__)


class PythonRunner:
    """A tool to safely run generated Python code in a controlled environment."""

    def __init__(self) -> None:
        self.workspace_root = Path(settings.execution.workspace_root)
        self.timeout = settings.execution.timeout_seconds
        self.python_exec = settings.execution.python_executable
        self.max_stdout = settings.execution.max_stdout_size_bytes
        self.max_stderr = settings.execution.max_stderr_size_bytes
        self.allowed_env = settings.execution.allowed_env_vars

    def run_code(self, code: str, input_files: dict[str, str] | None = None) -> ExecutionResult:
        """
        Execute Python code in a temporary workspace.
        
        Args:
            code: The Python code to run.
            input_files: A dictionary of {filename: content} to write into the workspace before running.
        
        Returns:
            ExecutionResult containing execution details.
        """
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp(dir=self.workspace_root, prefix="exec_")
        temp_path = Path(temp_dir)
        
        script_path = temp_path / "script.py"
        out_dir = temp_path / settings.execution.output_dir_name
        out_dir.mkdir()
        
        try:
            script_path.write_text(code, encoding="utf-8")
            
            if input_files:
                for fname, content in input_files.items():
                    # Security: Prevent writing outside temp path
                    target_path = (temp_path / fname).resolve()
                    if temp_path.resolve() not in target_path.parents:
                        raise ValueError(f"Invalid path {fname} escapes workspace.")
                    target_path.write_text(content, encoding="utf-8")

            # Prepare controlled environment
            env = {}
            for key in self.allowed_env:
                if key in os.environ:
                    env[key] = os.environ[key]

            start_time = time.time()
            
            # Execute
            try:
                process = subprocess.run(
                    [self.python_exec, script_path.name],
                    cwd=str(temp_path),
                    env=env,
                    capture_output=True,
                    check=False,
                    timeout=self.timeout,
                    text=True,
                    encoding="utf-8",
                    errors="replace"
                )
                success = process.returncode == 0
                return_code = process.returncode
                stdout = process.stdout
                stderr = process.stderr
                error_info = None

            except subprocess.TimeoutExpired as e:
                success = False
                return_code = None
                stdout = e.stdout.decode("utf-8", errors="replace") if e.stdout else ""
                stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
                error_info = f"Execution timed out after {self.timeout} seconds."

            duration = time.time() - start_time

            # Truncate output if too large
            if len(stdout.encode("utf-8")) > self.max_stdout:
                stdout = stdout[:self.max_stdout] + "\n...[TRUNCATED]"
            if len(stderr.encode("utf-8")) > self.max_stderr:
                stderr = stderr[:self.max_stderr] + "\n...[TRUNCATED]"

            # Collect output files
            output_files = []
            if out_dir.exists():
                for root, _, files in os.walk(out_dir):
                    for file in files:
                        rel_path = Path(root).joinpath(file).relative_to(out_dir)
                        output_files.append(str(rel_path))

            return ExecutionResult(
                success=success,
                return_code=return_code,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                output_files=output_files,
                error_info=error_info
            )

        except (OSError, ValueError) as e:
            return ExecutionResult(
                success=False,
                error_info=f"Runner encountered an error: {e}",
            )
        finally:
            # Clean up temporary resources
            try:
                shutil.rmtree(temp_path, ignore_errors=True)
            except OSError as e:
                logger.warning(f"Failed to clean up workspace {temp_path}: {e}")
