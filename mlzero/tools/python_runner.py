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

    def __init__(
        self,
        workspace_root: Path | str | None = None,
        default_timeout_seconds: int | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root) if workspace_root else Path(settings.execution.workspace_root)
        self.timeout = timeout_seconds or default_timeout_seconds or settings.execution.timeout_seconds
        self.python_exec = settings.execution.python_executable
        self.max_stdout = settings.execution.max_stdout_size_bytes
        self.max_stderr = settings.execution.max_stderr_size_bytes
        self.allowed_env = settings.execution.allowed_env_vars

    def run_code(
        self,
        code: str,
        input_files: dict[str, str] | None = None,
        bash_script: str | None = None,
        expected_output_files: list[str] | None = None,
        workspace_dir: str | Path | None = None,
        timeout_seconds: int | None = None,
    ) -> ExecutionResult:
        """
        Execute Python code in a controlled workspace.
        
        Args:
            code: The Python code to run.
            input_files: A dictionary of {filename: content} to write into the workspace before running.
            bash_script: Optional setup or execution shell script to store/run in workspace.
            expected_output_files: List of expected output filenames to validate.
            workspace_dir: Optional custom workspace directory to run within.
            timeout_seconds: Optional timeout override for this execution.
        
        Returns:
            ExecutionResult containing execution details and validation status.
        """
        if workspace_dir:
            temp_path = Path(workspace_dir)
            temp_path.mkdir(parents=True, exist_ok=True)
            temp_dir = str(temp_path)
        else:
            self.workspace_root.mkdir(parents=True, exist_ok=True)
            temp_dir = tempfile.mkdtemp(dir=self.workspace_root, prefix="exec_")
            temp_path = Path(temp_dir)
        
        script_path = temp_path / "script.py"
        out_dir = temp_path / settings.execution.output_dir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        effective_timeout = timeout_seconds if timeout_seconds is not None else self.timeout

        try:
            # Auto-sanitize code to ensure valid line endings and prevent corrupted literal \n
            if code and r"\n" in code and ("\n" not in code or code.count(r"\n") > code.count("\n")):
                code = code.replace(r"\r\n", "\n").replace(r"\n", "\n").replace(r"\t", "\t")

            script_path.write_text(code, encoding="utf-8")

            # Store optional bash script as an artifact
            if bash_script:
                bash_path = temp_path / "setup.sh"
                bash_path.write_text(bash_script, encoding="utf-8")
            
            if input_files:
                for fname, content in input_files.items():
                    # Security: Prevent writing outside temp path
                    target_path = (temp_path / fname).resolve()
                    if not str(target_path).startswith(str(temp_path.resolve())):
                        raise ValueError(f"Insecure file path detected: {fname}")
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_text(content, encoding="utf-8")

            # Filter environment variables for safety
            env = {
                k: os.environ[k]
                for k in self.allowed_env
                if k in os.environ
            }
            # Also pass PATH and SYSTEMROOT so subprocess can run python
            for key in ["PATH", "SystemRoot", "PYTHONPATH"]:
                if key in os.environ:
                    env[key] = os.environ[key]

            # Constrain parallel thread pools to avoid excessive RAM usage in cloud/constrained environments (e.g. 512MB RAM)
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            env.setdefault("NUMEXPR_NUM_THREADS", "1")
            env.setdefault("VECLIB_MAXIMUM_THREADS", "1")

            # Execute the script
            start_time = time.time()
            timed_out = False
            try:
                process = subprocess.run(
                    [self.python_exec, "script.py"],
                    cwd=temp_dir,
                    env=env,
                    capture_output=True,
                    timeout=effective_timeout,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
                success = process.returncode == 0
                status = "SUCCESS" if success else "FAILURE"
                return_code = process.returncode
                stdout = process.stdout
                stderr = process.stderr
                error_info = None

            except subprocess.TimeoutExpired as e:
                success = False
                timed_out = True
                status = "TIMEOUT"
                return_code = None
                stdout = e.stdout if isinstance(e.stdout, str) else (e.stdout.decode("utf-8", errors="replace") if e.stdout else "")
                stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr.decode("utf-8", errors="replace") if e.stderr else "")
                error_info = f"Execution timed out after {effective_timeout} seconds."
                if not stderr:
                    stderr = error_info

            duration = time.time() - start_time

            # Truncate output if too large
            if len(stdout.encode("utf-8")) > self.max_stdout:
                stdout = stdout[:self.max_stdout] + "\n...[TRUNCATED]"
            if len(stderr.encode("utf-8")) > self.max_stderr:
                stderr = stderr[:self.max_stderr] + "\n...[TRUNCATED]"

            # Collect output files and directories from output dir AND workspace root (excluding internal files)
            output_files: list[str] = []
            if out_dir.exists():
                for root, dirs, files in os.walk(out_dir):
                    for d in dirs:
                        rel_d_out = Path(root).joinpath(d).relative_to(out_dir).as_posix()
                        rel_d_ws = Path(root).joinpath(d).relative_to(temp_path).as_posix()
                        if rel_d_ws not in output_files:
                            output_files.append(rel_d_ws)
                        if rel_d_out not in output_files:
                            output_files.append(rel_d_out)
                    for file in files:
                        rel_f_out = Path(root).joinpath(file).relative_to(out_dir).as_posix()
                        rel_f_ws = Path(root).joinpath(file).relative_to(temp_path).as_posix()
                        if rel_f_ws not in output_files:
                            output_files.append(rel_f_ws)
                        if rel_f_out not in output_files:
                            output_files.append(rel_f_out)

            internal_files = {"script.py", "setup.sh"}
            for item in temp_path.iterdir():
                clean_name = item.name.replace("\\", "/")
                if item.name not in internal_files and clean_name not in output_files and item.name != "out":
                    output_files.append(clean_name)

            # Validate expected output files
            missing_expected: list[str] = []
            if expected_output_files and success:
                for exp_file in expected_output_files:
                    norm_exp = exp_file.replace("/", "\\") if os.name == "nt" else exp_file.replace("\\", "/")
                    base_name = Path(exp_file).name

                    # 1. Match against collected output files/dirs
                    file_found = any(
                        out_f == exp_file or out_f == norm_exp or Path(out_f).name == base_name
                        for out_f in output_files
                    )
                    # 2. Check disk directly in workspace / out directory
                    if not file_found:
                        file_found = (
                            (temp_path / exp_file).exists()
                            or (out_dir / exp_file).exists()
                            or (out_dir / base_name).exists()
                            or (temp_path / base_name).exists()
                        )
                    # 3. Model directory alias check (AutogluonModels)
                    if not file_found and "model" in exp_file.lower():
                        ag_candidates = [
                            temp_path / "AutogluonModels",
                            out_dir / "AutogluonModels",
                        ]
                        found_ag = next((c for c in ag_candidates if c.exists()), None)
                        if found_ag:
                            file_found = True
                            target_models = out_dir / "models"
                            if not target_models.exists() or not any(target_models.iterdir()):
                                try:
                                    shutil.copytree(found_ag, target_models, dirs_exist_ok=True)
                                except (OSError, shutil.Error):
                                    pass

                    if not file_found:
                        missing_expected.append(exp_file)

                if missing_expected:
                    success = False
                    status = "INVALID_OUTPUT"
                    error_info = f"Missing expected output files: {', '.join(missing_expected)}"

            return ExecutionResult(
                success=success,
                status=status,
                return_code=return_code,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                timed_out=timed_out,
                output_files=output_files,
                missing_expected_files=missing_expected,
                workspace_dir=temp_dir,
                error_info=error_info
            )

        except Exception as e:  # noqa: BLE001
            logger.error(f"Execution failed with unexpected exception: {e}")
            return ExecutionResult(
                success=False,
                status="FAILURE",
                return_code=-1,
                stdout="",
                stderr=str(e),
                duration_seconds=0.0,
                workspace_dir=temp_dir,
                error_info=f"Execution setup exception: {e}"
            )
        finally:
            import gc
            gc.collect()

            # If not configured to keep artifacts and directory was auto-created, clean up
            if not workspace_dir and not getattr(settings.execution, "keep_artifacts", True):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Failed to cleanup temp dir {temp_dir}: {e}")
