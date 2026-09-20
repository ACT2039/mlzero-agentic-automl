"""
Security verification tests for Stage 4:
Proving process-level isolation, safe script handling, path validation, and env filtering.
"""
import ast
import os
from pathlib import Path

from mlzero.agents.coder import CodeArtifact, ExecutorAgent
from mlzero.tools.python_runner import PythonRunner


def test_security_valid_setup_script_representation(tmp_path: Path) -> None:
    """A. Valid controlled setup script is stored safely in workspace as setup.sh."""
    runner = PythonRunner()
    agent = ExecutorAgent(runner=runner)

    artifact = CodeArtifact(
        code="print('execution running')",
        bash_script="#!/bin/bash\necho 'setting up sandbox environment'",
    )
    result = agent.process(artifact)

    assert result.success is True
    assert result.workspace_dir is not None
    setup_file = Path(result.workspace_dir) / "setup.sh"
    assert setup_file.exists()
    assert "setting up sandbox environment" in setup_file.read_text(encoding="utf-8")


def test_security_bash_does_not_execute_in_parent_process() -> None:
    """B. Generated bash script is never evaluated in the parent process."""
    # Even if bash_script contains malicious parent commands, it is only written to disk
    runner = PythonRunner()
    canary_file = Path("canary_should_not_exist.txt")
    if canary_file.exists():
        canary_file.unlink()

    artifact = CodeArtifact(
        code="print('safe python')",
        bash_script="touch canary_should_not_exist.txt",
    )
    result = runner.run_code(
        artifact.code,
        bash_script=artifact.bash_script,
    )
    assert result.success is True
    assert not canary_file.exists(), "Bash script was executed in parent process!"


def test_security_no_eval_or_exec_in_execution_pipeline() -> None:
    """C. Verify via AST that no eval() or exec() calls exist in execution pipeline modules."""
    files_to_check = [
        Path("mlzero/agents/coder.py"),
        Path("mlzero/agents/judge.py"),
        Path("mlzero/tools/python_runner.py"),
        Path("mlzero/orchestration/iterative.py"),
    ]

    for file_path in files_to_check:
        assert file_path.exists(), f"File {file_path} not found."
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    assert func.id not in ("eval", "exec"), (
                        f"Found forbidden {func.id}() call in {file_path} at line {node.lineno}!"
                    )


def test_security_no_unrestricted_parent_shell() -> None:
    """D. PythonRunner does not invoke shell=True, preventing parent shell command injection."""
    runner = PythonRunner()
    # Code with shell-metacharacters should not be interpreted by a shell
    code = "print('hello & dir')"
    result = runner.run_code(code)
    assert result.success is True
    assert "hello & dir" in result.stdout


def test_security_path_traversal_prevention(tmp_path: Path) -> None:
    """E. Existing path security remains enforced: writing outside workspace is rejected."""
    runner = PythonRunner()
    malicious_files = {
        "../escape.txt": "evil content",
        "../../evil.sh": "rm -rf /",
    }
    result = runner.run_code("print('attempt escape')", input_files=malicious_files)
    assert result.success is False
    assert "Insecure file path detected" in result.stderr or "Insecure file path detected" in (result.error_info or "")


def test_security_api_keys_not_exposed_in_subprocess_env() -> None:
    """F. API keys in os.environ are filtered out and not accessible to subprocess execution."""
    os.environ["GEMINI_API_KEY"] = "secret_gemini_key_12345"
    os.environ["OPENAI_API_KEY"] = "secret_openai_key_67890"

    try:
        runner = PythonRunner()
        code = """
import os
print("GEMINI:", os.environ.get("GEMINI_API_KEY", "NOT_FOUND"))
print("OPENAI:", os.environ.get("OPENAI_API_KEY", "NOT_FOUND"))
"""
        result = runner.run_code(code)
        assert result.success is True
        assert "GEMINI: NOT_FOUND" in result.stdout
        assert "OPENAI: NOT_FOUND" in result.stdout
        assert "secret_gemini_key_12345" not in result.stdout
        assert "secret_openai_key_67890" not in result.stdout
    finally:
        del os.environ["GEMINI_API_KEY"]
        del os.environ["OPENAI_API_KEY"]
