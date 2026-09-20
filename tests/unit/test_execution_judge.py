"""
Unit tests for STAGE 4: Execution Judge, Bash/Setup Artifacts, and Iterative Coding Loop.
"""
import pytest
from pydantic import ValidationError

from mlzero.agents.coder import CoderAgent, ErrorAnalyzerAgent
from mlzero.agents.judge import ExecutionJudgeAgent
from mlzero.core.llm import get_llm_client
from mlzero.memory.episodic import EpisodicMemory
from mlzero.memory.semantic import SemanticMemory
from mlzero.orchestration.iterative import IterativeCodingOrchestrator
from mlzero.schemas.coder import CodeArtifact, ExecutionResult
from mlzero.schemas.execution import ExecutionDecision
from mlzero.schemas.perception import PerceptualContext
from mlzero.tools.python_runner import PythonRunner

# ---------------------------------------------------------------------------
# 1. Schema Tests
# ---------------------------------------------------------------------------


def test_execution_decision_schema():
    """Test ExecutionDecision model serialization and validation."""
    dec_finish = ExecutionDecision(
        decision="FINISH",
        reason="Execution completed cleanly with all required outputs.",
        confidence=0.95,
    )
    assert dec_finish.decision == "FINISH"
    assert dec_finish.needs_fix is False
    assert dec_finish.confidence == 0.95

    dec_fix = ExecutionDecision(
        decision="FIX",
        reason="KeyError: 'target' not found in dataframe.",
        confidence=0.9,
        issue_summary="KeyError: 'target'",
    )
    assert dec_fix.decision == "FIX"
    assert dec_fix.needs_fix is True
    assert dec_fix.issue_summary == "KeyError: 'target'"

    # Invalid decision literal should raise ValidationError
    with pytest.raises(ValidationError):
        ExecutionDecision(decision="MAYBE", reason="uncertain")


def test_code_artifact_stage4_fields():
    """Test CodeArtifact with bash_script and expected_output_files."""
    art = CodeArtifact(
        code="print('hello')",
        language="python",
        bash_script="#!/bin/bash\npip install -q autogluon",
        expected_output_files=["predictions.csv", "summary.json"],
    )
    assert art.bash_script is not None
    assert "pip install" in art.bash_script
    assert art.expected_output_files == ["predictions.csv", "summary.json"]


def test_execution_result_stage4_fields():
    """Test ExecutionResult status, timed_out, and missing_expected_files."""
    res = ExecutionResult(
        success=False,
        status="TIMEOUT",
        timed_out=True,
        missing_expected_files=["predictions.csv"],
        duration_seconds=300.0,
    )
    assert res.status == "TIMEOUT"
    assert res.timed_out is True
    assert res.missing_expected_files == ["predictions.csv"]


# ---------------------------------------------------------------------------
# 2. ExecutionJudgeAgent Unit Tests
# ---------------------------------------------------------------------------


def test_judge_finish_on_clean_execution():
    """ExecutionJudgeAgent must decide FINISH when execution succeeds cleanly."""
    client = get_llm_client(use_mock=True)
    judge = ExecutionJudgeAgent(llm_client=client)

    artifact = CodeArtifact(code="print('done')", expected_output_files=[])
    result = ExecutionResult(
        success=True,
        status="SUCCESS",
        return_code=0,
        stdout="done\n",
        stderr="",
        output_files=[],
    )

    decision = judge.process(artifact, result)
    assert decision.decision == "FINISH"
    assert decision.needs_fix is False
    assert decision.confidence == 1.0


def test_judge_fix_on_timeout_fastpath():
    """ExecutionJudgeAgent fast-path decides FIX immediately on timeout without LLM call."""
    # Even with None client, fast-path must work
    judge = ExecutionJudgeAgent(llm_client=None)

    artifact = CodeArtifact(code="import time; time.sleep(10)")
    result = ExecutionResult(
        success=False,
        status="TIMEOUT",
        timed_out=True,
        return_code=-1,
        stderr="Execution timed out after 5.0 seconds.",
    )

    decision = judge.process(artifact, result)
    assert decision.decision == "FIX"
    assert decision.needs_fix is True
    assert "timed out" in decision.reason.lower()


def test_judge_fix_on_missing_expected_files_fastpath():
    """ExecutionJudgeAgent fast-path decides FIX when expected output files are missing."""
    judge = ExecutionJudgeAgent(llm_client=None)

    artifact = CodeArtifact(
        code="print('trained')",
        expected_output_files=["predictions.csv", "model.pkl"],
    )
    result = ExecutionResult(
        success=False,
        status="INVALID_OUTPUT",
        return_code=0,
        missing_expected_files=["predictions.csv", "model.pkl"],
        output_files=[],
    )

    decision = judge.process(artifact, result)
    assert decision.decision == "FIX"
    assert decision.needs_fix is True
    assert "missing" in decision.reason.lower()
    assert "predictions.csv" in decision.reason


def test_judge_fix_on_error_execution():
    """ExecutionJudgeAgent decides FIX on standard execution error."""
    client = get_llm_client(use_mock=True)
    judge = ExecutionJudgeAgent(llm_client=client)

    artifact = CodeArtifact(code="import pandas as pd; df['targt']")
    result = ExecutionResult(
        success=False,
        status="FAILURE",
        return_code=1,
        stderr="KeyError: 'targt'",
    )

    decision = judge.process(artifact, result)
    assert decision.decision == "FIX"
    assert decision.needs_fix is True
    assert "keyerror" in decision.reason.lower() or "targt" in decision.reason.lower()


def test_judge_structured_prompt_content():
    """ExecutionJudgeAgent formats comprehensive evaluation prompt."""
    judge = ExecutionJudgeAgent(llm_client=None)
    artifact = CodeArtifact(
        code="print('test')",
        bash_script="echo test",
        expected_output_files=["out.csv"],
    )
    result = ExecutionResult(
        success=False,
        status="FAILURE",
        return_code=1,
        stdout="some stdout",
        stderr="some stderr",
        output_files=[],
        missing_expected_files=["out.csv"],
    )
    pctx = PerceptualContext()

    prompt = judge._build_judge_prompt(artifact, result, pctx, iteration=1)
    assert "print('test')" in prompt
    assert "echo test" in prompt
    assert "out.csv" in prompt
    assert "some stderr" in prompt
    assert "FINISH" in prompt
    assert "FIX" in prompt


# ---------------------------------------------------------------------------
# 3. PythonRunner Enhancements (Timeout, Expected Files, Bash Artifact)
# ---------------------------------------------------------------------------


def test_python_runner_expected_output_files_success(tmp_path):
    """PythonRunner sets SUCCESS when all expected output files are generated."""
    runner = PythonRunner()
    code = """
with open('predictions.csv', 'w') as f:
    f.write('pred\\n1\\n')
"""
    result = runner.run_code(
        code,
        workspace_dir=str(tmp_path),
        expected_output_files=["predictions.csv"],
    )
    assert result.success is True
    assert result.status == "SUCCESS"
    assert result.missing_expected_files == []
    assert any("predictions.csv" in f for f in result.output_files)


def test_python_runner_missing_expected_output_files(tmp_path):
    """PythonRunner sets INVALID_OUTPUT when an expected file is not created."""
    runner = PythonRunner()
    code = "print('did not write predictions')"
    result = runner.run_code(
        code,
        workspace_dir=str(tmp_path),
        expected_output_files=["predictions.csv"],
    )
    assert result.success is False
    assert result.status == "INVALID_OUTPUT"
    assert "predictions.csv" in result.missing_expected_files


def test_python_runner_timeout(tmp_path):
    """PythonRunner cleanly handles timeouts with TIMEOUT status and timed_out=True."""
    runner = PythonRunner(default_timeout_seconds=1)
    code = """
import time
time.sleep(5)
"""
    result = runner.run_code(code, workspace_dir=str(tmp_path), timeout_seconds=1)
    assert result.success is False
    assert result.status == "TIMEOUT"
    assert result.timed_out is True
    assert "timed out" in result.stderr.lower()


def test_python_runner_writes_bash_script_artifact(tmp_path):
    """PythonRunner saves bash_script as setup.sh artifact in workspace."""
    runner = PythonRunner()
    code = "print('hello')"
    bash = "#!/bin/bash\necho 'setting up environment'"

    result = runner.run_code(code, workspace_dir=str(tmp_path), bash_script=bash)
    assert result.success is True
    setup_file = tmp_path / "setup.sh"
    assert setup_file.exists()
    assert "setting up environment" in setup_file.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 4. IterativeCodingOrchestrator Stage 4 Closed Loop
# ---------------------------------------------------------------------------


def test_orchestrator_closed_loop_judge_fix_then_finish():
    """
    Test full closed loop:
    Iteration 1: Executor fails -> Judge decides FIX -> ErrorAnalyzer runs -> Episodic Memory records FAIL
    Iteration 2: Executor succeeds -> Judge decides FINISH -> ErrorAnalyzer NOT called -> Loop terminates SUCCESS
    """
    client = get_llm_client(use_mock=True)
    coder = CoderAgent(llm_client=client)

    # Mock executor that fails on call 1, succeeds on call 2
    class MockLoopExecutor:
        def __init__(self):
            self.call_count = 0

        def process(self, art):
            self.call_count += 1
            if self.call_count == 1:
                return ExecutionResult(
                    success=False,
                    status="FAILURE",
                    return_code=1,
                    stderr="KeyError: 'targt'",
                )
            return ExecutionResult(
                success=True,
                status="SUCCESS",
                return_code=0,
                stdout="Evaluation complete. Accuracy: 0.85",
            )

    executor = MockLoopExecutor()
    analyzer = ErrorAnalyzerAgent(llm_client=client)
    judge = ExecutionJudgeAgent(llm_client=client)
    ep_mem = EpisodicMemory()
    sem_mem = SemanticMemory(llm_client=client)

    orch = IterativeCodingOrchestrator(
        coder=coder,
        executor=executor,
        error_analyzer=analyzer,
        judge=judge,
        max_iterations=3,
        episodic_memory=ep_mem,
        semantic_memory=sem_mem,
    )

    pctx = PerceptualContext()
    result = orch.process(pctx, run_id="test-judge-closed-loop")

    assert result.success is True
    assert result.total_iterations == 2
    assert len(result.iteration_history) == 2

    # Iteration 1 checks
    rec1 = result.iteration_history[0]
    assert rec1.judge_decision == "FIX"
    assert rec1.error_context is not None

    # Iteration 2 checks
    rec2 = result.iteration_history[1]
    assert rec2.judge_decision == "FINISH"
    assert rec2.error_context is None

    # Episodic memory checks
    episodes = ep_mem.get_history("test-judge-closed-loop")
    assert len(episodes) == 2
    assert episodes[0].status == "FAIL"
    assert episodes[0].judge_decision == "FIX"
    assert episodes[1].status == "SUCCESS"
    assert episodes[1].judge_decision == "FINISH"
    # Verify semantic memory was retrieved for iteration 2 based on error context
    assert episodes[1].retrieved_knowledge is not None


def test_orchestrator_terminates_at_max_iterations_with_persistent_fix():
    """Orchestrator cleanly terminates when Judge continually decides FIX up to max_iterations."""
    client = get_llm_client(use_mock=True)
    coder = CoderAgent(llm_client=client)

    class AlwaysFailMockExecutor:
        def process(self, art):
            return ExecutionResult(
                success=False,
                status="FAILURE",
                return_code=1,
                stderr="PersistentError: model diverges",
            )

    orch = IterativeCodingOrchestrator(
        coder=coder,
        executor=AlwaysFailMockExecutor(),
        error_analyzer=ErrorAnalyzerAgent(llm_client=client),
        judge=ExecutionJudgeAgent(llm_client=client),
        max_iterations=2,
    )

    result = orch.process(PerceptualContext())
    assert result.success is False
    assert result.total_iterations == 2
    assert result.final_error_context is not None
    assert all(r.judge_decision == "FIX" for r in result.iteration_history)
