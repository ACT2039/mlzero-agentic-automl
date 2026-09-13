"""Tests for Orchestration (Phase 4)."""
import json

from mlzero.agents.coder import CoderAgent, ErrorAnalyzerAgent, ExecutorAgent
from mlzero.core.llm import get_llm_client
from mlzero.orchestration.iterative import IterativeCodingOrchestrator
from mlzero.schemas.coder import (
    CodeArtifact,
    CodeGenerationRequest,
    ErrorContext,
    ExecutionResult,
)
from mlzero.schemas.orchestration import IterationRecord
from mlzero.schemas.perception import PerceptualContext
from mlzero.tools.python_runner import PythonRunner


def test_error_context_validation():
    ctx = ErrorContext(
        iteration=1,
        error_category="syntax",
        error_message="SyntaxError",
        stderr_excerpt="...",
        suggested_fix="Fix syntax"
    )
    assert ctx.iteration == 1
    assert ctx.error_category == "syntax"


def test_iteration_record_serialization():
    rec = IterationRecord(iteration_number=1, duration_seconds=1.5)
    data = json.loads(rec.model_dump_json())
    assert data["iteration_number"] == 1
    assert data["duration_seconds"] == 1.5


def test_error_analyzer_mock_llm():
    client = get_llm_client(use_mock=True)
    analyzer = ErrorAnalyzerAgent(llm_client=client)
    art = CodeArtifact(code="print(x)")
    res = ExecutionResult(success=False, stderr="NameError: name 'x' is not defined", error_info="crash")
    
    ctx = analyzer.process(art, res, iteration=2)
    assert isinstance(ctx, ErrorContext)
    # The mock returns specific deterministic values:
    assert ctx.error_category == "file_not_found"
    assert ctx.iteration == 2


def test_orchestrator_success_first_try(monkeypatch):
    """Test one successful iteration."""
    # We patch CoderAgent to return success immediately
    client = get_llm_client(use_mock=True)
    coder = CoderAgent(llm_client=client)
    
    # Monkeypatch coder to bypass mock's "PREVIOUS FAILURE" logic and just return good code
    def mock_process(req):
        return CodeArtifact(code="print('success')", language="python")
    monkeypatch.setattr(coder, "process", mock_process)
    
    executor = ExecutorAgent(runner=PythonRunner())
    analyzer = ErrorAnalyzerAgent(llm_client=client)
    
    orch = IterativeCodingOrchestrator(coder, executor, analyzer, max_iterations=3)
    p_ctx = PerceptualContext()
    
    result = orch.process(p_ctx)
    assert result.success is True
    assert result.total_iterations == 1
    assert len(result.iteration_history) == 1
    assert result.final_error_context is None


def test_orchestrator_failure_then_success(tmp_path):
    """Test failure followed by successful retry."""
    # The MockLLMClient is designed exactly for this: Iter1 = fail code, Iter2 = success code
    client = get_llm_client(use_mock=True)
    coder = CoderAgent(llm_client=client)
    ExecutorAgent(runner=PythonRunner())
    analyzer = ErrorAnalyzerAgent(llm_client=client)
    
    # We need a data.csv so the iteration 2 succeeds
    (tmp_path / "data.csv").write_text("id,val\n1,10\n2,20")
    
    # Because ExecutorAgent runs in a temp dir, it won't see data.csv unless passed.
    # But wait, our PythonRunner tests use `cwd=temp_path`. The mock LLM expects `data.csv`.
    # Let's monkeypatch PythonRunner to run in tmp_path or pass input_files.
    # For testing the loop, we can just monkeypatch the Executor.
    
    class FakeExecutor:
        def __init__(self):
            self.calls = 0
        def process(self, art):
            self.calls += 1
            if self.calls == 1:
                return ExecutionResult(success=False, return_code=1, stderr="FileNotFound")
            return ExecutionResult(success=True, return_code=0, stdout="30")
            
    orch = IterativeCodingOrchestrator(coder, FakeExecutor(), analyzer, max_iterations=3)
    result = orch.process(PerceptualContext())
    
    assert result.success is True
    assert result.total_iterations == 2
    assert len(result.iteration_history) == 2
    assert result.iteration_history[0].execution_result.success is False
    assert result.iteration_history[1].execution_result.success is True


def test_orchestrator_max_iterations():
    """Test persistent failure hits max iterations."""
    client = get_llm_client(use_mock=True)
    coder = CoderAgent(llm_client=client)
    
    # Always fail
    class AlwaysFailExecutor:
        def process(self, art):
            return ExecutionResult(success=False, return_code=1, stderr="Fail")
            
    orch = IterativeCodingOrchestrator(coder, AlwaysFailExecutor(), ErrorAnalyzerAgent(llm_client=client), max_iterations=3)
    result = orch.process(PerceptualContext())
    
    assert result.success is False
    assert result.total_iterations == 3
    assert result.final_error_context is not None


def test_orchestrator_coder_crash():
    """Test coder crash handled safely."""
    client = get_llm_client(use_mock=True)
    coder = CoderAgent(llm_client=client)
    
    def fail_process(req):
        raise ValueError("Coder crashed")
    coder.process = fail_process
    
    orch = IterativeCodingOrchestrator(coder, ExecutorAgent(), ErrorAnalyzerAgent(llm_client=client), max_iterations=3)
    result = orch.process(PerceptualContext())
    
    assert result.success is False
    assert result.total_iterations == 0


def test_preservation_of_task_context():
    """Test preservation of original task context."""
    req = CodeGenerationRequest(perceptual_context_json='{"test": 1}', user_instruction="do it")
    assert "do it" in req.user_instruction
    assert '{"test": 1}' in req.perceptual_context_json
