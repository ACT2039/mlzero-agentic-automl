"""Tests for coder and executor modules."""

import os

import pytest

from mlzero.agents.coder import CoderAgent, ExecutorAgent
from mlzero.core.llm import get_llm_client
from mlzero.schemas.coder import CodeArtifact, CodeGenerationRequest
from mlzero.tools.python_runner import PythonRunner


def test_code_generation_request_validation():
    req = CodeGenerationRequest(perceptual_context_json="{}")
    assert req.perceptual_context_json == "{}"
    assert req.user_instruction is None
    
    with pytest.raises(ValueError):
        CodeGenerationRequest()


def test_code_artifact_validation():
    art = CodeArtifact(code="print('hi')")
    assert art.code == "print('hi')"
    assert art.language == "python"
    
    with pytest.raises(ValueError):
        CodeArtifact()


def test_coder_agent_mock():
    client = get_llm_client(use_mock=True)
    agent = CoderAgent(llm_client=client)
    
    req = CodeGenerationRequest(perceptual_context_json="{}")
    artifact = agent.process(req)
    
    assert isinstance(artifact, CodeArtifact)
    assert artifact.code.startswith("import csv")
    assert artifact.language == "python"


def test_executor_success():
    runner = PythonRunner()
    agent = ExecutorAgent(runner=runner)
    
    artifact = CodeArtifact(code="print('Hello World')\nimport os\nos.makedirs('out', exist_ok=True)\nwith open('out/test.txt', 'w') as f: f.write('OK')")
    result = agent.process(artifact)
    
    assert result.success is True
    assert result.return_code == 0
    assert "Hello World" in result.stdout
    assert "test.txt" in result.output_files


def test_executor_failure():
    runner = PythonRunner()
    agent = ExecutorAgent(runner=runner)
    
    artifact = CodeArtifact(code="raise ValueError('Intentional error')")
    result = agent.process(artifact)
    
    assert result.success is False
    assert result.return_code == 1
    assert "ValueError: Intentional error" in result.stderr


def test_executor_timeout(monkeypatch):
    from mlzero.core.config import settings
    monkeypatch.setattr(settings.execution, "timeout_seconds", 1)
    
    runner = PythonRunner()
    agent = ExecutorAgent(runner=runner)
    
    artifact = CodeArtifact(code="import time\ntime.sleep(2)")
    result = agent.process(artifact)
    
    assert result.success is False
    assert result.error_info is not None
    assert "Execution timed out" in result.error_info


def test_executor_empty_code():
    agent = ExecutorAgent()
    artifact = CodeArtifact(code="   \n  ")
    result = agent.process(artifact)
    assert result.success is False
    assert "empty" in result.error_info



@pytest.mark.skip(reason="Phase 7 relaxed security isolation to support AutoGluon")
def test_executor_security_isolation(monkeypatch):
    """Test that execution does not have access to specific env vars."""
    os.environ["OPENAI_API_KEY"] = "secret_key"
    
    runner = PythonRunner()
    agent = ExecutorAgent(runner=runner)
    
    code = "import os\nprint('KEY:', os.environ.get('OPENAI_API_KEY', 'NOT_FOUND'))"
    artifact = CodeArtifact(code=code)
    
    result = agent.process(artifact)
    assert "KEY: NOT_FOUND" in result.stdout
    
    # Cleanup
    del os.environ["OPENAI_API_KEY"]


@pytest.mark.skip(reason="Mock LLM behavior changed in Phase 7")
def test_mock_code_execution_with_input_file():
    runner = PythonRunner()
    agent = ExecutorAgent(runner=runner)
    
    # Mock code from LLMClient mock requires a data.csv input and expects a retry for success
    client = get_llm_client(use_mock=True)
    coder = CoderAgent(llm_client=client)
    
    # We pass previous_code to trigger the "PREVIOUS FAILURE" success branch in mock LLM
    artifact = coder.process(CodeGenerationRequest(perceptual_context_json="{}", previous_code="fail", error_context_json="{}"))
    
    input_files = {"data.csv": "id,val\n1,10\n2,20"}
    
    result = agent.process(artifact, input_files=input_files)
    
    assert result.success is True
    assert "result.txt" in result.output_files
