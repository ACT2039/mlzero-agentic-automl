"""
Comprehensive unit tests for Stage 3: Paper-Faithful Episodic Memory Module.
All tests use MockLLMClient or mocked components (0 Gemini API quota consumed).
"""
import json
import uuid
from pathlib import Path

import pytest

from mlzero.agents.coder import CoderAgent, ErrorAnalyzerAgent
from mlzero.core.config import settings
from mlzero.core.llm import MockLLMClient
from mlzero.memory.episodic import EpisodicMemory
from mlzero.memory.episodic_store import EpisodicStore
from mlzero.orchestration.iterative import IterativeCodingOrchestrator
from mlzero.schemas.coder import (
    CodeArtifact,
    ErrorContext,
    ExecutionResult,
)
from mlzero.schemas.episodic import Episode, RunHistory
from mlzero.schemas.memory import KnowledgeChunk, RetrievedKnowledge
from mlzero.schemas.perception import (
    LibrarySelection,
    PerceptualContext,
    TaskContext,
)


@pytest.fixture
def temp_store(tmp_path: Path) -> EpisodicStore:
    return EpisodicStore(storage_dir=tmp_path / "episodes")


@pytest.fixture
def episodic_mem(temp_store: EpisodicStore) -> EpisodicMemory:
    return EpisodicMemory(store=temp_store)


def test_episode_schema_validation() -> None:
    """1. Test episode schema validation with full required fields."""
    ep = Episode(
        episode_id="ep-1",
        run_id="run-1",
        iteration=1,
        timestamp="2026-09-20T00:00:00Z",
        status="FAIL",
        generated_code="import pandas as pd",
        stdout="Loading data...",
        stderr="KeyError: 'targt'",
        exit_code=1,
        execution_duration=1.25,
        retrieval_query="Predict target classification",
        retrieved_knowledge=[{"chunk_id": "c1", "source": "quickstart.md"}],
        error_category="KeyError",
        error_summary="Target column 'targt' not found.",
        suggested_fix="Fix target column name.",
    )
    assert ep.run_id == "run-1"
    assert ep.iteration == 1
    assert ep.status == "FAIL"
    assert ep.exit_code == 1
    assert ep.execution_duration == 1.25
    assert ep.generated_code == "import pandas as pd"
    assert ep.code_content == "import pandas as pd" or ep.generated_code == ep.code_content
    assert ep.error_category == "KeyError"


def test_episode_json_serialization() -> None:
    """2. Test episode JSON serialization and round-trip parsing."""
    ep = Episode(
        episode_id=str(uuid.uuid4()),
        run_id="run-json",
        iteration=1,
        timestamp="2026-09-20T00:00:00Z",
        status="SUCCESS",
        generated_code="print('hello')",
        stdout="hello\n",
        exit_code=0,
        execution_duration=0.5,
    )
    json_str = ep.model_dump_json()
    data = json.loads(json_str)
    assert data["run_id"] == "run-json"
    assert data["exit_code"] == 0

    ep_loaded = Episode(**data)
    assert ep_loaded.run_id == ep.run_id
    assert ep_loaded.generated_code == ep.generated_code


def test_chronological_history(episodic_mem: EpisodicMemory) -> None:
    """3. Test that get_history preserves chronological iteration order."""
    run_id = "run-chrono"
    p_ctx = PerceptualContext(task=TaskContext(objective="Test", task_type="classification"))
    episodic_mem.start_run(run_id, p_ctx)

    for i in range(1, 4):
        status = "FAIL" if i < 3 else "SUCCESS"
        episodic_mem.record_iteration(
            run_id=run_id,
            iteration=i,
            status=status,
            artifact=CodeArtifact(code=f"code_iter_{i}"),
            result=ExecutionResult(success=(status == "SUCCESS"), return_code=0 if status == "SUCCESS" else 1),
        )

    history = episodic_mem.get_history(run_id)
    assert len(history) == 3
    assert [ep.iteration for ep in history] == [1, 2, 3]
    assert [ep.status for ep in history] == ["FAIL", "FAIL", "SUCCESS"]


def test_get_latest_context(episodic_mem: EpisodicMemory) -> None:
    """4. Test get_latest_context returns structured context for the latest iteration."""
    run_id = "run-latest"
    p_ctx = PerceptualContext(task=TaskContext(objective="Test", task_type="classification"))
    episodic_mem.start_run(run_id, p_ctx)

    err = ErrorContext(
        iteration=1,
        error_category="KeyError",
        error_summary="Target column not found",
        error_message="KeyError: 'targt'",
        suggested_fix="Use 'target'",
        stderr_excerpt="KeyError: 'targt'",
    )
    episodic_mem.record_iteration(
        run_id=run_id,
        iteration=1,
        status="FAIL",
        artifact=CodeArtifact(code="code_1"),
        result=ExecutionResult(success=False, return_code=1, stderr="KeyError: 'targt'"),
        error_ctx=err,
        retrieval_query="classification autogluon",
    )

    ctx = episodic_mem.get_latest_context(run_id)
    assert ctx["latest_iteration"] == 1
    assert ctx["latest_status"] == "FAIL"
    assert ctx["latest_code"] == "code_1"
    assert ctx["latest_error_summary"] == "Target column not found"
    assert ctx["latest_suggested_fix"] == "Use 'target'"
    assert ctx["latest_retrieval_query"] == "classification autogluon"


def test_code_artifacts_stored_for_all_iterations(episodic_mem: EpisodicMemory) -> None:
    """5. Test that code is stored for both failed and successful iterations."""
    run_id = "run-code-store"
    episodic_mem.start_run(run_id, PerceptualContext())

    art1 = CodeArtifact(code="failed_code_attempt()")
    res1 = ExecutionResult(success=False, return_code=1, stderr="Failure")
    episodic_mem.record_iteration(run_id, 1, "FAIL", artifact=art1, result=res1)

    art2 = CodeArtifact(code="successful_code_solution()")
    res2 = ExecutionResult(success=True, return_code=0, stdout="OK")
    episodic_mem.record_iteration(run_id, 2, "SUCCESS", artifact=art2, result=res2)

    history = episodic_mem.get_history(run_id)
    assert len(history) == 2
    assert history[0].generated_code == "failed_code_attempt()"
    assert history[0].code_content == "failed_code_attempt()"
    assert history[1].generated_code == "successful_code_solution()"
    assert history[1].code_content == "successful_code_solution()"


def test_execution_logs_stored(episodic_mem: EpisodicMemory) -> None:
    """6, 7, 8. Test stdout, stderr, and exit code storage."""
    run_id = "run-logs"
    episodic_mem.start_run(run_id, PerceptualContext())

    result = ExecutionResult(
        success=False,
        return_code=42,
        stdout="Starting data ingestion...\nData loaded: 100 rows.\n",
        stderr="Traceback: ValueError('Invalid schema')",
        duration_seconds=2.45,
    )
    episodic_mem.record_iteration(
        run_id,
        1,
        "FAIL",
        artifact=CodeArtifact(code="test()"),
        result=result,
    )

    ep = episodic_mem.get_history(run_id)[0]
    assert ep.exit_code == 42
    assert "Data loaded: 100 rows" in (ep.stdout or "")
    assert "ValueError('Invalid schema')" in (ep.stderr or "")
    assert ep.execution_duration == 2.45


def test_semantic_retrieval_stored(episodic_mem: EpisodicMemory) -> None:
    """9, 10. Test that semantic retrieval query and results are stored in episode."""
    run_id = "run-retrieval"
    episodic_mem.start_run(run_id, PerceptualContext())

    chunk = KnowledgeChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        content="Raw doc content",
        condensed_guidance="Use TabularPredictor with label parameter.",
        source="quickstart.md",
        library_name="autogluon.tabular",
    )
    retrieved = RetrievedKnowledge(
        chunks=[chunk],
        sources=["quickstart.md"],
        query_used="tabular classification error fix",
        selected_library="autogluon.tabular",
        iteration=1,
    )

    episodic_mem.record_iteration(
        run_id,
        1,
        "FAIL",
        artifact=CodeArtifact(code="code()"),
        result=ExecutionResult(success=False, return_code=1),
        retrieved_knowledge=retrieved,
        retrieval_query="tabular classification error fix",
    )

    ep = episodic_mem.get_history(run_id)[0]
    assert ep.retrieval_query == "tabular classification error fix"
    assert ep.retrieved_knowledge_references == ["quickstart.md"]
    assert ep.retrieved_knowledge is not None
    assert len(ep.retrieved_knowledge) == 1
    assert ep.retrieved_knowledge[0]["chunk_id"] == "chunk-1"
    assert "Use TabularPredictor" in ep.retrieved_knowledge[0]["guidance"]


def test_error_context_stored(episodic_mem: EpisodicMemory) -> None:
    """11. Test that ErrorContext fields (category, summary, fix) are stored."""
    run_id = "run-err-ctx"
    episodic_mem.start_run(run_id, PerceptualContext())

    err = ErrorContext(
        iteration=1,
        error_category="DataQuality",
        error_summary="Non-numeric string in numeric price column",
        error_message="ValueError: could not convert string to float: 'not_available'",
        suggested_fix="Coerce numeric values and drop invalid target rows",
        stderr_excerpt="ValueError: could not convert string to float",
    )

    episodic_mem.record_iteration(
        run_id,
        1,
        "FAIL",
        artifact=CodeArtifact(code="fit()"),
        result=ExecutionResult(success=False, return_code=1),
        error_ctx=err,
    )

    ep = episodic_mem.get_history(run_id)[0]
    assert ep.error_category == "DataQuality"
    assert ep.error_summary == "Non-numeric string in numeric price column"
    assert ep.suggested_fix == "Coerce numeric values and drop invalid target rows"
    assert ep.error_context_summary is not None
    assert ep.error_context_summary["error_category"] == "DataQuality"


def test_bounded_storage_limits(episodic_mem: EpisodicMemory) -> None:
    """12. Test that gigantic code, stdout, and stderr are safely bounded."""
    run_id = "run-bounded"
    episodic_mem.start_run(run_id, PerceptualContext())

    huge_code = "x = 1\n" * 5000  # 30,000 chars
    huge_stdout = "log message\n" * 2000  # 24,000 chars
    huge_stderr = "error line\n" * 2000  # 22,000 chars

    episodic_mem.record_iteration(
        run_id,
        1,
        "FAIL",
        artifact=CodeArtifact(code=huge_code),
        result=ExecutionResult(success=False, return_code=1, stdout=huge_stdout, stderr=huge_stderr),
    )

    ep = episodic_mem.get_history(run_id)[0]
    assert len(ep.generated_code or "") <= settings.episodic.max_stored_code_chars + 100
    assert len(ep.stdout or "") <= settings.episodic.max_stdout_chars + 10
    assert len(ep.stderr or "") <= settings.episodic.max_stderr_chars + 10


def test_error_analyzer_structured_output_with_episodic_inputs() -> None:
    """13, 15, 16. Test ErrorAnalyzer produces structured diagnosis given code, logs, and knowledge."""
    client = MockLLMClient()
    analyzer = ErrorAnalyzerAgent(llm_client=client)

    artifact = CodeArtifact(code="TabularPredictor(label='targt').fit(train_df)")
    result = ExecutionResult(
        success=False,
        return_code=1,
        stderr="KeyError: 'targt'",
        error_info="KeyError: 'targt'",
    )
    p_ctx = PerceptualContext(
        task=TaskContext(objective="Classification", task_type="classification", target_column="target"),
        library=LibrarySelection(selected_library="autogluon.tabular", confidence="high", explanation="test"),
    )
    retrieved = RetrievedKnowledge(
        chunks=[KnowledgeChunk(chunk_id="c1", document_id="d1", content="Check label spelling", source="quickstart.md")],
        sources=["quickstart.md"],
    )

    err_ctx = analyzer.process(
        artifact=artifact,
        result=result,
        iteration=1,
        perceptual_context=p_ctx,
        selected_library="autogluon.tabular",
        retrieved_knowledge=retrieved,
    )

    assert isinstance(err_ctx, ErrorContext)
    assert err_ctx.iteration == 1
    assert err_ctx.error_category == "KeyError"
    assert err_ctx.error_summary is not None
    assert "target" in err_ctx.suggested_fix.lower()


def test_feedback_loop_previous_error_passed_to_next_iteration(tmp_path: Path) -> None:
    """14, 18. Test full feedback loop: Iteration 1 failure -> Episodic record -> Iteration 2 recovery."""
    store = EpisodicStore(storage_dir=tmp_path / "loop_store")
    mem = EpisodicMemory(store=store)

    client = MockLLMClient()
    coder = CoderAgent(llm_client=client)
    analyzer = ErrorAnalyzerAgent(llm_client=client)

    # Mock executor: fails iteration 1 with KeyError 'targt', succeeds iteration 2 with 'target'
    class DynamicExecutor:
        def process(self, art: CodeArtifact) -> ExecutionResult:
            if "targt" in art.code:
                return ExecutionResult(success=False, return_code=1, stderr="KeyError: 'targt'")
            return ExecutionResult(success=True, return_code=0, stdout="Training complete. Accuracy: 0.85")

    orchestrator = IterativeCodingOrchestrator(
        coder=coder,
        executor=DynamicExecutor(),
        error_analyzer=analyzer,
        max_iterations=3,
        episodic_memory=mem,
    )

    p_ctx = PerceptualContext(
        task=TaskContext(
            objective="Train classification",
            task_type="classification",
            target_column="target",
            input_data_files=["tests/data/tiny_classification/train.csv"],
        ),
        library=LibrarySelection(selected_library="autogluon.tabular", confidence="high", explanation="Tabular"),
    )

    res = orchestrator.process(perceptual_context=p_ctx, run_id="test-feedback-loop")

    assert res.success is True
    assert res.total_iterations == 2

    # Check episodic storage
    history = mem.get_history("test-feedback-loop")
    assert len(history) == 2

    # Iteration 1: recorded as FAIL with code and error context
    assert history[0].iteration == 1
    assert history[0].status == "FAIL"
    assert history[0].error_category == "KeyError"
    assert "targt" in (history[0].generated_code or "")
    assert history[0].error_context_before is None

    # Iteration 2: recorded as SUCCESS, with error_context_before matching iteration 1 error
    assert history[1].iteration == 2
    assert history[1].status == "SUCCESS"
    assert history[1].error_context_before is not None
    assert history[1].error_context_before["error_category"] == "KeyError"
    assert "target" in (history[1].generated_code or "")


def test_atomic_file_storage(tmp_path: Path) -> None:
    """Test that EpisodicStore uses atomic writes and does not leave temporary files."""
    store = EpisodicStore(storage_dir=tmp_path / "atomic_store")
    run = RunHistory(
        run_id="run-atomic",
        start_time="2026-09-20T00:00:00Z",
    )
    store.save_run_history(run)

    run_path = store._get_run_path("run-atomic")
    tmp_file = run_path.with_suffix(".tmp")

    assert run_path.exists()
    assert not tmp_file.exists()  # .tmp should be cleanly replaced

    loaded = store.get_run_history("run-atomic")
    assert loaded is not None
    assert loaded.run_id == "run-atomic"
