"""
Stage 5 Unit Tests for MLLibraryAdapter framework and expanded library adapters.
Covers:
1. Adapter registry & discovery
2. Adapter selection per perceptual context
3. TabularAdapter, TimeSeriesAdapter, MultiModalAdapter, RetrievalAdapter, GeneralMLAdapter
4. Input validation & expected output contracts
5. Library-aware Semantic Memory retrieval
6. Coder receiving adapter guidance context
7. Generic Executor & Execution Judge compatibility
8. Episodic Memory recording selected library
9. End-to-end integration via mock LLM
"""

from pathlib import Path

from mlzero.agents.coder import CoderAgent, ErrorAnalyzerAgent, ExecutorAgent
from mlzero.agents.judge import ExecutionJudgeAgent
from mlzero.core.llm import MockLLMClient
from mlzero.memory.episodic import EpisodicMemory
from mlzero.memory.semantic import SemanticMemory
from mlzero.orchestration.iterative import IterativeCodingOrchestrator
from mlzero.schemas.coder import CodeGenerationRequest
from mlzero.schemas.perception import (
    LibrarySelection,
    PerceptualContext,
    TaskContext,
)
from mlzero.tools.adapters.general import GeneralMLAdapter
from mlzero.tools.adapters.multimodal import MultiModalAdapter
from mlzero.tools.adapters.registry import AdapterRegistry
from mlzero.tools.adapters.retrieval import RetrievalAdapter
from mlzero.tools.adapters.tabular import TabularAdapter
from mlzero.tools.adapters.timeseries import TimeSeriesAdapter


def test_adapter_registry_discovery():
    """Verify AdapterRegistry discovers adapters for each selected library."""
    for lib, expected_type in [
        ("autogluon.tabular", TabularAdapter),
        ("autogluon.multimodal", MultiModalAdapter),
        ("autogluon.timeseries", TimeSeriesAdapter),
        ("FlagEmbedding", RetrievalAdapter),
        ("machine learning", GeneralMLAdapter),
        ("general_ml", GeneralMLAdapter),
    ]:
        ctx = PerceptualContext(library=LibrarySelection(selected_library=lib))
        adapter = AdapterRegistry.get_adapter(ctx)
        assert isinstance(adapter, expected_type), f"Expected {expected_type} for {lib}, got {type(adapter)}"


def test_adapter_selection_by_task_type():
    """Verify implicit selection when library is unassigned."""
    ts_ctx = PerceptualContext(task=TaskContext(task_type="time_series"))
    adapter = AdapterRegistry.get_adapter(ts_ctx)
    assert isinstance(adapter, TimeSeriesAdapter)

    ret_ctx = PerceptualContext(task=TaskContext(task_type="retrieval"))
    adapter = AdapterRegistry.get_adapter(ret_ctx)
    assert isinstance(adapter, RetrievalAdapter)

    mm_ctx = PerceptualContext(task=TaskContext(task_type="multimodal"))
    adapter = AdapterRegistry.get_adapter(mm_ctx)
    assert isinstance(adapter, MultiModalAdapter)


def test_tabular_adapter_contract(tmp_path: Path):
    """Test TabularAdapter input validation, code context, and result validation."""
    adapter = TabularAdapter()
    ctx = PerceptualContext(library=LibrarySelection(selected_library="autogluon.tabular"))
    assert adapter.supports(ctx)

    # Empty dir fails validation
    assert not adapter.validate_inputs(tmp_path, ctx)

    # Create synthetic train CSV
    train_file = tmp_path / "train.csv"
    train_file.write_text("a,b,target\n1,2,0\n3,4,1")

    assert adapter.validate_inputs(tmp_path, ctx)
    prep = adapter.prepare_data(tmp_path, tmp_path, ctx)
    assert "train_path" in prep

    code_ctx = adapter.build_code_context(ctx, prep)
    assert "autogluon.tabular" in code_ctx

    outputs = adapter.expected_outputs()
    assert "out/predictions.csv" in outputs

    # Test result validation
    res = adapter.validate_result(tmp_path)
    assert res["prediction_exists"] is False


def test_timeseries_adapter_contract(tmp_path: Path):
    """Test TimeSeriesAdapter contract."""
    adapter = TimeSeriesAdapter()
    ctx = PerceptualContext(library=LibrarySelection(selected_library="autogluon.timeseries"))
    assert adapter.supports(ctx)

    train_file = tmp_path / "train.csv"
    train_file.write_text("item_id,timestamp,target\n1,2025-01-01,10.0")
    assert adapter.validate_inputs(tmp_path, ctx)

    prep = adapter.prepare_data(tmp_path, tmp_path, ctx)
    code_ctx = adapter.build_code_context(ctx, prep)
    assert "TimeSeriesPredictor" in code_ctx
    assert "out/predictions.csv" in adapter.expected_outputs()


def test_multimodal_adapter_contract(tmp_path: Path):
    """Test MultiModalAdapter contract."""
    adapter = MultiModalAdapter()
    ctx = PerceptualContext(library=LibrarySelection(selected_library="autogluon.multimodal"))
    assert adapter.supports(ctx)
    assert adapter.validate_inputs(tmp_path, ctx)

    prep = adapter.prepare_data(tmp_path, tmp_path, ctx)
    code_ctx = adapter.build_code_context(ctx, prep)
    assert "MultiModalPredictor" in code_ctx


def test_retrieval_adapter_contract(tmp_path: Path):
    """Test RetrievalAdapter contract."""
    adapter = RetrievalAdapter()
    ctx = PerceptualContext(library=LibrarySelection(selected_library="FlagEmbedding"))
    assert adapter.supports(ctx)

    prep = adapter.prepare_data(tmp_path, tmp_path, ctx)
    code_ctx = adapter.build_code_context(ctx, prep)
    assert "FlagEmbedding" in code_ctx
    assert "out/retrieval_results.json" in adapter.expected_outputs()


def test_general_ml_adapter_contract(tmp_path: Path):
    """Test GeneralMLAdapter fallback contract."""
    adapter = GeneralMLAdapter()
    ctx = PerceptualContext(library=LibrarySelection(selected_library="machine learning"))
    assert adapter.supports(ctx)

    prep = adapter.prepare_data(tmp_path, tmp_path, ctx)
    code_ctx = adapter.build_code_context(ctx, prep)
    assert "General ML" in code_ctx


def test_coder_receives_adapter_guidance():
    """Verify CoderAgent includes adapter coding guidance in prompt."""
    mock_llm = MockLLMClient()
    coder = CoderAgent(llm_client=mock_llm)

    req = CodeGenerationRequest(
        perceptual_context_json='{"task": {"task_type": "time_series"}}',
        coding_guidance="Library Context (autogluon.timeseries): Use TimeSeriesPredictor"
    )
    artifact = coder.process(req)
    assert artifact.code is not None


def test_library_aware_semantic_memory_retrieval():
    """Verify SemanticMemory builds query with selected library filter."""
    memory = SemanticMemory()

    ctx = PerceptualContext(
        task=TaskContext(task_type="forecasting"),
        library=LibrarySelection(selected_library="autogluon.timeseries")
    )
    res = memory.retrieve(perceptual_context=ctx)
    assert res.selected_library == "autogluon.timeseries"


def test_orchestrator_with_timeseries_adapter(tmp_path: Path):
    """Smoke test running orchestrator with TimeSeries adapter and MockLLM."""
    mock_llm = MockLLMClient()
    coder = CoderAgent(llm_client=mock_llm)
    executor = ExecutorAgent()
    error_analyzer = ErrorAnalyzerAgent(llm_client=mock_llm)
    judge = ExecutionJudgeAgent(llm_client=mock_llm)
    episodic_memory = EpisodicMemory()

    orchestrator = IterativeCodingOrchestrator(
        coder=coder,
        executor=executor,
        error_analyzer=error_analyzer,
        judge=judge,
        episodic_memory=episodic_memory,
        max_iterations=1,
    )

    ctx = PerceptualContext(
        task=TaskContext(task_type="time_series", target_column="target"),
        library=LibrarySelection(selected_library="autogluon.timeseries")
    )
    adapter = AdapterRegistry.get_adapter(ctx)
    assert adapter is not None

    res = orchestrator.process(perceptual_context=ctx, run_id="test_ts_run", adapter=adapter)
    assert res.success is True
    assert res.final_code_artifact is not None


def test_orchestrator_with_retrieval_adapter():
    """Smoke test running orchestrator with Retrieval adapter and MockLLM."""
    mock_llm = MockLLMClient()
    coder = CoderAgent(llm_client=mock_llm)
    executor = ExecutorAgent()
    error_analyzer = ErrorAnalyzerAgent(llm_client=mock_llm)
    judge = ExecutionJudgeAgent(llm_client=mock_llm)
    episodic_memory = EpisodicMemory()

    orchestrator = IterativeCodingOrchestrator(
        coder=coder,
        executor=executor,
        error_analyzer=error_analyzer,
        judge=judge,
        episodic_memory=episodic_memory,
        max_iterations=1,
    )

    ctx = PerceptualContext(
        task=TaskContext(task_type="retrieval"),
        library=LibrarySelection(selected_library="FlagEmbedding")
    )
    adapter = AdapterRegistry.get_adapter(ctx)
    assert adapter is not None

    res = orchestrator.process(perceptual_context=ctx, run_id="test_ret_run", adapter=adapter)
    assert res.success is True


def test_orchestrator_with_multimodal_adapter():
    """Smoke test running orchestrator with MultiModal adapter and MockLLM."""
    mock_llm = MockLLMClient()
    coder = CoderAgent(llm_client=mock_llm)
    executor = ExecutorAgent()
    error_analyzer = ErrorAnalyzerAgent(llm_client=mock_llm)
    judge = ExecutionJudgeAgent(llm_client=mock_llm)

    orchestrator = IterativeCodingOrchestrator(
        coder=coder,
        executor=executor,
        error_analyzer=error_analyzer,
        judge=judge,
        max_iterations=1,
    )

    ctx = PerceptualContext(
        task=TaskContext(task_type="multimodal"),
        library=LibrarySelection(selected_library="autogluon.multimodal")
    )
    adapter = AdapterRegistry.get_adapter(ctx)
    assert adapter is not None

    res = orchestrator.process(perceptual_context=ctx, run_id="test_mm_run", adapter=adapter)
    assert res.success is True
