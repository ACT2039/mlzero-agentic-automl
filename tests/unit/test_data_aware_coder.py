"""
Unit and integration tests for Data-Aware Coder, ErrorAnalyzer, and MockLLM recovery.
"""
import json
from pathlib import Path

from mlzero.agents.coder import CoderAgent, ErrorAnalyzerAgent, ExecutorAgent
from mlzero.core.llm import MockLLMClient, get_llm_client
from mlzero.orchestration.iterative import IterativeCodingOrchestrator
from mlzero.schemas.coder import (
    CodeArtifact,
    CodeGenerationRequest,
    ErrorContext,
    ExecutionResult,
)
from mlzero.schemas.perception import (
    DataQualityReport,
    FileContext,
    FileMetadata,
    InvalidNumericInfo,
    LibrarySelection,
    MissingValueInfo,
    PerceptualContext,
    TaskContext,
)


def _make_house_price_perceptual_context() -> PerceptualContext:
    """Helper to build a PerceptualContext representing house_price_faulty."""
    dq = DataQualityReport(
        target_column="price",
        task_type="regression",
        row_count=65,
        column_names=["house_id", "area", "bedrooms", "bathrooms", "location_score", "age", "price"],
        missing_values=[MissingValueInfo(column="bedrooms", missing_count=1, missing_pct=1.54)],
        invalid_numeric_values=[
            InvalidNumericInfo(column="area", invalid_values=["1200 sqft", "unknown"], invalid_count=2),
            InvalidNumericInfo(column="location_score", invalid_values=["high"], invalid_count=1),
            InvalidNumericInfo(column="price", invalid_values=["not_available"], invalid_count=1),
        ],
        malformed_target_values=["not_available"],
        candidate_targets=["price"],
    )
    train_abs = str(Path("tests/mlzero_faulty_datasets/house_price_faulty/train.csv").resolve())
    test_abs = str(Path("tests/mlzero_faulty_datasets/house_price_faulty/test.csv").resolve())
    task = TaskContext(
        objective="Predict house price",
        task_type="regression",
        target_column="price",
        input_data_files=[train_abs, test_abs],
        data_quality=dq,
    )
    lib = LibrarySelection(
        selected_library="autogluon.tabular",
        confidence="high",
        explanation="Tabular regression.",
    )
    meta = FileMetadata(
        path="train.csv",
        absolute_path=train_abs,
        extension=".csv",
        size_bytes=1851,
        file_type="tabular",
    )
    fc = FileContext(metadata=meta, columns=dq.column_names, row_count=65)
    return PerceptualContext(files=[fc], task=task, library=lib, data_quality=dq)


def _make_tiny_classification_perceptual_context() -> PerceptualContext:
    """Helper to build a PerceptualContext representing tiny_classification."""
    dq = DataQualityReport(
        target_column="target",
        task_type="classification",
        row_count=4,
        column_names=["feature1", "feature2", "target"],
        candidate_targets=["target"],
    )
    train_abs = str(Path("tests/data/tiny_classification/train.csv").resolve())
    test_abs = str(Path("tests/data/tiny_classification/test.csv").resolve())
    task = TaskContext(
        objective="Mock Objective",
        task_type="classification",
        target_column="target",
        input_data_files=[train_abs, test_abs],
        data_quality=dq,
    )
    lib = LibrarySelection(
        selected_library="autogluon.tabular",
        confidence="high",
        explanation="Tabular classification.",
    )
    meta = FileMetadata(
        path="train.csv",
        absolute_path=str(Path("tests/data/tiny_classification/train.csv").resolve()),
        extension=".csv",
        size_bytes=66,
        file_type="tabular",
    )
    fc = FileContext(metadata=meta, columns=dq.column_names, row_count=4)
    return PerceptualContext(files=[fc], task=task, library=lib, data_quality=dq)


# ===========================================================================
# A. Coder Tests
# ===========================================================================

def test_coder_uses_actual_target_column():
    """CoderAgent must use the target column reported by Perception, not hardcoded 'target'."""
    client = get_llm_client(use_mock=True)
    coder = CoderAgent(llm_client=client)
    pctx = _make_house_price_perceptual_context()

    req = CodeGenerationRequest(
        perceptual_context_json=pctx.model_dump_json(),
        user_instruction="Predict house price",
    )
    artifact = coder.process(req)

    # First iteration on house_price_faulty must use 'price', not 'target' or 'targt'
    assert "label='price'" in artifact.code
    assert "label='target'" not in artifact.code
    assert "label='targt'" not in artifact.code


def test_coder_regression_problem_type():
    """CoderAgent must configure problem_type='regression' for regression tasks."""
    client = get_llm_client(use_mock=True)
    coder = CoderAgent(llm_client=client)
    pctx = _make_house_price_perceptual_context()

    req = CodeGenerationRequest(perceptual_context_json=pctx.model_dump_json())
    artifact = coder.process(req)

    assert "problem_type='regression'" in artifact.code


def test_coder_includes_data_quality_in_prompt():
    """CoderAgent prompt must include data quality findings."""
    recorded_prompts = []

    class CapturingMockLLM(MockLLMClient):
        def generate_structured(self, prompt, schema):
            recorded_prompts.append(prompt)
            return super().generate_structured(prompt, schema)

    client = CapturingMockLLM()
    coder = CoderAgent(llm_client=client)
    pctx = _make_house_price_perceptual_context()

    req = CodeGenerationRequest(perceptual_context_json=pctx.model_dump_json())
    coder.process(req)

    assert len(recorded_prompts) == 1
    assert "Target Column: price" in recorded_prompts[0]
    assert "Task Type: regression" in recorded_prompts[0]
    assert "Data Quality Findings" in recorded_prompts[0]


# ===========================================================================
# B. MockLLM Tests
# ===========================================================================

def test_mock_llm_tiny_classification_preserves_targt_to_target():
    """tiny_classification must retain the intentional 'targt' -> 'target' recovery."""
    client = get_llm_client(use_mock=True)
    coder = CoderAgent(llm_client=client)
    pctx = _make_tiny_classification_perceptual_context()

    # Iteration 1: should generate typo 'targt'
    req1 = CodeGenerationRequest(perceptual_context_json=pctx.model_dump_json())
    art1 = coder.process(req1)
    assert "label='targt'" in art1.code

    # Iteration 2: with previous failure, should fix to 'target'
    err_ctx = ErrorContext(
        iteration=1,
        error_category="KeyError",
        error_message="KeyError: 'targt' not found in DataFrame.",
        stderr_excerpt="KeyError: 'targt'",
        suggested_fix="Correct the label column name to 'target'.",
    )
    req2 = CodeGenerationRequest(
        perceptual_context_json=pctx.model_dump_json(),
        previous_code=art1.code,
        error_context_json=err_ctx.model_dump_json(),
    )
    art2 = coder.process(req2)
    assert "label='target'" in art2.code
    assert "label='targt'" not in art2.code


def test_mock_llm_house_price_retry_includes_preprocessing():
    """Iteration 2 for house_price_faulty must generate data cleaning code."""
    client = get_llm_client(use_mock=True)
    coder = CoderAgent(llm_client=client)
    pctx = _make_house_price_perceptual_context()

    # Iteration 1: uncleaned baseline
    req1 = CodeGenerationRequest(perceptual_context_json=pctx.model_dump_json())
    art1 = coder.process(req1)
    assert "label='price'" in art1.code
    assert "dropna" not in art1.code

    # Iteration 2: after DataQuality error
    err_ctx = ErrorContext(
        iteration=1,
        error_category="DataQuality",
        error_message="Non-numeric values in numeric/target columns.",
        stderr_excerpt="ValueError: could not convert string to float: 'not_available'",
        suggested_fix="Coerce invalid numeric values to NaN, drop rows with invalid target values, and impute missing feature values before training.",
    )
    req2 = CodeGenerationRequest(
        perceptual_context_json=pctx.model_dump_json(),
        previous_code=art1.code,
        error_context_json=err_ctx.model_dump_json(),
    )
    art2 = coder.process(req2)

    # Must contain data cleaning operations
    assert "to_numeric" in art2.code
    assert "dropna" in art2.code
    assert "fillna" in art2.code
    assert "label='price'" in art2.code


# ===========================================================================
# C. ErrorAnalyzer Tests
# ===========================================================================

def test_error_analyzer_keyerror_with_custom_target():
    """KeyError analysis must use the actual detected target column from perceptual context."""
    client = get_llm_client(use_mock=True)
    analyzer = ErrorAnalyzerAgent(llm_client=client)
    pctx = _make_house_price_perceptual_context()

    art = CodeArtifact(code="predictor.fit(df)")
    res = ExecutionResult(
        success=False,
        stderr="KeyError: 'pric' not found in columns",
        error_info="crash",
    )

    ctx = analyzer.process(art, res, iteration=1, perceptual_context=pctx)
    assert ctx.error_category == "KeyError"
    assert "price" in ctx.suggested_fix
    # Must NOT say 'target' when the target is 'price'
    assert "Correct the label column name to 'price'." == ctx.suggested_fix


def test_error_analyzer_data_quality_error():
    """Conversion errors must be classified as DataQuality with appropriate fix suggestion."""
    client = get_llm_client(use_mock=True)
    analyzer = ErrorAnalyzerAgent(llm_client=client)
    pctx = _make_house_price_perceptual_context()

    art = CodeArtifact(code="predictor.fit(df)")
    res = ExecutionResult(
        success=False,
        stderr="ValueError: could not convert string to float: 'not_available'\nAssertionError: Trainer has no fit models that can infer.",
        error_info="crash",
    )

    ctx = analyzer.process(art, res, iteration=1, perceptual_context=pctx)
    assert ctx.error_category == "DataQuality"
    assert "Coerce" in ctx.suggested_fix or "NaN" in ctx.suggested_fix


# ===========================================================================
# D. Orchestration Tests
# ===========================================================================

def test_orchestration_passes_error_context_to_retry():
    """IterativeCodingOrchestrator must pass error context from iteration 1 into iteration 2 request."""
    captured_requests = []

    class MockCoderWithCapture(CoderAgent):
        def process(self, request):
            captured_requests.append(request)
            return super().process(request)

    client = get_llm_client(use_mock=True)
    coder = MockCoderWithCapture(llm_client=client)
    executor = ExecutorAgent()
    analyzer = ErrorAnalyzerAgent(llm_client=client)

    orch = IterativeCodingOrchestrator(coder, executor, analyzer, max_iterations=2)
    pctx = _make_tiny_classification_perceptual_context()

    result = orch.process(pctx)
    assert result.success is True
    assert result.total_iterations == 2

    # Second request must have previous_code and error_context_json populated
    assert len(captured_requests) == 2
    assert captured_requests[1].previous_code is not None
    assert captured_requests[1].error_context_json is not None
    err_data = json.loads(captured_requests[1].error_context_json)
    assert err_data["error_category"] == "KeyError"


# ===========================================================================
# E. End-to-End Integration Test for house_price_faulty
# ===========================================================================

def test_house_price_faulty_end_to_end_recovery():
    """
    End-to-end integration run on house_price_faulty:
    Iteration 1: fails authentically due to uncleaned data
    ErrorAnalyzer: classifies error as DataQuality
    Iteration 2: cleans data and successfully fits AutoGluon regression model
    """
    client = get_llm_client(use_mock=True)
    coder = CoderAgent(llm_client=client)
    executor = ExecutorAgent()
    analyzer = ErrorAnalyzerAgent(llm_client=client)

    orch = IterativeCodingOrchestrator(coder, executor, analyzer, max_iterations=5)
    pctx = _make_house_price_perceptual_context()

    result = orch.process(pctx, user_instruction="Predict house price", run_id="test-house-price-e2e")

    assert result.success is True
    assert result.total_iterations == 2
    assert len(result.iteration_history) == 2

    # Iteration 1 failed with DataQuality
    iter1 = result.iteration_history[0]
    assert iter1.execution_result.success is False
    assert iter1.error_context is not None
    assert iter1.error_context.error_category == "DataQuality"

    # Iteration 2 succeeded
    iter2 = result.iteration_history[1]
    assert iter2.execution_result.success is True
