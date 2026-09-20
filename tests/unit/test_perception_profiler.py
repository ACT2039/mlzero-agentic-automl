"""
Tests for data-aware perception: DataProfiler, target inference, task-type inference,
and data quality detection.
"""
import csv
from pathlib import Path

import pytest

from mlzero.agents.perception import (
    DataProfiler,
    TaskPerceptionAgent,
    _extract_instruction_target,
)
from mlzero.core.llm import MockLLMClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture()
def classification_dir(tmp_path: Path) -> Path:
    train = [
        {"feature1": "1.0", "feature2": "0.5", "target": "0"},
        {"feature1": "0.0", "feature2": "0.1", "target": "0"},
        {"feature1": "1.5", "feature2": "1.5", "target": "1"},
        {"feature1": "2.0", "feature2": "1.8", "target": "1"},
    ]
    test = [
        {"feature1": "0.5", "feature2": "0.3", "target": "0"},
        {"feature1": "1.8", "feature2": "1.7", "target": "1"},
    ]
    _write_csv(tmp_path / "train.csv", train)
    _write_csv(tmp_path / "test.csv", test)
    return tmp_path


@pytest.fixture()
def regression_dir(tmp_path: Path) -> Path:
    train = [
        {"area": "1200", "bedrooms": "3", "price": "350000.0"},
        {"area": "900",  "bedrooms": "2", "price": "250000.0"},
        {"area": "1500", "bedrooms": "4", "price": "480000.5"},
    ]
    test = [
        {"area": "1100", "bedrooms": "3"},
    ]
    _write_csv(tmp_path / "train.csv", train)
    _write_csv(tmp_path / "test.csv", test)
    return tmp_path


@pytest.fixture()
def house_price_faulty_dir() -> Path:
    return Path("tests/mlzero_faulty_datasets/house_price_faulty")


@pytest.fixture()
def tiny_classification_dir() -> Path:
    return Path("tests/data/tiny_classification")


# ---------------------------------------------------------------------------
# 1. Classification dataset correctly identified
# ---------------------------------------------------------------------------

def test_classification_task_type(classification_dir: Path) -> None:
    profiler = DataProfiler(classification_dir)
    report = profiler.profile()
    assert report.task_type == "classification"


# ---------------------------------------------------------------------------
# 2. Regression dataset correctly identified
# ---------------------------------------------------------------------------

def test_regression_task_type(regression_dir: Path) -> None:
    profiler = DataProfiler(regression_dir)
    report = profiler.profile()
    assert report.task_type == "regression"


# ---------------------------------------------------------------------------
# 3. House-price target inferred as 'price'
# ---------------------------------------------------------------------------

def test_house_price_target_inference(house_price_faulty_dir: Path) -> None:
    profiler = DataProfiler(house_price_faulty_dir)
    report = profiler.profile()
    assert report.target_column == "price"


# ---------------------------------------------------------------------------
# 4. Missing values detected (bedrooms has blank in row 37 of house_price)
# ---------------------------------------------------------------------------

def test_missing_values_detected(house_price_faulty_dir: Path) -> None:
    profiler = DataProfiler(house_price_faulty_dir)
    report = profiler.profile()
    missing_cols = {mv.column for mv in report.missing_values}
    assert "bedrooms" in missing_cols


# ---------------------------------------------------------------------------
# 5. Invalid numeric values detected (area has "1200 sqft" and "unknown")
# ---------------------------------------------------------------------------

def test_invalid_numeric_values_detected(house_price_faulty_dir: Path) -> None:
    profiler = DataProfiler(house_price_faulty_dir)
    report = profiler.profile()
    invalid_cols = {inv.column for inv in report.invalid_numeric_values}
    assert "area" in invalid_cols


# ---------------------------------------------------------------------------
# 6. Malformed target values detected ("not_available" in price column)
# ---------------------------------------------------------------------------

def test_malformed_target_values_detected(house_price_faulty_dir: Path) -> None:
    profiler = DataProfiler(house_price_faulty_dir)
    report = profiler.profile()
    assert len(report.malformed_target_values) > 0
    assert any("not_available" in v for v in report.malformed_target_values)


# ---------------------------------------------------------------------------
# 7. Train/test schema mismatch detected
# ---------------------------------------------------------------------------

def test_train_test_schema_mismatch(regression_dir: Path) -> None:
    # regression_dir: train has 'price', test does NOT => price identified as target
    profiler = DataProfiler(regression_dir)
    report = profiler.profile()
    # price is in train only => captured in candidate targets / target_column
    assert report.target_column == "price"
    # no mismatch other than the target column (which is expected)
    non_target_mismatches = [
        m for m in report.train_test_schema_mismatches
        if "price" not in m
    ]
    assert len(non_target_mismatches) == 0


# ---------------------------------------------------------------------------
# 8. Explicit user instruction selects the target
# ---------------------------------------------------------------------------

def test_instruction_selects_target(regression_dir: Path) -> None:
    profiler = DataProfiler(regression_dir)
    # instruction explicitly says "predict price"
    report = profiler.profile(user_instruction="predict price")
    assert report.target_column == "price"


def test_instruction_target_extraction_patterns() -> None:
    # Single-word after predict => captured
    assert _extract_instruction_target("predict price") == "price"
    # use X as target
    assert _extract_instruction_target("use price as target") == "price"
    # target is X
    assert _extract_instruction_target("target column is salary") == "salary"
    # None input
    assert _extract_instruction_target(None) is None
    # Multi-word natural language: "predict house price" — may not extract perfectly,
    # but must not return a stopword and must not crash
    result = _extract_instruction_target("predict house price")
    assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# 9. No false 'target' hallucination when target is not named 'target'
# ---------------------------------------------------------------------------

def test_no_false_target_hallucination(tmp_path: Path) -> None:
    train = [
        {"sepal_length": "5.1", "petal_width": "1.4", "species": "setosa"},
        {"sepal_length": "4.9", "petal_width": "1.4", "species": "versicolor"},
        {"sepal_length": "4.7", "petal_width": "1.3", "species": "virginica"},
    ]
    test = [
        {"sepal_length": "5.0", "petal_width": "1.5"},
    ]
    _write_csv(tmp_path / "train.csv", train)
    _write_csv(tmp_path / "test.csv", test)

    profiler = DataProfiler(tmp_path)
    report = profiler.profile()
    # 'target' column does not exist — should infer 'species' (train-only col)
    assert report.target_column == "species"
    assert report.target_column != "target"


# ---------------------------------------------------------------------------
# 10. Existing tiny_classification still identified correctly
# ---------------------------------------------------------------------------

def test_tiny_classification_perception(tiny_classification_dir: Path) -> None:
    profiler = DataProfiler(tiny_classification_dir)
    report = profiler.profile()
    assert report.target_column == "target"
    assert report.task_type == "classification"


# ---------------------------------------------------------------------------
# 11. TaskPerceptionAgent overrides mock LLM with deterministic evidence
# ---------------------------------------------------------------------------

def test_task_perception_overrides_mock(house_price_faulty_dir: Path) -> None:
    from mlzero.agents.perception import FilePerceptionAgent

    llm = MockLLMClient()  # Returns classification / target by default

    profiler = DataProfiler(house_price_faulty_dir)
    dq = profiler.profile()

    file_agent = FilePerceptionAgent(house_price_faulty_dir)
    fctx = file_agent.process()

    agent = TaskPerceptionAgent(llm_client=llm)
    result = agent.process(fctx, user_instruction=None, data_quality=dq)

    # Deterministic override must win over the mock's hardcoded "target"/"classification"
    assert result.target_column == "price"
    assert result.task_type == "regression"


# ---------------------------------------------------------------------------
# 12. Existing tiny_classification mock recovery — LLM mock still returns "target"
# ---------------------------------------------------------------------------

def test_mock_llm_still_returns_target_for_classification(tiny_classification_dir: Path) -> None:
    """
    Confirm that MockLLM.generate_structured(TaskContext) still returns the
    hardcoded target='target'. The DataProfiler override then keeps 'target'
    because it matches the actual data.
    """
    from mlzero.agents.perception import FilePerceptionAgent

    llm = MockLLMClient()
    profiler = DataProfiler(tiny_classification_dir)
    dq = profiler.profile()

    file_agent = FilePerceptionAgent(tiny_classification_dir)
    fctx = file_agent.process()

    agent = TaskPerceptionAgent(llm_client=llm)
    result = agent.process(fctx, user_instruction=None, data_quality=dq)

    assert result.target_column == "target"
    assert result.task_type == "classification"


# ---------------------------------------------------------------------------
# 13. location_score column with "high" value detected as invalid numeric
# ---------------------------------------------------------------------------

def test_location_score_invalid_numeric(house_price_faulty_dir: Path) -> None:
    profiler = DataProfiler(house_price_faulty_dir)
    report = profiler.profile()
    invalid_cols = {inv.column: inv for inv in report.invalid_numeric_values}
    assert "location_score" in invalid_cols
    loc = invalid_cols["location_score"]
    assert any("high" in v.lower() for v in loc.invalid_values)
