"""
Benchmark Evaluation Cases definition for MLZero Evaluation Suite.
"""
from evaluation.schemas import EvaluationCase

BENCHMARK_CASES: list[EvaluationCase] = [
    EvaluationCase(
        case_id="case_01_tiny_cls",
        name="Tiny Tabular Classification",
        dataset_path="tests/data/tiny_classification",
        task_type="classification",
        expected_library="autogluon.tabular",
        modality="tabular",
        target_column="target",
        description="Clean tabular binary classification dataset.",
        difficulty="easy",
        source_type="synthetic_local",
        tags=["tabular", "classification", "clean"],
    ),
    EvaluationCase(
        case_id="case_02_tiny_reg",
        name="Tiny Tabular Regression",
        dataset_path="tests/data/tiny_regression",
        task_type="regression",
        expected_library="autogluon.tabular",
        modality="tabular",
        target_column="target",
        description="Clean tabular regression dataset.",
        difficulty="easy",
        source_type="synthetic_local",
        tags=["tabular", "regression", "clean"],
    ),
    EvaluationCase(
        case_id="case_03_house_price_faulty",
        name="House Price Faulty Data Quality",
        dataset_path="tests/mlzero_faulty_datasets/house_price_faulty",
        task_type="regression",
        expected_library="autogluon.tabular",
        modality="tabular",
        target_column="price",
        description="Tabular real-estate dataset with missing values, malformed target string, and invalid numerics requiring data cleaning recovery.",
        difficulty="medium",
        source_type="faulty_synthetic",
        tags=["tabular", "regression", "faulty", "data_quality"],
    ),
    EvaluationCase(
        case_id="case_04_tiny_timeseries",
        name="Tiny Multi-Series Time-Series Forecasting",
        dataset_path="tests/data/tiny_timeseries",
        task_type="time_series_forecasting",
        expected_library="autogluon.timeseries",
        modality="tabular",
        target_column="target",
        description="Multi-series tabular dataset with timestamp and item_id columns.",
        difficulty="medium",
        source_type="synthetic_local",
        tags=["time_series", "forecasting", "multi_series"],
    ),
    EvaluationCase(
        case_id="case_05_tiny_image_cls",
        name="Tiny Image Classification",
        dataset_path="tests/data/tiny_image_classification",
        task_type="multimodal",
        expected_library="autogluon.multimodal",
        modality="image",
        target_column="label",
        description="PNG image dataset with image_path to label CSV mapping.",
        difficulty="medium",
        source_type="synthetic_local",
        tags=["image", "multimodal", "classification"],
    ),
    EvaluationCase(
        case_id="case_06_tiny_text_cls",
        name="Tiny Text Classification",
        dataset_path="tests/data/tiny_text_classification",
        task_type="classification",
        expected_library="autogluon.tabular",
        modality="tabular",
        target_column="label",
        description="Text classification dataset containing short text strings and sentiment labels.",
        difficulty="easy",
        source_type="synthetic_local",
        tags=["text", "classification"],
    ),
    EvaluationCase(
        case_id="case_07_tiny_multimodal",
        name="Tiny Multimodal Text & Tabular",
        dataset_path="tests/data/tiny_multimodal",
        task_type="classification",
        expected_library="autogluon.tabular",
        modality="tabular",
        target_column="label",
        description="Mixed text/tabular multimodal classification dataset.",
        difficulty="medium",
        source_type="synthetic_local",
        tags=["multimodal", "text", "tabular"],
    ),
    EvaluationCase(
        case_id="case_08_tiny_retrieval",
        name="Tiny Dense Retrieval Corpus",
        dataset_path="tests/data/tiny_retrieval",
        task_type="retrieval",
        expected_library="FlagEmbedding",
        modality="json",
        target_column=None,
        description="JSON search corpus and query set for text retrieval.",
        difficulty="medium",
        source_type="synthetic_local",
        tags=["retrieval", "search", "json"],
    ),
    EvaluationCase(
        case_id="case_09_readme_described",
        name="README Described Task",
        dataset_path="tests/data/readme_described_task",
        task_type="classification",
        expected_library="autogluon.tabular",
        modality="tabular",
        target_column="target",
        description="Dataset accompanied by README.md instruction file.",
        difficulty="easy",
        source_type="synthetic_local",
        tags=["readme", "tabular", "task_description"],
    ),
    EvaluationCase(
        case_id="case_10_mixed_directory",
        name="Mixed Directory Perception Case",
        dataset_path="tests/data/mixed_directory",
        task_type="classification",
        expected_library="autogluon.tabular",
        modality="mixed",
        target_column="target",
        description="Directory containing mixed CSV, JSON, TXT, PNG, WAV, and binary files.",
        difficulty="hard",
        source_type="synthetic_local",
        tags=["mixed", "multimodal", "perception"],
        evaluation_scope="perception_only",
    ),
]


def get_case(case_id_or_name: str) -> EvaluationCase | None:
    """Find case by case_id or short name."""
    query = case_id_or_name.lower().strip()
    for case in BENCHMARK_CASES:
        if case.case_id.lower() == query or case.name.lower() == query or query in case.case_id.lower():
            return case
    return None


def get_all_cases() -> list[EvaluationCase]:
    """Return all benchmark evaluation cases."""
    return list(BENCHMARK_CASES)
