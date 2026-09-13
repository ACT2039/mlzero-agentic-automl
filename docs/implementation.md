# Implementation Plan

## Current Project Structure
The repository is structured around a core `mlzero` Python package:
- `agents/`: Core perception and coding agent stubs
- `core/`: Configuration and logging utilities
- `schemas/`: Pydantic base data models
- `orchestration/`, `tools/`, `utils/`: Scaffolded for future phases

## Development Standards
- Python >=3.11,<3.12
- Pydantic for data validation
- Ruff for linting and formatting
- Mypy for static type checking
- Pytest for testing

## Phase-by-Phase Implementation Strategy
- **Phase 1 (Completed):** Project foundation. Basic folder structure, configuration management, schemas, and CLI initialization.
- **Phase 2 (Completed):** Implement Perception reasoning pipelines (Task, File, Library).
- **Phase 3 (Completed):** Coder and Executor components utilizing isolated subprocess boundaries.
- **Phase 4 (Completed):** Error Analysis and Iterative Retry Orchestration.
- **Phase 5 (Completed):** Semantic Memory / RAG capability.
- **Phase 6 (Completed):** Episodic Memory chronological history tracking.
- **Phase 7 (Completed):** Real ML Integration using AutoGluon Tabular for classification/regression.

## Phase 7 ML Integration Details
The first production machine learning backend runs on **AutoGluon Tabular**.
- **Execution Limits:** `PythonRunner` encapsulates LLM-generated ML code inside isolated `tempfile.mkdtemp` environments, trapping standard output/error, capping process sizes, and enforcing `time_limit` constraints on `TabularPredictor.fit`.
- **Artifacts:** Output results (including binary artifacts and generated prediction CSVs) are persisted via Python's `shutil` mechanisms under `./outputs/models/exec_<id>/`. The models are fully decoupled from Git tracking to preserve repo size.
- **Limitations:** Limited strictly to Tabular tasks. RAG/Semantic mechanisms handle AutoGluon knowledge dynamically, preventing context bloat. Deep vector embeddings or complex non-tabular capabilities remain uninstantiated by design.

## Testing Strategy
Tests are split into:
- `tests/unit/`: Fast, deterministic tests that mock LLM responses using regex mappings.
- `tests/integration/`: End-to-end component interaction tests simulating real ML failure/recovery using local CSV fixtures (e.g. `tests/data/tiny_classification`). Tests execute safely disconnected from the Internet.
