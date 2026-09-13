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
- **Phase 1 (Current):** Project foundation. Basic folder structure, configuration management, schemas, and CLI initialization.
- **Phase 2:** Implement basic Agent reasoning pipelines and LLM integration.
- **Phase 3:** Develop the execution sandbox and memory systems.
- **Phase 4:** End-to-end iteration loop and AutoGluon integration.

## Testing Strategy
Tests are split into:
- `tests/unit/`: Fast, deterministic tests that do not require external services or LLMs.
- `tests/integration/`: Slower tests that verify component interaction (to be added in later phases).
