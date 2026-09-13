# MLZero-Agentic-AutoML

**Phase 1 Foundation**

## Project Purpose
MLZero is a Multi-Agent System for End-to-end Machine Learning Automation. It aims to automate the end-to-end process of applied machine learning by combining multiple specialized agents that collaborate to perceive tasks, read documentation, write ML code, and iteratively debug their solutions.

This project is an academic implementation inspired by the MLZero architecture and is not claiming to reproduce the authors' exact infrastructure or benchmark environment.

## Research Inspiration
This project is inspired by the research paper: *"MLZero: A Multi-Agent System for End-to-end Machine Learning Automation."* (NeurIPS 2025).

## Problem Being Addressed
Automating end-to-end machine learning workflows typically requires heavy human intervention. This project builds a collaborative multi-agent system to automate code generation, debugging, and framework selection.

## Locked Architecture
The architecture is structured around four main pillars:
1. **Perception**: File Perception, Task Perception, and ML Library Selection.
2. **Semantic Memory**: Documentation ingestion, summarization, and retrieval.
3. **Episodic Memory**: Iteration history, execution logs, and error tracking.
4. **Iterative Coding**: Coder, Executor, and Error Analyzer agents orchestrated in a retry loop.

## Current Status: Phase 7 Completed
The project has successfully implemented:
- **Perception** (File, Task, Library Selection)
- **Coder & Executor** (Secure Subprocess execution)
- **Error Analysis & Iterative Retry** (Orchestrator loop)
- **Semantic Memory** (RAG documentation retrieval)
- **Episodic Memory** (Chronological run history tracking)
- **Real ML Integration** (AutoGluon tabular classification/regression)

## AutoGluon Integration & Limitations
The system currently implements **AutoGluon Tabular** as the primary ML framework.
- **Setup:** A deterministic `TabularDatasetAdapter` bridges data from raw locations to a managed `./outputs/data/` environment.
- **Artifacts:** Models and execution scripts are securely logged under `./outputs/models/exec_<id>/`. Binary models are excluded from Git.
- **Limitations:** Currently limited to Tabular Supervised Learning (classification/regression). Large unstructured formats (images/text) or deep vector databases are out-of-scope for the current implementation footprint. Training time is capped deliberately to avoid large-scale infrastructure costs.

## Development Setup
1. Clone the repository.
2. Python version required: `>=3.11,<3.12`.
3. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. For ML integration, install AutoGluon locally or in a secondary test environment:
   ```bash
   pip install autogluon.tabular==1.1.1
   ```

## Configuration
Configuration is managed through `configs/config.yaml` and `.env` variables via Pydantic.

## Testing
Run standard unit tests with `pytest`. They are deterministic and require no external APIs.
To run the ML End-to-End integration test (requires AutoGluon):
```bash
$env:RUN_ML_INTEGRATION="1"
pytest tests/integration/test_ml_pipeline.py -v -s
```

## Security
Secrets must only be placed in `.env`, which is strictly ignored by Git. No credentials should be committed.

## Future Implementation Phases
Later phases will introduce LLM integration, a secure execution sandbox, the core iterative coding retry loop, and full integration with ML libraries.

## Phase 8: API and UI Integration
A FastAPI backend and Gradio UI are available.

### Run the API Server
\\ash
python -m mlzero serve --port 8000
\Endpoints available:
- \GET /health- \POST /runs\ (Create a background execution)
- \GET /runs/{run_id}- \GET /runs/{run_id}/episodes- \GET /runs/{run_id}/artifacts
### Run the Gradio UI
\\ash
python -m mlzero ui --port 7860
\
### Limitations
The background run manager is process-local and is not intended for distributed production deployments. State is kept in memory.
