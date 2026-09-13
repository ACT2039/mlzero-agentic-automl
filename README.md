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

## Current Phase 1 Status
Phase 1 establishes the clean project foundation, schemas, logging, configuration, and stubbed agents. It does not yet perform ML automation or LLM calls.

## Development Setup
1. Clone the repository.
2. Python version required: `>=3.11,<3.12`.
3. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Copy `.env.example` to `.env` and fill in placeholders if necessary.

## Python Version
Target runtime: Python 3.11.

## Configuration
Configuration is managed through `configs/config.yaml` and `.env` variables via Pydantic.

## Testing
Run unit tests with `pytest`. They are deterministic and require no external APIs.

## Security
Secrets must only be placed in `.env`, which is strictly ignored by Git. No credentials should be committed.

## Future Implementation Phases
Later phases will introduce LLM integration, a secure execution sandbox, the core iterative coding retry loop, and full integration with ML libraries.
