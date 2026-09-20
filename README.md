# MLZero: Agentic AutoML System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: Pydantic v2](https://img.shields.io/badge/framework-Pydantic_v2-green.svg)](https://docs.pydantic.dev/)

> **A Paper-Aligned Agentic AutoML System inspired by MLZero (NeurIPS 2025)**  
> *Independent B.Tech Implementation by Lead Architect*

---

## 📌 Project Overview

**MLZero** is a multi-agent system designed to automate end-to-end applied machine learning workflows—from raw dataset perception and library selection to code generation, sandboxed execution, LLM-based outcome evaluation, and error-recovery retry loops.

The system addresses the brittleness of traditional AutoML pipelines by combining:
1. **Multimodal Perception**: Automatic file grouping, task classification, schema profiling, and ML framework selection.
2. **Semantic Memory**: Knowledge-base ingestion, summarization, condensation, and error-aware document retrieval.
3. **Episodic Memory**: Chronological run history tracking and error context distillation across iterations.
4. **Execution Judge & Retry Loop**: Structured outcome evaluation (`FINISH` vs `FIX`), root-cause failure analysis, and closed-loop code repair.
5. **Modular ML Adapters**: Executable integration with AutoGluon (Tabular, Multimodal, TimeSeries), FlagEmbedding, and scikit-learn fallbacks.

---

## 🏗️ System Architecture

```
                      Raw Dataset Directory
                                │
                                ▼
                       ┌─────────────────┐
                       │   Perception    │
                       │ (Files/Task/Lib)│
                       └────────┬────────┘
                                │ PerceptualContext
                                ▼
                       ┌─────────────────┐
                 ┌────►│ Semantic Memory │◄─── ErrorContext
                 │     │ (RAG Retrieval) │
                 │     └────────┬────────┘
                 │              │ Top-K Chunks ($K=5$)
                 │              ▼
                 │     ┌─────────────────┐
                 │     │   Coder Agent   │
                 │     └────────┬────────┘
                 │              │ CodeArtifact (script.py)
                 │              ▼
                 │     ┌─────────────────┐
                 │     │ Library Adapter │ (Tabular, Multimodal, TimeSeries,
                 │     └────────┬────────┘  FlagEmbedding, General ML)
                 │              │ Executable Script
                 │              ▼
                 │     ┌─────────────────┐
                 │     │ Executor Agent  │ (Sandboxed Subprocess)
                 │     └────────┬────────┘
                 │              │ ExecutionResult
                 │              ▼
                 │     ┌─────────────────┐
                 │     │ Execution Judge │
                 │     └───┬─────────┬───┘
                 │         │         │
                   `FIX`   │         │ `FINISH`
                 ┌─────────┘         └─────────┐
                 ▼                             ▼
       ┌───────────────────┐          ┌───────────────────┐
       │  Error Analyzer   │          │ Run Completed     │
       └─────────┬─────────┘          │ Artifact Export   │
                 │ ErrorContext       └───────────────────┘
                 ▼
       ┌───────────────────┐
       │  Episodic Memory  │ (Chronological History)
       └───────────────────┘
```

---

## 💡 Major Components & Agents

| Component | Responsibility |
| :--- | :--- |
| **`FilePerceptionAgent`** | Scans raw dataset folders, groups files, infers data formats, and detects schema columns. |
| **`TaskPerceptionAgent`** | Classifies ML task types (classification, regression, forecasting, multimodal, retrieval). |
| **`LibrarySelectorAgent`** | Recommends the optimal ML framework adapter based on task requirements and dataset context. |
| **`SemanticMemory`** | Ingests API docs, summarizes code patterns, and performs error-aware Top-K document retrieval. |
| **`EpisodicMemory`** | Maintains JSON-persisted chronological execution history and distills failure contexts across iterations. |
| **`CoderAgent`** | Synthesizes executable Python code using a segregated 4-section prompt. |
| **`LibraryAdapters`** | Modular wrappers providing executable interfaces for specialized ML frameworks. |
| **`ExecutorAgent`** | Runs generated code in a sandboxed subprocess with strict timeouts and output artifact validation. |
| **`ExecutionJudgeAgent`** | Evaluates execution logs to issue `FINISH` (success) or `FIX` (retry) decisions. |
| **`ErrorAnalyzerAgent`** | Diagnoses execution failures, categorizes errors, and formulates code-level remediation context. |

---

## ⚙️ Supported ML Frameworks & Modalities

| Library Adapter | Task Categories | Input Modalities |
| :--- | :--- | :--- |
| **`autogluon.tabular`** | Supervised Classification & Regression | Tabular CSV / Parquet |
| **`autogluon.multimodal`** | Image Classification, Text Classification, Mixed Tabular+Text+Images | Images, Text, Mixed Tabular |
| **`autogluon.timeseries`** | Multi-Item Time-Series Forecasting | Timestamp & Item ID DataFrames |
| **`flagembedding`** | Corpus Dense Retrieval & Reranking | Document Corpus & Queries JSON |
| **`general_ml`** | Scikit-learn Baseline Fallback | Generic Tabular / Matrix Data |

---

## 🧠 LLM Execution Modes

The system explicitly supports two operational modes:

1. **`MockLLM` (Offline & Deterministic Mode)**:
   - Uses zero external network calls or API keys.
   - Ideal for reproducible local unit testing, formal benchmarking, and rapid CI/CD runs.
   - Triggers simulated execution paths and deterministic error-recovery loops.

2. **`RealLLM` (Online API Mode)**:
   - Integrates with Google Gemini, OpenAI, or custom OpenAI-compatible endpoints.
   - Configured via environment variables (`GEMINI_API_KEY`, `OPENAI_API_KEY`).
   - Generates live LLM code synthesis, dynamic perception, and reasoning.

---

## 🚀 Quick Start Commands

### 1. Verify Environment & Test Suite
```bash
test_venv\Scripts\python -m pytest -q
```

### 2. Run Offline End-to-End Execution (Mock LLM)
```bash
test_venv\Scripts\python -m mlzero run \
    --input tests/data/tiny_classification \
    --mock-llm
```

### 3. Run Real LLM Perception
```bash
test_venv\Scripts\python -m mlzero perceive \
    --input tests/data/tiny_classification \
    --llm-mode real \
    --json
```

### 4. Run Evaluation Benchmark Suite
```bash
# Smoke test (1 fast run per case)
test_venv\Scripts\python -m evaluation.runner --mode smoke

# Local formal evaluation (3 statistical runs per case)
test_venv\Scripts\python -m evaluation.runner --mode local_formal --runs 3

# Ablation studies (Semantic, Episodic, Judge, Retrieval K)
test_venv\Scripts\python -m evaluation.runner --mode ablation

# Controlled data noise robustness suite
test_venv\Scripts\python -m evaluation.runner --mode robustness
```

### 5. Launch FastAPI Backend & Gradio Web UI
```bash
# Start FastAPI backend server
test_venv\Scripts\python -m mlzero serve --port 8000

# Launch interactive Gradio Web UI
test_venv\Scripts\python -m mlzero ui --port 7860
```

---

## 🛡️ Security Model

- **Sandboxed Execution**: Subprocess execution isolation with strict execution timeout (`MLZERO_EXECUTION__TIMEOUT_SECONDS=120`).
- **Path Traversal Protection**: Input data paths are validated against `MLZERO_APP__ALLOWED_DATA_ROOT` using strict path resolution.
- **Environment Filtering**: Subprocesses run with sanitized environment variables, preventing key leaks to execution scripts.
- **No Secret Exposure**: Logs, API endpoints, and artifacts filter raw credentials, tokens, and model binaries.

---

## 📊 Benchmark & Provenance Summary

Evaluation outputs are generated dynamically in machine-readable formats under `reports/`:
- `reports/evaluation_report.md` — Full technical evaluation summary.
- `reports/evaluation_runs.json` — Raw end-to-end execution traces and judge decisions.
- `reports/ablation_runs.json` — Machine-readable ablation records across all 11 component variants.
- `reports/robustness_runs.json` — Raw data noise perturbation records across 6 perturbation types.

---

## ⚠️ Limitations & Paper-Alignment Disclaimer

> [!IMPORTANT]
> **Paper-Alignment Disclaimer**
> 
> This repository represents a paper-aligned local B.Tech implementation inspired by the **MLZero NeurIPS 2025** paper (*MLZero: A Multi-Agent System for End-to-End Machine Learning Automation*).
> 
> - **Inspiration, Not Reproduction**: We reproduce the core architectural methodology (multi-agent perception, semantic retrieval, episodic memory, execution judge, and library adapters) but do **not** claim identical benchmark scale, GPU compute infrastructure, or raw numerical results to the original authors' paper.
> - **Local Hardware Limits**: Local formal evaluations rely on CPU-friendly small synthetic fixtures and deterministic `MockLLM` execution to remain lightweight and reproducible on standard workstation hardware.
