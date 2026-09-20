# MLZero Architecture

## Overview

MLZero is a paper-aligned multi-agent system for end-to-end machine learning automation.
It is inspired by the paper *"MLZero: A Multi-Agent System for End-to-End Machine Learning Automation"* (NeurIPS 2025), representing an independent local B.Tech implementation with distinct engineering choices.

---

## High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MLZero System                                 │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                       Application Layer                           │  │
│  │     CLI (mlzero run)    FastAPI (/runs)    Gradio Web UI          │  │
│  └───────────────────────────┬───────────────────────────────────────┘  │
│                              │ MLZeroService.run_mlzero()               │
│  ┌───────────────────────────▼───────────────────────────────────────┐  │
│  │                    Orchestration Layer                            │  │
│  │                IterativeCodingOrchestrator                        │  │
│  │     (max 10 iterations, error-recovery closed feedback loop)      │  │
│  └──┬────────────────────────────────────────────────────────────────┘  │
│     │                                                                   │
│  ┌──▼───────────┐  ┌──────────────┐  ┌──────────────┐                 │
│  │ Perception   │  │   Coder      │  │  Executor    │                 │
│  │ (Stage 6)    │  │  (Stage 4)   │  │  (Stage 4)   │                 │
│  └──┬───────────┘  └──────┬───────┘  └──────┬───────┘                 │
│     │                     │                 │                          │
│  ┌──▼─────────────────────┴─────────────────▼──────────────────────┐  │
│  │            Execution Judge & Error Analyzer (Stage 4)           │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌────────────────────────┐   ┌─────────────────────────────────────┐  │
│  │    Semantic Memory     │   │          Episodic Memory            │  │
│  │    (Stage 2)           │   │          (Stage 3)                  │  │
│  │    (TF-IDF / FAISS)    │   │          (JSON-based store)         │  │
│  └────────────────────────┘   └─────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │             Executable ML Library Adapters (Stage 5)              │  │
│  │  - autogluon.tabular    : TabularPredictor (classification/reg)  │  │
│  │  - autogluon.multimodal : MultiModalPredictor (text/image/mult)  │  │
│  │  - autogluon.timeseries : TimeSeriesPredictor (forecasting)      │  │
│  │  - FlagEmbedding        : Dense retrieval & reranking            │  │
│  │  - general_ml           : Generic scikit-learn fallback          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │             Evaluation Framework & Runner (Stage 7)              │  │
│  │  Modes: smoke | local_formal (3 runs) | ablation | robustness   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Agent Roles across Implementation Stages

| Agent | Stage | Primary Role |
|---|---|---|
| `FilePerceptionAgent` | 6 | Scans raw dataset directory, groups files, infers schemas and formats |
| `TaskPerceptionAgent` | 6 | Extracts task type, label target, and objective constraints |
| `LibrarySelectorAgent` | 6 | Selects appropriate ML library adapter from allowable registry |
| `SummarizationAgent` | 2 | Distills raw API docs into concise, searchable knowledge summaries |
| `CondensationAgent` | 2 | Distills actionable code patterns into executable knowledge chunks |
| `RetrievalAgent` | 2 | Online error-aware retrieval of Top-K condensed knowledge chunks ($K=5$) |
| `CoderAgent` | 4 | Generates executable Python scripts using segregated 4-section prompt |
| `ExecutorAgent` | 4 | Runs code in sandboxed subprocess with execution timeout and output limits |
| `ExecutionJudgeAgent` | 4 | Evaluates execution logs to decide outcome (`FINISH` vs `FIX`) |
| `ErrorAnalyzerAgent` | 4 | Diagnoses execution failures, categorises errors, and formulates fixes |

---

## Memory Systems Architecture

| System | Stage | Role | Storage / Mechanics |
|---|---|---|---|
| `SemanticMemory` | 2 | Prior domain knowledge ingestion & retrieval | Local TF-IDF / FAISS index on disk |
| `EpisodicMemory` | 3 | Chronological run execution tracking | JSON-based file store under `memory/episodic/` |

---

## Stage Breakdown

### Stage 1: Real LLM Provider Foundation
- Support for `MockLLMClient` (deterministic offline testing) and `RealLLMClient` (Google Gemini API / OpenAI API compatible).
- Robust error handling, rate-limit retries, and structured JSON parsing.

### Stage 2: Semantic Memory Architecture
- Ingests documentation for AutoGluon (Tabular, Multimodal, TimeSeries), FlagEmbedding, and scikit-learn.
- `SummarizationAgent` and `CondensationAgent` process raw docs into condensed guidance.
- `RetrievalAgent` retrieves top 5 relevant knowledge chunks per coding iteration.

### Stage 3: Episodic Memory Architecture
- Chronological run history tracking: $E_t(P, C_{t-1}, L_{t-1}, G_{t-1}, R_{t-1}) = R_t$.
- Stores generated code, execution logs, retrieved knowledge, and resulting error context per iteration.
- Atomic persistence ensures crash resistance without disk corruption.

### Stage 4: Execution Judge & Iterative Coding
- `ExecutionJudgeAgent` decides `FINISH` (success) or `FIX` (retry required).
- `ErrorAnalyzerAgent` executes only on `FIX` to categorize errors and formulate fixes.
- Closed loop: Perception $\rightarrow$ Semantic Retrieval $\rightarrow$ Coder $\rightarrow$ Executor $\rightarrow$ Judge $\rightarrow$ Error Analyzer $\rightarrow$ Episodic Memory $\rightarrow$ Semantic Retrieval $\rightarrow$ Retry.

### Stage 5: Executable ML Library Adapters
- Modular provider layer (`MLLibraryAdapter`) supporting:
  1. `autogluon.tabular` (Tabular classification/regression)
  2. `autogluon.multimodal` (Image, text, and multimodal classification)
  3. `autogluon.timeseries` (Multi-item time-series forecasting)
  4. `flagembedding` (Dense retrieval & reranking)
  5. `general_ml` (Scikit-learn fallback)

### Stage 6: Multimodal Dataset Perception
- Multi-file directory scanner and file grouping (`file_groups`).
- Automatic modal detection (tabular CSV, images, text, JSON corpus, time-series timestamp/item_id).
- Task perception mapping raw dataset layouts directly to library adapters.

### Stage 7: Evaluation Framework
- Paper-aligned evaluation suite with 10 cross-modality benchmark cases.
- Modes: `smoke`, `local_formal` (3-run statistical averaging), `ablation`, `robustness`, `real_llm`.
- Machine-readable raw artifacts: `evaluation_runs.json`, `ablation_runs.json`, `robustness_runs.json`.

---

> *Note: This is an independent B.Tech implementation inspired by MLZero (NeurIPS 2025), not an exact reproduction of the authors' private production system.*
