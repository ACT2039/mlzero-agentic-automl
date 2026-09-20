# Viva Q&A Reference Guide — MLZero Agentic AutoML

This document provides concise, technically precise answers for a B.Tech project presentation or viva defense.

---

### 1. What is MLZero?
**MLZero** is an agentic Automated Machine Learning (AutoML) system inspired by the NeurIPS 2025 paper. It uses a collaborative multi-agent architecture to automate applied machine learning end-to-end, including raw data perception, framework selection, code generation, sandboxed execution, outcome judgment, and closed-loop error recovery.

---

### 2. What problem does MLZero solve?
Traditional AutoML frameworks (like standard hyperparameter search scripts) are brittle—they assume pre-cleaned tabular data, single file structures, and error-free code execution. MLZero solves this by using LLM agents to handle dirty data, unformatted multimodal directories, framework selection across multiple domain libraries, syntax/runtime errors, and iterative solution repair.

---

### 3. What are the main MLZero modules?
The system consists of four primary pillars:
1. **Perception**: Scans files, groups dataset formats, perceives task type, and selects optimal ML framework.
2. **Semantic Memory**: Documentation RAG system providing top 5 relevant code patterns and rules.
3. **Episodic Memory**: Chronological run history tracking and failure context distillation across iterations.
4. **Iterative Coding & Execution**: Coder agent, subprocess executor, execution judge, and error analyzer.

---

### 4. How does Perception work?
Perception runs in three stages before code generation:
1. `FilePerceptionAgent`: Scans dataset directories, groups files, detects file extensions (`.csv`, `.jpg`, `.json`), and checks column schemas.
2. `TaskPerceptionAgent`: Classifies task type (`classification`, `regression`, `time_series_forecasting`, `multimodal`, `retrieval`) and identifies target labels.
3. `LibrarySelectorAgent`: Recommends the best framework adapter (e.g. `autogluon.tabular`, `autogluon.timeseries`, `flagembedding`).
Result: A validated `PerceptualContext` schema object.

---

### 5. How does Semantic Memory work?
Semantic Memory is a static prior knowledge system:
- **Offline Ingestion**: `SummarizationAgent` distills API docs, and `CondensationAgent` distills executable code snippets into indexed chunks.
- **Online Retrieval**: `RetrievalAgent` performs TF-IDF search using perceptual context and error history, returning the Top 5 most relevant knowledge chunks ($K=5$) to inject into the Coder prompt.

---

### 6. How does Episodic Memory work?
Episodic Memory is a dynamic, run-specific memory system following $E_t(P, C_{t-1}, L_{t-1}, G_{t-1}, R_{t-1}) = R_t$:
- It logs every iteration's perceptual context ($P$), generated code ($C$), execution logs ($L$), retrieved guidance ($G$), and error context ($R$).
- It serializes state atomically as JSON files under `memory/episodic/`.
- It feeds past failure history into iteration $t+1$ to ensure the Coder never repeats identical errors.

---

### 7. How does the Execution Judge work?
`ExecutionJudgeAgent` acts as an independent evaluator of execution results:
- It inspects process exit codes, stdout/stderr logs, output prediction files, and model artifacts.
- It returns a structured decision: **`FINISH`** (task successful, end loop) or **`FIX`** (execution failed, retry required).
- `ErrorAnalyzerAgent` is invoked **only** when the Judge decides `FIX`.

---

### 8. How does the retry/recovery loop work?
1. Iteration 1 code fails or produces invalid output.
2. `ExecutionJudgeAgent` issues `FIX`.
3. `ErrorAnalyzerAgent` categorizes the failure (e.g. `KeyError`, `DataQuality`) and formulates a suggested fix.
4. `EpisodicMemory` records the failed iteration record.
5. `SemanticMemory` performs error-aware retrieval using the new error context.
6. `CoderAgent` receives a 4-section prompt containing perceptual context, instruction, retrieved guidance, and episodic error history to produce a repaired code script.

---

### 9. Why are multiple ML libraries supported?
Real-world AutoML spans diverse ML domains that single frameworks cannot cover:
- Tabular data $\rightarrow$ `autogluon.tabular`
- Multimodal text/images $\rightarrow$ `autogluon.multimodal`
- Time-series sequences $\rightarrow$ `autogluon.timeseries`
- Dense text retrieval $\rightarrow$ `flagembedding`
- Baseline fallback $\rightarrow$ `general_ml` (scikit-learn)
`LibrarySelectorAgent` and `AdapterRegistry` route tasks dynamically to the correct library adapter.

---

### 10. How does Multimodal Perception work?
When presented with a dataset folder, `FilePerceptionAgent` groups files into structured schemas (`tabular_files`, `image_files`, `text_files`, `series_files`). It checks for timestamp and entity columns to recognize time-series, or image folder hierarchies to identify computer vision tasks, mapping them to `PerceptualContext`.

---

### 11. How is evaluation performed?
The evaluation framework (`evaluation/runner.py`) tests the system on 10 cross-modality benchmark cases:
- **Modes**: `smoke` (quick test), `local_formal` (3-run statistical averaging), `ablation` (testing memory/judge contributions), `robustness` (testing data noise resilience).
- Outputs machine-readable raw logs (`evaluation_runs.json`, `ablation_runs.json`, `robustness_runs.json`) and summary reports.

---

### 12. What is the difference between MockLLM and RealLLM?
- **`MockLLMClient`**: Deterministic offline mode using zero network requests or API costs. Returns pre-programmed responses for fast unit testing and reproducible formal evaluation.
- **`RealLLMClient`**: Connects via API to Google Gemini or OpenAI models for real-world LLM reasoning and code generation.

---

### 13. What is the security model?
- Subprocess isolation via `PythonRunner` with timeout limits (`120s`).
- Path traversal validation against `MLZERO_APP__ALLOWED_DATA_ROOT`.
- Sanitized environment variables for subprocess execution.
- Filtering of API keys, tokens, and binary model blobs from logs and git tracking.

---

### 14. What are the project limitations?
- Runs on local CPU workstation footprint using synthetic lightweight benchmarks.
- Uses process-local memory management rather than distributed Kubernetes clusters.
- Relies on `MockLLMClient` for offline benchmark evaluation to avoid LLM API costs and stochastic variance.

---

### 15. How does this differ from the original NeurIPS paper?
This repository is an **independent paper-aligned B.Tech implementation inspired by MLZero**. It implements the paper's multi-agent architecture, semantic/episodic memory formulations, execution judge, and evaluation methodologies, but uses lightweight CPU-friendly local benchmarks rather than the authors' private multi-GPU enterprise infrastructure.
