# MLZero Evaluation Report — MLZero_Local_Formal_Evaluation

## 1. Reproducibility & Environment Metadata

- **Timestamp:** `2026-09-20T17:13:12.214795+00:00`
- **Python Version:** `3.11.9`
- **Platform:** `Windows-10-10.0.26200-SP0`
- **Git Commit:** `Local Worktree`
- **LLM Mode:** `mock`
- **Runs Per Case:** `3`
- **Evaluator Version:** `1.0.0`

## 2. Provenance & Execution Trace Analysis

### Raw Measurement Provenance

> [!NOTE]
> **Measurement Provenance:**
> Every single numerical metric, iteration count, judge decision trace, and execution duration reported herein is calculated dynamically from live execution traces.
> Zero results are hardcoded or manually constructed.

- **Raw Run Records File:** [`reports/evaluation_runs.json`](file:///C:/Users/LENOVO/OneDrive/Desktop/mlzero-agentic-automl/reports/evaluation_runs.json)
- **Total Measured Run Traces:** `30`

### Pipeline Execution Trace

For every end-to-end benchmark run, the complete pipeline step sequence is dynamically verified:
`Perception → Semantic Retrieval → Coder → Adapter → Executor → Execution Judge → Episodic Memory`

| Case ID | Perception | Semantic Retrieval | Coder | Adapter Used | Executor | Judge Decisions | Episodic Memory |
|---|---|---|---|---|---|---|---|
| `case_01_tiny_cls` | `True` | `True` | `True` | `TabularAdapter` | `True` | `['FIX', 'FINISH']` | `True` |
| `case_01_tiny_cls` | `True` | `True` | `True` | `TabularAdapter` | `True` | `['FIX', 'FINISH']` | `True` |
| `case_01_tiny_cls` | `True` | `True` | `True` | `TabularAdapter` | `True` | `['FIX', 'FINISH']` | `True` |
| `case_02_tiny_reg` | `True` | `True` | `True` | `TabularAdapter` | `True` | `['FIX', 'FINISH']` | `True` |
| `case_02_tiny_reg` | `True` | `True` | `True` | `TabularAdapter` | `True` | `['FIX', 'FINISH']` | `True` |
| `case_02_tiny_reg` | `True` | `True` | `True` | `TabularAdapter` | `True` | `['FIX', 'FINISH']` | `True` |
| `case_03_house_price_faulty` | `True` | `True` | `True` | `TabularAdapter` | `True` | `['FIX', 'FIX', 'FIX', 'FIX', 'FIX', 'FIX', 'FIX', 'FIX', 'FIX', 'FIX']` | `True` |
| `case_03_house_price_faulty` | `True` | `True` | `True` | `TabularAdapter` | `True` | `['FIX', 'FINISH']` | `True` |
| `case_03_house_price_faulty` | `True` | `True` | `True` | `TabularAdapter` | `True` | `['FIX', 'FINISH']` | `True` |
| `case_04_tiny_timeseries` | `True` | `True` | `True` | `TimeSeriesAdapter` | `True` | `['FINISH']` | `True` |

### Mock Backend vs Real Backend

- **`MockLLMClient` Execution Mode:** Benchmark evaluation uses deterministic `MockLLMClient` code artifacts to ensure offline reproducibility without API cost.
- **Adapter Execution Backend:** Adapter execution operates in `execution_backend = 'mock'` mode for lightweight synthetic execution verification.

## 3. Evaluation Scope

Benchmark cases are explicitly categorized by evaluation scope:
- **`end_to_end`**: Evaluates complete agentic AutoML pipeline (Perception $\rightarrow$ Semantic Retrieval $\rightarrow$ Iterative Coder $\rightarrow$ Executor $\rightarrow$ Execution Judge $\rightarrow$ ML Solution Artifacts).
- **`perception_only`**: Evaluates File Grouping, File Perception, Task Perception, and Library Selection routing over arbitrary directory structures without executing code generation.

## 4. Overall Performance Metrics

- **Overall Pipeline Success Rate:** `96.67%`
- **Library Selection Routing Accuracy:** `100.00%`
- **Task Perception Accuracy:** `100.00%`
- **Average Iterations:** `1.77`
- **Average Execution Time:** `4.52s`

## 5. End-to-End Cross-Modality Benchmark Matrix

| Case ID | Case Name | Task | Expected Library | Scope | Library Acc | Success Rate | Avg Iters | Mean Time | Key Metric |
|---|---|---|---|---|---|---|---|---|---|
| `case_01_tiny_cls` | Tiny Tabular Classification | `classification` | `autogluon.tabular` | `end_to_end` | `100%` | `100%` | `2.0` | `6.45s` | `N/A` |
| `case_02_tiny_reg` | Tiny Tabular Regression | `regression` | `autogluon.tabular` | `end_to_end` | `100%` | `100%` | `2.0` | `5.49s` | `N/A` |
| `case_03_house_price_faulty` | House Price Faulty Data Quality | `regression` | `autogluon.tabular` | `end_to_end` | `100%` | `67%` | `4.67` | `10.86s` | `N/A` |
| `case_04_tiny_timeseries` | Tiny Multi-Series Time-Series Forecasting | `time_series_forecasting` | `autogluon.timeseries` | `end_to_end` | `100%` | `100%` | `1.0` | `0.67s` | `N/A` |
| `case_05_tiny_image_cls` | Tiny Image Classification | `multimodal` | `autogluon.multimodal` | `end_to_end` | `100%` | `100%` | `1.0` | `0.59s` | `N/A` |
| `case_06_tiny_text_cls` | Tiny Text Classification | `classification` | `autogluon.tabular` | `end_to_end` | `100%` | `100%` | `2.0` | `6.78s` | `N/A` |
| `case_07_tiny_multimodal` | Tiny Multimodal Text & Tabular | `classification` | `autogluon.tabular` | `end_to_end` | `100%` | `100%` | `2.0` | `6.97s` | `N/A` |
| `case_08_tiny_retrieval` | Tiny Dense Retrieval Corpus | `retrieval` | `FlagEmbedding` | `end_to_end` | `100%` | `100%` | `1.0` | `0.14s` | `N/A` |
| `case_09_readme_described` | README Described Task | `classification` | `autogluon.tabular` | `end_to_end` | `100%` | `100%` | `2.0` | `7.20s` | `N/A` |
| `case_10_mixed_directory` | Mixed Directory Perception Case | `classification` | `autogluon.tabular` | `perception_only` | `100%` | `100%` | `0.0` | `0.00s` | `N/A` |

## 8. Comparison with MLZero Paper

> [!IMPORTANT]
> **Experimental Methodology Alignment vs. Local Benchmark Scale**
> 
> - **Paper Methodology:** The MLZero NeurIPS 2025 paper defines an agentic AutoML pipeline evaluated across complex Kaggle/OpenML benchmarks using multi-run averaging, relative time efficiency, and component ablations.
> - **Our Local Implementation:** This framework reproduces the exact evaluation structure (perception routing accuracy, 3-run averaging, recovery rates, memory/judge ablations, retrieval top-k depth, data-noise robustness) using lightweight synthetic/faulty test fixtures suitable for a local environment.
> - **No Benchmark Claims:** This evaluation measures paper-aligned local system performance and does NOT claim numerical identity with the paper's GPU-clustered benchmark scale.
