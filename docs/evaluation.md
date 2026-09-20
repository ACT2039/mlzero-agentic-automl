# MLZero Evaluation Methodology & Framework

## Overview

The MLZero Evaluation Framework provides a systematic, scientifically reproducible evaluation methodology aligned with the **MLZero NeurIPS 2025** paper architecture.

It measures pipeline success rates, error recovery, library selection accuracy, task perception accuracy, task-aware ML metrics, ablation contributions, retrieval depth effects, and data-noise robustness.

---

## Evaluation Architecture

```
Benchmark Cases (10 Cross-Modality Datasets)
                   │
                   ▼
  Evaluation Runner (`evaluation/runner.py`)
    ├── Mode: smoke (Fast 1-run check)
    ├── Mode: local_formal (3-Run Statistical Averaging)
    ├── Mode: ablation (Memory, Judge, Retrieval Top-K)
    ├── Mode: robustness (Controlled Data Noise Perturbations)
    └── Mode: real_llm (Opt-in Gemini Evaluation)
                   │
                   ▼
  MLZero Agentic Pipeline Engine
                   │
                   ▼
  Statistical Aggregator (`evaluation/aggregator.py`)
                   │
                   ▼
  Reports & Artifacts Generator (`evaluation/report.py`)
    ├── reports/evaluation_summary.csv
    ├── reports/evaluation_summary.json
    └── reports/evaluation_report.md
```

---

## 10 Benchmark Cross-Modality Evaluation Cases

| Case ID | Case Name | Modality | Task Type | Expected Library |
|---|---|---|---|---|
| `case_01_tiny_cls` | Tiny Tabular Classification | Tabular | `classification` | `autogluon.tabular` |
| `case_02_tiny_reg` | Tiny Tabular Regression | Tabular | `regression` | `autogluon.tabular` |
| `case_03_house_price_faulty` | House Price Faulty Data Quality | Tabular | `regression` | `autogluon.tabular` |
| `case_04_tiny_timeseries` | Tiny Multi-Series Time-Series | Tabular | `time_series_forecasting` | `autogluon.timeseries` |
| `case_05_tiny_image_cls` | Tiny Image Classification | Image | `multimodal` | `autogluon.multimodal` |
| `case_06_tiny_text_cls` | Tiny Text Classification | Tabular | `classification` | `autogluon.tabular` |
| `case_07_tiny_multimodal` | Tiny Multimodal Text & Tabular | Tabular | `classification` | `autogluon.tabular` |
| `case_08_tiny_retrieval` | Tiny Dense Retrieval Corpus | JSON | `retrieval` | `FlagEmbedding` |
| `case_09_readme_described` | README Described Task | Tabular | `classification` | `autogluon.tabular` |
| `case_10_mixed_directory` | Mixed Directory Perception Case | Mixed | `classification` | `autogluon.tabular` |

---

## Evaluation Metrics

1. **Pipeline Metrics:**
   - **Success Rate:** Proportion of runs completing successfully.
   - **First-Attempt Success Rate:** Proportion of runs succeeding on iteration 1.
   - **Recovery Rate:** `recovered_runs / initially_failed_runs` (returns `null` if no initial failures).
   - **Average Iterations:** Mean number of code generation attempts.
   - **Relative Time Efficiency:** `baseline_time / configuration_time`.

2. **Perception Routing Metrics:**
   - **Library Selection Accuracy:** `correct_library_selections / total_cases`.
   - **Task Perception Accuracy:** `correct_task_type_perceptions / total_cases`.
   - **Target Detection Accuracy:** `correct_target_detections / labeled_cases`.

3. **Task-Aware ML Quality Metrics:**
   - **Classification:** Accuracy, F1, Balanced Accuracy, MCC.
   - **Regression / Time-Series:** MAE, RMSE, $R^2$.
   - **Retrieval:** Recall@K, Precision@K, MRR.

---

## Experimental Ablation Studies

- **Ablation 1: Semantic Memory (ON vs OFF):** Measures the contribution of semantic knowledge retrieval to iteration reduction and ML quality.
- **Ablation 2: Episodic Memory (ON vs OFF):** Measures the contribution of chronological error history to code recovery.
- **Ablation 3: Execution Judge (ON vs OFF):** Measures the impact of LLM execution judgment vs. raw execution exit codes.
- **Ablation 4: Retrieval Depth ($K \in \{0, 1, 3, 5, 10\}$):** Evaluates how retrieval depth impacts context relevance and code quality.

---

## Controlled Data Noise Robustness Framework

Evaluates system resilience under 4 controlled dataset anomalies:
1. `missing_values`: Blanks injected into feature columns.
2. `malformed_numerics`: Strings ("N/A") injected into numeric columns.
3. `extra_columns`: Irrelevant noise columns injected.
4. `schema_mismatch`: Column mismatch between train and test splits.

---

## 3-Run Methodology & Reproducibility

For formal local evaluation (`--mode local_formal`), the evaluator executes **3 independent runs per case** and aggregates metrics into mean $\pm$ std summaries.

Every report automatically captures reproducibility metadata:
- Timestamp (UTC)
- Python Version
- Operating System & Platform
- Git Commit Hash
- LLM Mode (`mock` or `real`)
- Runs per case

---

## Comparison with MLZero NeurIPS 2025 Paper

> [!IMPORTANT]
> **Methodological Alignment vs. Local Benchmark Scale**
> 
> - **Paper Methodology:** The MLZero NeurIPS 2025 paper defines an agentic AutoML pipeline evaluated across complex Kaggle/OpenML benchmarks using multi-run averaging, relative time efficiency, and component ablations.
> - **Our Implementation:** This evaluation framework reproduces the exact paper methodology (perception routing accuracy, 3-run statistical averaging, recovery rates, memory/judge ablations, retrieval top-k depth, data-noise robustness) using lightweight synthetic/faulty test fixtures suitable for a local environment.
> - **No Benchmark Claims:** This evaluation measures paper-aligned local system performance and does NOT claim numerical identity with the paper's GPU-clustered benchmark scale.
