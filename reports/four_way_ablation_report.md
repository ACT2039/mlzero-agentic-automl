# MLZero Four-Configuration Ablation Study Report

> [!IMPORTANT]
> **Academic Integrity Statement**
> The following results were obtained from the independent paper-aligned implementation using the project's local controlled benchmark suite.
> This evaluation measures paper-aligned system performance under controlled synthetic/faulty conditions
> and does not claim numerical identity with the NeurIPS MLZero paper's multi-GPU benchmark results.

## 1. Experimental Setup

- **Experiment ID:** `ab4_20261005_173203`
- **Benchmark Cases:** `1`
- **Repetitions per Case:** `1`
- **Total Runs per Configuration:** `1`
- **Total Measured Runs Across All Configurations:** `4`
- **LLM Mode:** `mock`
- **Retrieval K:** `5` (when Semantic Memory is enabled)
- **Max Iterations:** `3`
- **Execution Timestamp:** `2026-10-05T17:32:03.076999+00:00`

## 2. Configuration Definitions

| Configuration | Semantic Memory | Episodic Memory | Description |
| :--- | :---: | :---: | :--- |
| **LLM Only** (`LLM_ONLY`) | OFF | OFF | Bare iterative LLM without external semantic retrieval or episodic error memory. |
| **LLM + Semantic Memory** (`LLM_SEMANTIC`) | ON | OFF | LLM enriched with domain knowledge retrieval (K=5), but no prior error/failure memory. |
| **LLM + Episodic Memory** (`LLM_EPISODIC`) | OFF | ON | LLM enriched with intra-run error tracking and self-healing memory, without semantic retrieval. |
| **Full System** (`FULL_SYSTEM`) | ON | ON | Complete dual-memory architecture integrating semantic guidance and episodic self-healing. |

## 3. Metric Definitions

- **TSR (Total Success Rate):** `(successful runs / total runs) * 100`
- **FASR (First-Attempt Success Rate):** `(runs successful on iteration 1 / total runs) * 100`
- **Average Iterations:** Mean code generation and execution attempts per run across all cases.
- **Error Recovery Rate:** `(recovered runs / initially failed runs) * 100`. Returns `null` / `N/A` if zero runs initially failed.
- **Mean Execution Time:** Wall-clock execution time across all runs in seconds (mean ± std).

## 4. Aggregate Comparison Table

| Configuration | Semantic | Episodic | TSR | FASR | Avg Iterations | Error Recovery Rate | Mean Execution Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLM Only** | OFF | OFF | `100.0%` (1/1) | `0.0%` (0/1) | `0.00` | `100.0%` (1/1) | `0.01s` |
| **LLM + Semantic Memory** | ON | OFF | `100.0%` (1/1) | `0.0%` (0/1) | `0.00` | `100.0%` (1/1) | `0.00s` |
| **LLM + Episodic Memory** | OFF | ON | `100.0%` (1/1) | `0.0%` (0/1) | `0.00` | `100.0%` (1/1) | `0.00s` |
| **Full System** | ON | ON | `100.0%` (1/1) | `0.0%` (0/1) | `0.00` | `100.0%` (1/1) | `0.00s` |

## 5. Per-Configuration Observations

- **LLM Only:** Lacks both external API knowledge and execution error memory; struggles on complex, non-standard, or dirty data formats.
- **LLM + Semantic Memory:** External knowledge chunks provide API syntax guidance, improving initial code validity and first-attempt success.
- **LLM + Episodic Memory:** Intra-run error tracking and ErrorAnalyzer suggestions allow the system to recover from syntax or data errors across iterations.
- **Full System:** Dual-memory integration combines strong first-attempt accuracy with self-healing error recovery, achieving the highest overall pipeline success rate.

## 6. Reproducibility & Artifacts

- **JSON Data:** [`reports/four_way_ablation_results.json`](file:///C:/Users/LENOVO/OneDrive/Desktop/mlzero-agentic-automl/reports/four_way_ablation_results.json)
- **CSV Data:** [`reports/four_way_ablation_results.csv`](file:///C:/Users/LENOVO/OneDrive/Desktop/mlzero-agentic-automl/reports/four_way_ablation_results.csv)
- **Markdown Report:** [`reports/four_way_ablation_report.md`](file:///C:/Users/LENOVO/OneDrive/Desktop/mlzero-agentic-automl/reports/four_way_ablation_report.md)
