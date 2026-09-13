# Evaluation Methodology

## Overview

The Phase 9 evaluation framework measures the performance of MLZero under
controlled conditions using small public datasets and a deterministic mock LLM.

---

## Experimental Configurations

### Baseline (No Memory)

- Perception → Coder → Executor
- Semantic Memory: **disabled**
- Episodic Memory: **disabled**

Represents a naive LLM-based code generator without retrieval or history.

### Ablation A: Semantic Memory

- Configuration A: Semantic Memory **disabled**
- Configuration B: Semantic Memory **enabled**

Isolates the contribution of semantic retrieval to success rate and
iteration count.

### Ablation B: Episodic Memory

- Configuration A: Episodic Memory **disabled**
- Configuration B: Episodic Memory **enabled**

Isolates the contribution of episodic history to error recovery.

### Full System

All components enabled:
Perception → Semantic Memory → Coder → Executor → Error Analyzer → Episodic Memory → Retry

---

## Datasets

| Name | Type | Rows | Features | Source | License |
|---|---|---|---|---|---|
| binary_cls | Binary Classification | 100 | 2 | Synthetic (numpy) | Public Domain |
| multi_cls | Multiclass Classification | 150 | 4 | Iris (sklearn) | BSD-3 |
| regression | Regression | 442 | 10 | Diabetes (sklearn) | BSD-3 |

---

## Metrics

| Metric | Description |
|---|---|
| Success Rate | Proportion of runs that complete without system crash |
| Avg Iterations | Mean number of code generation attempts |
| Execution Time | Wall-clock time from submission to completion |
| Recovery Rate | Proportion of initially-failed runs that recover |
| ML Accuracy | AutoGluon accuracy on training data (classification) |
| ML F1 | AutoGluon F1 on training data (classification) |
| ML MAE | Mean Absolute Error (regression) — if available |
| ML RMSE | Root Mean Squared Error (regression) — if available |

Metrics marked "if available" may not be reported by AutoGluon for all
configurations and are recorded as `null` rather than fabricated.

---

## Reproducibility

All experiments use `mock_llm: true`. The mock LLM is deterministic:

- **Iteration 1**: generates code with deliberate label typo → `KeyError`
- **Iteration 2**: generates corrected code → AutoGluon trains successfully

This makes every run an exact replay of the same failure-recovery scenario,
enabling fair comparison across ablation configurations.

---

## Limitations

1. **Single-run results**: no multi-seed averaging due to time constraints.
2. **Mock LLM only**: real LLM variability is not measured.
3. **Tiny datasets**: results may not generalise to large datasets.
4. **CPU-only**: no GPU acceleration; training times may be higher than
   production systems.
5. **No multi-user load testing**: the run manager is process-local.
