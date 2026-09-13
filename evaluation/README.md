"""
Evaluation README
=================

This directory contains the MLZero evaluation framework for the Phase 9
B.Tech final delivery.

## Structure

```
evaluation/
├── datasets/          # Small benchmark datasets (generated locally)
│   ├── binary_cls/    # Binary classification (synthetic, 100 rows)
│   ├── multi_cls/     # Multiclass classification (Iris, 150 rows)
│   └── regression/    # Regression (Diabetes sklearn, 442 rows)
├── configs/           # JSON experiment configurations
│   ├── baseline.json          # Coder + Executor only (no memory)
│   ├── ablation_semantic.json # With/without semantic memory
│   ├── ablation_episodic.json # With/without episodic memory
│   └── full_system.json       # Full pipeline enabled
├── runners/           # (future: parallel experiment runners)
├── scenarios/         # (future: scenario-specific tests)
├── metrics.py         # Metric extraction helpers
├── experiments.py     # Core experiment runner
├── report.py          # Report and plot generator
└── README.md          # This file
```

## Datasets

| Name        | Type          | Source           | License     | Rows | Features |
|-------------|---------------|------------------|-------------|------|----------|
| binary_cls  | Binary cls    | Synthetic (numpy)| Public domain | 100 | 2 |
| multi_cls   | Multiclass    | Iris (sklearn)   | BSD-3       | 150  | 4        |
| regression  | Regression    | Diabetes (sklearn)| BSD-3      | 442  | 10       |

Datasets are NOT committed to git (listed in .gitignore).
Run `python -m mlzero evaluate --config evaluation/configs/full_system.json` to regenerate.

## Running Experiments

```bash
# Generate datasets first
python -c "
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.datasets import load_iris, load_diabetes
..."

# Run a specific config
python -m mlzero evaluate --config evaluation/configs/baseline.json

# Run full system
python -m mlzero evaluate --config evaluation/configs/full_system.json
```

## Reproducibility

All experiments use `mock_llm: true` so no API key is needed and
results are deterministic. The mock LLM always:
1. Generates intentionally faulty code on Iteration 1
2. Generates correct AutoGluon code on Iteration 2

This exercises the full error-recovery loop in a controlled manner.

## Output

Reports are written to `reports/`:
- `evaluation_summary.json`  — structured JSON of all results
- `evaluation_summary.csv`   — tabular CSV for spreadsheet analysis
- `evaluation_report.md`     — human-readable Markdown report
- `figures/`                 — matplotlib PNG plots

## Limitations

- All experiments use the same mock LLM, so LLM variation is not measured.
- Results reflect single-run, not multi-run averages.
- GPU-free execution; AutoGluon uses CPU-only models.
- Dataset sizes are intentionally tiny for hardware constraints.
"""
