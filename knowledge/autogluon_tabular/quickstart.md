# AutoGluon Tabular Quickstart

AutoGluon Tabular enables automated machine learning for tabular datasets.
It automatically performs feature preprocessing, model selection, hyperparameter tuning, and ensemble generation.

## Basic Usage
```python
import pandas as pd
from autogluon.tabular import TabularPredictor

# 1. Load data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# 2. Fit predictor
predictor = TabularPredictor(label="target", path="out/models").fit(
    train_data,
    time_limit=60,
    presets="medium_quality"
)

# 3. Predict on test set
predictions = predictor.predict(test_data)
predictions.to_csv("out/predictions.csv", index=False)
```

## Problem Types
AutoGluon automatically infers `problem_type` (e.g. `'binary'`, `'multiclass'`, `'regression'`).
To specify explicitly:
```python
predictor = TabularPredictor(label="price", problem_type="regression", path="out/models")
```
