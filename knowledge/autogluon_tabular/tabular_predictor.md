# AutoGluon Tabular: TabularPredictor API Reference

`TabularPredictor` is the primary entry point for training models on structured tabular data.

## API Specification
```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(
    label: str,                     # Column name of the prediction target
    problem_type: str = None,       # 'binary', 'multiclass', 'regression', or None (inferred)
    eval_metric: str = None,        # 'accuracy', 'f1', 'roc_auc', 'rmse', 'r2', etc.
    path: str = "out/models",       # Directory where trained models and artifacts are saved
    verbosity: int = 2              # Verbosity level (0 to 4)
)

predictor.fit(
    train_data: pd.DataFrame,       # Training dataset
    time_limit: int = 60,           # Total training budget in seconds
    presets: str = "medium_quality",# 'best_quality', 'high_quality', 'medium_quality'
    holdout_frac: float = None      # Validation split fraction
)

# Inference and Evaluation
predictions = predictor.predict(test_data)
eval_results = predictor.evaluate(test_data)
```
