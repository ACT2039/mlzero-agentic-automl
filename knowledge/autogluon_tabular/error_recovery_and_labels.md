# AutoGluon Tabular: Label Column Handling and KeyError Recovery

When initializing `TabularPredictor(label='...')`, the label column name MUST exist exactly as spelled in the training DataFrame.

## Common Error: KeyError on Label Column
If the training DataFrame has columns `['feature1', 'feature2', 'target']` and the code passes `label='targt'` or an incorrect column name:
```
KeyError: 'targt'
```
Pandas and AutoGluon will raise a `KeyError` because the key `'targt'` does not match `'target'`.

## Recovery and Fix
1. Inspect the columns of `train_df`:
   ```python
   print(train_df.columns.tolist())
   ```
2. Verify that `label` corresponds exactly to the actual target column name in the dataset (e.g. `'target'` or `'price'`).
3. If there was a typo in the label parameter, correct it:
   ```python
   # Corrected:
   predictor = TabularPredictor(label="target", path="out/models")
   ```
4. Do not drop or rename the label column before passing the DataFrame to `predictor.fit()`.
