# General ML: Preprocessing, Data Hygiene, and Evaluation

Standard machine learning practices for tabular classification and regression pipelines.

## Preprocessing Patterns
- **Imputation**: Missing values in continuous features should be imputed using training set statistics (e.g. median or mean).
- **Target Integrity**: Never impute missing target values with mean/median for classification or regression; drop rows where the ground truth label is missing or malformed.
- **Categorical Encoding**: Ensure unseen categories in test data are gracefully mapped to unknown or handled by high-cardinality encoders.

## Evaluation Metrics
- **Classification**: Accuracy, F1-score (macro/weighted), ROC-AUC, Log Loss.
- **Regression**: RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), R-squared.
