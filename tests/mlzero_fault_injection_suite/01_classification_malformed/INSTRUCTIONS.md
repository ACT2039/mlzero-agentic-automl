Optional user instruction:
"Build a binary classification model to predict purchased. Handle malformed numeric values and missing categorical values robustly. Save predictions to predictions.csv with columns id,purchased_pred."

Faults intentionally injected:
- income contains a non-numeric token
- age has a missing value
- city has a missing value

Expected successful outcome:
- Task = binary classification
- Target = purchased
- Pipeline should diagnose/coerce malformed income, handle missing values, train, and produce predictions.csv with 3 prediction rows.
- Expected artifact schema: id,purchased_pred
