Optional user instruction:
"Predict house price. Treat price as the target. Clean malformed area values and handle an unseen test category in location. Save id and predicted price."

Faults:
- train area contains text mixed with a number
- test contains unseen category D

Expected:
- Task = regression
- Target = price
- Pipeline should repair/coerce area and use an encoder/model that tolerates unseen category D.
- predictions.csv should contain 3 rows and columns id,price_pred.
