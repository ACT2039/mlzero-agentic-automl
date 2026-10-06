Optional user instruction:
"Train on train.csv and predict test.csv."

Intentional fault:
- train uses income
- test uses salary
- common feature schema is inconsistent

Expected:
- Schema validator should detect feature mismatch.
- Error category: schema_mismatch.
- Pipeline should request/recover a mapping only if it has enough evidence; otherwise stop safely.
- It must not silently rename salary to income without justification.
