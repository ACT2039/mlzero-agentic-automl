# MLZero Fault Injection Suite

This suite is designed to stress perception, routing, semantic retrieval, episodic memory, error analysis, iterative repair, and safe failure handling.

Recommended order:
1. 01_classification_malformed
2. 02_regression_target_fault
3. 03_timeseries_irregular
4. 07_multimodal_image_metadata
5. 08_retrieval_conflict
6. 09_schema_mismatch
7. 04_wrong_target
8. 05_missing_file
9. 06_malformed_csv

For each dataset, first run with NO user instruction to test autonomous perception. Then rerun with the optional instruction in INSTRUCTIONS.md to test instruction-conditioned planning.

Success criteria:
- correct task/modality detection
- correct target/time/group inference
- correct library routing
- useful retrieval context
- structured fault diagnosis
- autonomous repair when repair is safe
- safe failure when required information is genuinely missing
- no fabricated files/columns/data
- output artifacts and logs are persisted
