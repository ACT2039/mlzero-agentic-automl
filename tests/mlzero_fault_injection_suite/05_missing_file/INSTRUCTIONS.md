Optional user instruction:
"Use train.csv and test.csv to train and generate test predictions."

Intentional fault:
- test.csv is missing.

Expected:
- File Perception should report train.csv but missing test.csv.
- Pipeline should not fabricate test data.
- Error Analyzer category: missing_file / data availability.
- Expected final status: FAIL/NEEDS_INPUT, unless the pipeline has a documented fallback mode.
