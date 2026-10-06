Optional user instruction:
"Load broken.csv and train a classifier."

Intentional fault:
- Quoting/comma structure is malformed.

Expected:
- Loader/parser failure should be captured cleanly.
- Error Analyzer category: parse_error / malformed_file.
- No code execution loop should continue pretending a valid dataframe exists.
- Final status should be FAIL/NEEDS_INPUT unless a parser-repair step is implemented.
