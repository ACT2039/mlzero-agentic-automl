Optional user instruction:
"Train a classifier using target column target_label."

Intentional fault:
- Actual target is named label, but user requests target_label.

Expected:
- Task perception or validation should flag the target mismatch.
- Error category should be target/ schema mismatch.
- A good agent should ask for clarification or recover only if it can safely infer that label is the intended target.
- Do not silently invent a target column.
