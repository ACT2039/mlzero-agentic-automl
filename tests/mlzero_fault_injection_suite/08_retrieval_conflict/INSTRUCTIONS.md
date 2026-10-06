Optional user instruction:
"Retrieve the most relevant guidance for repairing a ValueError caused by mixed numeric strings in an AutoGluon classification pipeline."

Expected:
- Retrieval should rank doc_error.md highly.
- doc_bad.md should not dominate merely because it mentions AutoGluon.
- With K=5, returned context should include the most relevant repair guidance.
- If the pipeline logs retrieval scores, doc_error.md should be among the top results.
