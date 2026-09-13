# Viva Q&A Reference — MLZero Implementation

This document answers the key technical questions that may be asked during
a B.Tech viva for the MLZero Agentic AutoML project.

---

## 1. What is AutoML?

**AutoML** (Automated Machine Learning) is the process of automating the
end-to-end steps required to train and deploy a machine learning model —
including data preprocessing, feature engineering, model selection,
hyperparameter optimisation, and evaluation.

In this project, we use **AutoGluon Tabular** as the AutoML engine. Instead
of manually writing model training code, the agent generates Python code that
calls AutoGluon's `TabularPredictor`, which internally handles model selection
and tuning.

---

## 2. What is an LLM agent?

An **LLM agent** is a software component that uses a Large Language Model as
its reasoning engine. The agent receives structured context (e.g. dataset
schema, task description, error messages), sends a prompt to the LLM, and
parses the response into an action (e.g. generated code, error category).

In this project, agents such as `CoderAgent`, `TaskPerceptionAgent`, and
`ErrorAnalyzerAgent` are LLM agents. They wrap the `LLMClient` abstraction,
which supports both a real API client and a `MockLLMClient` for offline tests.

---

## 3. Why multiple agents?

**Separation of concerns.** Each agent has a single, well-defined
responsibility:

- `FilePerceptionAgent` — file I/O and schema detection
- `TaskPerceptionAgent` — natural-language task understanding
- `LibrarySelectorAgent` — library recommendation
- `CoderAgent` — code synthesis
- `ExecutorAgent` — safe execution
- `ErrorAnalyzerAgent` — failure diagnosis

This design makes the system easier to test, debug, and extend without
modifying unrelated components.

---

## 4. What is perception?

**Perception** is the system's ability to observe and understand the
input environment before acting. In MLZero, perception consists of three
agents that collectively:

1. Scan the dataset directory and infer file formats, column names, and
   approximate size (`FilePerceptionAgent`).
2. Understand the ML task — classification vs regression, target column,
   evaluation metric — using the LLM (`TaskPerceptionAgent`).
3. Select the most appropriate ML library (`LibrarySelectorAgent`).

The combined output is a `PerceptualContext` schema object that is passed
to all downstream agents.

---

## 5. What is semantic memory?

**Semantic memory** answers: *"What knowledge is relevant to this task?"*

It is implemented as a retrieval-augmented system. Text documents
(e.g. API docs, code examples) are ingested, chunked, and indexed using
TF-IDF vectors (or FAISS if available). At query time, the most relevant
chunks are retrieved and injected into the Coder prompt.

In this project the index is stored entirely in-process (no external
vector database). See `mlzero/memory/semantic.py`.

---

## 6. What is episodic memory?

**Episodic memory** answers: *"What happened in previous iterations of
this run?"*

It maintains a chronological log of every iteration: the code that was
generated, whether execution succeeded, the error category if it failed,
and the suggested fix. This history is serialised as JSON to disk so it
survives process restarts.

The coder receives a compact summary of recent failures so it does not
repeat the same mistake.

See `mlzero/memory/episodic.py` and `mlzero/memory/episodic_store.py`.

---

## 7. Why are semantic and episodic memory separate?

They serve fundamentally different purposes:

- **Semantic memory** is *static* prior knowledge — it is built once
  from documents and reused across many runs.
- **Episodic memory** is *dynamic* run-specific history — it grows with
  each iteration and is unique to a single execution.

Mixing them would couple unrelated concerns and make both harder to test
and replace.

---

## 8. How does iterative coding work?

The `IterativeCodingOrchestrator` runs a loop with a configurable maximum
(default 10 iterations):

```
for iteration in 1..max_iterations:
    retrieve semantic context
    generate code (CoderAgent)
    execute code (ExecutorAgent)
    if success → break
    analyse error (ErrorAnalyzerAgent)
    record episode (EpisodicMemory)
    # next iteration receives error context + episodic history
```

Each iteration's `CoderAgent` call includes:
- the original perceptual context
- the error context from the previous iteration
- a summary of recent episodic failures

---

## 9. How is generated code executed?

`ExecutorAgent` delegates to `PythonRunner`, which:

1. Creates a temporary subdirectory under `outputs/workspaces/exec_<id>/`.
2. Writes the generated code as `script.py`.
3. Writes any input data files into the workspace.
4. Calls `subprocess.run([python, "script.py"], ...)` with a configurable
   timeout (default 120 s).
5. Captures stdout/stderr (truncated at configured limits).
6. Collects output files from the `out/` subdirectory.
7. Returns an `ExecutionResult` schema object.

---

## 10. How is security handled?

- **Path traversal prevention**: input filenames are validated with
  `.resolve()` + `.parents` check before writing.
- **Execution timeout**: subprocess is killed after `timeout_seconds`.
- **Output size limit**: stdout/stderr are truncated at `max_stdout_size_bytes`.
- **API path constraint**: the FastAPI backend validates all dataset paths
  against `settings.app.allowed_data_root`.
- **No secret exposure**: API responses never include `.env` contents,
  source code, `.git` history, or model binary blobs.
- **No arbitrary command endpoint**: there is no generic shell execution
  endpoint — only the structured `/runs` POST which calls `MLZeroService`.

---

## 11. How does the library selector work?

`LibrarySelectorAgent` sends a prompt to the LLM containing the task
context (task type, dataset schema) and asks it to select the most
appropriate library from a configured allowlist.

In the mock LLM, it always returns `autogluon.tabular`. In production,
the real LLM selects based on task type, dataset size hints, and
available libraries.

---

## 12. Why AutoGluon?

- **State-of-the-art tabular performance** out of the box with a single
  `TabularPredictor.fit()` call.
- **No manual model selection** — AutoGluon ensembles multiple models
  internally.
- **CPU-compatible** — runs without a GPU, which suits the project's
  hardware constraints.
- **Active maintenance** — well-documented, regularly updated.

---

## 13. What happens when code fails?

1. `ExecutorAgent` returns `ExecutionResult(success=False)`.
2. `ErrorAnalyzerAgent` receives the failing code and stderr.
3. It sends a structured prompt to the LLM asking for an error category
   and suggested fix.
4. The result is an `ErrorContext` (error category, iteration number,
   suggested fix, relevant error lines).
5. `EpisodicMemory` records the failed iteration with the error context.
6. The orchestrator calls `CoderAgent` again, this time passing the
   error context and episodic history.

---

## 14. How is the error repaired?

The `CoderAgent` receives a prompt that includes:
- the original task description
- the previously failing code
- the `ErrorContext.suggested_fix` from the error analyser
- a compact list of recent failures from episodic memory

The LLM uses this context to generate a corrected version of the code.
In the mock LLM, the correction is hard-coded: Iteration 1 uses the
wrong label column; Iteration 2 fixes it.

---

## 15. What are the limitations?

| Limitation | Detail |
|---|---|
| Mock LLM | Real experiments require a valid API key and incur cost |
| CPU-only | AutoGluon uses CPU models; no deep learning |
| Process-local run manager | The `RunManager` state is not shared across processes or restarts |
| No distributed execution | Single-machine only; not Kubernetes/Celery |
| Single ML framework | Only AutoGluon Tabular is integrated; no image/text/NLP tasks |
| Dataset size | Tested only on tiny datasets; large datasets may time out |
| Security isolation | `PythonRunner` uses a subprocess but not a container or VM |
| Single-run evaluation | No multi-seed averaging; results may not be statistically robust |
