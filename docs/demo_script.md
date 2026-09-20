# 5–10 Minute Live Demonstration Script — MLZero

This script guides a step-by-step 5 to 10 minute live demonstration of the **MLZero Agentic AutoML** system for evaluators, instructors, or viva examiners.

---

## 📋 Demo Overview & Timeline

| Time | Demonstration Step | Goal |
|---|---|---|
| **0:00 – 1:30** | 1. System Architecture Overview | Introduce multi-agent pillars and paper inspiration. |
| **1:30 – 3:00** | 2. Offline CLI Execution | Run standard tiny tabular classification workflow. |
| **3:00 – 6:00** | 3. Error Recovery & Memory Walkthrough | Demonstrate deliberate failure, `FIX` decision, and iteration 2 repair. |
| **6:00 – 7:30** | 4. Gradio Web UI & API Backend | Launch interactive UI and inspect execution timeline. |
| **7:30 – 9:00** | 5. Evaluation Suite & Provenance Reports | Show evaluation runner and machine-readable JSON artifacts. |
| **9:00 – 10:00** | 6. RealLLM Mode & Conclusion | Explain Gemini API integration and paper-alignment bounds. |

---

## Step 1: System Architecture Overview (1.5 mins)

- **Action**: Open [docs/architecture.md](file:///c:/Users/LENOVO/OneDrive/Desktop/mlzero-agentic-automl/docs/architecture.md) or display the high-level architecture diagram.
- **Narrative**:
  > "MLZero is a multi-agent system designed to automate machine learning end-to-end. Rather than relying on static scripts, MLZero combines 4 major pillars:
  > 1. Perception to understand data formats and select libraries.
  > 2. Semantic Memory for documentation retrieval.
  > 3. Episodic Memory to track chronological failure history.
  > 4. Execution Judge and Retry Loop to evaluate code output and fix errors automatically."

---

## Step 2: Offline CLI Execution (1.5 mins)

- **Action**: Open terminal and execute:
  ```bash
  test_venv\Scripts\python -m mlzero run --input tests/data/tiny_classification --mock-llm
  ```
- **Narrative**:
  > "We start by running an offline end-to-end execution on a clean dataset using `MockLLM`. Perception scans `tests/data/tiny_classification`, identifies the task type as classification, routes to `autogluon.tabular`, generates code, executes in a sandboxed subprocess, and receives a `FINISH` decision from the Execution Judge on iteration 1."

---

## Step 3: Error Recovery & Closed-Loop Memory (3 mins)

- **Action**: Run the faulty dataset benchmark:
  ```bash
  test_venv\Scripts\python -m mlzero run --input tests/data/house_price_faulty --mock-llm
  ```
- **Highlight Logs**:
  1. **Iteration 1 Failure**: Code generated with missing target variable name fails execution.
  2. **Execution Judge `FIX`**: Judge evaluates error logs and requests a retry.
  3. **Error Analyzer Diagnosis**: Identifies `KeyError` and formulates code remediation guidance.
  4. **Episodic Memory Persistence**: Logs failed iteration 1 record into JSON store.
  5. **Semantic Retrieval**: Queries domain knowledge for error recovery patterns.
  6. **Iteration 2 Repair**: Coder receives error context and generates corrected script.
  7. **Execution Judge `FINISH`**: Iteration 2 succeeds cleanly; prediction artifact generated.
- **Narrative**:
  > "Here we see MLZero's error recovery in action. On iteration 1, the script encounters a data quality anomaly. The Execution Judge issues a `FIX` decision, triggering the Error Analyzer. Episodic Memory records the failure, and Semantic Memory retrieves targeted fix guidance. On iteration 2, Coder fixes the script, achieving a `FINISH` decision."

---

## Step 4: Gradio Web UI & API Backend (1.5 mins)

- **Action**: Launch the Web UI in terminal:
  ```bash
  test_venv\Scripts\python -m mlzero ui --port 7860
  ```
- Open browser at `http://127.0.0.1:7860`.
- Select `Mock LLM`, enter input directory `tests/data/tiny_classification`, and click **Run MLZero Execution**.
- Demonstrate:
  - Perceptual Context display (selected library, task type).
  - Iteration History Timeline (`Iteration 1: SUCCESS - FINISH`).
  - Generated Python script and predictions summary artifact.

---

## Step 5: Evaluation Suite & Provenance Reports (1.5 mins)

- **Action**: Execute the formal evaluation smoke runner in terminal:
  ```bash
  test_venv\Scripts\python -m evaluation.runner --mode smoke
  ```
- Show generated machine-readable report artifacts under `reports/`:
  - [reports/evaluation_report.md](file:///c:/Users/LENOVO/OneDrive/Desktop/mlzero-agentic-automl/reports/evaluation_report.md)
  - `reports/evaluation_runs.json`
  - `reports/ablation_runs.json`
  - `reports/robustness_runs.json`
- **Narrative**:
  > "To validate scientific reproducibility, our Stage 7 Evaluation Framework runs 10 cross-modality benchmark cases. It logs raw machine-readable JSON traces containing execution judge decisions, component ablation records, and data-noise robustness metrics."

---

## Step 6: RealLLM Mode & Conclusion (1 min)

- **Action**: Demonstrate the CLI `perceive` command using real LLM mode options:
  ```bash
  test_venv\Scripts\python -m mlzero perceive --input tests/data/tiny_classification --llm-mode real --json
  ```
- **Conclusion Statement**:
  > "In summary, MLZero successfully bridges multi-agent reasoning, multimodal dataset perception, domain memory, and automated error recovery across multiple ML frameworks. All unit tests, type checks, and evaluation suites are 100% verified."
