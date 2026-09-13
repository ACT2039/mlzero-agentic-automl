"""
MLZero Demo Script -- Phase 9
============================

Runs one complete end-to-end example through the full MLZero pipeline:

  Dataset -> Perception -> Library Selection -> Semantic Retrieval
    -> Coder -> Executor -> Error Analyzer -> Episodic Memory
    -> Retry -> Final Prediction

Designed to be deterministic (mock LLM) and concise for viva demonstrations.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# ── ensure project root is importable ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mlzero.application.service import MLZeroService
from mlzero.core.logger import setup_logger

logger = setup_logger("mlzero.demo")

DEMO_DATASET = str(ROOT / "evaluation" / "datasets" / "binary_cls")
DEMO_INSTRUCTION = (
    "Train a binary classifier. The label column is 'target'. "
    "Save predictions to out/predictions.csv."
)

DIVIDER = "=" * 60


def banner(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def run_demo() -> None:
    banner("MLZero Agentic AutoML -- Live Demo")
    print("\nDataset     :", DEMO_DATASET)
    print("Instruction :", DEMO_INSTRUCTION)
    print("LLM         : MockLLM (deterministic, offline)")

    # ── Phase 2: Perception (happens inside service) ──────────────────────
    banner("STEP 1 / 6 -- Perception")
    print("  FilePerceptionAgent   -> scanning dataset files")
    print("  TaskPerceptionAgent   -> extracting task context via LLM")
    print("  LibrarySelectorAgent  -> selecting ML library")

    # ── Phase 5: Semantic Memory (retrieval inside service) ───────────────
    banner("STEP 2 / 6 -- Semantic Memory Retrieval")
    print("  SemanticMemory.retrieve() -> finding relevant code patterns")
    print("  (empty index in demo -- no prior knowledge documents loaded)")

    # ── Phase 3+4+6: Coder -> Executor -> Error Analyzer -> Retry ──────────
    banner("STEP 3 / 6 -- Iterative Coding & Execution")
    print("  Iteration 1 -> CoderAgent generates AutoGluon code")
    print("  ExecutorAgent -> runs in sandboxed subprocess")
    print("  Expected: KeyError 'targt' (intentional deliberate failure)")
    print("  ErrorAnalyzerAgent -> categorises error, suggests fix")
    print("  EpisodicMemory -> records Iteration 1 failure")
    print()
    print("  Iteration 2 -> CoderAgent generates corrected code")
    print("  ExecutorAgent -> trains AutoGluon TabularPredictor")
    print("  EpisodicMemory -> records Iteration 2 SUCCESS")

    # ── Execute ─────────────────────────────────────────────────────────
    banner("STEP 4 / 6 -- Running Pipeline")
    service = MLZeroService(use_mock_llm=True)
    t0 = time.time()
    result = service.run_mlzero(
        dataset_path=DEMO_DATASET,
        user_instruction=DEMO_INSTRUCTION,
    )
    elapsed = time.time() - t0

    # ── Results ──────────────────────────────────────────────────────────
    banner("STEP 5 / 6 -- Results")
    print(f"  Status          : {result.status}")
    print(f"  Iterations      : {result.iterations}")
    print(f"  Execution Time  : {elapsed:.2f}s")
    print(f"  Library         : {result.selected_library}")

    if result.final_metrics:
        print("\n  ML Metrics:")
        for k, v in result.final_metrics.items():
            print(f"    {k:30s}: {v:.4f}" if isinstance(v, float) else f"    {k:30s}: {v}")

    if result.prediction_artifact_reference:
        pred_path = Path("outputs") / "models" / result.prediction_artifact_reference
        exists = pred_path.exists()
        print(f"\n  Prediction file : {result.prediction_artifact_reference}")
        print(f"  File on disk    : {'OK EXISTS' if exists else 'MISSING MISSING'}")

    if result.model_artifact_reference:
        print(f"  Model artifact  : {result.model_artifact_reference}")

    # ── Episodic Summary ─────────────────────────────────────────────────
    banner("STEP 6 / 6 -- Episodic Memory Summary")
    if service.episodic_memory:
        try:
            history = service.episodic_memory.store.get_run_history(result.run_id)
            if history:
                print(f"  Run ID          : {history.run_id}")
                print(f"  Total Iterations: {len(history.episodes)}")
                for ep in history.episodes:
                    err = (ep.error_context_summary or {}).get("error_category", "none")
                    print(f"    Iteration {ep.iteration}: {ep.status}  | error={err}")
            else:
                print("  (no episodic history -- episodic memory disabled or run ID not found)")
        except Exception as e:  # noqa: BLE001
            print(f"  (episodic history not available: {e})")
    else:
        print("  (episodic memory disabled)")

    banner("DEMO COMPLETE")
    print(f"  Total wall-clock time : {elapsed:.2f}s")
    print()


if __name__ == "__main__":
    run_demo()
