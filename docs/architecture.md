# MLZero Architecture

## Overview

MLZero is a multi-agent system for end-to-end machine learning automation.
It is inspired by the paper *"MLZero: A Multi-Agent System for End-to-End Machine Learning Automation"*
but represents an independent B.Tech implementation with distinct design choices.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MLZero System                               │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                     Application Layer                        │  │
│  │   CLI (mlzero run)   FastAPI (/runs)   Gradio UI             │  │
│  └─────────────────────────┬─────────────────────────────────────┘  │
│                            │ MLZeroService.run_mlzero()             │
│  ┌─────────────────────────▼─────────────────────────────────────┐  │
│  │                  Orchestration Layer                          │  │
│  │              IterativeCodingOrchestrator                      │  │
│  │   (max 10 iterations, error-recovery loop)                    │  │
│  └──┬────────────────────────────────────────────────────────────┘  │
│     │                                                               │
│  ┌──▼─────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Perception │  │   Coder      │  │  Executor    │               │
│  │  Phase 2   │  │  Phase 3     │  │  Phase 3     │               │
│  └──┬─────────┘  └──────┬───────┘  └──────┬───────┘               │
│     │                   │                 │                        │
│  ┌──▼─────────────────────────────────────▼───────────────────────┐  │
│  │               Error Analyzer (Phase 4)                        │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────┐   ┌───────────────────────────────────┐  │
│  │  Semantic Memory     │   │      Episodic Memory              │  │
│  │  Phase 5             │   │      Phase 6                      │  │
│  │  (FAISS / TF-IDF)    │   │      (JSON-based store)           │  │
│  └──────────────────────┘   └───────────────────────────────────┘  │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │               AutoGluon Tabular (Phase 7)                     │  │
│  │   TabularPredictor.fit() → predict() → predictions.csv        │  │
│  └────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Agent Roles

| Agent | Phase | Role |
|---|---|---|
| `FilePerceptionAgent` | 2 | Scans dataset files, infers schema |
| `TaskPerceptionAgent` | 2 | Extracts task type, label, constraints via LLM |
| `LibrarySelectorAgent` | 2 | Selects appropriate ML library |
| `CoderAgent` | 3 | Generates Python ML code via LLM |
| `ExecutorAgent` | 3 | Runs code in sandboxed subprocess |
| `ErrorAnalyzerAgent` | 4 | Categorises errors, proposes fixes via LLM |

## Memory Systems

| System | Role | Storage |
|---|---|---|
| `SemanticMemory` | Retrieves relevant prior knowledge | FAISS / TF-IDF in-process |
| `EpisodicMemory` | Records iteration history chronologically | JSON files on disk |

## Execution Flow

1. User submits dataset path + instruction
2. Perception agents analyse the dataset
3. Semantic memory retrieves relevant code patterns
4. `CoderAgent` generates ML code using perceptual context + retrieved knowledge
5. `ExecutorAgent` runs the code in an isolated subprocess (sandboxed env)
6. On failure: `ErrorAnalyzerAgent` categorises the error
7. `EpisodicMemory` records the failed iteration
8. Orchestrator retries (up to 10 iterations)
9. On success: result + prediction artefact path returned
10. `EpisodicMemory` records the successful iteration
