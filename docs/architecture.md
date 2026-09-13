# System Architecture

The MLZero system is designed as a collaborative multi-agent architecture divided into four primary components. This document describes the theoretical architecture; Phase 1 implements clean interfaces and placeholders for these components.

```
Input Dataset
      ↓
Perception
      ↓
Perceptual Context
      ↓
Semantic Memory
      +
Episodic Memory
      ↓
Coder
      ↓
Executor
      ↓
Error Analyzer
      ↓
Iterative Refinement
      ↓
Output
```

Later phases will implement these components incrementally.

## 1. Perception
Responsible for understanding the user's intent, the data provided, and the best tools to solve the problem.
- **File Perception Agent**: Analyzes datasets and scripts to extract metadata, schemas, and context.
- **Task Perception Agent**: Parses natural language instructions from the user to formalize goals and constraints.
- **ML Library Selection Agent**: Recommends the best frameworks based on task and data characteristics.

## 2. Semantic Memory
A retrieval-augmented knowledge base (RAG) providing the Coder with relevant external knowledge about ML libraries and current problems.

```
Documentation
     ↓
Chunking
     ↓
Summarization
     ↓
Condensation
     ↓
Index
     ↓
Retrieval
     ↓
Coder
```

- **Why it's needed**: LLMs often hallucinate specific library APIs or lack knowledge of very recent frameworks. Semantic Memory grounds the Coder in factual, localized documentation.
- **Differs from LLM internal knowledge**: Internal knowledge is frozen at training time and general-purpose. Semantic Memory is targeted, domain-specific, and dynamically updatable.
- **Retrieval Bounding**: The system truncates retrieved knowledge via top-K scoring and maximum character limits (e.g., 4000 chars) to prevent overwhelming the context window and diluting the core task instructions.
- **Limitations**: The current implementation utilizes a lightweight local TF-IDF index for dependency-free operation, meaning it relies on exact term matching rather than dense semantic embeddings (which would require downloading heavy model weights).

## 3. Episodic Memory
Tracks the state of the current and past execution attempts to avoid repeating mistakes.
- **Iteration History**: Maintains a log of each coding attempt.
- **Previous Code**: Stores previously generated code for comparison.
- **Execution Logs & Errors**: Keeps track of stdout/stderr from code execution.
- **Error Summaries & Fix Suggestions**: Analyzes past failures to suggest corrections.

## 4. Iterative Coding
The core loop that generates, runs, and fixes code.
- **Coder Agent**: Writes Python scripts using input from Perception and Memory.
- **Executor Agent**: Runs the generated scripts in an isolated environment and captures outputs.
- **Error Analyzer Agent**: Investigates any runtime or logic errors and suggests fixes, feeding back into the Episodic Memory for the next iteration.
- **Retry Loop**: Orchestrates the interaction between these agents until the code succeeds or limits are reached.
