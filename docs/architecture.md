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
A knowledge base that stores domain expertise and documentation.
- **Documentation Ingestion**: Capable of reading and parsing library docs.
- **Summarization/Condensation**: Compresses lengthy documentation into concise instructions.
- **Retrieval**: Provides contextually relevant snippets to the Coder Agent during generation.

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
