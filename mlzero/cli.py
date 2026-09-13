"""
Command-line interface for MLZero.
"""

import argparse
import sys
from pathlib import Path

from mlzero.agents.perception import (
    FilePerceptionAgent,
    LibrarySelectorAgent,
    TaskPerceptionAgent,
)
from mlzero.core.config import settings
from mlzero.core.llm import get_llm_client
from mlzero.core.logger import setup_logger
from mlzero.schemas.perception import PerceptualContext

logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MLZero: A Multi-Agent System for End-to-end Machine Learning Automation."
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {settings.app.version}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Perceive command
    perceive_parser = subparsers.add_parser("perceive", help="Run the perception module on a dataset directory.")
    perceive_parser.add_argument("--input", type=str, required=True, help="Path to the dataset directory.")
    perceive_parser.add_argument("--instruction", type=str, help="Optional user instruction for the task.")
    perceive_parser.add_argument("--json", action="store_true", help="Output in JSON format.")
    perceive_parser.add_argument("--mock-llm", action="store_true", help="Use deterministic mock LLM (default in Phase 2 unless overridden).")
    
    # Run-code command
    run_code_parser = subparsers.add_parser("run-code", help="Execute Python code safely.")
    run_code_parser.add_argument("--code-file", type=str, required=True, help="Path to the Python code file.")
    
    # Iterate command
    iterate_parser = subparsers.add_parser("iterate", help="Run the full iterative coding pipeline.")
    iterate_parser.add_argument("--input", type=str, required=True, help="Path to the dataset directory.")
    iterate_parser.add_argument("--instruction", type=str, help="Optional user instruction for the task.")
    iterate_parser.add_argument("--mock-llm", action="store_true", help="Use deterministic mock LLM.")
    
    # Memory command
    memory_parser = subparsers.add_parser("memory", help="Manage semantic memory.")
    memory_sub = memory_parser.add_subparsers(dest="memory_command", required=True)
    
    build_parser = memory_sub.add_parser("build", help="Build the semantic index.")
    build_parser.add_argument("--knowledge-dir", type=str, required=True, help="Path to knowledge directory.")
    build_parser.add_argument("--summarize", action="store_true", help="Enable LLM summarization.")
    build_parser.add_argument("--condense", action="store_true", help="Enable LLM condensation.")
    build_parser.add_argument("--mock-llm", action="store_true", help="Use deterministic mock LLM.")
    
    search_parser = memory_sub.add_parser("search", help="Search the semantic index.")
    search_parser.add_argument("--query", type=str, required=True, help="Search query.")
    search_parser.add_argument("--mock-llm", action="store_true", help="Use deterministic mock LLM.")
    
    # Placeholder task command from Phase 1
    task_parser = subparsers.add_parser("task", help="Execute a task (placeholder).")
    task_parser.add_argument("description", type=str, help="Task description")
    
    return parser.parse_args()

def handle_iterate(args: argparse.Namespace) -> int:
    """Handle the iterate command."""
    from mlzero.agents.coder import CoderAgent, ErrorAnalyzerAgent, ExecutorAgent
    from mlzero.agents.perception import (
        FilePerceptionAgent,
        LibrarySelectorAgent,
        TaskPerceptionAgent,
    )
    from mlzero.orchestration.iterative import IterativeCodingOrchestrator
    from mlzero.schemas.perception import PerceptualContext
    
    input_path = Path(args.input)
    if not input_path.exists() or not input_path.is_dir():
        logger.error(f"Input directory does not exist or is not a directory: {input_path}")
        return 1
        
    try:
        # We enforce mock-llm default logic for phase 4 demo
        # The prompt says: "For Phase 4 the command may use MockLLM by default"
        use_mock = True
        llm_client = get_llm_client(use_mock=use_mock)
        
        # Perception
        print("=== RUNNING PERCEPTION ===")
        file_agent = FilePerceptionAgent(dataset_dir=input_path)
        file_contexts = file_agent.process()
        task_agent = TaskPerceptionAgent(llm_client=llm_client)
        task_context = task_agent.process(file_contexts, args.instruction)
        lib_agent = LibrarySelectorAgent(llm_client=llm_client)
        lib_selection = lib_agent.process(task_context, file_contexts)
        
        perceptual_context = PerceptualContext(
            files=file_contexts,
            task=task_context,
            library=lib_selection
        )
        
        # Iteration
        print("\n=== RUNNING ITERATIVE CODING ===")
        coder = CoderAgent(llm_client=llm_client)
        executor = ExecutorAgent()
        analyzer = ErrorAnalyzerAgent(llm_client=llm_client)
        
        from mlzero.memory.semantic import SemanticMemory
        semantic_memory = SemanticMemory(llm_client=llm_client)
        
        orchestrator = IterativeCodingOrchestrator(
            coder=coder,
            executor=executor,
            error_analyzer=analyzer,
            max_iterations=5,
            semantic_memory=semantic_memory
        )
        
        result = orchestrator.process(perceptual_context, args.instruction)
        
        print("\n=== FINAL RESULT ===")
        print(f"Success:      {result.success}")
        print(f"Iterations:   {result.total_iterations}")
        print(f"Duration:     {result.total_duration_seconds:.2f}s")
        if not result.success and result.final_error_context:
            print(f"Final Error:  {result.final_error_context.error_category}")
            print(f"Suggestion:   {result.final_error_context.suggested_fix}")
        print("====================")
        
        return 0 if result.success else 1
    except Exception as e:
        logger.error(f"Iteration pipeline failed: {e}", exc_info=settings.app.debug)
        return 1
def handle_run_code(args: argparse.Namespace) -> int:
    """Handle the run-code command."""
    from mlzero.agents.coder import ExecutorAgent
    from mlzero.schemas.coder import CodeArtifact
    
    code_path = Path(args.code_file)
    if not code_path.exists() or not code_path.is_file():
        logger.error(f"Code file does not exist: {code_path}")
        return 1
        
    code_content = code_path.read_text(encoding="utf-8")
    
    artifact = CodeArtifact(
        code=code_content,
        language="python"
    )
    
    executor = ExecutorAgent()
    result = executor.process(artifact)
    
    print("\n=== EXECUTION RESULT ===")
    print(f"Success:      {result.success}")
    print(f"Return Code:  {result.return_code}")
    print(f"Duration:     {result.duration_seconds:.2f}s")
    
    if result.error_info:
        print(f"Error Info:   {result.error_info}")
        
    print("\n--- Output Files ---")
    if result.output_files:
        for f in result.output_files:
            print(f"  - {f}")
    else:
        print("  (none)")
        
    print("\n--- Stdout ---")
    print(result.stdout if result.stdout.strip() else "(empty)")
    
    print("\n--- Stderr ---")
    print(result.stderr if result.stderr.strip() else "(empty)")
    print("========================\n")
    
    return 0 if result.success else 1


def handle_memory(args: argparse.Namespace) -> int:
    """Handle memory build and search commands."""
    from mlzero.memory.semantic import SemanticMemory
    from mlzero.schemas.perception import PerceptualContext, TaskContext
    
    llm_client = None
    if getattr(args, "mock_llm", False) or getattr(args, "summarize", False) or getattr(args, "condense", False):
        use_mock = getattr(args, "mock_llm", True)
        llm_client = get_llm_client(use_mock=use_mock)
        
    memory = SemanticMemory(llm_client=llm_client)
    
    if args.memory_command == "build":
        knowledge_dir = Path(args.knowledge_dir)
        if not knowledge_dir.exists() or not knowledge_dir.is_dir():
            logger.error(f"Knowledge directory does not exist: {knowledge_dir}")
            return 1
            
        print(f"Building semantic index from {knowledge_dir}...")
        memory.ingest(
            knowledge_dir,
            summarize=args.summarize,
            condense=args.condense
        )
        print("Build complete.")
        return 0
        
    elif args.memory_command == "search":
        print(f"Searching for: '{args.query}'\n")
        # Dummy perceptual context to hold the query
        p_ctx = PerceptualContext(task=TaskContext(objective=args.query, task_type="search"))
        retrieved = memory.retrieve(p_ctx)
        
        if not retrieved.chunks:
            print("No relevant knowledge found.")
            return 0
            
        for i, (chunk, score) in enumerate(zip(retrieved.chunks, retrieved.relevance_scores)):
            print(f"--- Result {i+1} [Score: {score:.4f}] ---")
            print(f"Source: {chunk.source}")
            print(f"Content:\n{chunk.content}\n")
        return 0
        
    return 1

def handle_perceive(args: argparse.Namespace) -> int:
    """Handle the perceive command."""
    input_path = Path(args.input)
    if not input_path.exists() or not input_path.is_dir():
        logger.error(f"Input directory does not exist or is not a directory: {input_path}")
        return 1
        
    try:
        # Initialize LLM Client (Defaulting to mock for deterministic execution in Phase 2)
        llm_client = get_llm_client(use_mock=True)
        
        # 1. File Perception
        file_agent = FilePerceptionAgent(dataset_dir=input_path)
        file_contexts = file_agent.process()
        
        # 2. Task Perception
        task_agent = TaskPerceptionAgent(llm_client=llm_client)
        task_context = task_agent.process(file_contexts, args.instruction)
        
        # 3. Library Selection
        lib_agent = LibrarySelectorAgent(llm_client=llm_client)
        lib_selection = lib_agent.process(task_context, file_contexts)
        
        # Aggregate Context
        perceptual_context = PerceptualContext(
            files=file_contexts,
            task=task_context,
            library=lib_selection
        )
        
        # Output
        if args.json:
            print(perceptual_context.model_dump_json(indent=2))
        else:
            print("\n=== PERCEPTION SUMMARY ===")
            print(f"Files Inspected: {len(perceptual_context.files)}")
            for f in perceptual_context.files:
                status = "Error" if f.error else "OK"
                print(f"  - {f.metadata.path} ({f.metadata.file_type}, {f.metadata.size_bytes} bytes) [{status}]")
            
            print("\n--- Task Context ---")
            if perceptual_context.task:
                print(f"Objective: {perceptual_context.task.objective}")
                print(f"Type:      {perceptual_context.task.task_type}")
                print(f"Target:    {perceptual_context.task.target_column}")
            else:
                print("Task context could not be determined.")
            
            print("\n--- Library Selection ---")
            if perceptual_context.library:
                print(f"Library:    {perceptual_context.library.selected_library}")
                print(f"Confidence: {perceptual_context.library.confidence}")
                print(f"Reason:     {perceptual_context.library.explanation}")
            else:
                print("Library selection could not be determined.")
            print("==========================\n")
            
        return 0
        
    except Exception as e:
        logger.error(f"Perception failed: {e}", exc_info=settings.app.debug)
        return 1


def main() -> int:
    """Main entry point for the CLI."""
    args = parse_args()
    
    if args.command == "perceive":
        return handle_perceive(args)
    elif args.command == "iterate":
        return handle_iterate(args)
    elif args.command == "run-code":
        return handle_run_code(args)
    elif args.command == "memory":
        return handle_memory(args)
    elif args.command == "task":
        logger.info(f"Received task: {args.description}")
        logger.info("Task processing is a placeholder in Phase 1/2.")
        return 0
    else:
        # Fallback for old behaviour `python -m mlzero`
        print("MLZero-Agentic-AutoML")
        print("Phase 1 foundation initialized.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
