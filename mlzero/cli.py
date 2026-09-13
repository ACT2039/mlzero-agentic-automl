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
    parser = argparse.ArgumentParser(description="MLZero Agentic AutoML CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run the full ML pipeline")
    run_parser.add_argument("--input", required=True, help="Path to input dataset directory")
    run_parser.add_argument("--instruction", help="Optional instruction describing the task")
    run_parser.add_argument("--mock-llm", action="store_true", help="Use mock LLM responses")
    run_parser.add_argument("--json", action="store_true", help="Output final result as JSON")
    
    # API Server command
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI server")
    serve_parser.add_argument("--host", default=None, help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=None, help="Port to bind to")
    
    # UI Server command
    ui_parser = subparsers.add_parser("ui", help="Start the Gradio UI")
    ui_parser.add_argument("--host", default=None, help="Host to bind to")
    ui_parser.add_argument("--port", type=int, default=None, help="Port to bind to")
    
    # Perceive command (Legacy support)
    perceive_parser = subparsers.add_parser("perceive", help="Run the perception phase only")
    perceive_parser.add_argument("--input", required=True, help="Path to input dataset directory")
    perceive_parser.add_argument("--instruction", help="Optional task instruction")
    perceive_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # Iterate command (Legacy support)
    iterate_parser = subparsers.add_parser("iterate", help="Run the iterative coding phase")
    iterate_parser.add_argument("--perception-file", required=True, help="Path to saved perception JSON")
    iterate_parser.add_argument("--instruction", help="Optional task instruction")
    iterate_parser.add_argument("--mock-llm", action="store_true", help="Use mock LLM responses")
    
    # Memory command
    memory_parser = subparsers.add_parser("memory", help="Memory management")
    mem_subparsers = memory_parser.add_subparsers(dest="memory_command", help="Memory sub-command")
    
    build_parser = mem_subparsers.add_parser("build", help="Build semantic memory index")
    build_parser.add_argument("--knowledge-dir", required=True, help="Directory containing knowledge docs")
    build_parser.add_argument("--summarize", action="store_true", help="Generate summaries during ingestion")
    build_parser.add_argument("--condense", action="store_true", help="Condense chunks during ingestion")
    build_parser.add_argument("--mock-llm", action="store_true", help="Use mock LLM for summarization")
    
    search_parser = mem_subparsers.add_parser("search", help="Search semantic memory")
    search_parser.add_argument("--query", required=True, help="Search query")
    search_parser.add_argument("--mock-llm", action="store_true", help="Use mock LLM for embedding")
    
    episodes_parser = mem_subparsers.add_parser("episodes", help="List episodes for a run")
    episodes_parser.add_argument("--run-id", required=True, help="Run ID")
    
    episode_parser = mem_subparsers.add_parser("episode", help="Get specific episode")
    episode_parser.add_argument("--run-id", required=True, help="Run ID")
    episode_parser.add_argument("--iteration", type=int, required=True, help="Iteration number")
    
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
        
        from mlzero.memory.episodic import EpisodicMemory
        ep_mem = EpisodicMemory()
        
        orchestrator = IterativeCodingOrchestrator(
            coder=coder,
            executor=executor,
            error_analyzer=analyzer,
            max_iterations=5,
            semantic_memory=semantic_memory,
            episodic_memory=ep_mem
        )
        
        result = orchestrator.process(perceptual_context, args.instruction)
        
        print("\n=== FINAL RESULT ===")
        if result.run_id:
            print(f"Run ID:       {result.run_id}")
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

def handle_run(args: argparse.Namespace) -> int:
    """Run the complete ML pipeline using the Application Service."""

    from mlzero.application.service import MLZeroService
    
    try:
        service = MLZeroService(use_mock_llm=getattr(args, "mock_llm", False))
        result = service.run_mlzero(
            dataset_path=args.input,
            user_instruction=getattr(args, "instruction", None)
        )
        
        if getattr(args, "json", False):
            print(result.model_dump_json(indent=2))
        else:
            print(f"\n=== RUN STATUS: {result.status} ===")
            print(f"Run ID: {result.run_id}")
            print(f"Success: {result.success}")
            print(f"Iterations: {result.iterations}")
            print(f"Duration: {result.execution_duration}")
            
            if result.task_summary:
                print(f"Task: {result.task_summary.get('objective')}")
                
            if result.selected_library:
                print(f"Library: {result.selected_library}")
                
            if result.final_metrics:
                print("\nMetrics:")
                for k, v in result.final_metrics.items():
                    print(f"  - {k}: {v:.4f}" if isinstance(v, float) else f"  - {k}: {v}")
                    
            if result.prediction_artifact_reference:
                print(f"Predictions: {result.prediction_artifact_reference}")
            if result.model_artifact_reference:
                print(f"Model: {result.model_artifact_reference}")
                
            if result.final_error:
                print(f"\nError: {result.final_error}")
                
            print("=================================\n")
            
        return 0 if result.success else 1
    except Exception as e:
        logger.error(f"Run failed: {e}", exc_info=settings.app.debug)
        return 1

def handle_serve(args: argparse.Namespace) -> int:
    """Start the FastAPI server."""
    import uvicorn

    from mlzero.api.app import app
    from mlzero.core.config import settings
    
    host = args.host or settings.app.api_host
    port = args.port or settings.app.api_port
    
    print(f"Starting MLZero API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
    return 0

def handle_ui(args: argparse.Namespace) -> int:
    """Start the Gradio UI."""
    from mlzero.core.config import settings
    from mlzero.ui.app import create_ui
    
    host = args.host or settings.app.ui_host
    port = args.port or settings.app.ui_port
    
    print(f"Starting MLZero UI on {host}:{port}")
    app = create_ui()
    app.launch(server_name=host, server_port=port)
    return 0

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
        
    elif args.memory_command == "episodes":
        from mlzero.memory.episodic import EpisodicMemory
        ep_mem = EpisodicMemory()
        run = ep_mem.store.get_run_history(args.run_id)
        if not run:
            print(f"Run {args.run_id} not found.")
            return 1
            
        print(f"\n=== EPISODES FOR RUN {args.run_id} ===")
        print(f"Total Iterations: {len(run.episodes)}")
        print(f"Final Result:     {run.final_result or 'UNKNOWN'}")
        
        for ep in run.episodes:
            cat = ep.error_context_summary.get('error_category') if ep.error_context_summary else "N/A"
            fix = ep.suggested_fix or "N/A"
            print(f" - Iteration {ep.iteration}: {ep.status} | Category: {cat} | Fix: {fix}")
        return 0
        
    elif args.memory_command == "episode":

        from mlzero.memory.episodic import EpisodicMemory
        ep_mem = EpisodicMemory()
        target_ep = ep_mem.store.get_episode(args.run_id, args.iteration)
        if not target_ep:
            print(f"Episode {args.iteration} not found for run {args.run_id}.")
            return 1
            
        print(f"\n=== EPISODE {target_ep.iteration} (Run {args.run_id}) ===")
        print(f"Timestamp: {target_ep.timestamp}")
        print(f"Status:    {target_ep.status}")
        if target_ep.error_context_summary:
            print(f"\nError: {target_ep.error_context_summary.get('error_category')}")
            print(f"Fix:   {target_ep.suggested_fix}")
        print("======================\n")
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
    elif args.command == "run":
        return handle_run(args)
    elif args.command == "serve":
        return handle_serve(args)
    elif args.command == "ui":
        return handle_ui(args)
    elif args.command == "run-code":
        return handle_run_code(args)
    elif args.command == "memory":
        return handle_memory(args)
    elif args.command == "task":
        logger.info(f"Received task: {args.description}")
        return 0
    else:
        print("MLZero-Agentic-AutoML")
        print("Phase 8 production initialized.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
