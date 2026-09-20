import json

import gradio as gr

from mlzero.application.manager import run_manager
from mlzero.memory.episodic_store import EpisodicStore
from mlzero.schemas.episodic import RunHistory


def submit_task(dataset_path: str, instruction: str, llm_mode: str | bool) -> tuple[str, str, gr.Timer]:
    try:
        if isinstance(llm_mode, bool):
            is_mock = llm_mode
            run_mode = "mock" if is_mock else "real"
        elif isinstance(llm_mode, str):
            run_mode = "real" if "real" in llm_mode.lower() else "mock"
            is_mock = (run_mode == "mock")
        else:
            run_mode = "mock"
            is_mock = True

        run_id = run_manager.submit_run(
            dataset_path=dataset_path.strip(),
            user_instruction=instruction.strip() if instruction.strip() else None,
            options={"mock_llm": is_mock, "llm_mode": run_mode}
        )
        return run_id, f"Run {run_id} started.", gr.Timer(active=True)
    except Exception as e:  # noqa: BLE001
        return "", f"Failed to start run: {e}", gr.Timer(active=False)

def refresh_status(run_id: str) -> tuple[str, str, str, str, str, str, str, gr.Timer]:
    if not run_id:
        return "No run selected.", "", "", "", "", "", "No run selected.", gr.Timer(active=False)
        
    status = run_manager.get_run_status(run_id)
    if not status:
        return "Run not found.", "", "", "", "", "", "Run not found.", gr.Timer(active=False)
        
    dur = f"{status.execution_duration:.2f} seconds" if status.execution_duration is not None else "None"
    status_text = f"Status: {status.status}\nSuccess: {status.success}\nIterations: {status.iterations}\nDuration: {dur}"
    perception_text = json.dumps(status.task_summary, indent=2) if status.task_summary else ""
    library_text = status.selected_library or ""
    metrics_text = json.dumps(status.final_metrics, indent=2) if status.final_metrics else ""
    error_text = status.final_error or ""
    artifacts_text = f"Prediction: {status.prediction_artifact_reference}\nModel: {status.model_artifact_reference}"
    
    history_text = format_history_and_recovery(EpisodicStore().get_run_history(run_id), status.status)
    timer_active = status.status in ("QUEUED", "RUNNING")
    
    return status_text, perception_text, library_text, metrics_text, error_text, artifacts_text, history_text, gr.Timer(active=timer_active)

def format_history_and_recovery(run_history: RunHistory | None, current_status: str) -> str:
    if not run_history or not run_history.episodes:
        if current_status in ("QUEUED", "RUNNING"):
            return "Waiting for iteration results..."
        return "No history available."
        
    lines = []
    lines.append("Iteration History")
    lines.append("-" * 48)
    
    had_failure = False
    
    for ep in run_history.episodes:
        lines.append(f"Iteration {ep.iteration}")
        if ep.status == "SUCCESS":
            lines.append("Status: SUCCESS")
            if had_failure:
                lines.append("Correction Applied: YES")
        else:
            had_failure = True
            lines.append("Status: FAILED")
            if ep.error_context_summary:
                cat = ep.error_context_summary.get("error_category", "Unknown")
                msg = ep.error_context_summary.get("error_message", "Unknown")
                lines.append(f"Error Category: {cat}")
                lines.append(f"Problem: {msg}")
            if ep.suggested_fix:
                lines.append(f"Correction: {ep.suggested_fix}")
                
        lines.append("")
        
    lines.append("Recovery Summary")
    lines.append("-" * 48)
    
    total = len(run_history.episodes)
    last_ep = run_history.episodes[-1]
    
    if total == 1 and last_ep.status == "SUCCESS":
        lines.append("✓ Execution succeeded on first iteration")
    elif had_failure and last_ep.status == "SUCCESS":
        lines.append("✓ Error detected")
        lines.append("✓ Error analyzed")
        lines.append("✓ Correction generated")
        lines.append("✓ Corrected code executed successfully")
        lines.append(f"✓ Recovery completed in {total} iterations")
    elif current_status == "FAIL":
        lines.append(f"✗ Run ultimately failed after {total} iterations.")
    elif current_status in ("QUEUED", "RUNNING"):
        lines.append("Run is still in progress...")
        
    return "\n".join(lines).strip()

def create_ui() -> gr.Blocks:
    with gr.Blocks(title="MLZero Agentic AutoML") as demo:
        gr.Markdown("# MLZero Agentic AutoML")
        
        with gr.Row():
            with gr.Column():
                dataset_path = gr.Textbox(label="Dataset Path", value="tests/data/tiny_classification")
                instruction = gr.Textbox(label="Optional Task Instruction (e.g. 'Predict target_column')")
                llm_mode_input = gr.Radio(
                    label="LLM Mode",
                    choices=["Mock LLM (Offline / deterministic)", "Real LLM (Gemini 3.8 Flash)"],
                    value="Mock LLM (Offline / deterministic)"
                )
                run_btn = gr.Button("Start Run", variant="primary")
                run_id_box = gr.Textbox(label="Run ID", interactive=False)
                run_msg = gr.Textbox(label="Message", interactive=False)
                history_box = gr.Textbox(label="Iteration History", lines=6, interactive=False)
                
            with gr.Column():
                refresh_btn = gr.Button("Refresh Status")
                status_box = gr.Textbox(label="Current Status", lines=4, interactive=False)
                lib_box = gr.Textbox(label="Selected Library", interactive=False)
                perc_box = gr.Code(label="Perception Summary", language="json", interactive=False)
                err_box = gr.Textbox(label="Final Error", interactive=False)
                metrics_box = gr.Code(label="Final Metrics", language="json", interactive=False)
                artifacts_box = gr.Textbox(label="Artifact References", interactive=False)
                
        # Add auto-refresh timer (starts inactive)
        timer = gr.Timer(value=2, active=False)
        
        run_btn.click(
            submit_task,
            inputs=[dataset_path, instruction, llm_mode_input],
            outputs=[run_id_box, run_msg, timer]
        )
        
        refresh_events = [refresh_btn.click, timer.tick]
        for event in refresh_events:
            event(
                refresh_status,
                inputs=[run_id_box],
                outputs=[status_box, perc_box, lib_box, metrics_box, err_box, artifacts_box, history_box, timer]
            )
        
    return demo  # type: ignore[no-any-return]
