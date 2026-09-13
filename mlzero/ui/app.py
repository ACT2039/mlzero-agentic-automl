import json

import gradio as gr

from mlzero.application.manager import run_manager


def submit_task(dataset_path: str, instruction: str, mock_llm: bool) -> tuple[str, str]:
    try:
        run_id = run_manager.submit_run(
            dataset_path=dataset_path.strip(),
            user_instruction=instruction.strip() if instruction.strip() else None,
            options={"mock_llm": mock_llm}
        )
        return run_id, f"Run {run_id} started."
    except Exception as e:  # noqa: BLE001
        return "", f"Failed to start run: {e}"

def refresh_status(run_id: str) -> tuple[str, str, str, str, str, str]:
    if not run_id:
        return "No run selected.", "", "", "", "", ""
        
    status = run_manager.get_run_status(run_id)
    if not status:
        return "Run not found.", "", "", "", "", ""
        
    status_text = f"Status: {status.status}\nSuccess: {status.success}\nIterations: {status.iterations}\nDuration: {status.execution_duration}"
    perception_text = json.dumps(status.task_summary, indent=2) if status.task_summary else ""
    library_text = status.selected_library or ""
    metrics_text = json.dumps(status.final_metrics, indent=2) if status.final_metrics else ""
    error_text = status.final_error or ""
    artifacts_text = f"Prediction: {status.prediction_artifact_reference}\nModel: {status.model_artifact_reference}"
    
    return status_text, perception_text, library_text, metrics_text, error_text, artifacts_text

def create_ui() -> gr.Blocks:
    with gr.Blocks(title="MLZero Agentic AutoML") as demo:
        gr.Markdown("# MLZero Agentic AutoML")
        
        with gr.Row():
            with gr.Column():
                dataset_path = gr.Textbox(label="Dataset Path", value="tests/data/tiny_classification")
                instruction = gr.Textbox(label="Optional Task Instruction (e.g. 'Predict target_column')")
                mock_llm = gr.Checkbox(label="Use Mock LLM (Fast, offline testing)", value=True)
                run_btn = gr.Button("Start Run", variant="primary")
                run_id_box = gr.Textbox(label="Run ID", interactive=False)
                run_msg = gr.Textbox(label="Message", interactive=False)
                
            with gr.Column():
                refresh_btn = gr.Button("Refresh Status")
                status_box = gr.Textbox(label="Current Status", lines=4, interactive=False)
                lib_box = gr.Textbox(label="Selected Library", interactive=False)
                perc_box = gr.Code(label="Perception Summary", language="json", interactive=False)
                err_box = gr.Textbox(label="Final Error", interactive=False)
                metrics_box = gr.Code(label="Final Metrics", language="json", interactive=False)
                artifacts_box = gr.Textbox(label="Artifact References", interactive=False)
                
        run_btn.click(
            submit_task,
            inputs=[dataset_path, instruction, mock_llm],
            outputs=[run_id_box, run_msg]
        )
        
        refresh_btn.click(
            refresh_status,
            inputs=[run_id_box],
            outputs=[status_box, perc_box, lib_box, metrics_box, err_box, artifacts_box]
        )
        
    return demo
