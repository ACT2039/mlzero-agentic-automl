"""Report generator."""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend — must be before pyplot import
from typing import Any

import matplotlib.pyplot as plt


def generate_report(results: list[dict[str, Any]], out_dir: str = "reports") -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path = out_path / "figures"
    fig_path.mkdir(exist_ok=True)
    
    # 1. JSON
    with open(out_path / "evaluation_summary.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # 2. CSV
    if results:
        keys = ["experiment_id", "name", "dataset", "mock_llm", "success", "iterations", "execution_time", "error_category"]
        with open(out_path / "evaluation_summary.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for r in results:
                m = r["metrics"]
                c = r["config"]
                writer.writerow([
                    r["experiment_id"], r["name"], c.get("dataset"), c.get("mock_llm"),
                    m.get("success"), m.get("iterations"), m.get("execution_time"), m.get("error_category")
                ])
                
    # 3. Markdown
    md = "# MLZero Evaluation Report\n\n"
    md += "## Summary of Results\n"
    for r in results:
        md += f"### {r['name']}\n"
        md += f"- **Success:** {r['metrics'].get('success')}\n"
        md += f"- **Iterations:** {r['metrics'].get('iterations')}\n"
        et = r['metrics'].get('execution_time')
        md += (f"- **Execution Time:** {et:.2f}s\n" if et is not None else "- **Execution Time:** N/A\n")
        md += f"- **Error:** {r['metrics'].get('error_category')}\n\n"
        
    with open(out_path / "evaluation_report.md", "w") as f:
        f.write(md)
        
    # 4. Plots (simple bar chart of success)
    names = [r["name"] for r in results]
    successes = [1 if r["metrics"].get("success") else 0 for r in results]
    plt.figure(figsize=(10, 5))
    plt.bar(names, successes, color=['blue' if s else 'red' for s in successes])
    plt.title("Experiment Success")
    plt.ylabel("Success (1=Yes, 0=No)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(fig_path / "success_rate.png")
    plt.close()
    
    # Iterations Plot
    iters = [r["metrics"].get("iterations") or 0 for r in results]
    plt.figure(figsize=(10, 5))
    plt.bar(names, iters, color='green')
    plt.title("Average Iterations")
    plt.ylabel("Iterations")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(fig_path / "iterations.png")
    plt.close()
