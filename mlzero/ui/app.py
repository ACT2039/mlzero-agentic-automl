import json

import gradio as gr

from mlzero.application.manager import run_manager
from mlzero.memory.episodic_store import EpisodicStore
from mlzero.schemas.episodic import RunHistory

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 8D — FULL REFERENCE-STYLE UI REBUILD
# Presentation layer only — no backend changes.
# ──────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Reset & Tokens ─────────────────────────────────────────────────────── */
:root {
  --bg:           #0B0F19;
  --surface:      #0F1522;
  --surface2:     #111827;
  --border:       #273250;
  --violet:       #8B5CF6;
  --cyan:         #06B6D4;
  --green:        #22C55E;
  --red:          #EF4444;
  --amber:        #F59E0B;
  --text:         #F8FAFC;
  --text2:        #94A3B8;
  --text3:        #64748B;
  --radius-sm:    6px;
  --radius-md:    10px;
  --radius-lg:    16px;
  --shell-w:      min(95vw, 1440px);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ── Gradio Container Reset ──────────────────────────────────────────────── */
body {
  background: var(--bg) !important;
  font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif !important;
}

body::before {
  content: '';
  position: fixed;
  top: -20%;
  left: 50%;
  transform: translateX(-50%);
  width: 900px;
  height: 500px;
  background: radial-gradient(ellipse at center,
    rgba(139,92,246,0.06) 0%,
    rgba(6,182,212,0.04) 40%,
    transparent 70%);
  pointer-events: none;
  z-index: 0;
}

.gradio-container {
  background: transparent !important;
  max-width: var(--shell-w) !important;
  width: var(--shell-w) !important;
  margin: 0 auto !important;
  padding: 24px 24px 48px !important;
}

/* ── Application Shell ───────────────────────────────────────────────────── */
.app-shell {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden;
  position: relative;
}

/* ── Top Navigation Bar ──────────────────────────────────────────────────── */
.topnav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 28px;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
}

.topnav-brand {
  display: flex;
  align-items: baseline;
  gap: 10px;
}

.topnav-logo {
  font-size: 1.05rem;
  font-weight: 800;
  color: var(--text);
  letter-spacing: -0.02em;
}

.topnav-logo span { color: var(--violet); }

.topnav-divider {
  width: 1px;
  height: 16px;
  background: var(--border);
}

.topnav-product {
  font-size: 0.8rem;
  color: var(--text3);
  font-weight: 500;
  letter-spacing: 0.02em;
}

.topnav-links {
  display: flex;
  gap: 4px;
}

.topnav-link {
  padding: 5px 12px;
  border-radius: var(--radius-sm);
  font-size: 0.825rem;
  font-weight: 500;
  color: var(--text3);
  cursor: default;
  transition: color 0.15s;
}

.topnav-link.active {
  color: var(--text);
  background: rgba(139,92,246,0.12);
}

.topnav-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.engine-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--green);
  display: inline-block;
  margin-right: 5px;
}

.engine-tag {
  font-size: 0.78rem;
  color: var(--text2);
  font-weight: 500;
}

/* ── Page Interior ───────────────────────────────────────────────────────── */
.page-inner {
  padding: 32px 36px 36px;
}

/* ── Hero ────────────────────────────────────────────────────────────────── */
.hero {
  margin-bottom: 28px;
}

.hero-title {
  font-size: 1.75rem;
  font-weight: 800;
  color: var(--text);
  letter-spacing: -0.03em;
  line-height: 1.2;
}

.hero-sub {
  font-size: 0.88rem;
  color: var(--text3);
  margin-top: 5px;
  font-weight: 400;
}

/* ── Action Bar ──────────────────────────────────────────────────────────── */
.action-bar {
  display: flex;
  gap: 10px;
  margin-bottom: 32px;
  align-items: stretch;
}

.action-bar-input {
  flex: 1;
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
  font-size: 0.9rem !important;
  padding: 10px 16px !important;
  outline: none !important;
}

.action-bar-input:focus {
  border-color: var(--violet) !important;
  box-shadow: 0 0 0 2px rgba(139,92,246,0.15) !important;
}

/* ── Section Header ──────────────────────────────────────────────────────── */
.section-label {
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text3);
  margin-bottom: 8px;
}

.section-divider {
  border: none;
  border-top: 1px solid var(--border);
  margin: 20px 0;
}

/* ── Two-Column Top Row ──────────────────────────────────────────────────── */
.top-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  margin-bottom: 28px;
}

@media (max-width: 900px) {
  .top-row { grid-template-columns: 1fr; }
}

/* ── Panel Card ──────────────────────────────────────────────────────────── */
.panel {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 22px 24px;
}

.panel-title {
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text3);
  margin-bottom: 16px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
  position: relative;
}

.panel-title::after {
  content: '';
  position: absolute;
  bottom: -1px;
  left: 0;
  width: 28px;
  height: 1px;
  background: var(--violet);
}

/* ── Status Badge ────────────────────────────────────────────────────────── */
.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  font-size: 1.1rem;
  font-weight: 800;
  letter-spacing: -0.01em;
  margin-bottom: 10px;
}

.status-success { color: var(--green); }
.status-running { color: var(--cyan); }
.status-failed  { color: var(--red); }
.status-queued  { color: var(--amber); }
.status-empty   { color: var(--text3); }

/* ── Run Meta Grid ───────────────────────────────────────────────────────── */
.run-meta {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px 20px;
  margin-top: 12px;
}

.run-meta-item {}
.run-meta-label { font-size: 0.72rem; color: var(--text3); font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; }
.run-meta-value { font-size: 0.88rem; color: var(--text); font-weight: 600; margin-top: 2px; }

/* ── Pipeline ────────────────────────────────────────────────────────────── */
.pipeline-section {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 20px 24px;
  margin-bottom: 24px;
}

.pipeline-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  margin-top: 12px;
}

.pipe-step {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--text3);
  padding: 6px 12px;
  border-radius: 20px;
  border: 1px solid transparent;
  white-space: nowrap;
}

.pipe-step.done {
  color: var(--cyan);
  border-color: rgba(6,182,212,0.3);
  background: rgba(6,182,212,0.07);
}

.pipe-step.active {
  color: var(--violet);
  border-color: rgba(139,92,246,0.4);
  background: rgba(139,92,246,0.1);
}

.pipe-arrow {
  color: var(--text3);
  font-size: 0.7rem;
  flex-shrink: 0;
}

/* ── Run Details ──────────────────────────────────────────────────────────── */
.details-section {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 20px 24px;
  margin-bottom: 24px;
}

.kv-grid {
  display: grid;
  grid-template-columns: 160px 1fr;
  row-gap: 0;
}

.kv-row {
  display: contents;
}

.kv-key {
  font-size: 0.8rem;
  color: var(--text3);
  padding: 9px 0;
  border-bottom: 1px solid rgba(39,50,80,0.5);
  font-weight: 500;
}

.kv-val {
  font-size: 0.88rem;
  color: var(--text);
  padding: 9px 0;
  border-bottom: 1px solid rgba(39,50,80,0.5);
  font-weight: 600;
}

.kv-key:last-of-type, .kv-val:last-of-type {
  border-bottom: none;
}

/* ── Metrics Strip ────────────────────────────────────────────────────────── */
.metrics-strip {
  display: flex;
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  overflow: hidden;
  margin-bottom: 24px;
}

.metric-cell {
  flex: 1;
  background: var(--surface2);
  padding: 20px 24px;
  text-align: left;
}

.metric-number {
  font-size: 1.9rem;
  font-weight: 800;
  color: var(--text);
  letter-spacing: -0.03em;
  line-height: 1;
}

.metric-label {
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--text3);
  margin-top: 5px;
}

/* ── Iteration Rows ───────────────────────────────────────────────────────── */
.iter-section {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  overflow: hidden;
  margin-bottom: 24px;
}

.iter-row {
  display: flex;
  gap: 20px;
  align-items: flex-start;
  padding: 16px 24px;
  border-bottom: 1px solid var(--border);
}

.iter-row:last-child { border-bottom: none; }

.iter-num {
  font-size: 0.75rem;
  font-weight: 800;
  color: var(--text3);
  min-width: 26px;
  padding-top: 2px;
}

.iter-status {
  font-size: 0.78rem;
  font-weight: 700;
  padding: 3px 9px;
  border-radius: 20px;
  min-width: 64px;
  text-align: center;
  flex-shrink: 0;
}

.iter-success { color: var(--green); background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.25); }
.iter-failed  { color: var(--red);   background: rgba(239,68,68,0.1);  border: 1px solid rgba(239,68,68,0.25); }

.iter-body { flex: 1; }
.iter-summary { font-size: 0.85rem; color: var(--text); font-weight: 500; }
.iter-detail  { font-size: 0.78rem; color: var(--text3); margin-top: 3px; }

.iter-judge {
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--text3);
  letter-spacing: 0.05em;
  min-width: 48px;
  text-align: right;
  padding-top: 2px;
}

/* ── Diagnostics & Artifacts ─────────────────────────────────────────────── */
.diag-artifact-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-bottom: 24px;
}

@media (max-width: 700px) {
  .diag-artifact-row { grid-template-columns: 1fr; }
}

.diag-ok  { color: var(--green); font-weight: 700; font-size: 0.9rem; }
.diag-err { color: var(--red);   font-weight: 700; font-size: 0.9rem; }

.artifact-list { display: flex; flex-direction: column; gap: 8px; }
.artifact-item { font-size: 0.84rem; color: var(--text2); }
.artifact-label { font-weight: 600; color: var(--text); }

/* ── Accordion (raw JSON) ────────────────────────────────────────────────── */
.accordion-raw > .label-wrap > .icon-wrap { display: none; }
.accordion-raw {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  margin-bottom: 16px !important;
}

/* ── Gradio Input Overrides ──────────────────────────────────────────────── */
input, textarea, select {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: var(--radius-sm) !important;
  font-size: 0.88rem !important;
}

input::placeholder, textarea::placeholder { color: var(--text3) !important; }

input:focus, textarea:focus {
  border-color: var(--violet) !important;
  outline: none !important;
  box-shadow: 0 0 0 2px rgba(139,92,246,0.12) !important;
}

label span {
  font-size: 0.75rem !important;
  font-weight: 600 !important;
  color: var(--text3) !important;
  letter-spacing: 0.03em !important;
  text-transform: uppercase !important;
}

.gr-button-primary, .cta-run {
  background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%) !important;
  color: #fff !important;
  font-weight: 700 !important;
  border: 1px solid rgba(139,92,246,0.4) !important;
  border-radius: var(--radius-sm) !important;
  height: 46px !important;
  min-width: 140px !important;
  font-size: 0.875rem !important;
  box-shadow: 0 4px 14px rgba(139,92,246,0.2) !important;
  transition: box-shadow 0.18s, transform 0.18s !important;
}

.gr-button-primary:hover, .cta-run:hover {
  box-shadow: 0 6px 20px rgba(139,92,246,0.35) !important;
  transform: translateY(-1px) !important;
}

.btn-secondary {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  color: var(--text2) !important;
  border-radius: var(--radius-sm) !important;
  font-size: 0.8rem !important;
  font-weight: 600 !important;
}

.block, .gr-form { background: transparent !important; border: none !important; padding: 0 !important; }

.cm-editor { background: var(--surface2) !important; }

.gr-radio { background: transparent !important; border: none !important; }

/* Tabs inside details section */
.inner-tabs > .tab-nav {
  background: transparent !important;
  border: none !important;
  border-bottom: 1px solid var(--border) !important;
  padding: 0 !important;
  gap: 0 !important;
}

.inner-tabs > .tab-nav button {
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  padding: 8px 16px !important;
  color: var(--text3) !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  margin-bottom: -1px !important;
}

.inner-tabs > .tab-nav button.selected {
  color: var(--text) !important;
  border-bottom-color: var(--violet) !important;
}

.inner-tabs > .tabitem {
  background: transparent !important;
  border: none !important;
  padding: 16px 0 0 !important;
}
"""

# ──────────────────────────────────────────────────────────────────────────────
# HELPER: Build HTML components from runtime data
# ──────────────────────────────────────────────────────────────────────────────

def _status_html(status_text: str) -> str:
    """Render a compact status badge from the textbox value."""
    if not status_text or status_text.startswith("No run"):
        return '<span class="status-badge status-empty">— No run selected</span>'
    # Parse the status line
    for line in status_text.splitlines():
        if line.startswith("Status:"):
            st = line.replace("Status:", "").strip()
            icon_map = {
                "SUCCESS": ("✓", "status-success"),
                "RUNNING": ("●", "status-running"),
                "QUEUED":  ("◌", "status-queued"),
                "FAILED":  ("✕", "status-failed"),
                "FAIL":    ("✕", "status-failed"),
                "TIMEOUT": ("⌛", "status-queued"),
            }
            icon, cls = icon_map.get(st, ("●", "status-running"))
            return f'<span class="status-badge {cls}">{icon} {st}</span>'
    return '<span class="status-badge status-empty">—</span>'


def _parse_status_fields(status_text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in status_text.splitlines():
        if ": " in line:
            k, _, v = line.partition(": ")
            out[k.strip()] = v.strip()
    return out


def _perception_kv_html(perc_json: str, library_text: str) -> str:
    """Render perception data as a clean kv-grid. Falls back to empty state."""
    if not perc_json or perc_json in ("Waiting for run", ""):
        txt = "Start a run to view task context." if library_text == "N/A" else f"Library: {library_text}"
        return f'<p style="color:var(--text3); font-size:0.85rem;">{txt}</p>'
    try:
        d: dict[str, object] = json.loads(perc_json)
    except (json.JSONDecodeError, TypeError):
        return f'<p style="color:var(--text3); font-size:0.85rem;">{perc_json}</p>'

    field_labels = {
        "task_type": "Task Type",
        "target_column": "Target",
        "timestamp_column": "Timestamp",
        "id_column": "ID",
        "modality": "Modality",
        "selected_library": "Library",
        "dataset_path": "Dataset",
    }

    rows_html = ""
    for k, label in field_labels.items():
        val = d.get(k)
        if val is not None and val != "" and val is not False:
            safe_val = str(val).replace("<", "&lt;").replace(">", "&gt;")
            rows_html += (
                f'<div class="kv-key">{label}</div>'
                f'<div class="kv-val">{safe_val}</div>'
            )
    if library_text and library_text != "N/A":
        rows_html += (
            f'<div class="kv-key">Library</div>'
            f'<div class="kv-val">{library_text}</div>'
        )
    if not rows_html:
        rows_html = (
            '<div class="kv-key" style="border-bottom:none; color:var(--text3)">No structured data</div>'
            '<div class="kv-val" style="border-bottom:none;"></div>'
        )
    return f'<div class="kv-grid">{rows_html}</div>'


def _metrics_html(fields: dict[str, str], metrics_json: str) -> str:
    iterations = fields.get("Iterations", "—")
    duration = fields.get("Duration", "—")

    accuracy = "—"
    if metrics_json and metrics_json not in ("No metrics available yet.",):
        try:
            m = json.loads(metrics_json)
            if isinstance(m, dict):
                for key in ("accuracy", "roc_auc", "f1", "rmse", "mae", "r2"):
                    if key in m:
                        accuracy = f"{m[key]:.3f}"
                        break
        except (json.JSONDecodeError, TypeError):
            pass

    def _cell(number: str, label: str) -> str:
        return (
            f'<div class="metric-cell">'
            f'<div class="metric-number">{number}</div>'
            f'<div class="metric-label">{label}</div>'
            f'</div>'
        )

    return (
        f'<div class="metrics-strip">'
        f'{_cell(iterations, "Iterations")}'
        f'{_cell(duration, "Duration")}'
        f'{_cell(accuracy, "Best Score")}'
        f'</div>'
    )


def _pipeline_html(fields: dict[str, str]) -> str:
    status = fields.get("Status", "")

    # All steps completed if SUCCESS
    if status == "SUCCESS":
        states = ["done", "done", "done", "done", "done", "done"]
    elif status in ("RUNNING", "QUEUED"):
        states = ["done", "done", "done", "active", "pending", "pending"]
    elif status in ("FAILED", "FAIL"):
        states = ["done", "done", "done", "done", "failed", "failed"]
    else:
        states = ["pending"] * 6

    steps = [
        ("✓", "Perception"),
        ("✓", "Retrieval"),
        ("✓", "Code Gen"),
        ("✓", "Execution"),
        ("✓", "Judge"),
        ("✓", "Complete"),
    ]

    icon_map = {"done": "✓", "active": "●", "pending": "○", "failed": "✕"}

    items = []
    for i, (_, label) in enumerate(steps):
        state = states[i] if i < len(states) else "pending"
        icon = icon_map.get(state, "○")
        cls = "done" if state == "done" else ("active" if state == "active" else "")
        items.append(f'<span class="pipe-step {cls}">{icon} {label}</span>')
        if i < len(steps) - 1:
            items.append('<span class="pipe-arrow">→</span>')

    inner = "\n".join(items)
    return (
        f'<div class="pipeline-section">'
        f'<div class="panel-title">AGENT EXECUTION</div>'
        f'<div class="pipeline-row">{inner}</div>'
        f'</div>'
    )


def _iterations_html(history_text: str) -> str:
    if not history_text or history_text.startswith(("Start an", "No run")):
        return '<div style="color:var(--text3); font-size:0.85rem; padding:12px 0;">No iterations yet.</div>'
    if history_text.startswith("Waiting"):
        return '<div style="color:var(--text3); font-size:0.85rem; padding:12px 0;">Waiting for iteration results…</div>'

    # Parse text-based history
    lines = history_text.splitlines()
    rows_html = ""
    i = 0
    iter_num = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Iteration "):
            iter_num += 1
            num_str = f"{iter_num:02d}"
            status_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            status = status_line.replace("Status:", "").strip() if "Status:" in status_line else "—"
            summary_parts = []
            j = i + 2
            while j < len(lines) and not lines[j].strip().startswith("Iteration ") \
                    and not lines[j].strip().startswith("Recovery"):
                l = lines[j].strip()
                if l:
                    summary_parts.append(l)
                j += 1
            i = j
            summary = summary_parts[0] if summary_parts else "—"
            detail = " · ".join(summary_parts[1:]) if len(summary_parts) > 1 else ""

            if status == "SUCCESS":
                scls = "iter-success"
                judge = "FINISH"
            else:
                scls = "iter-failed"
                judge = "FIX"

            detail_html = f'<div class="iter-detail">{detail}</div>' if detail else ""
            rows_html += (
                f'<div class="iter-row">'
                f'<div class="iter-num">{num_str}</div>'
                f'<span class="iter-status {scls}">{status}</span>'
                f'<div class="iter-body">'
                f'<div class="iter-summary">{summary}</div>'
                f'{detail_html}'
                f'</div>'
                f'<div class="iter-judge">{judge}</div>'
                f'</div>'
            )
        else:
            i += 1

    if not rows_html:
        rows_html = '<div class="iter-row"><div style="color:var(--text3);font-size:0.85rem;">No structured iteration data.</div></div>'

    return f'<div class="iter-section">{rows_html}</div>'


def _diagnostics_html(error_text: str) -> str:
    if not error_text or "No errors" in error_text:
        return '<div class="diag-ok">✓ No errors reported</div>'
    # Short summary + technical detail
    lines = [l for l in error_text.splitlines() if l.strip()]
    headline = lines[0] if lines else error_text
    rest = "\n".join(lines[1:]) if len(lines) > 1 else ""
    safe_rest = rest[:600].replace("<", "&lt;").replace(">", "&gt;")
    detail = f'<pre style="margin-top:8px; font-size:0.75rem; color:var(--text3); white-space:pre-wrap; overflow-wrap:break-word;">{safe_rest}</pre>' if safe_rest else ""
    return f'<div class="diag-err">{headline}</div>{detail}'


def _artifacts_html(artifacts_text: str) -> str:
    if not artifacts_text or "No artifacts" in artifacts_text:
        return '<div style="color:var(--text3); font-size:0.85rem;">No artifacts available.</div>'
    items = []
    for line in artifacts_text.splitlines():
        if ": " in line:
            label, _, val = line.partition(": ")
            safe_val = str(val).replace("<", "&lt;").replace(">", "&gt;")
            if val and val != "None":
                items.append(
                    f'<div class="artifact-item">'
                    f'<span class="artifact-label">{label}</span> — {safe_val}'
                    f'</div>'
                )
    if items:
        return f'<div class="artifact-list">{"".join(items)}</div>'
    return '<div style="color:var(--text3); font-size:0.85rem;">No artifacts available.</div>'


def _run_meta_html(fields: dict[str, str], run_id: str, run_msg: str, lib: str) -> str:
    if not fields.get("Status"):
        return '<div style="color:var(--text3); font-size:0.88rem; padding: 6px 0;">No run selected.</div>'
    items = []
    for label, key in [("Library", None), ("Iterations", "Iterations"), ("Duration", "Duration")]:
        if label == "Library":
            val = lib or "—"
        else:
            val = fields.get(key or "", "—")
        items.append(
            f'<div class="run-meta-item">'
            f'<div class="run-meta-label">{label}</div>'
            f'<div class="run-meta-value">{val}</div>'
            f'</div>'
        )
    if run_id:
        items.append(
            f'<div class="run-meta-item" style="grid-column: span 2;">'
            f'<div class="run-meta-label">Run ID</div>'
            f'<div class="run-meta-value" style="font-size:0.78rem; color:var(--text3);">{run_id}</div>'
            f'</div>'
        )
    if run_msg:
        items.append(
            f'<div class="run-meta-item" style="grid-column: span 2;">'
            f'<div class="run-meta-label">Submission</div>'
            f'<div class="run-meta-value" style="font-size:0.8rem; color:var(--text3);">{run_msg}</div>'
            f'</div>'
        )
    return f'<div class="run-meta">{"".join(items)}</div>'


# ──────────────────────────────────────────────────────────────────────────────
# BACKEND CALLBACKS
# ──────────────────────────────────────────────────────────────────────────────

def submit_task(
    dataset_path: str, instruction: str, llm_mode: str | bool
) -> tuple[str, str, gr.Timer]:
    try:
        if isinstance(llm_mode, bool):
            is_mock = llm_mode
            run_mode = "mock" if is_mock else "real"
        elif isinstance(llm_mode, str):
            run_mode = "real" if "real" in llm_mode.lower() else "mock"
            is_mock = run_mode == "mock"
        else:
            run_mode = "mock"
            is_mock = True

        run_id = run_manager.submit_run(
            dataset_path=dataset_path.strip(),
            user_instruction=instruction.strip() if instruction.strip() else None,
            options={"mock_llm": is_mock, "llm_mode": run_mode},
        )
        return run_id, f"Run {run_id} started.", gr.Timer(active=True)
    except Exception as e:  # noqa: BLE001
        return "", f"Failed to start run: {e}", gr.Timer(active=False)


def refresh_status(
    run_id: str,
) -> tuple[str, str, str, str, str, str, str, gr.Timer]:
    if not run_id:
        return (
            "No run selected.",
            "Waiting for run",
            "N/A",
            "No metrics available yet.",
            "✓ No errors reported",
            "No artifacts available.",
            "No run selected.",
            gr.Timer(active=False),
        )

    status = run_manager.get_run_status(run_id)
    if not status:
        return (
            "Run not found.",
            "Waiting for run",
            "N/A",
            "No metrics available yet.",
            "✓ No errors reported",
            "No artifacts available.",
            "Run not found.",
            gr.Timer(active=False),
        )

    dur = (
        f"{status.execution_duration:.2f} seconds"
        if status.execution_duration is not None
        else "None"
    )
    status_text = (
        f"Status: {status.status}\n"
        f"Success: {status.success}\n"
        f"Iterations: {status.iterations}\n"
        f"Duration: {dur}"
    )
    perception_text = (
        json.dumps(status.task_summary, indent=2) if status.task_summary else "Waiting for run"
    )
    library_text = status.selected_library or "N/A"
    metrics_text = (
        json.dumps(status.final_metrics, indent=2)
        if status.final_metrics
        else "No metrics available yet."
    )

    if status.final_error and status.final_error.strip():
        error_text = f"✕ Execution failed:\n\n{status.final_error}"
    else:
        error_text = "✓ No errors reported"

    artifacts_text = (
        f"Prediction: {status.prediction_artifact_reference}\n"
        f"Model: {status.model_artifact_reference}"
    )

    history_text = format_history_and_recovery(
        EpisodicStore().get_run_history(run_id), status.status
    )
    timer_active = status.status in ("QUEUED", "RUNNING")

    return (
        status_text,
        perception_text,
        library_text,
        metrics_text,
        error_text,
        artifacts_text,
        history_text,
        gr.Timer(active=timer_active),
    )


def format_history_and_recovery(
    run_history: RunHistory | None, current_status: str
) -> str:
    if not run_history or not run_history.episodes:
        if current_status in ("QUEUED", "RUNNING"):
            return "Waiting for iteration results..."
        return "Start an agentic run to view execution history."

    lines: list[str] = []
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


# ──────────────────────────────────────────────────────────────────────────────
# COMPOSITE RENDER FUNCTION
# Called on every refresh — takes raw state, emits HTML for display panels.
# ──────────────────────────────────────────────────────────────────────────────

def render_all(
    status_text: str,
    perception_text: str,
    library_text: str,
    metrics_text: str,
    error_text: str,
    artifacts_text: str,
    history_text: str,
    run_id: str,
    run_msg: str,
) -> tuple[str, str, str, str, str, str, str]:
    fields = _parse_status_fields(status_text)

    status_html    = _status_html(status_text) + _run_meta_html(fields, run_id, run_msg, library_text)
    pipeline_html  = _pipeline_html(fields)
    details_html   = _perception_kv_html(perception_text, library_text)
    metrics_html   = _metrics_html(fields, metrics_text)
    iterations_html = _iterations_html(history_text)
    diag_html      = _diagnostics_html(error_text)
    artifacts_html = _artifacts_html(artifacts_text)

    return (
        status_html,
        pipeline_html,
        details_html,
        metrics_html,
        iterations_html,
        diag_html,
        artifacts_html,
    )


# ──────────────────────────────────────────────────────────────────────────────
# UI COMPOSITION
# ──────────────────────────────────────────────────────────────────────────────

GRADIO_THEME = gr.themes.Soft(primary_hue="purple", neutral_hue="slate")


def create_ui() -> gr.Blocks:
    with gr.Blocks(
        title="MLZero — Agentic AutoML Platform",
    ) as demo:

        # ── Hidden state stores (8 raw values from backend) ─────────────────
        _status_text   = gr.State("No run selected.")
        _perc_text     = gr.State("Waiting for run")
        _lib_text      = gr.State("N/A")
        _metrics_text  = gr.State("No metrics available yet.")
        _error_text    = gr.State("✓ No errors reported")
        _artifacts_text = gr.State("No artifacts available.")
        _history_text  = gr.State("No run selected.")

        # ── Application Shell ─────────────────────────────────────────────
        gr.HTML("""
        <div class="app-shell">

          <!-- Top Navigation -->
          <nav class="topnav">
            <div class="topnav-brand">
              <span class="topnav-logo"><span>ML</span>Zero</span>
              <div class="topnav-divider"></div>
              <span class="topnav-product">Agentic AutoML</span>
            </div>
            <div class="topnav-links">
              <span class="topnav-link active">Overview</span>
              <span class="topnav-link">Runs</span>
              <span class="topnav-link">Memory</span>
              <span class="topnav-link">Evaluation</span>
            </div>
            <div class="topnav-right">
              <span class="engine-tag"><span class="engine-dot"></span>Engine Ready</span>
            </div>
          </nav>

          <!-- Page Interior opens here — closed by footer HTML below -->
          <div class="page-inner">

            <!-- Hero -->
            <div class="hero">
              <div class="hero-title">Agentic AutoML</div>
              <div class="hero-sub">Autonomous Machine Learning Engineering</div>
            </div>
        """)

        # ── Two-Column Top Row ────────────────────────────────────────────
        with gr.Row(equal_height=False):
            # LEFT: New Run Form
            with gr.Column(scale=1):
                gr.HTML('<div class="panel-title">⚡ NEW AGENTIC RUN</div>')
                dataset_path = gr.Textbox(
                    label="Dataset Path",
                    value="tests/data/tiny_classification",
                    placeholder="e.g. tests/data/tiny_classification",
                    info="Directory containing train.csv and test.csv",
                )
                instruction = gr.Textbox(
                    label="Task Instruction (Optional)",
                    placeholder="e.g. Predict target column using binary classification",
                )
                llm_mode_input = gr.Radio(
                    label="LLM Provider",
                    choices=[
                        "Mock LLM (Offline / deterministic)",
                        "Real LLM (Groq + OpenRouter Fallback)",
                    ],
                    value="Mock LLM (Offline / deterministic)",
                )
                run_btn = gr.Button(
                    "🚀 Start Agentic Run",
                    variant="primary",
                    elem_classes=["cta-run"],
                )
                # Hidden state for run ID and message
                run_id_box = gr.Textbox(
                    label="Run ID",
                    interactive=False,
                    visible=True,
                )
                run_msg = gr.Textbox(
                    label="Submission",
                    interactive=False,
                    visible=True,
                )

            # RIGHT: Current Run status panel
            with gr.Column(scale=1):
                with gr.Row():
                    gr.HTML('<div class="panel-title" style="flex:1;">📊 CURRENT RUN</div>')
                    refresh_btn = gr.Button(
                        "↻ Refresh",
                        variant="secondary",
                        size="sm",
                        elem_classes=["btn-secondary"],
                    )
                # Status badge + meta grid rendered as HTML
                current_run_html = gr.HTML(
                    '<span class="status-badge status-empty">— No run selected</span>'
                )

        # ── Pipeline (full-width HTML) ────────────────────────────────────
        pipeline_html = gr.HTML(_pipeline_html({}))

        # ── Run Details + Raw JSON ────────────────────────────────────────
        gr.HTML('<div class="panel-title" style="margin-bottom:8px;">📋 RUN DETAILS</div>')
        details_html = gr.HTML(
            '<p style="color:var(--text3); font-size:0.85rem;">Start a run to view task context.</p>'
        )
        with gr.Accordion("View raw perception JSON", open=False, elem_classes=["accordion-raw"]):
            perc_raw = gr.Code(
                label="",
                language="json",
                interactive=False,
                lines=8,
            )

        gr.HTML('<hr class="section-divider">')

        # ── Metrics Strip ─────────────────────────────────────────────────
        metrics_html = gr.HTML(
            '<div class="metrics-strip">'
            '<div class="metric-cell"><div class="metric-number">—</div><div class="metric-label">Iterations</div></div>'
            '<div class="metric-cell"><div class="metric-number">—</div><div class="metric-label">Duration</div></div>'
            '<div class="metric-cell"><div class="metric-number">—</div><div class="metric-label">Best Score</div></div>'
            '</div>'
        )

        # ── Iteration History ─────────────────────────────────────────────
        gr.HTML('<div class="panel-title" style="margin-bottom:8px; margin-top:4px;">📜 ITERATION HISTORY</div>')
        iterations_html = gr.HTML(
            '<div style="color:var(--text3); font-size:0.85rem; padding: 8px 0;">No iterations yet.</div>'
        )

        # ── Diagnostics + Artifacts ───────────────────────────────────────
        gr.HTML('<hr class="section-divider">')
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<div class="panel-title" style="margin-bottom:10px;">🛠 DIAGNOSTICS</div>')
                diag_html = gr.HTML('<div class="diag-ok">✓ No errors reported</div>')

            with gr.Column(scale=1):
                gr.HTML('<div class="panel-title" style="margin-bottom:10px;">📦 ARTIFACTS</div>')
                artifacts_html = gr.HTML(
                    '<div style="color:var(--text3); font-size:0.85rem;">No artifacts available.</div>'
                )

        # ── Close shell HTML ──────────────────────────────────────────────
        gr.HTML("</div></div>")  # close .page-inner and .app-shell

        # ── Timer ─────────────────────────────────────────────────────────
        timer = gr.Timer(value=2, active=False)

        # ──────────────────────────────────────────────────────────────────
        # WIRING: Start Run → update raw state → render HTML panels
        # ──────────────────────────────────────────────────────────────────

        def _do_start(
            dataset: str, instr: str, mode: str | bool
        ) -> tuple[str, str, gr.Timer]:
            return submit_task(dataset, instr, mode)

        run_btn.click(
            _do_start,
            inputs=[dataset_path, instruction, llm_mode_input],
            outputs=[run_id_box, run_msg, timer],
        )

        # On refresh: pull raw backend state, store in hidden State, render HTML
        def _do_refresh(
            run_id: str, run_msg_val: str
        ) -> tuple[str, str, str, str, str, str, str,  # HTML panels (7)
                   str, str, str, str, str, str, str,  # raw state (7)
                   gr.Timer]:
            raw = refresh_status(run_id)
            (
                status_text, perc_text, lib_text,
                m_text, err_text, art_text, hist_text,
                tmr,
            ) = raw

            html_vals = render_all(
                status_text, perc_text, lib_text,
                m_text, err_text, art_text, hist_text,
                run_id, run_msg_val,
            )

            return (
                *html_vals,            # 7 HTML outputs
                status_text, perc_text, lib_text,
                m_text, err_text, art_text, hist_text,  # 7 raw state outputs
                tmr,
            )

        _html_outputs = [
            current_run_html,
            pipeline_html,
            details_html,
            metrics_html,
            iterations_html,
            diag_html,
            artifacts_html,
        ]
        _raw_state_outputs = [
            _status_text, _perc_text, _lib_text,
            _metrics_text, _error_text, _artifacts_text, _history_text,
        ]

        for event_fn in [refresh_btn.click, timer.tick]:
            event_fn(
                _do_refresh,
                inputs=[run_id_box, run_msg],
                outputs=[*_html_outputs, *_raw_state_outputs, timer],
            )

        # Also expose raw JSON in accordion
        def _update_raw_perc(perc_text: str) -> str:
            return perc_text if perc_text != "Waiting for run" else ""

        _perc_text.change(
            _update_raw_perc,
            inputs=[_perc_text],
            outputs=[perc_raw],
        )

    return demo  # type: ignore[no-any-return]
