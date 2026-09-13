"""Tests for Phase 9: evaluation framework, metrics, report generation, CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# ── Metrics ──────────────────────────────────────────────────────────────────


def test_extract_metrics_success() -> None:
    from evaluation.metrics import extract_metrics

    status = SimpleNamespace(
        success=True,
        iterations=2,
        execution_duration=3.5,
        final_error=None,
        final_metrics={"accuracy": 0.9, "f1": 0.85},
    )
    m = extract_metrics(status)
    assert m["success"] is True
    assert m["iterations"] == 2
    assert m["ml_accuracy"] == pytest.approx(0.9)
    assert m["ml_f1"] == pytest.approx(0.85)
    assert m["ml_mae"] is None
    assert m["ml_rmse"] is None


def test_extract_metrics_failure() -> None:
    from evaluation.metrics import extract_metrics

    status = SimpleNamespace(
        success=False,
        iterations=1,
        execution_duration=1.0,
        final_error="KeyError",
        final_metrics=None,
    )
    m = extract_metrics(status)
    assert m["success"] is False
    assert m["ml_accuracy"] is None


def test_extract_metrics_regression() -> None:
    from evaluation.metrics import extract_metrics

    status = SimpleNamespace(
        success=True,
        iterations=2,
        execution_duration=5.0,
        final_error=None,
        final_metrics={"mae": 42.1, "rmse": 60.3},
    )
    m = extract_metrics(status)
    assert m["ml_mae"] == pytest.approx(42.1)
    assert m["ml_rmse"] == pytest.approx(60.3)


# ── Experiment runner ────────────────────────────────────────────────────────


def test_run_experiment_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluation.experiments import run_experiment
    from mlzero.application.service import MLZeroService
    from mlzero.schemas.application import RunStatusResponse

    def fake_run_mlzero(
        self: MLZeroService,
        dataset_path: str,
        user_instruction: str | None = None,
        options: dict | None = None,
    ) -> RunStatusResponse:
        return RunStatusResponse(
            run_id="test-eval",
            status="SUCCESS",
            success=True,
            iterations=2,
            execution_duration=3.0,
            final_metrics={"accuracy": 0.80},
        )

    monkeypatch.setattr(MLZeroService, "run_mlzero", fake_run_mlzero)

    result = run_experiment(
        "test_experiment",
        {"dataset": str(tmp_path), "mock_llm": True},
    )
    assert result["name"] == "test_experiment"
    assert result["metrics"]["success"] is True
    assert result["metrics"]["ml_accuracy"] == pytest.approx(0.80)
    assert "experiment_id" in result


def test_run_experiment_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluation.experiments import run_experiment
    from mlzero.application.service import MLZeroService

    def boom(self: MLZeroService, *a: object, **kw: object) -> None:
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(MLZeroService, "run_mlzero", boom)

    result = run_experiment("fail_exp", {"dataset": str(tmp_path), "mock_llm": True})
    assert result["metrics"]["success"] is False
    assert result["metrics"]["error_category"] == "system_crash"


# ── Config loading ────────────────────────────────────────────────────────────


def test_config_json_valid() -> None:
    """All built-in configs should parse as valid JSON lists."""
    config_dir = Path("evaluation/configs")
    for cfg_file in config_dir.glob("*.json"):
        with open(cfg_file) as f:
            data = json.load(f)
        assert isinstance(data, list), f"{cfg_file} should be a list"
        for entry in data:
            assert "name" in entry
            assert "dataset" in entry


# ── Report generation ────────────────────────────────────────────────────────


def test_generate_report(tmp_path: Path) -> None:
    from evaluation.report import generate_report

    results = [
        {
            "experiment_id": "aaa",
            "name": "exp1",
            "config": {"dataset": "d1", "mock_llm": True},
            "metrics": {
                "success": True,
                "iterations": 2,
                "execution_time": 3.5,
                "error_category": None,
            },
        },
        {
            "experiment_id": "bbb",
            "name": "exp2",
            "config": {"dataset": "d2", "mock_llm": True},
            "metrics": {
                "success": False,
                "iterations": 1,
                "execution_time": 1.0,
                "error_category": "KeyError",
            },
        },
    ]
    generate_report(results, out_dir=str(tmp_path))

    assert (tmp_path / "evaluation_summary.json").exists()
    assert (tmp_path / "evaluation_summary.csv").exists()
    assert (tmp_path / "evaluation_report.md").exists()
    assert (tmp_path / "figures" / "success_rate.png").exists()
    assert (tmp_path / "figures" / "iterations.png").exists()


# ── Evaluate CLI ──────────────────────────────────────────────────────────────


def test_evaluate_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mlzero.application.service import MLZeroService
    from mlzero.cli import main
    from mlzero.schemas.application import RunStatusResponse

    config_file = tmp_path / "test_config.json"
    config_file.write_text(
        json.dumps(
            [{"name": "cli_test", "dataset": str(tmp_path), "mock_llm": True}]
        )
    )

    def fake_run(
        self: MLZeroService,
        dataset_path: str,
        user_instruction: str | None = None,
        options: dict | None = None,
    ) -> RunStatusResponse:
        return RunStatusResponse(run_id="cli-eval", status="SUCCESS", success=True)

    monkeypatch.setattr(MLZeroService, "run_mlzero", fake_run)

    with patch.object(sys, "argv", ["mlzero", "evaluate", "--config", str(config_file)]):
        result = main()
    assert result == 0


# ── Demo CLI ──────────────────────────────────────────────────────────────────


def test_demo_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    from mlzero.application.service import MLZeroService
    from mlzero.cli import main
    from mlzero.schemas.application import RunStatusResponse

    def fake_run(
        self: MLZeroService,
        dataset_path: str,
        user_instruction: str | None = None,
        options: dict | None = None,
    ) -> RunStatusResponse:
        return RunStatusResponse(run_id="demo-run", status="SUCCESS", success=True, iterations=2)

    monkeypatch.setattr(MLZeroService, "run_mlzero", fake_run)

    with patch.object(sys, "argv", ["mlzero", "demo"]):
        result = main()
    assert result == 0
