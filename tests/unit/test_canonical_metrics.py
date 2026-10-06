"""
Unit tests for global metric canonicalization and semantic directionality.

Verifies:
1. Lower-is-better metrics (RMSE, MSE, MAE, MedAE, Log Loss) are converted from
   negative framework scores to canonical positive human-interpretable values.
2. Negated framework scorer names (e.g. `neg_root_mean_squared_error`) strip `neg_`
   and convert negative scores to positive.
3. Higher-is-better metrics (Accuracy, F1, Precision, Recall, ROC-AUC) remain unchanged.
4. Naturally signed metrics (R², Pearson, Spearman, MCC) remain strictly unchanged,
   preserving genuine negative values.
5. Unknown metrics remain unchanged without blind guessing.
6. Canonicalization is 100% idempotent.
7. Raw metrics are preserved intact in summary.json, Service, and API models.
8. Evaluation extraction produces canonical values.
"""
from unittest.mock import MagicMock

from evaluation.metrics import extract_metrics
from mlzero.schemas.application import RunStatusResponse
from mlzero.utils.metrics import (
    can_be_negative_metric,
    canonicalize_metric,
    canonicalize_metrics,
    is_higher_is_better_metric,
    is_lower_is_better_metric,
)


def test_lower_is_better_conversion():
    """Verify lower-is-better metrics convert negative framework scores to positive."""
    raw = {
        "rmse": -35.9010,
        "root_mean_squared_error": -35.9010,
        "mse": -1288.8829,
        "mean_squared_error": -1288.8829,
        "mae": -27.6667,
        "mean_absolute_error": -27.6667,
        "median_absolute_error": -19.5000,
        "medae": -19.5000,
        "msle": -0.45,
        "rmsle": -0.45,
        "log_loss": -0.45,
        "cross_entropy": -0.45,
        "mape": -5.2,
    }
    canonical = canonicalize_metrics(raw)

    assert canonical["rmse"] == 35.9010
    assert canonical["root_mean_squared_error"] == 35.9010
    assert canonical["mse"] == 1288.8829
    assert canonical["mean_squared_error"] == 1288.8829
    assert canonical["mae"] == 27.6667
    assert canonical["mean_absolute_error"] == 27.6667
    assert canonical["median_absolute_error"] == 19.5000
    assert canonical["medae"] == 19.5000
    assert canonical["msle"] == 0.45
    assert canonical["rmsle"] == 0.45
    assert canonical["log_loss"] == 0.45
    assert canonical["cross_entropy"] == 0.45
    assert canonical["mape"] == 5.2

    # Direct canonicalize_metric test
    k, v = canonicalize_metric("rmse", -35.9010)
    assert k == "rmse"
    assert v == 35.9010

    # Already positive values must remain positive and unchanged
    already_pos = {"rmse": 35.9010, "mae": 27.6667}
    assert canonicalize_metrics(already_pos) == already_pos


def test_negated_framework_scorer_names():
    """Verify framework scorer names starting with neg_ are stripped and made positive."""
    raw = {
        "neg_root_mean_squared_error": -35.9010,
        "neg_mean_squared_error": -1288.8829,
        "neg_mean_absolute_error": -27.6667,
        "neg_log_loss": -0.45,
    }
    canonical = canonicalize_metrics(raw)

    assert canonical == {
        "root_mean_squared_error": 35.9010,
        "mean_squared_error": 1288.8829,
        "mean_absolute_error": 27.6667,
        "log_loss": 0.45,
    }


def test_higher_is_better_unchanged():
    """Verify higher-is-better metrics remain completely unchanged."""
    raw = {
        "accuracy": 0.85,
        "f1": 0.92,
        "precision": 0.88,
        "recall": 0.91,
        "roc_auc": 0.96,
        "balanced_accuracy": 0.84,
        "mrr": 0.89,
        "ndcg": 0.93,
    }
    canonical = canonicalize_metrics(raw)
    assert canonical == raw


def test_naturally_signed_metrics_unchanged():
    """Verify naturally signed metrics (R², Pearson, Spearman, MCC) preserve genuine negative values."""
    raw = {
        "r2": -0.25,
        "r_squared": -0.25,
        "r2_score": -0.25,
        "pearson": -0.42,
        "pearson_correlation": -0.42,
        "spearman": -0.38,
        "spearman_correlation": -0.38,
        "mcc": -0.15,
        "matthews_corrcoef": -0.15,
    }
    canonical = canonicalize_metrics(raw)

    assert canonical["r2"] == -0.25
    assert canonical["r_squared"] == -0.25
    assert canonical["r2_score"] == -0.25
    assert canonical["pearson"] == -0.42
    assert canonical["pearson_correlation"] == -0.42
    assert canonical["spearman"] == -0.38
    assert canonical["spearman_correlation"] == -0.38
    assert canonical["mcc"] == -0.15
    assert canonical["matthews_corrcoef"] == -0.15


def test_unknown_metric_preservation():
    """Verify unknown metrics remain untouched without blind guessing."""
    raw = {
        "custom_metric": -8.0,
        "my_score": -12.5,
        "unknown_eval": -99.9,
    }
    canonical = canonicalize_metrics(raw)
    assert canonical == raw


def test_idempotency():
    """Verify that canonicalizing metrics multiple times is strictly idempotent."""
    raw = {
        "neg_root_mean_squared_error": -35.9010,
        "rmse": -35.9010,
        "r2": -0.25,
        "accuracy": 0.85,
        "custom_metric": -8.0,
    }
    pass1 = canonicalize_metrics(raw)
    pass2 = canonicalize_metrics(pass1)
    pass3 = canonicalize_metrics(pass2)

    assert pass1 == pass2
    assert pass2 == pass3
    assert pass1 == {
        "root_mean_squared_error": 35.9010,
        "rmse": 35.9010,
        "r2": -0.25,
        "accuracy": 0.85,
        "custom_metric": -8.0,
    }


def test_metric_direction_classifiers():
    """Test direction helper functions."""
    assert is_lower_is_better_metric("rmse") is True
    assert is_lower_is_better_metric("mean_squared_error") is True
    assert is_lower_is_better_metric("log_loss") is True
    assert is_lower_is_better_metric("neg_custom_loss") is True
    assert is_lower_is_better_metric("r2") is False
    assert is_lower_is_better_metric("accuracy") is False
    assert is_lower_is_better_metric("custom_metric") is False

    assert can_be_negative_metric("r2") is True
    assert can_be_negative_metric("r_squared") is True
    assert can_be_negative_metric("pearson") is True
    assert can_be_negative_metric("spearman") is True
    assert can_be_negative_metric("mcc") is True
    assert can_be_negative_metric("rmse") is False
    assert can_be_negative_metric("accuracy") is False

    assert is_higher_is_better_metric("accuracy") is True
    assert is_higher_is_better_metric("f1") is True
    assert is_higher_is_better_metric("roc_auc") is True
    assert is_higher_is_better_metric("rmse") is False


def test_run_status_response_raw_and_canonical():
    """Verify RunStatusResponse exposes both canonical final_metrics and raw_metrics."""
    raw = {"rmse": -35.9010, "r2": -0.25, "accuracy": 0.85}
    canonical = canonicalize_metrics(raw)

    response = RunStatusResponse(
        run_id="run-test-123",
        status="SUCCESS",
        success=True,
        final_metrics=canonical,
        raw_metrics=raw,
    )

    assert response.final_metrics["rmse"] == 35.9010
    assert response.final_metrics["r2"] == -0.25
    assert response.final_metrics["accuracy"] == 0.85

    assert response.raw_metrics["rmse"] == -35.9010
    assert response.raw_metrics["r2"] == -0.25
    assert response.raw_metrics["accuracy"] == 0.85


def test_evaluation_extract_metrics_canonicalization():
    """Verify evaluation.extract_metrics returns canonical positive error metrics."""
    mock_run_status = MagicMock()
    mock_run_status.success = True
    mock_run_status.iterations = 2
    mock_run_status.execution_duration = 15.4
    mock_run_status.final_error = None
    mock_run_status.final_metrics = {
        "rmse": -35.9010,
        "mae": -27.6667,
        "r2": -0.25,
        "accuracy": 0.85,
        "f1": 0.84,
    }

    extracted = extract_metrics(mock_run_status, task_type="regression")

    assert extracted["rmse"] == 35.9010
    assert extracted["ml_rmse"] == 35.9010
    assert extracted["mae"] == 27.6667
    assert extracted["ml_mae"] == 27.6667
    assert extracted["r2"] == -0.25
    assert extracted["accuracy"] == 0.85
    assert extracted["f1"] == 0.84


def test_service_enrich_metrics_with_autogluon_scores(tmp_path):
    """Verify service._enrich_metrics handles AutoGluon negative maximization scores properly."""
    from mlzero.application.service import MLZeroService

    # AutoGluon negative regression scores
    raw_ag_metrics = {
        "root_mean_squared_error": -35.9010,
        "mean_squared_error": -1288.8829,
        "mean_absolute_error": -27.6667,
        "median_absolute_error": -19.5000,
        "r2": 0.8276,
        "pearson": 0.9621,
    }

    enriched = MLZeroService._enrich_metrics(
        metrics=raw_ag_metrics,
        workspace=tmp_path,
        found_pred=None,
        pctx=None,
    )

    assert enriched is not None
    assert enriched["root_mean_squared_error"] == 35.9010
    assert enriched["mean_squared_error"] == 1288.8829
    assert enriched["mean_absolute_error"] == 27.6667
    assert enriched["median_absolute_error"] == 19.5000
    assert enriched["r2"] == 0.8276
    assert enriched["pearson"] == 0.9621


def test_negative_r2_remains_negative():
    """Verify models worse than baseline with negative R² remain negative."""
    raw = {"r2": -0.45, "rmse": -12.3}
    canonical = canonicalize_metrics(raw)
    assert canonical["r2"] == -0.45
    assert canonical["rmse"] == 12.3

