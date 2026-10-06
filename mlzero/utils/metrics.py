"""
Centralized metric canonicalization and semantic directionality module.

Normalizes framework-internal metric scores (such as AutoGluon/scikit-learn
negative maximization scores for RMSE, MSE, MAE, Log Loss) into canonical,
human-interpretable positive values while strictly preserving genuine negative
metrics (R², Pearson, Spearman, MCC) and leaving higher-is-better metrics unchanged.
"""
from __future__ import annotations

import re
from typing import Any

# Metrics that legitimately permit negative values in standard statistics.
# These must NEVER be converted to positive or sign-flipped.
CAN_BE_NEGATIVE_METRICS: frozenset[str] = frozenset({
    "r2",
    "r_2",
    "r_squared",
    "r2_score",
    "pearson",
    "pearsonr",
    "pearson_correlation",
    "pearson_score",
    "spearman",
    "spearmanr",
    "spearman_correlation",
    "spearman_score",
    "mcc",
    "matthews_corrcoef",
    "matthews_correlation",
    "matthews_correlation_coefficient",
    "kendall_tau",
    "kendalltau",
})

# Known lower-is-better metrics (error, loss, distance, divergence).
# In standard human-facing convention, these represent positive magnitudes.
KNOWN_LOWER_IS_BETTER_METRICS: frozenset[str] = frozenset({
    "rmse",
    "root_mean_squared_error",
    "mse",
    "mean_squared_error",
    "mae",
    "mean_absolute_error",
    "median_absolute_error",
    "medae",
    "msle",
    "mean_squared_log_error",
    "rmsle",
    "root_mean_squared_log_error",
    "mape",
    "mean_absolute_percentage_error",
    "smape",
    "symmetric_mean_absolute_percentage_error",
    "wape",
    "mase",
    "log_loss",
    "cross_entropy",
    "cross_entropy_loss",
    "nll",
    "negative_log_likelihood",
    "brier_score_loss",
    "brier_score",
    "brier_loss",
    "hinge_loss",
    "pinball_loss",
    "quantile_loss",
    "per_sample_loss",
    "loss",
})

# Known higher-is-better metrics (accuracy, agreement, quality, ranking).
KNOWN_HIGHER_IS_BETTER_METRICS: frozenset[str] = frozenset({
    "accuracy",
    "acc",
    "balanced_accuracy",
    "bal_acc",
    "f1",
    "f1_score",
    "f1_macro",
    "f1_micro",
    "f1_weighted",
    "macro_f1",
    "micro_f1",
    "weighted_f1",
    "precision",
    "precision_score",
    "macro_precision",
    "micro_precision",
    "weighted_precision",
    "recall",
    "recall_score",
    "macro_recall",
    "micro_recall",
    "weighted_recall",
    "roc_auc",
    "auc",
    "roc_auc_score",
    "average_precision",
    "pr_auc",
    "cohen_kappa",
    "jaccard",
    "jaccard_score",
    "mrr",
    "mean_reciprocal_rank",
    "ndcg",
    "ndcg_score",
    "map",
    "mean_average_precision",
    "hit_rate",
    "recall_at_k",
    "precision_at_k",
})


def _normalize_name(name: str) -> str:
    """Normalize metric name for consistent lookup (lowercase, underscores)."""
    s = name.strip().lower()
    return re.sub(r"[\s\-]+", "_", s)


def can_be_negative_metric(name: str) -> bool:
    """
    Check if a metric legitimately permits negative values.
    
    Examples:
    - R² can be negative when a regression model performs worse than the mean.
    - Pearson/Spearman correlations range from -1.0 to +1.0.
    - MCC ranges from -1.0 to +1.0.
    """
    norm = _normalize_name(name)
    if norm in CAN_BE_NEGATIVE_METRICS:
        return True
    # Strip any accidental 'neg_' prefix if someone wrote neg_r2
    return bool(norm.startswith("neg_") and norm[4:] in CAN_BE_NEGATIVE_METRICS)


def is_higher_is_better_metric(name: str) -> bool:
    """Check if a metric is known to be higher-is-better."""
    norm = _normalize_name(name)
    if norm in KNOWN_HIGHER_IS_BETTER_METRICS:
        return True
    for suffix in ("_accuracy", "_acc", "_f1", "_precision", "_recall", "_auc", "_mrr", "_ndcg"):
        if norm.endswith(suffix):
            return True
    return False


def is_lower_is_better_metric(name: str) -> bool:
    """
    Check if a metric is lower-is-better (error, loss, deviation).
    
    Uses explicit registry and conservative pattern matching.
    Naturally signed metrics (R², Pearson, MCC) are strictly excluded.
    """
    norm = _normalize_name(name)

    # Exclude metrics that can legitimately be negative
    if can_be_negative_metric(norm):
        return False

    # Check for framework negated scorer prefix: neg_*
    if norm.startswith("neg_"):
        stem = norm[4:]
        if can_be_negative_metric(stem):
            return False
        if stem in KNOWN_LOWER_IS_BETTER_METRICS:
            return True
        if stem.endswith(("_error", "_loss")):
            return True
        return True

    # Check explicit known lower-is-better registry
    if norm in KNOWN_LOWER_IS_BETTER_METRICS:
        return True

    # Conservative pattern matching for error and loss metrics
    if norm.endswith(("_error", "_loss")):
        return True

    # Token match for common error acronyms
    tokens = set(norm.split("_"))
    return bool(tokens.intersection({"rmse", "mse", "mae", "medae", "msle", "rmsle", "mape", "smape", "wape"}))


def canonicalize_metric(name: str, value: Any) -> tuple[str, Any]:
    """
    Canonicalize a single metric name and value into standard human-interpretable representation.
    
    Rules:
    1. Framework negated scorer names (e.g. `neg_root_mean_squared_error`) have
       their `neg_` prefix removed and their value flipped to positive.
    2. Lower-is-better error metrics (RMSE, MSE, MAE, MedAE, Log Loss) with negative
       framework scores are converted to their standard positive values.
    3. Naturally signed metrics (R², Pearson, Spearman, MCC) remain strictly unchanged,
       preserving genuine negative values.
    4. Higher-is-better metrics (Accuracy, F1, Precision, Recall, ROC-AUC) remain unchanged.
    5. Unknown metrics remain unchanged (no blind guessing).
    6. Non-numeric values (e.g. string summaries) remain unchanged.
    
    Returns:
        tuple[str, Any]: (canonical_name, canonical_value)
    """
    norm = _normalize_name(name)

    # Rule 1: Framework negated scorer prefix (e.g. neg_root_mean_squared_error)
    if norm.startswith("neg_"):
        stem = name[4:] if name.lower().startswith("neg_") else name
        stem_norm = _normalize_name(stem)
        if not can_be_negative_metric(stem_norm):
            canonical_name = stem
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                canonical_val = abs(value)
            else:
                canonical_val = value
            return canonical_name, canonical_val

    # Rule 3: Naturally signed metrics must NEVER be sign-flipped
    if can_be_negative_metric(norm):
        return name, value

    # Rule 2: Lower-is-better metrics with negative scores
    if is_lower_is_better_metric(norm):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            canonical_val = abs(value) if value < 0 else value
            return name, canonical_val
        return name, value

    # Rules 4 & 5: Higher-is-better or Unknown metrics remain unchanged
    return name, value


def canonicalize_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    """
    Globally canonicalize a dictionary of metrics into human-facing values.
    
    This operation is fully idempotent:
    canonicalize_metrics(canonicalize_metrics(m)) == canonicalize_metrics(m)
    """
    if not metrics or not isinstance(metrics, dict):
        return {}

    canonical: dict[str, Any] = {}
    for k, v in metrics.items():
        can_k, can_v = canonicalize_metric(k, v)
        canonical[can_k] = can_v

    return canonical
