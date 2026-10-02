"""Regression metrics in original units (USD) and, separately labelled, in log1p scale."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

#: Metric names. Original-scale metrics are in Market Cap units (USD);
#: ``*_log1p`` metrics are computed on ``log1p(Market Cap)``.
ORIGINAL_SCALE_METRICS: tuple[str, ...] = ("rmse", "mae", "r2", "mape")
LOG_SCALE_METRICS: tuple[str, ...] = ("rmse_log1p", "mae_log1p", "r2_log1p")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error over rows with ``y_true > 0``.

    Rows with a non-positive target are skipped (training already excludes them);
    returns ``NaN`` if no such rows exist.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask]) / y_true[mask]))


def _log1p_clipped(values: np.ndarray) -> np.ndarray:
    # Predictions of expm1(model) are > -1 by construction; clipping guards odd inputs.
    clipped: np.ndarray = np.clip(np.asarray(values, dtype=float), 0.0, None)
    return np.asarray(np.log1p(clipped), dtype=float)


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute all metrics for predictions in original units.

    Args:
        y_true: True Market Cap values.
        y_pred: Predicted Market Cap values (same units).

    Returns:
        ``rmse``, ``mae``, ``r2``, ``mape`` in original scale and ``*_log1p`` metrics.
    """
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {true.shape} vs {pred.shape}")
    log_true, log_pred = _log1p_clipped(true), _log1p_clipped(pred)
    return {
        "rmse": _rmse(true, pred),
        "mae": float(mean_absolute_error(true, pred)),
        "r2": float(r2_score(true, pred)),
        "mape": mape(true, pred),
        "rmse_log1p": _rmse(log_true, log_pred),
        "mae_log1p": float(mean_absolute_error(log_true, log_pred)),
        "r2_log1p": float(r2_score(log_true, log_pred)),
    }


def _rmse_log1p(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return _rmse(_log1p_clipped(y_true), _log1p_clipped(y_pred))


def _mae_log1p(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(_log1p_clipped(y_true), _log1p_clipped(y_pred)))


def _r2_log1p(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(_log1p_clipped(y_true), _log1p_clipped(y_pred)))


def cv_scorers() -> dict[str, Callable[..., float]]:
    """Scorers for ``cross_validate``; error metrics are negated (higher is better)."""
    return {
        "rmse": make_scorer(_rmse, greater_is_better=False),
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "r2": make_scorer(r2_score),
        "mape": make_scorer(mape, greater_is_better=False),
        "rmse_log1p": make_scorer(_rmse_log1p, greater_is_better=False),
        "mae_log1p": make_scorer(_mae_log1p, greater_is_better=False),
        "r2_log1p": make_scorer(_r2_log1p),
    }


def summarize_cv(results: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Turn ``cross_validate`` output into ``{metric: {mean, std}}`` with positive errors."""
    summary: dict[str, dict[str, float]] = {}
    for name in (*ORIGINAL_SCALE_METRICS, *LOG_SCALE_METRICS):
        key = f"test_{name}"
        if key not in results:
            continue
        values = np.asarray(results[key], dtype=float)
        if not name.startswith("r2"):
            values = -values
        summary[name] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return summary
