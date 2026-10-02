"""Train/holdout split, cross-validation on train, final fit and artifact saving."""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate, train_test_split

from company_capitalization import __version__
from company_capitalization.data import (
    CATEGORICAL_INPUTS,
    NUMERIC_INPUTS,
    REQUIRED_INPUTS,
    TARGET,
    load_training_data,
)
from company_capitalization.evaluation import cv_scorers, regression_metrics, summarize_cv
from company_capitalization.features import FEATURE_NAMES, week52_diagnostics
from company_capitalization.models import build_model, get_spec

logger = logging.getLogger(__name__)

MODEL_FILE = "model.joblib"
METRICS_FILE = "metrics.json"
METADATA_FILE = "metadata.json"
HOLDOUT_PREDICTIONS_FILE = "holdout_predictions.csv"
ARTIFACT_FORMAT_VERSION = 1


@dataclass
class TrainConfig:
    """Parameters of one training run."""

    data_path: Path
    output_dir: Path
    model: str = "ridge"
    seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    n_jobs: int = 1
    params: dict[str, Any] = field(default_factory=dict)
    tune_trials: int = 0
    tune_timeout: float | None = None

    def validate(self) -> None:
        """Raise ``ValueError`` for inconsistent settings."""
        get_spec(self.model)
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be in (0, 1)")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be >= 2")
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be >= 1")
        if self.tune_trials < 0:
            raise ValueError("tune_trials must be >= 0")


@dataclass
class TrainResult:
    """Paths and metrics produced by :func:`train`."""

    output_dir: Path
    metrics: dict[str, Any]
    metadata: dict[str, Any]


def split_indices(index: pd.Index, *, test_size: float, seed: int) -> tuple[pd.Index, pd.Index]:
    """Shuffle-split row labels into disjoint train and holdout sets."""
    train_idx, holdout_idx = train_test_split(
        index.to_numpy(), test_size=test_size, random_state=seed, shuffle=True
    )
    train, holdout = pd.Index(train_idx), pd.Index(holdout_idx)
    if train.intersection(holdout).size:
        raise RuntimeError("Train and holdout indices overlap")
    return train, holdout


def library_versions(extra: tuple[str, ...] = ()) -> dict[str, str]:
    """Return installed versions of the libraries that affect a saved model."""
    names = ("numpy", "pandas", "scikit-learn", "joblib", *extra)
    versions: dict[str, str] = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return repr(value)


def _model_params(estimator: Any) -> dict[str, Any]:
    final_step = estimator.regressor.named_steps["model"]
    params = final_step.get_params(deep=False)
    params.pop("estimators", None)  # nested pipelines are summarised by name
    if hasattr(final_step, "estimators"):
        params["estimators"] = [name for name, _ in final_step.estimators]
    return dict(_json_safe(params))


def run_cross_validation(
    estimator: Any, features: pd.DataFrame, target: pd.Series, *, folds: int, seed: int
) -> dict[str, dict[str, float]]:
    """Shuffled KFold CV on the given (training) rows; preprocessing refits per fold."""
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    results = cross_validate(
        estimator,
        features,
        target,
        cv=splitter,
        scoring=cv_scorers(),
        n_jobs=1,
        error_score="raise",
    )
    return summarize_cv(results)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a UTF-8 JSON file with stable formatting."""
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def train(config: TrainConfig) -> TrainResult:
    """Run the full training procedure and save artifacts to ``config.output_dir``.

    Procedure: load + validate data, split train/holdout, optionally tune on train with CV,
    cross-validate on train, refit on the whole train split, evaluate once on holdout.
    The holdout is never used for tuning, early stopping or model selection.
    """
    config.validate()
    spec = get_spec(config.model)
    features, target, data_info = load_training_data(config.data_path)
    train_idx, holdout_idx = split_indices(
        features.index, test_size=config.test_size, seed=config.seed
    )
    x_train, y_train = features.loc[train_idx], target.loc[train_idx]
    x_holdout, y_holdout = features.loc[holdout_idx], target.loc[holdout_idx]
    logger.info("Split: %d train rows, %d holdout rows", len(train_idx), len(holdout_idx))

    params = dict(config.params)
    tuning: dict[str, Any] | None = None
    if config.tune_trials:
        from company_capitalization.tuning import tune_hyperparameters

        tuning = tune_hyperparameters(
            config.model,
            x_train,
            y_train,
            n_trials=config.tune_trials,
            timeout=config.tune_timeout,
            cv_folds=config.cv_folds,
            seed=config.seed,
            n_jobs=config.n_jobs,
        )
        params.update(tuning["best_params"])

    estimator = build_model(config.model, seed=config.seed, n_jobs=config.n_jobs, params=params)
    logger.info("Cross-validating %s with %d folds on train", config.model, config.cv_folds)
    cv_summary = run_cross_validation(
        estimator, x_train, y_train, folds=config.cv_folds, seed=config.seed
    )

    estimator.fit(x_train, y_train)
    holdout_pred = estimator.predict(x_holdout)
    metrics: dict[str, Any] = {
        "model": config.model,
        "units": {"original": "Market Cap (USD)", "log1p": "log1p(Market Cap)"},
        "cv_train": cv_summary,
        "holdout": regression_metrics(y_holdout, holdout_pred),
        "train_fit": regression_metrics(y_train, estimator.predict(x_train)),
        "n_train": len(train_idx),
        "n_holdout": len(holdout_idx),
    }

    symbols = data_info.pop("symbols", None)
    symbol_map = dict(zip(features.index, symbols, strict=True)) if symbols else {}
    metadata: dict[str, Any] = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "package_version": __version__,
        "versions": library_versions((spec.optional_module,) if spec.optional_module else ()),
        "platform": platform.platform(),
        "config": {
            "model": config.model,
            "seed": config.seed,
            "test_size": config.test_size,
            "cv_folds": config.cv_folds,
            "n_jobs": config.n_jobs,
            "tune_trials": config.tune_trials,
            "tune_timeout": config.tune_timeout,
            "params_override": params,
        },
        "model_params": _model_params(estimator),
        "tuning": tuning,
        "target": {"column": TARGET, "transform": "log1p", "inverse_transform": "expm1"},
        "input_schema": {
            "required_columns": list(REQUIRED_INPUTS),
            "numeric_columns": list(NUMERIC_INPUTS),
            "categorical_columns": list(CATEGORICAL_INPUTS),
            "engineered_features": list(FEATURE_NAMES),
        },
        "data": data_info,
        "week52_diagnostics": week52_diagnostics(features),
        "split": {
            "method": "train_test_split(shuffle=True)",
            "train_index": [int(i) for i in train_idx],
            "holdout_index": [int(i) for i in holdout_idx],
            "train_symbols": [symbol_map[i] for i in train_idx] if symbol_map else None,
            "holdout_symbols": [symbol_map[i] for i in holdout_idx] if symbol_map else None,
        },
    }

    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, output / MODEL_FILE)
    write_json(output / METRICS_FILE, metrics)
    write_json(output / METADATA_FILE, metadata)
    holdout_frame = pd.DataFrame(
        {
            "row_index": holdout_idx,
            "Symbol": [symbol_map.get(i) for i in holdout_idx],
            "actual_market_cap": y_holdout.to_numpy(),
            "predicted_market_cap": holdout_pred,
        }
    )
    holdout_frame.to_csv(output / HOLDOUT_PREDICTIONS_FILE, index=False)
    logger.info("Artifacts saved to %s", output)
    return TrainResult(output_dir=output, metrics=metrics, metadata=metadata)
