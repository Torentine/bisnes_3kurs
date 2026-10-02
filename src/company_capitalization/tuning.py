"""Optional, explicitly budgeted Optuna search (CV on the training split only)."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

from company_capitalization.evaluation import cv_scorers
from company_capitalization.models import MissingDependencyError, build_model, get_spec

logger = logging.getLogger(__name__)

SearchSpace = Callable[[Any], dict[str, Any]]


def _ridge_space(trial: Any) -> dict[str, Any]:
    return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}


def _svr_space(trial: Any) -> dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 0.1, 100.0, log=True),
        "epsilon": trial.suggest_float("epsilon", 0.01, 0.5, log=True),
    }


def _tree_space(trial: Any) -> dict[str, Any]:
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
    }


def _forest_space(trial: Any) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_float("max_features", 0.2, 1.0),
    }


def _xgboost_space(trial: Any) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _lightgbm_space(trial: Any) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 64, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }


def _catboost_space(trial: Any) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 200, 1200, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 3, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
    }


SEARCH_SPACES: dict[str, SearchSpace] = {
    "ridge": _ridge_space,
    "svr": _svr_space,
    "decision_tree": _tree_space,
    "random_forest": _forest_space,
    "xgboost": _xgboost_space,
    "lightgbm": _lightgbm_space,
    "catboost": _catboost_space,
}


def tune_hyperparameters(
    model_name: str,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    n_trials: int,
    timeout: float | None = None,
    cv_folds: int = 5,
    seed: int = 42,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Search hyperparameters minimising CV RMSE in log1p scale on the given (training) data.

    Args:
        model_name: Registered model with a search space in :data:`SEARCH_SPACES`.
        features: Training inputs only; the holdout must not be passed here.
        target: Training target in original units.
        n_trials: Maximum number of trials (required budget).
        timeout: Optional wall-clock budget in seconds.
        cv_folds: Number of shuffled KFold splits.
        seed: Seed for the sampler, CV splits and models.
        n_jobs: Threads per model.

    Returns:
        ``{"best_params": ..., "best_cv_rmse_log1p": ..., "n_trials": ...}``.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if model_name not in SEARCH_SPACES:
        raise ValueError(
            f"No search space for {model_name!r}. Tunable: {', '.join(sorted(SEARCH_SPACES))}."
        )
    get_spec(model_name)
    try:
        optuna = importlib.import_module("optuna")
    except ImportError as error:
        raise MissingDependencyError(
            "Tuning requires optuna: `poetry install --extras tuning`."
        ) from error

    space = SEARCH_SPACES[model_name]
    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scorer = cv_scorers()["rmse_log1p"]

    def objective(trial: Any) -> float:
        estimator = build_model(model_name, seed=seed, n_jobs=n_jobs, params=space(trial))
        scores = cross_val_score(estimator, features, target, cv=splitter, scoring=scorer, n_jobs=1)
        return float(-np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    logger.info("Best CV rmse_log1p %.4f with %s", study.best_value, study.best_params)
    return {
        "best_params": dict(study.best_params),
        "best_cv_rmse_log1p": float(study.best_value),
        "n_trials": len(study.trials),
    }
