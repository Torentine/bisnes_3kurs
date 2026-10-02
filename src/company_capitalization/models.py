"""Model factories: every learned step lives inside one scikit-learn estimator.

The returned estimator is always::

    TransformedTargetRegressor(func=log1p, inverse_func=expm1,
        regressor=Pipeline([features -> preprocess -> model]))

so imputation, scaling and one-hot encoding are refit on the training part of every CV fold,
and ``predict`` returns Market Cap in the original currency units.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from company_capitalization.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    FeatureBuilder,
)

logger = logging.getLogger(__name__)

INSTALL_HINT = (
    "Install optional dependencies with `poetry install --extras boosting` "
    "(or `pip install 'company-capitalization[boosting]'`)."
)


class UnknownModelError(ValueError):
    """Raised for a model name that is not registered."""


class MissingDependencyError(ImportError):
    """Raised when an optional library required by a model is not installed."""


def target_forward(values: np.ndarray) -> np.ndarray:
    """Forward target transform: ``log1p`` (defined for the positive Market Cap)."""
    return np.log1p(values)


def target_inverse(values: np.ndarray) -> np.ndarray:
    """Inverse of :func:`target_forward`: ``expm1``."""
    return np.expm1(values)


def make_preprocessor(*, scale: bool) -> ColumnTransformer:
    """Imputation + optional robust scaling for numbers, one-hot encoding for ``Sector``.

    Sectors unseen during fit map to an all-zero code (missing ones are already
    ``__missing__`` after :func:`~company_capitalization.features.build_features`).
    """
    numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", RobustScaler()))
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_native_category_preprocessor() -> ColumnTransformer:
    """Pandas-output column selection for CatBoost: native Sector, NaN numbers kept.

    CatBoost handles missing numeric values itself; nothing is learned in this step.
    """
    transformer = ColumnTransformer(
        [
            ("categorical", "passthrough", list(CATEGORICAL_FEATURES)),
            ("numeric", "passthrough", list(NUMERIC_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    transformer.set_output(transform="pandas")
    return transformer


def _optional(module: str, attribute: str, model_name: str) -> Any:
    try:
        return getattr(importlib.import_module(module), attribute)
    except ImportError as error:
        raise MissingDependencyError(
            f"Model {model_name!r} requires the optional package {module!r}. {INSTALL_HINT}"
        ) from error


def _merge(defaults: Mapping[str, Any], params: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update(params or {})
    return merged


Factory = Callable[[int, int, Mapping[str, Any] | None], Any]


@dataclass(frozen=True)
class ModelSpec:
    """A registered algorithm: how to build it and how to preprocess its inputs."""

    name: str
    factory: Factory
    scale: bool = False
    native_categories: bool = False
    optional_module: str | None = None
    description: str = ""


def _dummy(seed: int, n_jobs: int, params: Mapping[str, Any] | None) -> Any:
    return DummyRegressor(**_merge({"strategy": "mean"}, params))


def _ridge(seed: int, n_jobs: int, params: Mapping[str, Any] | None) -> Any:
    return Ridge(**_merge({"alpha": 1.0}, params))


def _linear(seed: int, n_jobs: int, params: Mapping[str, Any] | None) -> Any:
    return LinearRegression(**_merge({}, params))


def _decision_tree(seed: int, n_jobs: int, params: Mapping[str, Any] | None) -> Any:
    defaults = {"max_depth": 6, "min_samples_leaf": 5, "random_state": seed}
    return DecisionTreeRegressor(**_merge(defaults, params))


def _random_forest(seed: int, n_jobs: int, params: Mapping[str, Any] | None) -> Any:
    defaults = {
        "n_estimators": 400,
        "min_samples_leaf": 2,
        "max_features": 0.5,
        "random_state": seed,
        "n_jobs": n_jobs,
    }
    return RandomForestRegressor(**_merge(defaults, params))


def _svr(seed: int, n_jobs: int, params: Mapping[str, Any] | None) -> Any:
    return SVR(**_merge({"C": 3.0, "epsilon": 0.1, "gamma": "scale"}, params))


def _xgboost(seed: int, n_jobs: int, params: Mapping[str, Any] | None) -> Any:
    regressor = _optional("xgboost", "XGBRegressor", "xgboost")
    defaults = {
        "n_estimators": 600,
        "learning_rate": 0.03,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": n_jobs,
    }
    return regressor(**_merge(defaults, params))


def _lightgbm(seed: int, n_jobs: int, params: Mapping[str, Any] | None) -> Any:
    regressor = _optional("lightgbm", "LGBMRegressor", "lightgbm")
    defaults = {
        "n_estimators": 600,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 10,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": n_jobs,
        "verbose": -1,
    }
    return regressor(**_merge(defaults, params))


def _catboost(seed: int, n_jobs: int, params: Mapping[str, Any] | None) -> Any:
    regressor = _optional("catboost", "CatBoostRegressor", "catboost")
    defaults = {
        "iterations": 800,
        "learning_rate": 0.05,
        "depth": 6,
        "loss_function": "RMSE",
        "random_seed": seed,
        "thread_count": n_jobs,
        # A tuple keeps sklearn.clone working (CatBoost copies lists, breaking identity).
        "cat_features": tuple(CATEGORICAL_FEATURES),
        "verbose": 0,
        "allow_writing_files": False,
    }
    return regressor(**_merge(defaults, params))


def _stacking(seed: int, n_jobs: int, params: Mapping[str, Any] | None) -> Any:
    """Stacking whose base estimators each own their preprocessing (fit per inner fold)."""
    base = [
        ("ridge", Pipeline([("preprocess", make_preprocessor(scale=True)), ("model", Ridge())])),
        (
            "svr",
            Pipeline(
                [("preprocess", make_preprocessor(scale=True)), ("model", _svr(seed, 1, None))]
            ),
        ),
        (
            "random_forest",
            Pipeline(
                [
                    ("preprocess", make_preprocessor(scale=False)),
                    ("model", _random_forest(seed, n_jobs, None)),
                ]
            ),
        ),
        (
            "hist_gradient_boosting",
            Pipeline(
                [
                    ("preprocess", make_preprocessor(scale=False)),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            max_iter=300, learning_rate=0.05, random_state=seed
                        ),
                    ),
                ]
            ),
        ),
    ]
    defaults = {
        "estimators": base,
        "final_estimator": RidgeCV(alphas=np.logspace(-3, 3, 13)),
        "cv": KFold(n_splits=5, shuffle=True, random_state=seed),
        "n_jobs": 1,
    }
    return StackingRegressor(**_merge(defaults, params))


MODEL_REGISTRY: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        ModelSpec("dummy", _dummy, description="Mean of log1p(Market Cap); sanity baseline"),
        ModelSpec("ridge", _ridge, scale=True, description="Ridge regression (baseline)"),
        ModelSpec("linear", _linear, scale=True, description="Ordinary least squares"),
        ModelSpec("decision_tree", _decision_tree, description="Single regression tree"),
        ModelSpec("random_forest", _random_forest, description="Random forest"),
        ModelSpec("svr", _svr, scale=True, description="RBF support vector regression"),
        ModelSpec("xgboost", _xgboost, optional_module="xgboost", description="XGBoost"),
        ModelSpec("lightgbm", _lightgbm, optional_module="lightgbm", description="LightGBM"),
        ModelSpec(
            "catboost",
            _catboost,
            native_categories=True,
            optional_module="catboost",
            description="CatBoost with native Sector category",
        ),
        ModelSpec(
            "stacking",
            _stacking,
            description="Stacking of ridge/SVR/RF/HistGB; preprocessing inside each base model",
        ),
    )
}


def get_spec(name: str) -> ModelSpec:
    """Return the registered spec for ``name``.

    Raises:
        UnknownModelError: If the name is not registered.
    """
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise UnknownModelError(f"Unknown model {name!r}. Available: {known}.") from None


def is_available(name: str) -> bool:
    """Whether the optional dependency of a model (if any) can be imported."""
    module = get_spec(name).optional_module
    if module is None:
        return True
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True


def build_model(
    name: str,
    *,
    seed: int = 42,
    n_jobs: int = 1,
    params: Mapping[str, Any] | None = None,
) -> TransformedTargetRegressor:
    """Build an unfitted end-to-end estimator for raw input tables.

    Args:
        name: Registered model name (see :data:`MODEL_REGISTRY`).
        seed: Random seed for the model and internal CV splits.
        n_jobs: Threads for the model itself; outer CV runs sequentially to avoid
            nested parallelism.
        params: Hyperparameters overriding the defaults of the final model.

    Returns:
        A ``TransformedTargetRegressor`` that predicts Market Cap in original units.
    """
    spec = get_spec(name)
    model = spec.factory(seed, n_jobs, params)
    if name == "stacking":
        steps: list[tuple[str, Any]] = [("features", FeatureBuilder()), ("model", model)]
    elif spec.native_categories:
        steps = [
            ("features", FeatureBuilder()),
            ("preprocess", make_native_category_preprocessor()),
            ("model", model),
        ]
    else:
        steps = [
            ("features", FeatureBuilder()),
            ("preprocess", make_preprocessor(scale=spec.scale)),
            ("model", model),
        ]
    return TransformedTargetRegressor(
        regressor=Pipeline(steps),
        func=target_forward,
        inverse_func=target_inverse,
        check_inverse=True,
    )
