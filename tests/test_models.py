from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from company_capitalization.data import TARGET, split_features_target, validate_inputs
from company_capitalization.features import build_features
from company_capitalization.models import (
    MODEL_REGISTRY,
    MissingDependencyError,
    UnknownModelError,
    build_model,
    is_available,
    target_forward,
    target_inverse,
)

FAST_MODELS = ["dummy", "ridge", "linear", "decision_tree", "svr"]
FAST_PARAMS: dict[str, dict[str, Any]] = {
    "random_forest": {"n_estimators": 20},
    "xgboost": {"n_estimators": 30},
    "lightgbm": {"n_estimators": 30},
    "catboost": {"iterations": 30},
}


def _xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return split_features_target(validate_inputs(frame, require_target=True))


def test_target_transform_pair_is_consistent() -> None:
    values = np.array([1.0, 1e6, 3.5e9, 2.9e12])
    np.testing.assert_allclose(target_inverse(target_forward(values)), values, rtol=1e-12)
    model = build_model("ridge")
    assert model.func is target_forward
    assert model.inverse_func is target_inverse
    assert model.check_inverse


def test_unknown_model_name() -> None:
    with pytest.raises(UnknownModelError, match="Available"):
        build_model("gradient_magic")


def test_missing_optional_dependency_explains_install(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = importlib.import_module

    def fake_import(name: str, *args: Any) -> Any:
        if name == "xgboost":
            raise ImportError("simulated")
        return real_import(name, *args)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    assert not is_available("xgboost")
    with pytest.raises(MissingDependencyError, match="--extras boosting"):
        build_model("xgboost")


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_every_model_handles_unknown_sector_and_nan(name: str, synthetic_frame: Any) -> None:
    if not is_available(name):
        pytest.skip(f"{name} optional dependency not installed")
    x, y = _xy(synthetic_frame)
    model = build_model(name, seed=0, params=FAST_PARAMS.get(name))
    model.fit(x, y)

    new = x.head(5).copy()
    new.loc[new.index[0], "Sector"] = "Brand New Sector"
    new.loc[new.index[1], "Sector"] = None
    new.loc[new.index[2], ["EBITDA", "Price/Book", "52 Week Low"]] = np.nan
    predictions = model.predict(new)
    assert predictions.shape == (5,)
    assert np.isfinite(predictions).all()
    assert (predictions > -1).all()  # expm1 output is always > -1


def test_preprocessing_is_fit_inside_each_cv_fold(synthetic_frame: Any) -> None:
    x, y = _xy(synthetic_frame)
    x.loc[x.index[:15], "EBITDA"] = np.nan  # make medians fold-dependent
    splitter = KFold(n_splits=4, shuffle=True, random_state=0)
    result = cross_validate(build_model("ridge"), x, y, cv=splitter, return_estimator=True)
    full_median = build_features(x)["ebitda_log_positive"].median()
    fold_medians = []
    for estimator, (train_idx, _) in zip(result["estimator"], splitter.split(x), strict=True):
        imputer = estimator.regressor_.named_steps["preprocess"].named_transformers_["numeric"]
        position = list(imputer.feature_names_in_).index("ebitda_log_positive")
        fitted = imputer.named_steps["impute"].statistics_[position]
        expected = build_features(x.iloc[train_idx])["ebitda_log_positive"].median()
        assert fitted == pytest.approx(expected)
        fold_medians.append(fitted)
    assert any(m != pytest.approx(full_median) for m in fold_medians)


def test_stacking_base_estimators_own_their_preprocessing() -> None:
    model = build_model("stacking")
    pipeline = model.regressor
    assert [name for name, _ in pipeline.steps] == ["features", "model"]
    stacking = pipeline.named_steps["model"]
    assert isinstance(stacking.cv, KFold)
    for name, base in stacking.estimators:
        assert isinstance(base, Pipeline), name
        assert base.steps[0][0] == "preprocess", name


@pytest.mark.parametrize("name", ["ridge", "random_forest"])
def test_predictions_do_not_depend_on_target_column(name: str, synthetic_frame: Any) -> None:
    x, y = _xy(synthetic_frame)
    model = build_model(name, params=FAST_PARAMS.get(name)).fit(x, y)
    with_target = x.assign(**{TARGET: 1.0})
    np.testing.assert_allclose(model.predict(x), model.predict(with_target))
