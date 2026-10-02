from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from company_capitalization.data import TARGET, SchemaError, validate_inputs
from company_capitalization.features import (
    FEATURE_NAMES,
    FeatureBuilder,
    build_features,
    safe_divide,
    week52_bounds,
    week52_diagnostics,
)


def test_target_never_used_as_feature(synthetic_frame: pd.DataFrame) -> None:
    frame = validate_inputs(synthetic_frame, require_target=True)
    features = build_features(frame)
    assert TARGET not in features.columns
    assert list(features.columns) == list(FEATURE_NAMES)
    assert not any("market" in name.lower() for name in features.columns)


def test_changing_target_does_not_change_features(synthetic_frame: pd.DataFrame) -> None:
    frame = validate_inputs(synthetic_frame, require_target=True)
    shuffled = frame.assign(**{TARGET: frame[TARGET].sample(frac=1, random_state=1).to_numpy()})
    pd.testing.assert_frame_equal(build_features(frame), build_features(shuffled))
    without = frame.drop(columns=[TARGET])
    pd.testing.assert_frame_equal(build_features(frame), build_features(without))


def test_feature_builder_ignores_y(synthetic_frame: pd.DataFrame) -> None:
    frame = validate_inputs(synthetic_frame, require_target=True)
    y = frame[TARGET]
    first = FeatureBuilder().fit(frame, y).transform(frame)
    second = FeatureBuilder().fit(frame, -y * 7).transform(frame)
    pd.testing.assert_frame_equal(first, second)


def test_week52_policy_orders_bounds_and_flags_swaps() -> None:
    low = pd.Series([259.77, 10.0, 5.0, np.nan])
    high = pd.Series([175.49, 20.0, 5.0, 3.0])
    bounds = week52_bounds(low, high)
    assert bounds["lower"].tolist()[:3] == [175.49, 10.0, 5.0]
    assert bounds["upper"].tolist()[:3] == [259.77, 20.0, 5.0]
    assert (bounds["upper"] - bounds["lower"]).dropna().ge(0).all()
    assert bounds["swapped"].tolist()[:3] == [1.0, 0.0, 0.0]
    assert np.isnan(bounds["swapped"].iloc[3])


def test_week52_derived_features_handle_zero_range(synthetic_frame: pd.DataFrame) -> None:
    frame = validate_inputs(synthetic_frame.head(3), require_target=False)
    frame.loc[0, ["52 Week Low", "52 Week High"]] = 50.0
    features = build_features(frame)
    assert np.isnan(features.loc[0, "price_position_week52"])
    assert features.loc[0, "week52_range_rel"] == 0.0
    assert (features["week52_range_rel"] >= 0).all()
    assert (features.loc[1:, "week52_bounds_swapped"] == 1.0).all()


def test_week52_diagnostics_counts(synthetic_frame: pd.DataFrame) -> None:
    diagnostics = week52_diagnostics(synthetic_frame)
    assert diagnostics["n_low_greater_than_high"] == len(synthetic_frame)
    assert diagnostics["n_missing_bounds"] == 0


def test_safe_divide_zero_and_missing_denominators() -> None:
    result = safe_divide(pd.Series([1.0, 1.0, 1.0, 2.0]), pd.Series([0.0, np.nan, np.inf, 4.0]))
    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == 0.0
    assert result.iloc[3] == 0.5


def test_ebitda_policy_keeps_sign_and_flags_zero(synthetic_frame: pd.DataFrame) -> None:
    frame = validate_inputs(synthetic_frame.head(4), require_target=False)
    frame["EBITDA"] = [1e9, 0.0, -2e8, np.nan]
    features = build_features(frame)
    assert features["ebitda_log_positive"].iloc[0] == pytest.approx(np.log1p(1e9))
    assert features["ebitda_log_positive"].iloc[1:].isna().all()
    assert features["ebitda_sign"].iloc[:3].tolist() == [1.0, 0.0, -1.0]
    assert features["ebitda_missing_or_zero"].tolist() == [0.0, 1.0, 0.0, 1.0]


def test_no_infinite_values_in_features(synthetic_frame: pd.DataFrame) -> None:
    frame = validate_inputs(synthetic_frame, require_target=False)
    frame.loc[0, "Price/Earnings"] = 0.0
    frame.loc[1, "Price/Book"] = 0.0
    numeric = build_features(frame).drop(columns=["Sector"])
    assert not np.isinf(numeric.to_numpy()).any()


def test_feature_builder_rejects_missing_columns(synthetic_frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="Price/Book"):
        FeatureBuilder().fit(synthetic_frame.drop(columns=["Price/Book"]))
    with pytest.raises(SchemaError):
        FeatureBuilder().fit(synthetic_frame.to_numpy())  # type: ignore[arg-type]
