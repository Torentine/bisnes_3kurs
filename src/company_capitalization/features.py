"""Deterministic, stateless feature engineering on prediction-time inputs only.

Every engineered feature is computed row-wise from :data:`~company_capitalization.data.
REQUIRED_INPUTS`; the target column is never read, so the same code runs for training and
for prediction on tables without ``Market Cap``.

Feature catalogue (name: formula):

* ``Sector``: kept as a category; missing values become ``"__missing__"``.
* ``price_log``: ``log1p(Price)``; price per share, positive.
* ``pe_ratio``, ``dividend_yield``, ``eps``, ``ps_ratio``, ``pb_ratio``: raw inputs.
* ``earnings_yield``: ``1 / Price/Earnings`` (E/P); continuous through loss-making companies,
  unlike P/E itself. ``NaN`` if the ratio is 0 or missing.
* ``sales_yield``: ``1 / Price/Sales`` (S/P per share); ``book_yield``: ``1 / Price/Book``.
* EBITDA: 58 of 505 rows report exactly ``0`` (51 of them Financials, where EBITDA is not a
  meaningful metric), so ``0`` is treated as "not reported", not as zero earnings.
  ``ebitda_log_positive = log1p(EBITDA)`` for ``EBITDA > 0`` (else ``NaN``, imputed later),
  ``ebitda_sign`` in ``{-1, 0, 1}`` keeps the sign of loss-making companies (no ``log(|x|)``),
  ``ebitda_missing_or_zero`` flags missing or zero values.
* 52-week bounds: the source files report ``52 Week Low > 52 Week High`` in every row, so the
  bounds are re-derived as ``week52_lower = min(Low, High)`` and ``week52_upper = max(...)``.
  ``week52_bounds_swapped`` flags rows where ``Low > High`` in the input,
  ``week52_range_rel = (upper - lower) / upper`` is a non-negative relative price range (not a
  volatility estimate), ``price_to_week52_upper = Price / upper`` and
  ``price_position_week52 = (Price - lower) / (upper - lower)`` (``NaN`` for a zero range).
"""

from __future__ import annotations

import logging
from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from company_capitalization.data import REQUIRED_INPUTS, SECTOR, SchemaError

logger = logging.getLogger(__name__)

MISSING_CATEGORY = "__missing__"
CATEGORICAL_FEATURES: tuple[str, ...] = (SECTOR,)
NUMERIC_FEATURES: tuple[str, ...] = (
    "price_log",
    "pe_ratio",
    "dividend_yield",
    "eps",
    "ps_ratio",
    "pb_ratio",
    "earnings_yield",
    "sales_yield",
    "book_yield",
    "ebitda_log_positive",
    "ebitda_sign",
    "ebitda_missing_or_zero",
    "week52_bounds_swapped",
    "week52_range_rel",
    "price_to_week52_upper",
    "price_position_week52",
)
FEATURE_NAMES: tuple[str, ...] = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division that yields ``NaN`` for zero/missing denominators."""
    denominator = denominator.where(denominator != 0)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def week52_bounds(low: pd.Series, high: pd.Series) -> pd.DataFrame:
    """Derive ordered 52-week bounds and a flag for inputs where ``Low > High``.

    The source values are not modified; ``lower``/``upper`` are the element-wise min/max,
    so ``range = upper - lower`` is never negative.
    """
    lower = np.fmin(low, high)
    upper = np.fmax(low, high)
    swapped = (low > high).astype("float64").where(low.notna() & high.notna())
    return pd.DataFrame({"lower": lower, "upper": upper, "swapped": swapped})


def week52_diagnostics(frame: pd.DataFrame) -> dict[str, int]:
    """Count rows with inverted, equal or missing 52-week bounds (for run metadata)."""
    low, high = frame["52 Week Low"], frame["52 Week High"]
    complete = low.notna() & high.notna()
    return {
        "n_rows": len(frame),
        "n_low_greater_than_high": int((complete & (low > high)).sum()),
        "n_equal_bounds": int((complete & (low == high)).sum()),
        "n_missing_bounds": int((~complete).sum()),
    }


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the model feature table from raw prediction-time inputs.

    Args:
        frame: Table containing at least :data:`REQUIRED_INPUTS` (numeric columns already
            coerced to float). Any other columns, including the target, are ignored.

    Returns:
        A new frame with columns :data:`FEATURE_NAMES` and the same index.

    Raises:
        SchemaError: If required input columns are missing.
    """
    missing = [column for column in REQUIRED_INPUTS if column not in frame.columns]
    if missing:
        raise SchemaError(f"Cannot build features, missing columns: {missing}")

    def num(column: str) -> pd.Series:
        return pd.to_numeric(frame[column], errors="coerce").astype("float64")

    price = num("Price")
    ebitda = num("EBITDA")
    bounds = week52_bounds(num("52 Week Low"), num("52 Week High"))
    width = bounds["upper"] - bounds["lower"]
    ones = pd.Series(1.0, index=frame.index)

    features = pd.DataFrame(
        {
            SECTOR: frame[SECTOR].astype(object).where(frame[SECTOR].notna(), MISSING_CATEGORY),
            "price_log": np.log1p(price.where(price >= 0)),
            "pe_ratio": num("Price/Earnings"),
            "dividend_yield": num("Dividend Yield"),
            "eps": num("Earnings/Share"),
            "ps_ratio": num("Price/Sales"),
            "pb_ratio": num("Price/Book"),
            "earnings_yield": safe_divide(ones, num("Price/Earnings")),
            "sales_yield": safe_divide(ones, num("Price/Sales")),
            "book_yield": safe_divide(ones, num("Price/Book")),
            "ebitda_log_positive": np.log1p(ebitda.where(ebitda > 0)),
            "ebitda_sign": np.sign(ebitda),
            "ebitda_missing_or_zero": (ebitda.isna() | (ebitda == 0)).astype("float64"),
            "week52_bounds_swapped": bounds["swapped"],
            "week52_range_rel": safe_divide(width, bounds["upper"]),
            "price_to_week52_upper": safe_divide(price, bounds["upper"]),
            "price_position_week52": safe_divide(price - bounds["lower"], width),
        },
        index=frame.index,
    )
    numeric = list(NUMERIC_FEATURES)
    features[numeric] = features[numeric].replace([np.inf, -np.inf], np.nan)
    return features.loc[:, list(FEATURE_NAMES)]


class FeatureBuilder(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Scikit-learn wrapper around :func:`build_features`.

    The transformer is stateless: ``fit`` only records input column names, so it is safe to
    place in front of cross-validation, stacking or any learned preprocessing.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """Validate the schema of ``X``; nothing is learned. ``y`` is ignored."""
        if not isinstance(X, pd.DataFrame):
            raise SchemaError("FeatureBuilder expects a pandas DataFrame with named columns.")
        build_features(X.head(0))
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return engineered features for ``X``."""
        if not isinstance(X, pd.DataFrame):
            raise SchemaError("FeatureBuilder expects a pandas DataFrame with named columns.")
        return build_features(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return the names of the engineered features."""
        return np.asarray(FEATURE_NAMES, dtype=object)
