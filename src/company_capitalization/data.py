"""Reading CSV files, schema validation and row-selection rules."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET = "Market Cap"
SECTOR = "Sector"
ID_COLUMNS: tuple[str, ...] = ("Symbol", "Name", "SEC Filings")

#: Raw numeric inputs that are known at prediction time (no target involved).
NUMERIC_INPUTS: tuple[str, ...] = (
    "Price",
    "Price/Earnings",
    "Dividend Yield",
    "Earnings/Share",
    "52 Week Low",
    "52 Week High",
    "EBITDA",
    "Price/Sales",
    "Price/Book",
)
CATEGORICAL_INPUTS: tuple[str, ...] = (SECTOR,)
REQUIRED_INPUTS: tuple[str, ...] = CATEGORICAL_INPUTS + NUMERIC_INPUTS

#: Pre-computed columns of ``financials.csv``. They equal ``Price / ratio`` (per share),
#: while the research notebook defines them as ``Market Cap / ratio`` (target leakage).
#: Because of this ambiguity they are never used as model inputs.
AMBIGUOUS_DERIVED_COLUMNS: tuple[str, ...] = ("Earnings", "Sales", "Book")


class SchemaError(ValueError):
    """Raised when an input table does not satisfy the expected schema."""


@dataclass(frozen=True)
class RowSelectionReport:
    """How many rows were removed from a training table and why."""

    n_input_rows: int
    n_missing_target: int
    n_non_positive_target: int
    n_duplicate_symbol: int
    n_kept_rows: int

    def to_dict(self) -> dict[str, int]:
        """Return the report as a JSON-serialisable dictionary."""
        return asdict(self)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV file with company financials.

    Args:
        path: Path to a comma-separated file with a header row.

    Returns:
        The raw table; values are not modified.

    Raises:
        FileNotFoundError: If the file does not exist.
        SchemaError: If the file is empty.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise SchemaError(f"Data file has no rows: {path}")
    logger.info("Loaded %d rows x %d columns from %s", len(frame), frame.shape[1], path)
    return frame


def validate_inputs(frame: pd.DataFrame, *, require_target: bool) -> pd.DataFrame:
    """Check required columns and coerce numeric columns.

    Non-numeric strings in numeric columns and ``±inf`` become ``NaN`` (imputed later inside
    the model pipeline); the number of coerced cells is logged.

    Args:
        frame: Raw input table.
        require_target: Whether the ``Market Cap`` column must be present.

    Returns:
        A copy of ``frame`` with numeric columns as ``float64`` and ``Sector`` as ``object``.

    Raises:
        SchemaError: If required columns are missing.
    """
    required = list(REQUIRED_INPUTS) + ([TARGET] if require_target else [])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}. Expected at least {required}.")

    result = frame.copy()
    numeric = [*NUMERIC_INPUTS, *([TARGET] if TARGET in result.columns else [])]
    for column in numeric:
        original = result[column]
        coerced = pd.to_numeric(original, errors="coerce").astype("float64")
        coerced = coerced.replace([np.inf, -np.inf], np.nan)
        n_bad = int((coerced.isna() & original.notna()).sum())
        if n_bad:
            logger.warning("Column %r: %d non-numeric or infinite values set to NaN", column, n_bad)
        result[column] = coerced
    sector = result[SECTOR]
    result[SECTOR] = sector.astype(object).where(sector.notna(), None)
    return result


def select_training_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, RowSelectionReport]:
    """Apply the documented row-selection rules for training.

    Rules (in order): drop rows with missing/non-finite target, drop rows with target <= 0
    (``log1p`` target transform and MAPE require a positive target), keep the first row of
    each duplicated ``Symbol`` (if the column exists) so one company cannot be in both splits.

    Args:
        frame: Validated table that contains the target column.

    Returns:
        The selected rows (original index preserved) and a report with exclusion counts.

    Raises:
        SchemaError: If the target column is missing or no rows remain.
    """
    if TARGET not in frame.columns:
        raise SchemaError(f"Training requires the target column {TARGET!r}.")
    n_input = len(frame)
    target = frame[TARGET]
    missing_mask = target.isna()
    non_positive_mask = ~missing_mask & (target <= 0)
    kept = frame.loc[~missing_mask & ~non_positive_mask]

    n_duplicates = 0
    if "Symbol" in kept.columns:
        duplicated = kept["Symbol"].notna() & kept["Symbol"].duplicated(keep="first")
        n_duplicates = int(duplicated.sum())
        kept = kept.loc[~duplicated]

    report = RowSelectionReport(
        n_input_rows=n_input,
        n_missing_target=int(missing_mask.sum()),
        n_non_positive_target=int(non_positive_mask.sum()),
        n_duplicate_symbol=n_duplicates,
        n_kept_rows=len(kept),
    )
    logger.info("Row selection: %s", report.to_dict())
    if report.n_kept_rows < 10:
        raise SchemaError(
            f"Only {report.n_kept_rows} rows with a valid positive {TARGET!r} remain; "
            "at least 10 are required for training."
        )
    return kept, report


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model inputs from the target.

    Only :data:`REQUIRED_INPUTS` are returned as ``X``; identifiers, the target and the
    ambiguous derived columns are excluded.
    """
    features = frame.loc[:, list(REQUIRED_INPUTS)].copy()
    target = frame[TARGET].astype("float64")
    return features, target


def load_training_data(path: Path) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Read, validate and select rows of a labelled table.

    Returns:
        ``X``, ``y`` and a dictionary with data provenance (path, SHA-256, row report).
    """
    raw = read_table(path)
    ignored = [column for column in AMBIGUOUS_DERIVED_COLUMNS if column in raw.columns]
    if ignored:
        logger.info("Ignoring ambiguous derived columns %s", ignored)
    validated = validate_inputs(raw, require_target=True)
    selected, report = select_training_rows(validated)
    features, target = split_features_target(selected)
    info: dict[str, Any] = {
        "path": str(path),
        "sha256": file_sha256(path),
        "columns": list(raw.columns),
        "ignored_columns": [*ID_COLUMNS, *ignored],
        "row_selection": report.to_dict(),
    }
    if "Symbol" in selected.columns:
        info["symbols"] = selected["Symbol"].astype(str).tolist()
    return features, target, info
