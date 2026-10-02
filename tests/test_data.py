from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from company_capitalization.data import (
    REQUIRED_INPUTS,
    TARGET,
    SchemaError,
    file_sha256,
    load_training_data,
    read_table,
    select_training_rows,
    split_features_target,
    validate_inputs,
)


def test_missing_required_column_raises(synthetic_frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="EBITDA"):
        validate_inputs(synthetic_frame.drop(columns=["EBITDA"]), require_target=False)


def test_missing_target_rejected_for_training(synthetic_frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="Market Cap"):
        validate_inputs(synthetic_frame.drop(columns=[TARGET]), require_target=True)


def test_target_optional_for_prediction(synthetic_frame: pd.DataFrame) -> None:
    validated = validate_inputs(synthetic_frame.drop(columns=[TARGET]), require_target=False)
    assert TARGET not in validated.columns


def test_non_numeric_and_infinite_values_become_nan(synthetic_frame: pd.DataFrame) -> None:
    frame = synthetic_frame.astype({"Price": object, "EBITDA": object})
    frame.loc[0, "Price"] = "n/a"
    frame.loc[1, "EBITDA"] = np.inf
    validated = validate_inputs(frame, require_target=True)
    assert np.isnan(validated.loc[0, "Price"])
    assert np.isnan(validated.loc[1, "EBITDA"])
    assert validated["Price"].dtype == np.float64


def test_row_selection_rules_and_counts(synthetic_frame: pd.DataFrame) -> None:
    frame = synthetic_frame.copy()
    frame.loc[0, TARGET] = np.nan
    frame.loc[1, TARGET] = 0.0
    frame.loc[2, TARGET] = -5.0
    frame.loc[3, "Symbol"] = frame.loc[4, "Symbol"]
    selected, report = select_training_rows(validate_inputs(frame, require_target=True))
    assert report.n_missing_target == 1
    assert report.n_non_positive_target == 2
    assert report.n_duplicate_symbol == 1
    assert report.n_kept_rows == len(frame) - 4 == len(selected)
    assert (selected[TARGET] > 0).all()


def test_too_few_valid_rows_raise(synthetic_frame: pd.DataFrame) -> None:
    frame = synthetic_frame.head(12).copy()
    frame.loc[:5, TARGET] = np.nan
    with pytest.raises(SchemaError, match="rows"):
        select_training_rows(validate_inputs(frame, require_target=True))


def test_split_excludes_target_ids_and_derived(synthetic_frame: pd.DataFrame) -> None:
    frame = synthetic_frame.assign(Earnings=1.0, Sales=2.0, Book=3.0)
    features, target = split_features_target(frame)
    assert list(features.columns) == list(REQUIRED_INPUTS)
    for column in (TARGET, "Symbol", "Name", "SEC Filings", "Earnings", "Sales", "Book"):
        assert column not in features.columns
    assert target.name == TARGET


def test_load_training_data_provenance(synthetic_csv: Path) -> None:
    features, target, info = load_training_data(synthetic_csv)
    assert len(features) == len(target) == info["row_selection"]["n_kept_rows"]
    assert info["sha256"] == file_sha256(synthetic_csv)
    assert len(info["sha256"]) == 64


def test_read_table_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_table(tmp_path / "missing.csv")
    empty = tmp_path / "empty.csv"
    empty.write_text("Symbol,Price\n", encoding="utf-8")
    with pytest.raises(SchemaError):
        read_table(empty)
