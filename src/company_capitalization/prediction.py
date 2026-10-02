"""Loading a trained artifact and predicting Market Cap for tables without the target."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from company_capitalization.data import ID_COLUMNS, read_table, validate_inputs
from company_capitalization.training import (
    ARTIFACT_FORMAT_VERSION,
    METADATA_FILE,
    MODEL_FILE,
)

logger = logging.getLogger(__name__)

PREDICTION_COLUMN = "predicted_market_cap"


def resolve_model_path(path: Path) -> Path:
    """Accept either an artifact directory or a direct path to ``model.joblib``."""
    return path / MODEL_FILE if path.is_dir() else path


def load_metadata(model_path: Path) -> dict[str, Any] | None:
    """Read ``metadata.json`` saved next to the model, if present."""
    metadata_path = model_path.parent / METADATA_FILE
    if not metadata_path.is_file():
        return None
    payload: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    return payload


def load_model(path: Path) -> Any:
    """Load a trained estimator.

    Security: joblib uses pickle, which can execute code. Only load artifacts that you
    produced yourself or obtained from a trusted source.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the artifact format version is not supported.
    """
    model_path = resolve_model_path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    metadata = load_metadata(model_path)
    if metadata is not None:
        found = metadata.get("artifact_format_version")
        if found != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported artifact format {found!r}; expected {ARTIFACT_FORMAT_VERSION}."
            )
    else:
        logger.warning("No %s next to %s; schema cannot be cross-checked", METADATA_FILE, path)
    return joblib.load(model_path)


def predict_frame(model: Any, frame: pd.DataFrame) -> pd.DataFrame:
    """Predict Market Cap for a raw input table.

    The target column is not required; if present it is ignored by the model.

    Returns:
        Identifier columns present in ``frame`` plus ``predicted_market_cap`` (USD).
    """
    validated = validate_inputs(frame, require_target=False)
    predictions = model.predict(validated)
    result = frame.loc[:, [c for c in ID_COLUMNS if c in frame.columns and c != "SEC Filings"]]
    result = result.copy()
    result[PREDICTION_COLUMN] = predictions
    return result


def predict_file(model_path: Path, data_path: Path, output_path: Path) -> pd.DataFrame:
    """Load a model, predict for a CSV file and write the predictions as CSV."""
    model = load_model(model_path)
    predictions = predict_frame(model, read_table(data_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    logger.info("Wrote %d predictions to %s", len(predictions), output_path)
    return predictions
