from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from company_capitalization.cli import main
from company_capitalization.evaluation import mape, regression_metrics
from company_capitalization.prediction import PREDICTION_COLUMN, load_model, predict_frame
from company_capitalization.training import (
    HOLDOUT_PREDICTIONS_FILE,
    METADATA_FILE,
    METRICS_FILE,
    MODEL_FILE,
    TrainConfig,
    split_indices,
    train,
)


def test_split_indices_are_disjoint_and_complete() -> None:
    index = pd.RangeIndex(100)
    train_idx, holdout_idx = split_indices(index, test_size=0.2, seed=3)
    assert train_idx.intersection(holdout_idx).empty
    assert sorted([*train_idx, *holdout_idx]) == list(index)
    assert len(holdout_idx) == 20


def test_train_saves_artifacts_and_roundtrip(synthetic_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "ridge"
    result = train(TrainConfig(data_path=synthetic_csv, output_dir=out, model="ridge", seed=7))
    for name in (MODEL_FILE, METRICS_FILE, METADATA_FILE, HOLDOUT_PREDICTIONS_FILE):
        assert (out / name).is_file()

    metadata = json.loads((out / METADATA_FILE).read_text(encoding="utf-8"))
    split = metadata["split"]
    assert set(split["train_index"]).isdisjoint(split["holdout_index"])
    assert set(split["train_symbols"]).isdisjoint(split["holdout_symbols"])
    assert metadata["config"]["seed"] == 7
    assert len(metadata["data"]["sha256"]) == 64
    assert metadata["target"] == {
        "column": "Market Cap",
        "transform": "log1p",
        "inverse_transform": "expm1",
    }
    assert "rmse" in result.metrics["holdout"]
    assert "rmse_log1p" in result.metrics["cv_train"]

    frame = pd.read_csv(synthetic_csv).drop(columns=["Market Cap"])
    loaded = load_model(out)
    holdout = pd.read_csv(out / HOLDOUT_PREDICTIONS_FILE)
    rows = frame.loc[holdout["row_index"]]
    np.testing.assert_allclose(
        predict_frame(loaded, rows)[PREDICTION_COLUMN], holdout["predicted_market_cap"]
    )


def test_training_is_reproducible(synthetic_csv: Path, tmp_path: Path) -> None:
    def run(folder: str) -> dict[str, float]:
        config = TrainConfig(
            data_path=synthetic_csv, output_dir=tmp_path / folder, model="random_forest"
        )
        config.params = {"n_estimators": 20}
        return train(config).metrics["holdout"]

    assert run("a") == run("b")


def test_invalid_config_rejected(synthetic_csv: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="test_size"):
        train(TrainConfig(data_path=synthetic_csv, output_dir=tmp_path, test_size=1.5))


def test_cli_train_then_predict_without_target(synthetic_csv: Path, tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    code = main(
        [
            "--log-level",
            "WARNING",
            "train",
            "--data",
            str(synthetic_csv),
            "--model",
            "ridge",
            "--cv-folds",
            "3",
            "--seed",
            "1",
            "--output-dir",
            str(artifacts),
            "--param",
            "alpha=2.5",
        ]
    )
    assert code == 0
    metadata = json.loads((artifacts / METADATA_FILE).read_text(encoding="utf-8"))
    assert metadata["model_params"]["alpha"] == 2.5

    no_target = tmp_path / "no_target.csv"
    pd.read_csv(synthetic_csv).drop(columns=["Market Cap"]).to_csv(no_target, index=False)
    output = tmp_path / "pred.csv"
    code = main(
        ["predict", "--model", str(artifacts), "--data", str(no_target), "--output", str(output)]
    )
    assert code == 0
    predictions = pd.read_csv(output)
    assert len(predictions) == len(pd.read_csv(no_target))
    assert list(predictions.columns) == ["Symbol", "Name", PREDICTION_COLUMN]
    assert (predictions[PREDICTION_COLUMN] > 0).all()


def test_cli_reports_schema_errors(synthetic_csv: Path, tmp_path: Path) -> None:
    broken = tmp_path / "broken.csv"
    pd.read_csv(synthetic_csv).drop(columns=["Sector"]).to_csv(broken, index=False)
    assert main(["train", "--data", str(broken), "--output-dir", str(tmp_path / "x")]) == 2
    assert main(["train", "--data", str(tmp_path / "missing.csv")]) == 2


def test_metrics_values() -> None:
    metrics = regression_metrics([100.0, 200.0], [110.0, 190.0])
    assert metrics["rmse"] == pytest.approx(10.0)
    assert metrics["mae"] == pytest.approx(10.0)
    assert metrics["mape"] == pytest.approx((0.1 + 0.05) / 2)
    assert set(metrics) == {"rmse", "mae", "r2", "mape", "rmse_log1p", "mae_log1p", "r2_log1p"}
    assert mape(np.array([0.0, 10.0]), np.array([5.0, 12.0])) == pytest.approx(0.2)
    assert np.isnan(mape(np.array([0.0]), np.array([1.0])))


def test_tuning_uses_train_only_and_respects_budget(synthetic_csv: Path, tmp_path: Path) -> None:
    pytest.importorskip("optuna")
    config = TrainConfig(
        data_path=synthetic_csv,
        output_dir=tmp_path / "tuned",
        model="ridge",
        cv_folds=3,
        tune_trials=3,
    )
    result = train(config)
    tuning = result.metadata["tuning"]
    assert tuning["n_trials"] == 3
    assert result.metadata["model_params"]["alpha"] == pytest.approx(tuning["best_params"]["alpha"])
