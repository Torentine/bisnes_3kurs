"""Command-line interface: ``capitalization train | predict | models``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from company_capitalization.data import SchemaError
from company_capitalization.models import (
    MODEL_REGISTRY,
    MissingDependencyError,
    UnknownModelError,
    is_available,
)

logger = logging.getLogger("company_capitalization")

DEFAULT_DATA = Path("financials_src.csv")
DEFAULT_ARTIFACTS = Path("artifacts")


def _parse_param(text: str) -> tuple[str, Any]:
    """Parse ``key=value``; the value is read as JSON when possible (numbers, bools)."""
    key, sep, raw = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"Expected key=value, got {text!r}")
    try:
        value: Any = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key, value


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="capitalization",
        description="Train and apply Market Cap regression models.",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train a model and save artifacts.")
    train.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Labelled CSV file.")
    train.add_argument("--model", default="ridge", choices=sorted(MODEL_REGISTRY))
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction.")
    train.add_argument("--cv-folds", type=int, default=5)
    train.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Artifact directory (default: artifacts/<model>).",
    )
    train.add_argument("--n-jobs", type=int, default=1, help="Threads per model.")
    train.add_argument(
        "--param",
        type=_parse_param,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a model hyperparameter; may be repeated.",
    )
    train.add_argument(
        "--tune-trials",
        type=int,
        default=0,
        help="Optuna trials on train CV (0 = no tuning; needs the 'tuning' extra).",
    )
    train.add_argument("--tune-timeout", type=float, default=None, help="Tuning budget, s.")

    predict = commands.add_parser("predict", help="Predict Market Cap for a CSV file.")
    predict.add_argument(
        "--model", type=Path, required=True, help="Artifact dir or model.joblib (trusted)."
    )
    predict.add_argument("--data", type=Path, required=True, help="CSV without the target.")
    predict.add_argument("--output", type=Path, required=True, help="Output CSV path.")

    commands.add_parser("models", help="List available models.")
    return parser


def _run_train(args: argparse.Namespace) -> int:
    from company_capitalization.training import TrainConfig, train

    config = TrainConfig(
        data_path=args.data,
        output_dir=args.output_dir or DEFAULT_ARTIFACTS / args.model,
        model=args.model,
        seed=args.seed,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        n_jobs=args.n_jobs,
        params=dict(args.param),
        tune_trials=args.tune_trials,
        tune_timeout=args.tune_timeout,
    )
    result = train(config)
    summary = {
        "model": args.model,
        "output_dir": str(result.output_dir),
        "cv_train": result.metrics["cv_train"],
        "holdout": result.metrics["holdout"],
    }
    print(json.dumps(summary, indent=2))
    return 0


def _run_predict(args: argparse.Namespace) -> int:
    from company_capitalization.prediction import predict_file

    predictions = predict_file(args.model, args.data, args.output)
    print(f"Wrote {len(predictions)} predictions to {args.output}")
    return 0


def _run_models() -> int:
    for name, spec in sorted(MODEL_REGISTRY.items()):
        status = "available" if is_available(name) else f"needs extra ({spec.optional_module})"
        print(f"{name:15s} {status:28s} {spec.description}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``capitalization`` command."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        if args.command == "train":
            return _run_train(args)
        if args.command == "predict":
            return _run_predict(args)
        return _run_models()
    except (
        FileNotFoundError,
        SchemaError,
        MissingDependencyError,
        UnknownModelError,
        ValueError,
    ) as error:
        logger.error("%s", error)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
