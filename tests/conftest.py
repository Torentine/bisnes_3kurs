from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SECTORS = ("Industrials", "Financials", "Health Care", "Energy")


def make_synthetic_frame(n_rows: int = 120, seed: int = 0) -> pd.DataFrame:
    """Small table with the raw schema of financials_src.csv and a learnable target."""
    rng = np.random.default_rng(seed)
    price = rng.uniform(10, 400, n_rows)
    ebitda = rng.lognormal(mean=21, sigma=1.0, size=n_rows)
    ebitda[rng.random(n_rows) < 0.08] = 0.0
    ebitda[rng.random(n_rows) < 0.04] *= -0.1
    multiple = rng.uniform(6, 20, n_rows)
    market_cap = np.abs(ebitda) * multiple + rng.lognormal(22, 0.5, n_rows)
    high = price * rng.uniform(0.6, 0.95, n_rows)  # mimics the source: Low > High
    low = price * rng.uniform(1.05, 1.4, n_rows)
    pe = rng.normal(20, 8, n_rows)
    return pd.DataFrame(
        {
            "Symbol": [f"S{i:03d}" for i in range(n_rows)],
            "Name": [f"Company {i}" for i in range(n_rows)],
            "Sector": rng.choice(SECTORS, n_rows),
            "Price": price,
            "Price/Earnings": pe,
            "Dividend Yield": rng.uniform(0, 5, n_rows),
            "Earnings/Share": price / pe,
            "52 Week Low": low,
            "52 Week High": high,
            "Market Cap": market_cap,
            "EBITDA": ebitda,
            "Price/Sales": rng.uniform(0.5, 10, n_rows),
            "Price/Book": rng.uniform(0.5, 20, n_rows),
            "SEC Filings": "http://example.invalid",
        }
    )


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    return make_synthetic_frame()


@pytest.fixture
def synthetic_csv(tmp_path: Path, synthetic_frame: pd.DataFrame) -> Path:
    path = tmp_path / "train.csv"
    synthetic_frame.to_csv(path, index=False)
    return path
