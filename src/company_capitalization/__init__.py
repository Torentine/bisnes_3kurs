"""Leakage-safe Market Cap regression for S&P 500 company financials."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("company-capitalization")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"

__all__ = ["__version__"]
