"""falsealpha: search formulaic alphas in C++, then find out which ones are fake."""

from importlib.metadata import PackageNotFoundError, version

from ._core import (
    Data,
    FormulaError,
    backtest,
    backtest_many,
    canonical,
    evaluate,
    expected_max_sharpe,
    formula_size,
    moments,
    operators,
    pbo,
    probabilistic_sharpe,
    snoop_tests,
    summarize,
)
from .panel import Panel

try:
    __version__ = version("falsealpha")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0"

__all__ = [
    "Data", "FormulaError", "Panel", "backtest", "backtest_many", "canonical", "evaluate",
    "expected_max_sharpe", "formula_size", "moments", "operators", "pbo", "probabilistic_sharpe",
    "snoop_tests", "summarize",
]
