"""Synthetic panels with a known answer, for self-tests and demos.

`planted` hides one real alpha in random data: returns two days after day t
are pushed by the planted formula's cross-sectional rank on day t (exactly
what a delay-1 strategy trading it earns). `noise` has no alpha at all, so
everything a search finds there is false by construction.
"""

from __future__ import annotations

import re

import numpy as np

from . import _core
from .panel import Panel


def _base(T: int, N: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {"volume": np.exp(rng.normal(10, 1, (T, N)))}


def _finish(T: int, N: int, fields: dict[str, np.ndarray], ret: np.ndarray, source: str) -> Panel:
    close = 100 * np.cumprod(1 + ret, axis=0)
    fields = {**fields, "close": close, "returns": ret}
    dates = np.datetime64("2018-01-01") + np.arange(T)
    return Panel(dates, [f"A{n:03d}" for n in range(N)], fields, np.ones((T, N), bool), source)


def noise(T: int = 1500, N: int = 60, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    fields = _base(T, N, rng)
    ret = rng.normal(0, 0.02, (T, N))
    return _finish(T, N, fields, ret, f"synthetic noise seed={seed}")


def planted(formula: str = "rank(ts_mean(volume, 5))", beta: float = 0.004, T: int = 1500, N: int = 60,
            seed: int = 0) -> Panel:
    if re.search(r"\b(close|returns)\b", formula):
        raise ValueError("a planted formula may only use volume: close and returns are generated from it")
    rng = np.random.default_rng(seed)
    fields = _base(T, N, rng)
    probe = Panel(np.datetime64("2018-01-01") + np.arange(T), [str(n) for n in range(N)],
                  {**fields, "returns": np.zeros((T, N))}, np.ones((T, N), bool))
    sig = _core.evaluate(formula, probe.core())
    ranked = (np.argsort(np.argsort(np.where(np.isfinite(sig), sig, -np.inf), axis=1), axis=1) / (N - 1)) - 0.5
    ranked[~np.isfinite(sig)] = 0
    ret = rng.normal(0, 0.02, (T, N))
    ret[2:] += beta * ranked[:-2]
    return _finish(T, N, fields, ret, f"synthetic planted {formula!r} beta={beta} seed={seed}")
