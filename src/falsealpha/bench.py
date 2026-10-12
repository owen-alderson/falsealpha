"""Speed benchmark: the same formulas in falsealpha's C++ core and in pandas.

Canonical falsealpha formulas are valid Python expressions, so the pandas side
evaluates the identical text with pandas implementations of each operator
(the usual way Alpha101 is written in pandas: rolling(), rank(axis=1), ...).
Needs pandas: pip install "falsealpha[dev]".
"""

from __future__ import annotations

import ast
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import _core
from .panel import Panel

# Kakushadze (2016) "101 Formulaic Alphas", rewritten with falsealpha operators.
FORMULAS = {
    "alpha002": "-1 * ts_corr(rank(ts_delta(log(volume), 2)), rank((close - open) / open), 6)",
    "alpha003": "-1 * ts_corr(rank(open), rank(volume), 10)",
    "alpha004": "-1 * ts_rank(rank(low), 9)",
    "alpha006": "-1 * ts_corr(open, volume, 10)",
    "alpha012": "sign(ts_delta(volume, 1)) * (-1 * ts_delta(close, 1))",
    "alpha014": "-1 * rank(ts_delta(returns, 3)) * ts_corr(open, volume, 10)",
    "alpha033": "rank(-1 * (1 - open / close))",
    "alpha034": "rank((1 - rank(ts_std(returns, 2) / ts_std(returns, 5))) + (1 - rank(ts_delta(close, 1))))",
    "alpha044": "-1 * ts_corr(high, rank(volume), 5)",
    "alpha053": "-1 * ts_delta(((close - low) - (high - close)) / (close - low), 9)",
    "alpha101": "(close - open) / ((high - low) + 0.001)",
    "decay": "rank(decay_linear(ts_delta(vwap, 4), 7)) - rank(ts_argmax(close, 10))",
}


# One operator each, for the per-operator table.
SINGLE = ["rank(close)", "ts_mean(close, 20)", "ts_std(close, 20)", "ts_corr(close, volume, 20)",
          "ts_max(close, 20)", "ts_rank(close, 20)", "decay_linear(close, 20)", "ts_argmax(close, 20)"]


def random_panel(days: int, assets: int, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.02, (days, assets))
    close = 100 * np.cumprod(1 + ret, axis=0)
    open_ = close * (1 + rng.normal(0, 0.005, close.shape))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.01, close.shape)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.01, close.shape)))
    volume = rng.lognormal(12, 1, close.shape)
    vwap = (high + low + close) / 3
    fields = dict(open=open_, high=high, low=low, close=close, volume=volume, vwap=vwap, returns=ret)
    return Panel(np.datetime64("2015-01-01") + np.arange(days), [f"S{i}" for i in range(assets)], fields,
                 np.ones((days, assets), bool), "benchmark random walk")


def pandas_namespace(panel: Panel) -> dict:
    import pandas as pd

    def roll_apply(fn):
        return lambda x, d: x.rolling(d).apply(fn, raw=True)

    decay_w = {}

    def decay(x, d):
        w = decay_w.setdefault(d, np.arange(1, d + 1) / (d * (d + 1) / 2))
        return x.rolling(d).apply(lambda a: a @ w, raw=True)

    ns = {name: pd.DataFrame(arr) for name, arr in panel.fields.items()}
    ns.update(
        log=np.log, abs=np.abs, sign=np.sign,
        rank=lambda x: x.rank(axis=1, pct=True),
        zscore=lambda x: x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0),
        demean=lambda x: x.sub(x.mean(axis=1), axis=0),
        scale=lambda x: x.div(x.abs().sum(axis=1), axis=0),
        ts_delay=lambda x, d: x.shift(d),
        ts_delta=lambda x, d: x - x.shift(d),
        ts_sum=lambda x, d: x.rolling(d).sum(),
        ts_mean=lambda x, d: x.rolling(d).mean(),
        ts_std=lambda x, d: x.rolling(d).std(),
        ts_min=lambda x, d: x.rolling(d).min(),
        ts_max=lambda x, d: x.rolling(d).max(),
        ts_corr=lambda x, y, d: x.rolling(d).corr(y),
        ts_cov=lambda x, y, d: x.rolling(d).cov(y),
        ts_rank=roll_apply(lambda a: (a < a[-1]).sum() / (len(a) - 1)),
        ts_argmax=roll_apply(lambda a: np.argmax(a[::-1])),  # days since the max, ties -> most recent
        ts_argmin=roll_apply(lambda a: np.argmin(a[::-1])),
        decay_linear=decay,
    )
    return ns


def run(days: int = 2000, assets: int = 500, repeat: int = 3, log=print) -> dict:
    panel = random_panel(days, assets)
    data = panel.core()
    formulas = {k: _core.canonical(f) for k, f in FORMULAS.items()}
    threads = os.cpu_count() or 1

    def best(fn) -> float:
        times = []
        for _ in range(repeat):
            t = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t)
        return min(times)

    single = best(lambda: [_core.evaluate(f, data) for f in formulas.values()])
    with ThreadPoolExecutor(threads) as pool:
        parallel = best(lambda: list(pool.map(lambda f: _core.evaluate(f, data), formulas.values())))
    out = {"days": days, "assets": assets, "formulas": len(formulas), "falsealpha_1_thread": single,
           f"falsealpha_{threads}_threads": parallel}
    log(f"{len(formulas)} Alpha101-style formulas on {days} days x {assets} assets")
    log(f"  falsealpha, 1 thread    {single * 1000:9.1f} ms")
    log(f"  falsealpha, {threads} threads  {parallel * 1000:9.1f} ms")
    try:
        ns = pandas_namespace(panel)
    except ImportError:
        log("  pandas not installed: skipped")
        return out
    code = {k: compile(ast.parse(f, mode="eval"), k, "eval") for k, f in formulas.items()}
    slow = best(lambda: [eval(c, {"__builtins__": {}}, ns) for c in code.values()])  # noqa: S307
    out["pandas"] = slow
    log(f"  pandas                  {slow * 1000:9.1f} ms   ({slow / single:.0f}x slower than 1 thread, "
        f"{slow / parallel:.0f}x slower than {threads})")
    log("\n  per operator (1 thread)          falsealpha      pandas")
    out["operators"] = {}
    for f in SINGLE:
        fast = best(lambda: _core.evaluate(f, data))
        c = compile(ast.parse(f, mode="eval"), f, "eval")
        ref = best(lambda: eval(c, {"__builtins__": {}}, ns))  # noqa: S307
        out["operators"][f] = (fast, ref)
        log(f"  {f:30s} {fast * 1000:8.1f} ms {ref * 1000:9.1f} ms  {ref / fast:6.0f}x")
    return out
