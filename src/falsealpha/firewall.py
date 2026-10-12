"""The overfitting firewall: decide which mined formulas are more than luck.

Days are split train | validation | holdout. The search only ever sees train,
and validation and holdout are scored after the fact. Tests, in order:

1. Train Deflated Sharpe (Bailey & Lopez de Prado 2014). Each finalist's train
   Sharpe is deflated by the best Sharpe expected from *all* the trials the
   search ran, not just the finalists. The benchmark is the expected maximum of
   N unskilled strategies, whose Sharpe estimates have variance 1/(T-1). The
   paper plugs in the variance *across* trials instead; in a search that finds
   something real, dozens of variants of the real signal inflate that variance
   and the test ends up punishing the discovery. We report it as a diagnostic.
   Counting every distinct formula as a trial overstates the number of
   independent trials (near-copies are correlated), so the test is conservative.
2. Validation Deflated Sharpe. The finalists are fresh trials on unseen days,
   deflated for the number of finalists.
3. Hansen's SPA and White's Reality Check on validation. Is the best finalist
   better than flat after allowing for the fact that we looked at all of them?
   This is a global gate: if it fails, nothing survives.
4. PBO by CSCV over train + validation, a property of the whole procedure:
   how often the in-sample winner lands in the bottom half out of sample.

A finalist survives if it passes 1 and 2 with a positive validation Sharpe
and the gate in 3 is open.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import _core


@dataclass
class Thresholds:
    dsr: float = 0.95
    snoop_alpha: float = 0.05
    pbo_blocks: int = 16
    bootstraps: int = 2000
    mean_block: float = 10.0
    seed: int = 1


def deflated(net: np.ndarray, n_trials: float) -> dict:
    """Sharpe stats for one return series, deflated for n_trials unskilled trials."""
    m = _core.moments(net)
    sr = m["mean"] / m["sd"] if m["sd"] > 0 else 0.0
    sr0 = _core.expected_max_sharpe(1 / (m["n"] - 1), n_trials) if m["n"] > 2 else float("nan")
    dsr = _core.probabilistic_sharpe(sr, sr0, m["n"], m["skew"], m["kurt"]) if m["n"] > 2 else float("nan")
    return {"sharpe": sr, "sr0": sr0, "dsr": dsr, "n": m["n"], "skew": m["skew"], "kurt": m["kurt"]}


def run(ledger_sharpe: np.ndarray, nets: np.ndarray, train: slice, val: slice, t: Thresholds) -> dict:
    """nets: finalists x days (pre-holdout panel), NaN where a day has no position or target."""
    trials = ledger_sharpe[np.isfinite(ledger_sharpe)]
    n_trials = len(trials)
    var_trials = float(np.var(trials, ddof=1)) if n_trials > 1 else 0.0
    K = len(nets)

    rows = [{"train": deflated(nets[k, train], n_trials), "validation": deflated(nets[k, val], K)}
            for k in range(K)]
    T_train = train.stop - train.start

    out = {
        "n_trials": n_trials,
        "trial_sharpe_sd": math.sqrt(var_trials),
        "null_sharpe_sd": 1 / math.sqrt(T_train - 1),
        "sr0_train": _core.expected_max_sharpe(1 / (T_train - 1), n_trials),
        "sr0_train_paper": _core.expected_max_sharpe(var_trials, n_trials),
        "n_finalists": K,
        "finalists": rows,
        "snoop": None,
        "pbo": None,
        "survivors": [],
    }
    if K == 0:
        return out

    excess = np.nan_to_num(nets[:, val], nan=0.0)
    out["snoop"] = _core.snoop_tests(excess, t.bootstraps, t.mean_block, t.seed)
    gate = out["snoop"]["spa_p"] < t.snoop_alpha

    span = np.nan_to_num(nets[:, train.start:val.stop], nan=0.0)
    if K >= 2 and span.shape[1] >= 2 * t.pbo_blocks:
        p, logits = _core.pbo(span, t.pbo_blocks)
        out["pbo"] = {"pbo": p, "logits": logits.tolist(), "blocks": t.pbo_blocks}

    for k, row in enumerate(rows):
        tr, va = row["train"], row["validation"]
        reasons = []
        if not tr["dsr"] >= t.dsr:
            reasons.append(f"train DSR {tr['dsr']:.3f} < {t.dsr}")
        if not va["sharpe"] > 0:
            reasons.append("validation Sharpe <= 0")
        elif not va["dsr"] >= t.dsr:
            reasons.append(f"validation DSR {va['dsr']:.3f} < {t.dsr}")
        if not gate:
            reasons.append(f"SPA p {out['snoop']['spa_p']:.3f} >= {t.snoop_alpha} (gate closed)")
        row["passed"] = not reasons
        row["reasons"] = reasons
        if not reasons:
            out["survivors"].append(k)
    return out


def combine(nets: np.ndarray, val: slice) -> np.ndarray:
    """Equal-risk weights from validation volatility; weights sum to 1."""
    sd = np.array([np.nanstd(n[val], ddof=1) for n in nets])
    w = np.where(sd > 0, 1 / sd, 0.0)
    return w / w.sum() if w.sum() > 0 else w
