"""A mining run on disk: search, firewall, and a holdout that opens once.

runs/<name>/
  config.json    panel path + fingerprint, splits, search settings
  ledger.csv.gz  every distinct formula the search scored (train stats)
  finalists.json the hall of fame with firewall verdicts
  nets.npy       finalists x pre-holdout days, daily net returns
  holdout.json   written by `unseal`; every look at the holdout is logged
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import _core, firewall
from .panel import Panel


@dataclass
class Splits:
    train: float = 0.6
    validation: float = 0.2  # holdout gets the rest

    def bounds(self, T: int) -> tuple[int, int]:
        """(validation start, holdout start) as day indices."""
        if not (0 < self.train and 0 < self.validation and self.train + self.validation < 1):
            raise ValueError("train and validation fractions must be positive and leave room for a holdout")
        return int(T * self.train), int(T * (self.train + self.validation))


@dataclass
class MineConfig:
    population: int = 400
    generations: int = 25
    max_size: int = 20
    max_depth: int = 6
    windows: list[int] = field(default_factory=lambda: [3, 5, 10, 20, 40])
    parsimony: float = 0.0005
    max_corr: float = 0.7
    hall_size: int = 50
    delay: int = 1
    cost_bps: float = 10.0
    min_assets: int = 5
    seed: int = 1
    threads: int = 0
    fields: list[str] | None = None  # default: every panel field


def _segments(T_pre: int, val_start: int, delay: int) -> tuple[slice, slice]:
    # A position on day t earns returns on day t + delay + 1, so the last
    # delay + 1 train days would peek into validation: drop them.
    lag = delay + 1
    return slice(0, val_start - lag), slice(val_start, T_pre - lag)


def mine(panel_path: str | Path, run_dir: str | Path, cfg: MineConfig, splits: Splits = Splits(),
         thresholds: firewall.Thresholds = firewall.Thresholds(), progress=None) -> dict:
    run_dir = Path(run_dir)
    if (run_dir / "config.json").exists():
        raise FileExistsError(f"{run_dir} already holds a run; pick a new name (re-mining is a new trial)")
    full = Panel.load(panel_path)
    T = full.shape[0]
    val_start, hold_start = splits.bounds(T)
    pre = full.head(hold_start)  # the search never receives the holdout
    train, val = _segments(hold_start, val_start, cfg.delay)
    fields = cfg.fields or sorted(pre.fields)

    started = time.time()
    result = _core.search(
        pre.core(), "returns", fields, train.start, train.stop,
        population=cfg.population, generations=cfg.generations, max_size=cfg.max_size,
        max_depth=cfg.max_depth, windows=cfg.windows, parsimony=cfg.parsimony, max_corr=cfg.max_corr,
        hall_size=cfg.hall_size, delay=cfg.delay, cost_bps=cfg.cost_bps, min_assets=cfg.min_assets,
        seed=cfg.seed, threads=cfg.threads, progress=progress,
    )
    elapsed = time.time() - started

    hall = result["hall"]
    nets = np.array([h["net"] for h in hall]) if hall else np.zeros((0, hold_start))
    verdict = firewall.run(result["ledger"]["sharpe"], nets, train, val, thresholds)

    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "panel": str(Path(panel_path).resolve()),
        "panel_fingerprint": full.fingerprint(),
        "source": full.source,
        "periods_per_year": full.periods_per_year,
        "dates": {
            "train": [str(full.dates[0]), str(full.dates[val_start - 1])],
            "validation": [str(full.dates[val_start]), str(full.dates[hold_start - 1])],
            "holdout": [str(full.dates[hold_start]), str(full.dates[-1])],
        },
        "index": {"val_start": val_start, "hold_start": hold_start, "T": T,
                  "train": [train.start, train.stop], "validation": [val.start, val.stop]},
        "assets": full.shape[1],
        "fields": fields,
        "search": asdict(cfg),
        "splits": asdict(splits),
        "thresholds": asdict(thresholds),
        "elapsed_seconds": elapsed,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    ledger = result["ledger"]
    with gzip.open(run_dir / "ledger.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["formula", "train_sharpe", "train_mean_ic", "train_turnover", "size", "generation"])
        for row in zip(ledger["formula"], ledger["sharpe"], ledger["mean_ic"], ledger["turnover"],
                       ledger["size"], ledger["generation"]):
            w.writerow(row)

    finalists = []
    for h, v in zip(hall, verdict["finalists"]):
        finalists.append({"formula": h["formula"], "fitness": h["fitness"], **v})
    verdict = {**verdict, "finalists": finalists}
    verdict["weights"] = firewall.combine(nets[verdict["survivors"]], val).tolist() if verdict["survivors"] else []
    (run_dir / "finalists.json").write_text(json.dumps(_strict(verdict), indent=2, allow_nan=False))
    np.save(run_dir / "nets.npy", nets)
    return verdict


def unseal(run_dir: str | Path, force: bool = False) -> dict:
    """Score the survivors (and their combination) on the holdout, once."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    verdict = json.loads((run_dir / "finalists.json").read_text())
    log_path = run_dir / "holdout.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"looks": []}
    if log["looks"] and not force:
        raise PermissionError("the holdout has already been opened; another look is another trial "
                              "(use --force to look again; the report will show every look)")

    panel = Panel.load(config["panel"])
    if panel.fingerprint() != config["panel_fingerprint"]:
        raise ValueError("the panel file changed since this run was mined")
    s = config["search"]
    hold_start, T = config["index"]["hold_start"], config["index"]["T"]
    hold = slice(hold_start, T - (s["delay"] + 1))
    ppy = config["periods_per_year"]

    survivors = [verdict["finalists"][k]["formula"] for k in verdict["survivors"]]
    look = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "survivors": survivors, "strategies": [], "combined": None}
    if survivors:
        bts = _core.backtest_many(survivors, "returns", panel.core(), delay=s["delay"], cost_bps=s["cost_bps"],
                                  min_assets=s["min_assets"], threads=s["threads"])
        nets = np.array([b["net"][hold] for b in bts])
        for f, n in zip(survivors, nets):
            look["strategies"].append({"formula": f, **_holdout_stats(n, ppy)})
        combo = np.nansum(nets * np.array(verdict["weights"])[:, None], axis=0)
        look["combined"] = _holdout_stats(combo, ppy)
        look["combined_net"] = combo.tolist()
    log["looks"].append(look)
    log_path.write_text(json.dumps(_strict(log), indent=2, allow_nan=False))
    return look


def _holdout_stats(net: np.ndarray, ppy: float) -> dict:
    m = _core.moments(net)
    sr = m["mean"] / m["sd"] if m["sd"] > 0 else 0.0
    psr = _core.probabilistic_sharpe(sr, 0.0, m["n"], m["skew"], m["kurt"]) if m["n"] > 2 else float("nan")
    curve = np.nancumsum(net)
    return {"sharpe": sr, "sharpe_annual": sr * math.sqrt(ppy), "psr_vs_zero": psr,
            "total_return": float(curve[-1]) if len(curve) else 0.0, "days": m["n"]}


def _strict(x):
    """NaN/inf -> null, so the run files are strict JSON."""
    if isinstance(x, dict):
        return {k: _strict(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_strict(v) for v in x]
    if isinstance(x, (np.ndarray, np.generic)):
        return _strict(x.tolist())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x

