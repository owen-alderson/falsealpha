"""falsealpha command line."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from . import _core, run, synthetic
from .panel import Panel

DEFAULT_CACHE = Path.home() / ".cache" / "falsealpha" / "binance"


def _ann(sr: float, ppy: float) -> str:
    return f"{sr * math.sqrt(ppy):+.2f}"


def cmd_data(a: argparse.Namespace) -> None:
    if a.source == "binance":
        from . import binance

        if not a.no_download:
            binance.download(Path(a.cache), workers=a.workers)
        panel = binance.build_panel(Path(a.cache), top=a.top, start=a.start, end=a.end)
    elif a.source == "planted":
        panel = synthetic.planted(a.formula, a.beta, a.days, a.assets, a.seed)
    else:
        panel = synthetic.noise(a.days, a.assets, a.seed)
    panel.save(a.out)
    T, N = panel.shape
    print(f"wrote {a.out}: {T} days x {N} assets, {panel.dates[0]}..{panel.dates[-1]}")
    print(f"  {panel.source}")


def cmd_info(a: argparse.Namespace) -> None:
    p = Panel.load(a.panel)
    T, N = p.shape
    members = p.universe.sum(axis=1)
    val, hold = run.Splits(a.train, a.validation).bounds(T)
    print(f"{a.panel}: {T} days x {N} assets   fingerprint {p.fingerprint()}")
    print(f"  source      {p.source}")
    print(f"  fields      {', '.join(sorted(p.fields))}")
    print(f"  members/day min {members.min()}  median {int(np.median(members))}  max {members.max()}")
    print(f"  train       {p.dates[0]} .. {p.dates[val - 1]}  ({val} days)")
    print(f"  validation  {p.dates[val]} .. {p.dates[hold - 1]}  ({hold - val} days)")
    print(f"  holdout     {p.dates[hold]} .. {p.dates[-1]}  ({T - hold} days, sealed)")


def cmd_eval(a: argparse.Namespace) -> None:
    p = Panel.load(a.panel)
    T = p.shape[0]
    val, hold = run.Splits(a.train, a.validation).bounds(T)
    pre = p.head(hold)  # never the holdout
    train, valid = run._segments(hold, val, a.delay)
    print(_core.canonical(a.formula))
    bt = _core.backtest(_core.evaluate(a.formula, pre.core()), "returns", pre.core(), delay=a.delay,
                        cost_bps=a.cost_bps)
    for name, seg in (("train", train), ("validation", valid)):
        s = _core.summarize(bt, seg.start, seg.stop)
        print(f"  {name:<11} Sharpe {_ann(s['sharpe'], p.periods_per_year)} /yr   IC {s['mean_ic']:+.4f}   "
              f"IC IR {s['ic_ir']:+.3f}   turnover {s['turnover']:.2f}/day   days {s['n']}")
    print("  (holdout not touched: it opens once, via `falsealpha unseal`)")


def cmd_mine(a: argparse.Namespace) -> None:
    cfg = run.MineConfig(population=a.population, generations=a.generations, seed=a.seed, threads=a.threads,
                         cost_bps=a.cost_bps, delay=a.delay, max_size=a.max_size,
                         fields=a.fields.split(",") if a.fields else None)
    panel = Panel.load(a.panel)
    ppy = panel.periods_per_year

    def progress(gen: int, trials: int, best: float, formula: str) -> None:
        print(f"  gen {gen + 1:>3}/{a.generations}  trials {trials:>7}  best train Sharpe {_ann(best, ppy)}  {formula}",
              flush=True)

    print(f"mining {a.panel} -> {a.run}")
    v = run.mine(a.panel, a.run, cfg, run.Splits(a.train, a.validation), progress=progress)
    _print_verdict(v, ppy)
    from . import report

    print(f"report: {report.write(a.run)}")


def _print_verdict(v: dict, ppy: float) -> None:
    print(f"\n{v['n_trials']} formulas tried -> {v['n_finalists']} finalists -> {len(v['survivors'])} survivors")
    print(f"  luck bar (train): best of {v['n_trials']} unskilled strategies ~ Sharpe {_ann(v['sr0_train'], ppy)} /yr")
    if v["snoop"]:
        print(f"  SPA p {v['snoop']['spa_p']:.3f}   Reality Check p {v['snoop']['reality_check_p']:.3f}")
    if v["pbo"]:
        print(f"  PBO {v['pbo']['pbo']:.2f}  (share of splits where the in-sample winner lands in the bottom half)")
    for f in v["finalists"][:10]:
        mark = "PASS" if f["passed"] else "fail"
        print(f"  [{mark}] train {_ann(f['train']['sharpe'], ppy)} (DSR {f['train']['dsr']:.2f})  "
              f"val {_ann(f['validation']['sharpe'], ppy)} (DSR {f['validation']['dsr']:.2f})  {f['formula']}")


def cmd_unseal(a: argparse.Namespace) -> None:
    look = run.unseal(a.run, force=a.force)
    ppy = json.loads((Path(a.run) / "config.json").read_text())["periods_per_year"]
    if not look["survivors"]:
        print("no survivors: nothing to test on the holdout (the look is still logged)")
    for s in look["strategies"]:
        print(f"  holdout Sharpe {_ann(s['sharpe'], ppy)} /yr  PSR {s['psr_vs_zero']:.2f}  {s['formula']}")
    if look["combined"]:
        c = look["combined"]
        print(f"combined: holdout Sharpe {_ann(c['sharpe'], ppy)} /yr, P(true Sharpe > 0) {c['psr_vs_zero']:.2f}, "
              f"{c['days']} days")
    from . import report

    print(f"report: {report.write(a.run)}")


def cmd_report(a: argparse.Namespace) -> None:
    from . import report

    print(report.write(a.run))


def cmd_ops(a: argparse.Namespace) -> None:
    for o in _core.operators():
        args = ", ".join(["x", "y"][: o["arity"]] + (["d"] if o["window"] else []))
        print(f"  {o['name']}({args})".ljust(28) + o["kind"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="falsealpha", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def splits(p: argparse.ArgumentParser) -> None:
        p.add_argument("--train", type=float, default=0.6, help="fraction of days for training (default 0.6)")
        p.add_argument("--validation", type=float, default=0.2, help="fraction for validation (default 0.2)")

    d = sub.add_parser("data", help="build a panel file")
    d.add_argument("source", choices=["binance", "planted", "noise"])
    d.add_argument("--out", required=True)
    d.add_argument("--top", type=int, default=100, help="binance: universe size (default 100)")
    d.add_argument("--start", default="2018-01-01")
    d.add_argument("--end")
    d.add_argument("--cache", default=str(DEFAULT_CACHE))
    d.add_argument("--workers", type=int, default=16)
    d.add_argument("--no-download", action="store_true", help="binance: use the cache as is")
    d.add_argument("--formula", default="rank(ts_mean(volume, 5))", help="planted: the hidden alpha")
    d.add_argument("--beta", type=float, default=0.004, help="planted: its strength")
    d.add_argument("--days", type=int, default=1500)
    d.add_argument("--assets", type=int, default=60)
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(fn=cmd_data)

    i = sub.add_parser("info", help="describe a panel and its splits")
    i.add_argument("panel")
    splits(i)
    i.set_defaults(fn=cmd_info)

    e = sub.add_parser("eval", help="backtest one formula on train and validation")
    e.add_argument("panel")
    e.add_argument("formula")
    e.add_argument("--delay", type=int, default=1)
    e.add_argument("--cost-bps", type=float, default=10.0)
    splits(e)
    e.set_defaults(fn=cmd_eval)

    m = sub.add_parser("mine", help="search for alphas, then run the overfitting firewall")
    m.add_argument("panel")
    m.add_argument("run", help="new run directory, e.g. runs/first")
    m.add_argument("--population", type=int, default=400)
    m.add_argument("--generations", type=int, default=25)
    m.add_argument("--max-size", type=int, default=20)
    m.add_argument("--fields", help="comma-separated inputs (default: all panel fields)")
    m.add_argument("--delay", type=int, default=1)
    m.add_argument("--cost-bps", type=float, default=10.0)
    m.add_argument("--seed", type=int, default=1)
    m.add_argument("--threads", type=int, default=0)
    splits(m)
    m.set_defaults(fn=cmd_mine)

    u = sub.add_parser("unseal", help="score the survivors on the holdout (once)")
    u.add_argument("run")
    u.add_argument("--force", action="store_true", help="look again; every look is logged in the report")
    u.set_defaults(fn=cmd_unseal)

    r = sub.add_parser("report", help="regenerate a run's HTML report")
    r.add_argument("run")
    r.set_defaults(fn=cmd_report)

    o = sub.add_parser("ops", help="list the formula operators")
    o.set_defaults(fn=cmd_ops)

    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except (ValueError, FileExistsError, FileNotFoundError, PermissionError) as err:
        print(f"falsealpha: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
