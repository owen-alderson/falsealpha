"""End to end: the firewall keeps a planted alpha and rejects pure noise."""

import json

import numpy as np
import pytest

from falsealpha import run, synthetic

SMALL = run.MineConfig(population=200, generations=6, cost_bps=0, threads=2)


@pytest.fixture(scope="module")
def planted_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("planted")
    synthetic.planted(T=1200, N=50, seed=3).save(d / "p.npz")
    verdict = run.mine(d / "p.npz", d / "run", SMALL)
    return d, verdict


def test_planted_alpha_survives(planted_run):
    _, v = planted_run
    assert v["survivors"], "the planted alpha should survive"
    best = v["finalists"][v["survivors"][0]]
    assert "volume" in best["formula"]
    assert v["snoop"]["reality_check_p"] < 0.01
    assert v["pbo"]["pbo"] < 0.1


@pytest.mark.parametrize("seed", [11, 12])
def test_noise_has_no_survivors(tmp_path, seed):
    synthetic.noise(T=1200, N=50, seed=seed).save(tmp_path / "n.npz")
    v = run.mine(tmp_path / "n.npz", tmp_path / "run", SMALL)
    assert v["n_trials"] > 500
    assert v["survivors"] == []
    # The search still found formulas that look good in sample: that is the point.
    assert max(f["train"]["sharpe"] for f in v["finalists"]) > 0


def test_holdout_is_out_of_reach_and_opens_once(planted_run):
    d, v = planted_run
    cfg = json.loads((d / "run" / "config.json").read_text())
    nets = np.load(d / "run" / "nets.npy")
    assert nets.shape[1] == cfg["index"]["hold_start"]  # the search never saw a holdout day
    look = run.unseal(d / "run")
    assert look["combined"]["sharpe"] > 0
    with pytest.raises(PermissionError):
        run.unseal(d / "run")
    run.unseal(d / "run", force=True)
    log = json.loads((d / "run" / "holdout.json").read_text())
    assert len(log["looks"]) == 2


def test_rerunning_into_a_run_dir_is_refused(planted_run):
    d, _ = planted_run
    with pytest.raises(FileExistsError):
        run.mine(d / "p.npz", d / "run", SMALL)


def test_changed_panel_is_detected(tmp_path):
    p = synthetic.planted(T=900, N=30, seed=4)
    p.save(tmp_path / "p.npz")
    run.mine(tmp_path / "p.npz", tmp_path / "run", run.MineConfig(population=60, generations=2, threads=2))
    p.fields["volume"][0, 0] += 1
    p.save(tmp_path / "p.npz")
    with pytest.raises(ValueError, match="changed"):
        run.unseal(tmp_path / "run")


def test_report_renders(planted_run):
    from falsealpha import report

    d, _ = planted_run
    html = report.build(d / "run")
    assert "<svg" not in html  # charts are drawn client-side from embedded data
    assert "const D = {" in html and "NaN" not in html.split("const D = ")[1].split(";\n")[0]


def test_too_short_segments_are_refused(tmp_path):
    synthetic.noise(T=1000, N=20, seed=1).save(tmp_path / "n.npz")
    with pytest.raises(ValueError, match="validation segment"):
        run.mine(tmp_path / "n.npz", tmp_path / "run", SMALL, run.Splits(0.6, 0.001))


def test_report_survives_missing_holdout_stats(planted_run):
    from falsealpha import report

    d, _ = planted_run
    path = d / "run" / "holdout.json"
    log = json.loads(path.read_text()) if path.exists() else {"looks": []}
    log["looks"].append({"time": "t", "survivors": ["x"], "strategies": [],
                         "combined": {"sharpe": None, "sharpe_annual": None, "psr_vs_zero": None,
                                      "total_return": 0.0, "days": 2}})
    path.write_text(json.dumps(log))
    assert "–" in report.build(d / "run")


def test_report_uses_the_panels_own_dates(tmp_path):
    from falsealpha import report
    from falsealpha.panel import Panel

    p = synthetic.planted(T=900, N=20, seed=5)
    bdays = np.busday_offset("2015-01-01", np.arange(900), roll="forward")  # business days
    Panel(bdays, p.assets, p.fields, p.universe, p.source, 252.0).save(tmp_path / "b.npz")
    run.mine(tmp_path / "b.npz", tmp_path / "run", run.MineConfig(population=40, generations=2, threads=2))
    html = report.build(tmp_path / "run")
    assert str(bdays[-1]) in html and str(np.datetime64("2015-01-01") + 899) not in html
