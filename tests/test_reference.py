"""Every operator in C++ matches the slow numpy reference to 1e-9."""

import numpy as np
import pytest
import reference

import falsealpha as fa

OPS = {o["name"]: o for o in fa.operators()}
WINDOWS = [2, 3, 7, 20]


def masked(x, universe):
    return np.where(universe, x, np.nan)


def check(got, want):
    assert np.array_equal(np.isnan(got), np.isnan(want)), "NaN pattern differs"
    ok = ~np.isnan(want)
    np.testing.assert_allclose(got[ok], want[ok], rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("name", [n for n, o in OPS.items() if o["kind"] == "time-series"])
@pytest.mark.parametrize("d", WINDOWS)
def test_time_series(random_panel, name, d):
    op = OPS[name]
    if d < op["min_window"]:
        pytest.skip("below minimum window")
    p = random_panel
    args = "close, volume" if op["arity"] == 2 else "close"
    got = fa.evaluate(f"{name}({args}, {d})", p.core())
    y = p.fields["volume"] if op["arity"] == 2 else None
    check(got, masked(reference.series(name, p.fields["close"], d, y), p.universe))


@pytest.mark.parametrize("name", [n for n, o in OPS.items() if o["kind"] == "cross-sectional"])
@pytest.mark.parametrize("field", ["close", "returns"])
def test_cross_sectional(random_panel, name, field):
    p = random_panel
    got = fa.evaluate(f"{name}({field})", p.core())
    check(got, masked(reference.cross(name, p.fields[field], p.universe), p.universe))


@pytest.mark.parametrize("name", [n for n, o in OPS.items() if o["kind"] == "elementwise"])
def test_elementwise(random_panel, name):
    p = random_panel
    binary = OPS[name]["arity"] == 2
    got = fa.evaluate(f"{name}(returns, volume)" if binary else f"{name}(returns)", p.core())
    want = reference.elem(name, p.fields["returns"], p.fields["volume"] if binary else None)
    check(got, masked(want, p.universe))


def test_composite(random_panel):
    p = random_panel
    f = "rank(ts_corr(rank(close), decay_linear(volume, 6), 9)) - ts_rank(returns, 5)"
    c, v, r = p.fields["close"], p.fields["volume"], p.fields["returns"]
    inner = reference.series("ts_corr", reference.cross("rank", c, p.universe), 9,
                             reference.series("decay_linear", v, 6))
    want = reference.elem("sub", reference.cross("rank", inner, p.universe), reference.series("ts_rank", r, 5))
    check(fa.evaluate(f, p.core()), masked(want, p.universe))


@pytest.mark.parametrize("d", [2, 5, 30])
def test_long_random_walk(d):
    # Longer than the 512-day refresh; a trending price at a high level, where
    # running moments would lose precision without the exact fallback.
    rng = np.random.default_rng(d)
    T, N = 3000, 6
    close = 5e6 + np.cumsum(rng.normal(0, 100, (T, N)), axis=0)
    volume = rng.lognormal(12, 2, (T, N))
    p = fa.Panel(np.datetime64("2010-01-01") + np.arange(T), list("abcdef"),
                 {"close": close, "volume": volume, "returns": np.zeros((T, N))}, np.ones((T, N), bool))
    for name in ["ts_mean", "ts_sum", "ts_std", "ts_zscore", "decay_linear", "ts_corr", "ts_cov"]:
        if d < OPS[name]["min_window"]:
            continue
        args = "close, volume" if OPS[name]["arity"] == 2 else "close"
        y = volume if OPS[name]["arity"] == 2 else None
        check(fa.evaluate(f"{name}({args}, {d})", p.core()), reference.series(name, close, d, y))
