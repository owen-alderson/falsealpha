import numpy as np

from falsealpha import bench, evaluate


def test_bench_runs_and_pandas_agrees_where_definitions_match():
    out = bench.run(days=120, assets=20, repeat=1, log=lambda *_: None)
    assert out["falsealpha_1_thread"] > 0 and "pandas" in out

    # Same text, same numbers, for operators whose definitions coincide.
    p = bench.random_panel(120, 20)
    ns = bench.pandas_namespace(p)
    for f in ["ts_corr(open, volume, 10)", "ts_std(returns, 5)", "ts_delta(close, 3)", "decay_linear(vwap, 7)",
              "(close - open) / ((high - low) + 0.001)"]:
        want = eval(f, {"__builtins__": {}}, ns).to_numpy()  # noqa: S307
        got = evaluate(f, p.core())
        ok = np.isfinite(want) & np.isfinite(got)
        assert ok.sum() > 1000
        np.testing.assert_allclose(got[ok], want[ok], rtol=1e-8, atol=1e-10)
