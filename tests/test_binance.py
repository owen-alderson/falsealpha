"""Binance panel building from a fake cache (no network)."""

import io
import zipfile

import numpy as np

from falsealpha import binance

DAY_MS = 86_400_000


def write_month(cache, symbol, start_day, closes, volumes, micros=False):
    """One monthly zip with consecutive daily candles from start_day (YYYY-MM-DD)."""
    t0 = np.datetime64(start_day, "D").astype(int) * DAY_MS
    lines = []
    for i, (c, v) in enumerate(zip(closes, volumes)):
        ts = t0 + i * DAY_MS
        if micros:
            ts *= 1000
        lines.append(f"{ts},{c},{c * 1.01},{c * 0.99},{c},{v / c},{ts + DAY_MS - 1},{v},100,0,0,0")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{symbol}-1d-{start_day[:7]}.csv", "\n".join(lines))
    (cache / symbol).mkdir(parents=True, exist_ok=True)
    (cache / symbol / f"{symbol}-1d-{start_day[:7]}.zip").write_bytes(buf.getvalue())


def test_tradable_filter():
    assert binance.tradable("BTCUSDT")
    assert binance.tradable("JUPUSDT")  # ends in UP but is a real coin
    assert not binance.tradable("BTCUPUSDT")
    assert not binance.tradable("ETHBEARUSDT")
    assert not binance.tradable("USDCUSDT")
    assert not binance.tradable("EURUSDT")
    assert not binance.tradable("BTCBUSD")


def test_lives_split_on_long_gaps():
    days = np.array(["2022-01-01", "2022-01-02", "2022-01-20", "2022-06-01", "2022-06-02"], dtype="datetime64[D]")
    lives = binance.lives(days, gap_days=30)
    assert [len(x) for x in lives] == [3, 2]


def test_panel_from_cache(tmp_path):
    n = 120
    # BIG trades a lot for the whole period; DEAD is liquid then delisted after 90 days;
    # SMALL is illiquid; LUNA dies and a new coin reuses the ticker 60 days later.
    write_month(tmp_path, "BIGUSDT", "2020-01-01", np.linspace(100, 120, n), [1e9] * n)
    write_month(tmp_path, "DEADUSDT", "2020-01-01", np.linspace(50, 1, 90), [5e8] * 90)
    write_month(tmp_path, "SMALLUSDT", "2020-01-01", [10.0] * n, [1e3] * n)
    write_month(tmp_path, "LUNAUSDT", "2020-01-01", [80.0] * 40, [2e8] * 40)
    write_month(tmp_path, "LUNAUSDT", "2020-04-10", [1.0] * 20, [2e8] * 20, micros=True)  # 2025-style timestamps
    write_month(tmp_path, "USDCUSDT", "2020-01-01", [1.0] * n, [9e9] * n)

    p = binance.build_panel(tmp_path, top=3, start="2020-01-01", min_history=10)
    assert "USDCUSDT" not in p.assets
    assert {"BIGUSDT", "DEADUSDT", "LUNAUSDT"} <= set(p.assets)
    assert "LUNAUSDT~2" in p.assets  # the relaunch is a different asset
    assert p.universe.sum(axis=1).max() <= 3

    dead = p.assets.index("DEADUSDT")
    in_univ = np.nonzero(p.universe[:, dead])[0]
    assert in_univ.max() == 89  # a member up to its last trading day, never after
    assert np.isnan(p.fields["close"][90:, dead]).all()

    big = p.assets.index("BIGUSDT")
    assert not p.universe[:9, big].any()  # needs min_history days first
    np.testing.assert_allclose(p.fields["returns"][1:, big], np.diff(p.fields["close"][:, big]) / p.fields["close"][:-1, big])
    np.testing.assert_allclose(p.fields["vwap"][5, big], p.fields["close"][5, big])  # volume = quote / close here
