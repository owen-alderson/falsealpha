"""Daily Binance spot candles for every USDT pair, delisted ones included.

Source: Binance's public dump at data.binance.vision (monthly zip per pair).
Delisted pairs stay in the dump, which is what makes a survivorship-free
universe possible. Three traps handled here:

- Timestamps switch from milliseconds to microseconds in 2025.
- Tickers get reused: LUNAUSDT runs straight through the May 2022 collapse into
  the relaunched LUNA. A gap of more than `gap_days` without candles splits a
  symbol into separate assets ("LUNAUSDT", "LUNAUSDT~2").
- Stablecoins, fiat and leveraged tokens are not crypto risk; they are dropped.

The universe on day t is the `top` assets by trailing 30-day dollar volume,
using only data up to day t.
"""

from __future__ import annotations

import csv
import io
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import certifi
import numpy as np

from .panel import Panel

LISTING = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FILES = "https://data.binance.vision"
PREFIX = "data/spot/monthly/klines/"
S3 = "{http://s3.amazonaws.com/doc/2006-03-01/}"

STABLE_OR_FIAT = {
    "USDC", "BUSD", "TUSD", "USDP", "PAX", "DAI", "FDUSD", "USDS", "USDSB", "UST", "USTC", "SUSD", "PYUSD",
    "USD1", "RLUSD", "XUSD", "BFUSD", "USDE", "AEUR", "EUR", "EURI", "GBP", "AUD", "BRL", "TRY", "RUB", "NGN",
    "UAH", "ZAR", "IDRT", "BIDR", "BVND", "PLN", "RON", "ARS", "JPY", "MXN", "COP", "CZK", "BKRW",
}
# Underlyings of Binance's (discontinued) leveraged tokens, e.g. BTCUP, ETHDOWN.
LEVERAGED = {"BTC", "ETH", "BNB", "XRP", "ADA", "LINK", "DOT", "TRX", "LTC", "XTZ", "EOS", "BCH", "YFI",
             "UNI", "SXP", "FIL", "AAVE", "SUSHI", "1INCH", "XLM"}

FIELDS = ["open", "high", "low", "close", "volume", "quote_volume", "vwap", "returns", "trades"]


def tradable(symbol: str) -> bool:
    if not symbol.endswith("USDT"):
        return False
    base = symbol[: -len("USDT")]
    if not base or base in STABLE_OR_FIAT:
        return False
    if re.fullmatch(r".*(BULL|BEAR)", base):
        return False
    m = re.fullmatch(r"(.+)(UP|DOWN)", base)
    return not (m and m.group(1) in LEVERAGED)


# ---------------------------------------------------------------- download


# python.org builds on macOS ship without system certificates; use certifi's.
_TLS = ssl.create_default_context(cafile=certifi.where())


def _get(url: str, tries: int = 4) -> bytes:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60, context=_TLS) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404 or attempt == tries - 1:
                raise
        except OSError:
            if attempt == tries - 1:
                raise
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _list(prefix: str) -> tuple[list[str], list[str]]:
    """All (sub-prefixes, keys) under an S3 prefix, following pagination."""
    prefixes, keys, marker = [], [], ""
    while True:
        q = {"prefix": prefix, "delimiter": "/", **({"marker": marker} if marker else {})}
        url = f"{LISTING}?{urllib.parse.urlencode(q)}"  # some pair names are not ASCII
        root = ET.fromstring(_get(url))
        prefixes += [e.text for e in root.iter(f"{S3}Prefix") if e.text and e.text != prefix]
        keys += [e.text for e in root.iter(f"{S3}Key") if e.text]
        if root.findtext(f"{S3}IsTruncated") != "true":
            return prefixes, keys
        marker = root.findtext(f"{S3}NextMarker") or (keys or prefixes)[-1]


def symbols() -> list[str]:
    prefixes, _ = _list(PREFIX)
    names = [p[len(PREFIX):].strip("/") for p in prefixes]
    return sorted(s for s in names if tradable(s))


def download(cache: Path, start: str = "2017-01", workers: int = 16, log=print) -> int:
    """Fetch every monthly daily-candle zip not already cached. Returns files fetched."""
    cache = Path(cache)
    syms = symbols()
    log(f"{len(syms)} tradable USDT pairs listed (delisted included)")

    def months(sym: str) -> list[str]:
        _, keys = _list(f"{PREFIX}{sym}/1d/")
        return [k for k in keys if k.endswith(".zip") and k[-11:-4] >= start]

    with ThreadPoolExecutor(workers) as pool:
        keys = [k for ks in pool.map(months, syms) for k in ks]
    todo = [k for k in keys if not (cache / Path(k).parent.parent.name / Path(k).name).exists()]
    log(f"{len(keys)} monthly files, {len(todo)} to fetch")

    def fetch(key: str) -> None:
        data = _get(f"{FILES}/{urllib.parse.quote(key)}")
        zipfile.ZipFile(io.BytesIO(data)).testzip()  # CRC check before caching
        dest = cache / Path(key).parent.parent.name / Path(key).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(dest)

    with ThreadPoolExecutor(workers) as pool:
        for i, _ in enumerate(pool.map(fetch, todo), 1):
            if i % 2000 == 0:
                log(f"  {i}/{len(todo)}")
    return len(todo)


# ---------------------------------------------------------------- panel


def read_symbol(folder: Path) -> dict[np.datetime64, list[float]]:
    """day -> [open, high, low, close, volume, quote_volume, trades] from cached zips."""
    rows: dict[np.datetime64, list[float]] = {}
    for z in sorted(folder.glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                text = zf.read(name).decode()
                for r in csv.reader(io.StringIO(text)):
                    if not r or not r[0].isdigit():
                        continue  # header line in some files
                    ts = int(r[0])
                    ms = ts // 1000 if ts > 10**14 else ts  # microseconds from 2025
                    day = np.datetime64(ms // 86_400_000, "D")
                    rows[day] = [float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]),
                                 float(r[7]), float(r[8])]
    return rows


def lives(days: np.ndarray, gap_days: int) -> list[np.ndarray]:
    """Split sorted candle days into separate assets at gaps longer than gap_days."""
    if len(days) == 0:
        return []
    cuts = np.nonzero(np.diff(days).astype(int) > gap_days)[0] + 1
    return np.split(days, cuts)


def build_panel(cache: Path, top: int = 100, start: str = "2018-01-01", end: str | None = None,
                min_history: int = 60, gap_days: int = 30, liquidity_days: int = 30) -> Panel:
    cache = Path(cache)
    series: dict[str, dict[np.datetime64, list[float]]] = {}
    for folder in sorted(p for p in cache.iterdir() if p.is_dir() and tradable(p.name)):
        rows = read_symbol(folder)
        days = np.array(sorted(rows), dtype="datetime64[D]")
        for i, life in enumerate(lives(days, gap_days)):
            series[folder.name if i == 0 else f"{folder.name}~{i + 1}"] = {d: rows[d] for d in life}
    if not series:
        raise ValueError(f"no cached candles under {cache}; run the download first")

    first = np.datetime64(start, "D")
    last = max(max(s) for s in series.values()) if end is None else np.datetime64(end, "D")
    dates = np.arange(first, last + 1, dtype="datetime64[D]")
    T = len(dates)
    names = sorted(series)
    raw = np.full((7, T, len(names)), np.nan)
    for n, name in enumerate(names):
        for day, vals in series[name].items():
            t = int((day - first).astype(int))
            if 0 <= t < T:
                raw[:, t, n] = vals
    o, h, lo, c, vol, qv, trades = raw
    bad = ~(c > 0)
    for arr in (o, h, lo, c, vol, qv, trades):
        arr[bad] = np.nan

    # Point-in-time universe: enough history, traded today, top by trailing dollar volume.
    have = np.isfinite(c)
    history = np.cumsum(have, axis=0)
    qv0 = np.nan_to_num(qv)
    csum = np.cumsum(qv0, axis=0)
    trailing = (csum - np.vstack([np.zeros((liquidity_days, len(names))), csum[:-liquidity_days]])) / liquidity_days
    eligible = have & (history >= min_history) & (qv0 > 0)
    score = np.where(eligible, trailing, -np.inf)
    universe = np.zeros_like(eligible)
    k = min(top, len(names))
    idx = np.argpartition(-score, k - 1, axis=1)[:, :k]
    np.put_along_axis(universe, idx, True, axis=1)
    universe &= eligible

    keep = universe.any(axis=0)
    names = [nm for nm, k_ in zip(names, keep) if k_]
    sel = lambda a: np.ascontiguousarray(a[:, keep])  # noqa: E731
    o, h, lo, c, vol, qv, trades, universe = (sel(a) for a in (o, h, lo, c, vol, qv, trades, universe))
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(vol > 0, qv / vol, np.nan)
        returns = np.full_like(c, np.nan)
        returns[1:] = c[1:] / c[:-1] - 1
    fields = dict(zip(FIELDS, [o, h, lo, c, vol, qv, vwap, returns, trades]))
    source = f"binance spot USDT daily, top {top} by {liquidity_days}d dollar volume, {start}..{last}"
    return Panel(dates, names, fields, universe, source, periods_per_year=365.0)
