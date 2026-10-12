"""A slow, obviously-correct numpy version of every operator.

The C++ kernels use running sums, pivots and block tricks; this file uses
plain loops over each window, so the two share no code. Same NaN rules.
"""

import numpy as np

ZERO_VAR = 1e-20


def _windows(x, d):
    T, N = x.shape
    for t in range(d - 1, T):
        yield t, x[t - d + 1 : t + 1]


def _var(w):
    var = np.var(w, ddof=1, axis=0)
    ms = np.mean(w * w, axis=0)
    return np.where(var <= ZERO_VAR * ms, 0.0, var)


def series(op, x, d, y=None):
    out = np.full(x.shape, np.nan)
    if op in ("ts_delay", "ts_delta"):
        for t in range(d, x.shape[0]):
            ok = np.isfinite(x[t - d : t + 1]).all(axis=0)
            now, then = x[t], x[t - d]
            out[t] = np.where(ok, then if op == "ts_delay" else now - then, np.nan)
        return out
    for t, w in _windows(x, d):
        wy = y[t - d + 1 : t + 1] if y is not None else None
        ok = np.isfinite(w).all(axis=0)
        if wy is not None:
            ok &= np.isfinite(wy).all(axis=0)
        with np.errstate(all="ignore"):
            if op == "ts_sum":
                v = w.sum(axis=0)
            elif op == "ts_mean":
                v = w.mean(axis=0)
            elif op == "ts_std":
                v = np.sqrt(_var(w))
            elif op == "ts_zscore":
                var = _var(w)
                v = np.where(var > 0, (w[-1] - w.mean(axis=0)) / np.sqrt(var), np.nan)
            elif op == "ts_min":
                v = w.min(axis=0)
            elif op == "ts_max":
                v = w.max(axis=0)
            elif op in ("ts_argmin", "ts_argmax"):
                rev = w[::-1]  # row 0 = today; argmin/argmax return the first hit = most recent
                v = (rev.argmin(axis=0) if op == "ts_argmin" else rev.argmax(axis=0)).astype(float)
            elif op == "ts_rank":
                cur = w[-1]
                v = ((w < cur).sum(axis=0) + 0.5 * ((w == cur).sum(axis=0) - 1)) / (d - 1)
            elif op == "decay_linear":
                weights = np.arange(1, d + 1, dtype=float)
                v = (w * weights[:, None]).sum(axis=0) / weights.sum()
            elif op == "ts_cov":
                v = ((w - w.mean(0)) * (wy - wy.mean(0))).sum(0) / (d - 1)
            elif op == "ts_corr":
                vx, vy = _var(w), _var(wy)
                cov = ((w - w.mean(0)) * (wy - wy.mean(0))).sum(0) / (d - 1)
                v = np.where((vx > 0) & (vy > 0), np.clip(cov / np.sqrt(vx * vy), -1, 1), np.nan)
            else:
                raise ValueError(op)
        out[t] = np.where(ok, v, np.nan)
    return out


def cross(op, x, universe):
    out = np.full(x.shape, np.nan)
    for t in range(x.shape[0]):
        m = universe[t] & np.isfinite(x[t])
        v = x[t][m]
        if len(v) == 0:
            continue
        if op == "rank":
            order = np.argsort(v, kind="stable")
            r = np.empty(len(v))
            sv = v[order]
            i = 0
            while i < len(v):
                j = i
                while j + 1 < len(v) and sv[j + 1] == sv[i]:
                    j += 1
                r[order[i : j + 1]] = 0.5 if len(v) == 1 else 0.5 * (i + j) / (len(v) - 1)
                i = j + 1
            out[t, m] = r
        elif op == "demean":
            out[t, m] = v - v.mean()
        elif op == "zscore":
            if len(v) < 2:
                continue
            var = np.var(v, ddof=1)
            if var <= ZERO_VAR * np.mean(v * v):
                continue
            out[t, m] = (v - v.mean()) / np.sqrt(var)
        elif op == "scale":
            g = np.abs(v).sum()
            if g > 1e-12:
                out[t, m] = v / g
    return out


def elem(op, x, y=None):
    with np.errstate(all="ignore"):
        out = {
            "neg": lambda: -x,
            "abs": lambda: np.abs(x),
            "log": lambda: np.where(x > 0, np.log(np.where(x > 0, x, 1)), np.nan),
            "sign": lambda: np.sign(x),
            "add": lambda: x + y,
            "sub": lambda: x - y,
            "mul": lambda: x * y,
            "div": lambda: np.where(np.abs(y) > 1e-12, x / y, np.nan),
            "max": lambda: np.maximum(x, y),
            "min": lambda: np.minimum(x, y),
        }[op]()
    return np.where(np.isfinite(out), out, np.nan)
