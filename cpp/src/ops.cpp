#include "falsealpha/ops.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace fa {

namespace {

constexpr double kZeroVar = 1e-20;  // catches float noise on constant series only
constexpr double kTinyDen = 1e-12;

bool fin(double x) { return std::isfinite(x); }

// Rolling sums are updated in O(1) per step; to stop floating-point drift they
// are rebuilt from scratch every kRefresh steps.
constexpr std::size_t kRefresh = 512;

// Running sums are kept relative to a per-asset pivot (an observed value) so
// that a large price level does not cancel away the variance.
// Zero-variance rule on raw values: var <= kZeroVar * mean(x^2).
double variance(double s, double ss, double pivot, int d) {
    double var = (ss - s * s / d) / (d - 1);
    double raw_ms = (ss + 2 * pivot * s) / d + pivot * pivot;
    return var <= kZeroVar * raw_ms ? 0.0 : var;
}

// Shared machinery for windowed sum-type ops: tracks, per asset, the sums of
// x, x^2 (and y, y^2, xy when b is given) over the trailing window, shifted
// by pivots px/py, plus how many non-finite entries the window holds.
struct Rolling {
    const Mat& a;
    const Mat* b;
    int d;
    std::size_t N;
    std::vector<double> px, py, sx, sxx, sy, syy, sxy;
    std::vector<int> bad;

    Rolling(const Mat& a_, const Mat* b_, int d_)
        : a(a_), b(b_), d(d_), N(a_.N), px(N, NaN), py(N, NaN), sx(N), sxx(N), sy(N), syy(N), sxy(N), bad(N) {}

    void add(std::size_t t, double sign) {
        const double* x = a.row(t);
        const double* y = b ? b->row(t) : nullptr;
        for (std::size_t n = 0; n < N; ++n) {
            double xv = x[n], yv = y ? y[n] : 0.0;
            if (!fin(xv) || (y && !fin(yv))) { bad[n] += sign > 0 ? 1 : -1; continue; }
            // A pivot is only ever set while this asset's sums are still empty.
            if (std::isnan(px[n])) { px[n] = xv; py[n] = yv; }
            xv -= px[n]; yv -= py[n];
            sx[n] += sign * xv;
            sxx[n] += sign * xv * xv;
            if (y) { sy[n] += sign * yv; syy[n] += sign * yv * yv; sxy[n] += sign * xv * yv; }
        }
    }
    // Advance so the window ends at t.
    void step(std::size_t t) {
        if (t % kRefresh == 0 && t >= static_cast<std::size_t>(d)) {
            std::fill(sx.begin(), sx.end(), 0.0); std::fill(sxx.begin(), sxx.end(), 0.0);
            std::fill(sy.begin(), sy.end(), 0.0); std::fill(syy.begin(), syy.end(), 0.0);
            std::fill(sxy.begin(), sxy.end(), 0.0); std::fill(bad.begin(), bad.end(), 0);
            std::fill(px.begin(), px.end(), NaN); std::fill(py.begin(), py.end(), NaN);
            for (std::size_t u = t - d + 1; u <= t; ++u) add(u, +1);
            return;
        }
        add(t, +1);
        if (t >= static_cast<std::size_t>(d)) add(t - d, -1);
    }
    bool full(std::size_t t, std::size_t n) const { return t + 1 >= static_cast<std::size_t>(d) && bad[n] == 0; }
    double mean_x(std::size_t n) const { return px[n] + sx[n] / d; }
};

// van Herk / Gil-Werman sliding max (or min): O(1) per cell, vectorised over
// assets. g = running extreme from each block start, h = from each block end.
Mat sliding_extreme(const Mat& a, int d, bool want_max) {
    std::size_t T = a.T, N = a.N;
    Mat g(T, N), h(T, N), out(T, N);
    auto pick = [want_max](double x, double y) { return want_max ? std::max(x, y) : std::min(x, y); };
    for (std::size_t t = 0; t < T; ++t) {
        bool start = t % d == 0;
        for (std::size_t n = 0; n < N; ++n) g(t, n) = start ? a(t, n) : pick(g(t - 1, n), a(t, n));
    }
    for (std::size_t t = T; t-- > 0;) {
        bool end = (t + 1) % d == 0 || t + 1 == T;
        for (std::size_t n = 0; n < N; ++n) h(t, n) = end ? a(t, n) : pick(h(t + 1, n), a(t, n));
    }
    Rolling nan_count(a, nullptr, d);
    for (std::size_t t = 0; t < T; ++t) {
        nan_count.step(t);
        if (t + 1 < static_cast<std::size_t>(d)) continue;
        std::size_t s = t + 1 - d;
        for (std::size_t n = 0; n < N; ++n)
            if (nan_count.bad[n] == 0) out(t, n) = pick(h(s, n), g(t, n));
    }
    return out;
}

}  // namespace

Mat broadcast(double value, std::size_t T, std::size_t N) { return Mat(T, N, value); }

Mat apply_elem(Op op, const Mat& a, const Mat* b) {
    Mat out(a.T, a.N);
    std::size_t sz = a.v.size();
    const double* x = a.v.data();
    const double* y = b ? b->v.data() : nullptr;
    double* o = out.v.data();
    switch (op) {
        case Op::Neg: for (std::size_t i = 0; i < sz; ++i) o[i] = -x[i]; break;
        case Op::Abs: for (std::size_t i = 0; i < sz; ++i) o[i] = std::fabs(x[i]); break;
        case Op::Log: for (std::size_t i = 0; i < sz; ++i) o[i] = x[i] > 0 ? std::log(x[i]) : NaN; break;
        case Op::Sign: for (std::size_t i = 0; i < sz; ++i) o[i] = fin(x[i]) ? (x[i] > 0) - (x[i] < 0) : NaN; break;
        case Op::Add: for (std::size_t i = 0; i < sz; ++i) o[i] = x[i] + y[i]; break;
        case Op::Sub: for (std::size_t i = 0; i < sz; ++i) o[i] = x[i] - y[i]; break;
        case Op::Mul: for (std::size_t i = 0; i < sz; ++i) o[i] = x[i] * y[i]; break;
        case Op::Div: for (std::size_t i = 0; i < sz; ++i) o[i] = std::fabs(y[i]) > kTinyDen ? x[i] / y[i] : NaN; break;
        case Op::Max: for (std::size_t i = 0; i < sz; ++i) o[i] = fin(x[i]) && fin(y[i]) ? std::max(x[i], y[i]) : NaN; break;
        case Op::Min: for (std::size_t i = 0; i < sz; ++i) o[i] = fin(x[i]) && fin(y[i]) ? std::min(x[i], y[i]) : NaN; break;
        default: throw std::logic_error("not an elementwise op");
    }
    for (std::size_t i = 0; i < sz; ++i) if (!fin(o[i])) o[i] = NaN;
    return out;
}

Mat apply_cross(Op op, const Mat& a, const Data& d) {
    std::size_t T = a.T, N = a.N;
    Mat out(T, N);
    std::vector<std::size_t> idx;
    idx.reserve(N);
    for (std::size_t t = 0; t < T; ++t) {
        const double* x = a.row(t);
        double* o = out.row(t);
        idx.clear();
        for (std::size_t n = 0; n < N; ++n)
            if (d.in_universe(t, n) && fin(x[n])) idx.push_back(n);
        std::size_t m = idx.size();
        if (m == 0) continue;
        if (op == Op::Rank) {
            // Percentile rank in [0, 1]; ties share their average rank.
            std::sort(idx.begin(), idx.end(), [x](std::size_t i, std::size_t j) { return x[i] < x[j]; });
            for (std::size_t i = 0; i < m;) {
                std::size_t j = i;
                while (j + 1 < m && x[idx[j + 1]] == x[idx[i]]) ++j;
                double r = m == 1 ? 0.5 : 0.5 * (i + j) / (m - 1);
                for (std::size_t k = i; k <= j; ++k) o[idx[k]] = r;
                i = j + 1;
            }
            continue;
        }
        double s = 0, ss = 0;
        for (auto n : idx) { s += x[n]; ss += x[n] * x[n]; }
        double mean = s / m;
        if (op == Op::Demean) {
            for (auto n : idx) o[n] = x[n] - mean;
        } else if (op == Op::Zscore) {
            if (m < 2) continue;
            double var = 0;
            for (auto n : idx) var += (x[n] - mean) * (x[n] - mean);
            var /= m - 1;
            if (var <= kZeroVar * (ss / m)) continue;
            double sd = std::sqrt(var);
            for (auto n : idx) o[n] = (x[n] - mean) / sd;
        } else if (op == Op::Scale) {
            double gross = 0;
            for (auto n : idx) gross += std::fabs(x[n]);
            if (gross <= kTinyDen) continue;
            for (auto n : idx) o[n] = x[n] / gross;
        } else {
            throw std::logic_error("not a cross-sectional op");
        }
    }
    return out;
}

Mat apply_series(Op op, const Mat& a, const Mat* b, int d) {
    std::size_t T = a.T, N = a.N;
    std::size_t D = static_cast<std::size_t>(d);
    Mat out(T, N);

    if (op == Op::Delay || op == Op::Delta) {
        for (std::size_t t = D; t < T; ++t)
            for (std::size_t n = 0; n < N; ++n) {
                double now = a(t, n), then = a(t - D, n);
                if (fin(now) && fin(then)) out(t, n) = op == Op::Delay ? then : now - then;
            }
        return out;
    }
    if (op == Op::TsMax || op == Op::TsMin) return sliding_extreme(a, d, op == Op::TsMax);

    if (op == Op::TsArgmax || op == Op::TsArgmin || op == Op::TsRank) {
        // O(d) per cell; windows are short and these ops are rare in the search.
        Rolling nan_count(a, nullptr, d);
        for (std::size_t t = 0; t < T; ++t) {
            nan_count.step(t);
            if (t + 1 < D) continue;
            for (std::size_t n = 0; n < N; ++n) {
                if (nan_count.bad[n]) continue;
                double cur = a(t, n);
                if (op == Op::TsRank) {
                    double below = 0, equal = 0;
                    for (std::size_t k = 0; k < D; ++k) {
                        double v = a(t - k, n);
                        below += v < cur;
                        equal += v == cur;
                    }
                    out(t, n) = (below + 0.5 * (equal - 1)) / (d - 1);
                } else {
                    // Days since the extreme (0 = today); ties go to the most recent.
                    std::size_t best = 0;
                    for (std::size_t k = 1; k < D; ++k) {
                        double v = a(t - k, n), bv = a(t - best, n);
                        if (op == Op::TsArgmax ? v > bv : v < bv) best = k;
                    }
                    out(t, n) = static_cast<double>(best);
                }
            }
        }
        return out;
    }

    if (op == Op::DecayLinear) {
        // Weights d, d-1, ..., 1 (today heaviest). With S_{t-1} the plain sum of
        // the window ending yesterday, W_t = W_{t-1} + d*x_t - S_{t-1}.
        // Non-finite values enter both sums as 0; such windows are masked.
        Rolling nan_count(a, nullptr, d);
        std::vector<double> w(N, 0.0), s(N, 0.0);
        double norm = d * (d + 1) / 2.0;
        auto val = [&](std::size_t t, std::size_t n) { double v = a(t, n); return fin(v) ? v : 0.0; };
        for (std::size_t t = 0; t < T; ++t) {
            nan_count.step(t);
            if (t % kRefresh == 0 && t >= D) {
                std::fill(w.begin(), w.end(), 0.0);
                std::fill(s.begin(), s.end(), 0.0);
                for (std::size_t k = 0; k < D; ++k)
                    for (std::size_t n = 0; n < N; ++n) {
                        w[n] += (d - static_cast<double>(k)) * val(t - k, n);
                        s[n] += val(t - k, n);
                    }
            } else {
                for (std::size_t n = 0; n < N; ++n) {
                    w[n] += d * val(t, n) - s[n];
                    s[n] += val(t, n) - (t >= D ? val(t - D, n) : 0.0);
                }
            }
            for (std::size_t n = 0; n < N; ++n)
                if (nan_count.full(t, n)) out(t, n) = w[n] / norm;
        }
        return out;
    }

    Rolling r(a, b, d);
    for (std::size_t t = 0; t < T; ++t) {
        r.step(t);
        if (t + 1 < D) continue;
        const double* x = a.row(t);
        double* o = out.row(t);
        for (std::size_t n = 0; n < N; ++n) {
            if (!r.full(t, n)) continue;
            double sx = r.sx[n], sxx = r.sxx[n], px = r.px[n];
            switch (op) {
                case Op::TsSum: o[n] = sx + d * px; break;
                case Op::TsMean: o[n] = r.mean_x(n); break;
                case Op::TsStd: o[n] = std::sqrt(variance(sx, sxx, px, d)); break;
                case Op::TsZscore: {
                    double var = variance(sx, sxx, px, d);
                    if (var > 0) o[n] = (x[n] - r.mean_x(n)) / std::sqrt(var);
                    break;
                }
                case Op::TsCov: o[n] = (r.sxy[n] - sx * r.sy[n] / d) / (d - 1); break;
                case Op::TsCorr: {
                    double vx = variance(sx, sxx, px, d), vy = variance(r.sy[n], r.syy[n], r.py[n], d);
                    if (vx > 0 && vy > 0) {
                        double cov = (r.sxy[n] - sx * r.sy[n] / d) / (d - 1);
                        o[n] = std::clamp(cov / std::sqrt(vx * vy), -1.0, 1.0);
                    }
                    break;
                }
                default: throw std::logic_error("not a time-series op");
            }
        }
    }
    return out;
}

}  // namespace fa
