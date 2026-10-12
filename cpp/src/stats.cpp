#include "falsealpha/stats.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace fa {

namespace {

constexpr double kEulerGamma = 0.5772156649015329;

// splitmix64: small, seedable, identical on every platform.
std::uint64_t next(std::uint64_t& s) {
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
double uniform(std::uint64_t& s) { return (next(s) >> 11) * 0x1.0p-53; }

double poly(const double* c, int n, double x) {
    double r = 0;
    for (int i = n - 1; i >= 0; --i) r = r * x + c[i];
    return r;
}

}  // namespace

double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

double norm_ppf(double p) {
    if (!(p > 0 && p < 1)) {
        if (p == 0) return -std::numeric_limits<double>::infinity();
        if (p == 1) return std::numeric_limits<double>::infinity();
        return std::numeric_limits<double>::quiet_NaN();
    }
    // Wichura (1988) AS241, PPND16: about 1e-16 relative accuracy.
    static const double a[] = {3.387132872796366608, 133.14166789178437745, 1971.5909503065514427,
                               13731.693765509461125, 45921.953931549871457, 67265.770927008700853,
                               33430.575583588128105, 2509.0809287301226727};
    static const double b[] = {1.0, 42.313330701600911252, 687.1870074920579083, 5394.1960214247511077,
                               21213.794301586595867, 39307.89580009271061, 28729.085735721942674,
                               5226.495278852545925};
    static const double c[] = {1.42343711074968357734, 4.6303378461565452959, 5.7694972214606914055,
                               3.64784832476320460504, 1.27045825245236838258, 0.24178072517745061177,
                               0.0227238449892691845833, 7.7454501427834140764e-4};
    static const double d[] = {1.0, 2.05319162663775882187, 1.6763848301838038494, 0.68976733498510000455,
                               0.14810397642748007459, 0.0151986665636164571966, 5.475938084995344946e-4,
                               1.05075007164441684324e-9};
    static const double e[] = {6.6579046435011037772, 5.4637849111641143699, 1.7848265399172913358,
                               0.29656057182850489123, 0.026532189526576123093, 0.0012426609473880784386,
                               2.71155556874348757815e-5, 2.01033439929228813265e-7};
    static const double f[] = {1.0, 0.59983220655588793769, 0.13692988092273580531, 0.0148753612908506148525,
                               7.868691311456132591e-4, 1.8463183175100546818e-5, 1.4215117583164458887e-7,
                               2.04426310338993978564e-15};
    double q = p - 0.5;
    if (std::fabs(q) <= 0.425) {
        double r = 0.180625 - q * q;
        return q * poly(a, 8, r) / poly(b, 8, r);
    }
    double r = std::sqrt(-std::log(q < 0 ? p : 1 - p));
    double x = r <= 5 ? poly(c, 8, r - 1.6) / poly(d, 8, r - 1.6) : poly(e, 8, r - 5) / poly(f, 8, r - 5);
    return q < 0 ? -x : x;
}

Moments moments(const std::vector<double>& x) {
    Moments m;
    double s = 0;
    for (double v : x) if (std::isfinite(v)) { s += v; ++m.n; }
    if (m.n < 2) return m;
    m.mean = s / m.n;
    double m2 = 0, m3 = 0, m4 = 0;
    for (double v : x) {
        if (!std::isfinite(v)) continue;
        double z = v - m.mean;
        m2 += z * z; m3 += z * z * z; m4 += z * z * z * z;
    }
    m2 /= m.n; m3 /= m.n; m4 /= m.n;
    m.sd = std::sqrt(m2 * m.n / (m.n - 1));
    if (m2 > 0) { m.skew = m3 / std::pow(m2, 1.5); m.kurt = m4 / (m2 * m2); }
    return m;
}

double expected_max_sharpe(double var_sr, double n_trials) {
    if (n_trials < 2 || var_sr <= 0) return 0;
    double z = (1 - kEulerGamma) * norm_ppf(1 - 1 / n_trials) +
               kEulerGamma * norm_ppf(1 - 1 / (n_trials * std::exp(1.0)));
    return std::sqrt(var_sr) * z;
}

double probabilistic_sharpe(double sr, double sr0, int T, double skew, double kurt) {
    double den = 1 - skew * sr + (kurt - 1) / 4 * sr * sr;
    if (T < 2 || den <= 0) return std::numeric_limits<double>::quiet_NaN();
    return norm_cdf((sr - sr0) * std::sqrt(T - 1.0) / std::sqrt(den));
}

Pbo pbo_cscv(const std::vector<std::vector<double>>& returns, int S) {
    if (S < 2 || S % 2) throw std::invalid_argument("S must be even and >= 2");
    std::size_t K = returns.size();
    if (K < 2) throw std::invalid_argument("PBO needs at least 2 strategies");
    std::size_t T = returns[0].size();
    if (T < static_cast<std::size_t>(S) * 2) throw std::invalid_argument("too few days for S blocks");

    // Per strategy and block: count, sum, sum of squares (NaN days skipped).
    std::vector<double> cnt(K * S), sum(K * S), sq(K * S);
    for (std::size_t k = 0; k < K; ++k) {
        if (returns[k].size() != T) throw std::invalid_argument("ragged return matrix");
        for (std::size_t t = 0; t < T; ++t) {
            double v = returns[k][t];
            if (!std::isfinite(v)) continue;
            std::size_t blk = std::min<std::size_t>(t * S / T, S - 1);
            cnt[k * S + blk] += 1; sum[k * S + blk] += v; sq[k * S + blk] += v * v;
        }
    }
    auto sharpe = [&](std::size_t k, const std::vector<int>& blocks) {
        double c = 0, s = 0, q = 0;
        for (int b : blocks) { c += cnt[k * S + b]; s += sum[k * S + b]; q += sq[k * S + b]; }
        if (c < 2) return 0.0;
        double var = (q - s * s / c) / (c - 1);
        return var > 0 ? (s / c) / std::sqrt(var) : 0.0;
    };

    Pbo out;
    std::vector<bool> pick(S, false);
    std::fill(pick.begin(), pick.begin() + S / 2, true);
    std::vector<int> is, oos;
    std::vector<double> oos_sr(K);
    do {
        is.clear(); oos.clear();
        for (int b = 0; b < S; ++b) (pick[b] ? is : oos).push_back(b);
        std::size_t best = 0;
        double best_sr = -std::numeric_limits<double>::infinity();
        for (std::size_t k = 0; k < K; ++k) {
            double sr = sharpe(k, is);
            if (sr > best_sr) { best_sr = sr; best = k; }
            oos_sr[k] = sharpe(k, oos);
        }
        // Relative rank of the IS winner out of sample, in (0, 1); ties count half.
        double below = 0;
        for (std::size_t k = 0; k < K; ++k)
            below += oos_sr[k] < oos_sr[best] ? 1.0 : (oos_sr[k] == oos_sr[best] && k != best ? 0.5 : 0.0);
        double w = (below + 1) / (K + 1);
        out.logits.push_back(std::log(w / (1 - w)));
    } while (std::prev_permutation(pick.begin(), pick.end()));
    out.pbo = std::count_if(out.logits.begin(), out.logits.end(), [](double l) { return l <= 0; }) /
              static_cast<double>(out.logits.size());
    return out;
}

std::vector<std::size_t> stationary_bootstrap(std::size_t T, double mean_block, std::uint64_t& state) {
    std::vector<std::size_t> idx(T);
    double p = 1.0 / mean_block;
    std::size_t cur = next(state) % T;
    for (std::size_t t = 0; t < T; ++t) {
        if (t > 0) cur = uniform(state) < p ? next(state) % T : (cur + 1) % T;
        idx[t] = cur;
    }
    return idx;
}

SnoopTest snoop_tests(const std::vector<std::vector<double>>& excess, int bootstraps, double mean_block,
                      std::uint64_t seed) {
    std::size_t K = excess.size();
    if (K == 0) throw std::invalid_argument("no strategies");
    std::size_t T = excess[0].size();
    for (const auto& e : excess)
        if (e.size() != T) throw std::invalid_argument("ragged return matrix");
    double sn = std::sqrt(static_cast<double>(T));

    std::vector<double> mean(K, 0.0);
    for (std::size_t k = 0; k < K; ++k) {
        for (double v : excess[k]) mean[k] += std::isfinite(v) ? v : 0.0;
        mean[k] /= T;
    }

    // Bootstrap means, reused for the variance estimate and both p-values.
    std::vector<double> boot(static_cast<std::size_t>(bootstraps) * K);
    std::uint64_t state = seed;
    for (int b = 0; b < bootstraps; ++b) {
        auto idx = stationary_bootstrap(T, mean_block, state);
        for (std::size_t k = 0; k < K; ++k) {
            double s = 0;
            for (auto i : idx) { double v = excess[k][i]; s += std::isfinite(v) ? v : 0.0; }
            boot[b * K + k] = s / T;
        }
    }
    std::vector<double> omega(K, 0.0);
    for (std::size_t k = 0; k < K; ++k) {
        double v = 0;
        for (int b = 0; b < bootstraps; ++b) {
            double z = sn * (boot[b * K + k] - mean[k]);
            v += z * z;
        }
        omega[k] = std::sqrt(v / bootstraps);
    }

    SnoopTest out;
    double rc_stat = -std::numeric_limits<double>::infinity(), best_z = -std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k < K; ++k) {
        rc_stat = std::max(rc_stat, sn * mean[k]);
        if (omega[k] > 0 && sn * mean[k] / omega[k] > best_z) {
            best_z = sn * mean[k] / omega[k];
            out.best = static_cast<int>(k);
        }
    }
    double spa_stat = std::max(0.0, best_z);
    // Hansen's consistent recentring g(x) = x * 1{x >= -A}: a strategy far
    // below the benchmark is not recentred, so its bootstrap draws stay very
    // negative and it cannot inflate the p-value.
    double loglog = std::log(std::log(static_cast<double>(T)));
    std::vector<double> mu_c(K);
    for (std::size_t k = 0; k < K; ++k) {
        double a = std::sqrt(omega[k] * omega[k] / T * 2 * loglog);
        mu_c[k] = mean[k] >= -a ? mean[k] : 0.0;
    }
    int rc_hits = 0, spa_hits = 0;
    for (int b = 0; b < bootstraps; ++b) {
        double rc_b = -std::numeric_limits<double>::infinity(), spa_b = 0;
        for (std::size_t k = 0; k < K; ++k) {
            double bm = boot[b * K + k];
            rc_b = std::max(rc_b, sn * (bm - mean[k]));
            if (omega[k] > 0) spa_b = std::max(spa_b, sn * (bm - mu_c[k]) / omega[k]);
        }
        rc_hits += rc_b >= rc_stat;
        spa_hits += spa_b >= spa_stat;
    }
    out.reality_check_p = static_cast<double>(rc_hits) / bootstraps;
    out.spa_p = static_cast<double>(spa_hits) / bootstraps;
    return out;
}

}  // namespace fa
