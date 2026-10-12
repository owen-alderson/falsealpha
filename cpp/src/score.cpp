#include "falsealpha/score.hpp"

#include "falsealpha/ops.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace fa {

namespace {

double pearson(const std::vector<double>& a, const std::vector<double>& b) {
    double n = static_cast<double>(a.size());
    double ma = std::accumulate(a.begin(), a.end(), 0.0) / n;
    double mb = std::accumulate(b.begin(), b.end(), 0.0) / n;
    double sab = 0, saa = 0, sbb = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sab += (a[i] - ma) * (b[i] - mb);
        saa += (a[i] - ma) * (a[i] - ma);
        sbb += (b[i] - mb) * (b[i] - mb);
    }
    return saa > 0 && sbb > 0 ? sab / std::sqrt(saa * sbb) : NaN;
}

}  // namespace

Backtest backtest(const Mat& signal, const Mat& returns, const Data& d, const ScoreConfig& c) {
    std::size_t T = signal.T, N = signal.N;
    Backtest b;
    b.ic.assign(T, NaN); b.gross.assign(T, NaN); b.cost.assign(T, NaN);
    b.net.assign(T, NaN); b.turnover.assign(T, NaN);
    std::size_t lag = static_cast<std::size_t>(c.delay) + 1;

    std::vector<double> w(N, 0.0), prev(N, 0.0), sig, tgt;
    std::vector<std::size_t> members;
    for (std::size_t t = 0; t + lag < T; ++t) {
        const double* s = signal.row(t);
        const double* r = returns.row(t + lag);

        members.clear();
        for (std::size_t n = 0; n < N; ++n)
            if (d.in_universe(t, n) && std::isfinite(s[n])) members.push_back(n);

        std::fill(w.begin(), w.end(), 0.0);
        if (members.size() >= static_cast<std::size_t>(c.min_assets)) {
            sig.clear();
            for (auto n : members) sig.push_back(s[n]);
            std::vector<double> rk = average_ranks(sig);
            double mean = std::accumulate(rk.begin(), rk.end(), 0.0) / rk.size();
            double gross = 0;
            for (auto& x : rk) { x -= mean; gross += std::fabs(x); }
            if (gross > 0)
                for (std::size_t i = 0; i < members.size(); ++i) w[members[i]] = rk[i] / gross;

            sig.clear(); tgt.clear();
            for (auto n : members)
                if (std::isfinite(r[n])) { sig.push_back(s[n]); tgt.push_back(r[n]); }
            if (sig.size() >= static_cast<std::size_t>(c.min_assets)) b.ic[t] = pearson(average_ranks(sig), average_ranks(tgt));
        }

        double pnl = 0, turn = 0;
        for (std::size_t n = 0; n < N; ++n) {
            if (w[n] != 0 && std::isfinite(r[n])) pnl += w[n] * r[n];
            turn += std::fabs(w[n] - prev[n]);
        }
        b.gross[t] = pnl;
        b.turnover[t] = turn;
        b.cost[t] = turn * c.cost_bps / 1e4;
        b.net[t] = pnl - b.cost[t];
        std::swap(w, prev);
    }
    return b;
}

Summary summarize(const Backtest& b, std::size_t begin, std::size_t end) {
    Summary s;
    end = std::min(end, b.net.size());
    double sum = 0, sq = 0, ic = 0, icsq = 0, turn = 0;
    int nic = 0;
    for (std::size_t t = begin; t < end; ++t) {
        if (std::isfinite(b.net[t])) {
            sum += b.net[t]; sq += b.net[t] * b.net[t]; turn += b.turnover[t]; ++s.n;
        }
        if (std::isfinite(b.ic[t])) { ic += b.ic[t]; icsq += b.ic[t] * b.ic[t]; ++nic; }
    }
    if (s.n > 1) {
        s.mean = sum / s.n;
        s.sd = std::sqrt(std::max(0.0, (sq - sum * sum / s.n) / (s.n - 1)));
        s.sharpe = s.sd > 0 ? s.mean / s.sd : 0.0;
        s.turnover = turn / s.n;
    }
    if (nic > 1) {
        s.mean_ic = ic / nic;
        double sd = std::sqrt(std::max(0.0, (icsq - ic * ic / nic) / (nic - 1)));
        s.ic_ir = sd > 0 ? s.mean_ic / sd : 0.0;
    }
    return s;
}

}  // namespace fa
