#pragma once
#include <vector>

#include "falsealpha/mat.hpp"

namespace fa {

struct ScoreConfig {
    int delay = 1;          // signal from data up to close t is traded at close t+delay
    double cost_bps = 10;   // charged per unit of turnover (one side)
    int min_assets = 5;     // fewer members with a signal => flat that day
};

// Per signal day t. The position built from day t's signal earns
// returns(t + delay + 1). Days whose target lies past the panel are NaN.
struct Backtest {
    std::vector<double> ic;        // Spearman(signal_t, target), NaN if undefined
    std::vector<double> gross;     // sum_n w_t[n] * target[n]
    std::vector<double> cost;      // turnover_t * cost_bps / 1e4
    std::vector<double> net;       // gross - cost
    std::vector<double> turnover;  // sum_n |w_t[n] - w_{t-1}[n]|
};

// Weights: cross-sectional rank of the signal, demeaned, scaled to gross 1
// (dollar-neutral long-short). A member with no target return (delisted, gap)
// earns 0, i.e. it is assumed sold at its last price.
Backtest backtest(const Mat& signal, const Mat& returns, const Data& d, const ScoreConfig& c);

struct Summary {
    double mean = 0, sd = 0, sharpe = 0;  // per-period, NaN days skipped
    double mean_ic = 0, ic_ir = 0, turnover = 0;
    int n = 0;
};

// Summary of net returns over days [begin, end).
Summary summarize(const Backtest& b, std::size_t begin, std::size_t end);

}  // namespace fa
