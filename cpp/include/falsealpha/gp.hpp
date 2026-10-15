#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "falsealpha/expr.hpp"
#include "falsealpha/mat.hpp"
#include "falsealpha/score.hpp"

namespace fa {

struct SearchConfig {
    int population = 400;
    int generations = 25;
    int tournament = 4;
    int elite = 5;
    int init_depth = 4;
    int max_depth = 6;
    int max_size = 20;
    // Variation mix; whatever is left over is plain reproduction.
    double p_crossover = 0.55, p_subtree = 0.15, p_point = 0.15, p_hoist = 0.05;
    std::vector<std::string> fields;
    std::vector<int> windows{3, 5, 10, 20, 40};
    std::size_t train_begin = 0, train_end = 0;  // fitness is measured on days [begin, end)
    ScoreConfig score;
    double parsimony = 0.0005;  // per node, in per-period Sharpe units
    double max_corr = 0.7;      // above this |corr| a formula is a near-copy (hall of fame, trial clusters)
    int hall_size = 50;
    std::uint64_t seed = 1;
    int threads = 0;            // 0 = all cores
};

// One row of the trial ledger: every distinct formula the search ever scored.
// The Deflated Sharpe Ratio needs all of them, not just the winners.
// Near-copies are not independent trials, so trials are also clustered by
// their train returns: a trial joins the first cluster whose leader it
// matches with |corr| > max_corr, or leads a new one. The number of clusters
// estimates the number of independent trials.
struct Trial {
    std::string formula;
    double sharpe = 0, mean_ic = 0, turnover = 0;  // train window, per period; NaN if degenerate
    int size = 0, generation = 0;
    int cluster = -1;  // -1 if degenerate
};

struct Finalist {
    std::string formula;
    double fitness = 0;
    Summary train;
    std::vector<double> net;  // full-length daily net returns (all days, not just train)
};

struct SearchResult {
    std::vector<Trial> ledger;
    std::vector<Finalist> hall;  // diverse best formulas, best first
};

using Progress = std::function<void(int generation, const SearchResult&)>;

SearchResult search(const Data& d, const Mat& returns, const SearchConfig& c, const Progress& progress = {});

// Backtests many formulas in parallel (formula-level parallelism).
std::vector<Backtest> backtest_many(const std::vector<std::string>& formulas, const Data& d, const Mat& returns,
                                    const ScoreConfig& c, int threads = 0);

// Pearson correlation over days where both are finite.
double series_corr(const std::vector<double>& a, const std::vector<double>& b, std::size_t begin, std::size_t end);

}  // namespace fa
