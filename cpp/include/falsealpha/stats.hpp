#pragma once
#include <cstdint>
#include <vector>

namespace fa {

double norm_cdf(double x);
double norm_ppf(double p);  // inverse normal CDF (Wichura AS241)

// Sample moments of a series (NaN entries skipped). Kurtosis is raw (3 for a normal).
struct Moments { double mean = 0, sd = 0, skew = 0, kurt = 3; int n = 0; };
Moments moments(const std::vector<double>& x);

// Bailey & Lopez de Prado (2014). Sharpe ratios are per period, not annualised.
// Expected maximum Sharpe among n_trials unskilled strategies whose Sharpe
// ratios have variance var_sr:
//   sqrt(var_sr) * [(1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e))], g = Euler-Mascheroni.
double expected_max_sharpe(double var_sr, double n_trials);
// Probabilistic Sharpe ratio: P(true Sharpe > sr0) given T observations.
// With sr0 = expected_max_sharpe(...) this is the Deflated Sharpe Ratio.
double probabilistic_sharpe(double sr, double sr0, int T, double skew, double kurt);

// Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado & Zhu 2017)
// by combinatorially symmetric cross-validation. returns = K strategies x T days.
// The T days are cut into S equal blocks (S even); for every half/half split the
// in-sample best strategy's out-of-sample rank w gives logit ln(w / (1 - w)).
struct Pbo {
    double pbo = 0;              // share of splits with logit <= 0
    std::vector<double> logits;  // one per split
};
Pbo pbo_cscv(const std::vector<std::vector<double>>& returns, int S = 16);

// Stationary bootstrap (Politis & Romano 1994) resample of 0..T-1.
std::vector<std::size_t> stationary_bootstrap(std::size_t T, double mean_block, std::uint64_t& state);

// Data-snooping tests on K strategies' excess returns over a benchmark
// (here: zero, i.e. flat). H0: no strategy beats the benchmark.
struct SnoopTest {
    double reality_check_p = 1;  // White (2000)
    double spa_p = 1;            // Hansen (2005), consistent recentring, studentised
    int best = -1;               // index of the strategy with the best studentised mean
};
SnoopTest snoop_tests(const std::vector<std::vector<double>>& excess, int bootstraps = 1000,
                      double mean_block = 10, std::uint64_t seed = 1);

}  // namespace fa
