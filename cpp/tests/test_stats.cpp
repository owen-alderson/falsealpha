#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <random>

#include "falsealpha/stats.hpp"

using namespace fa;
using Catch::Approx;

TEST_CASE("inverse normal") {
    REQUIRE(norm_ppf(0.5) == Approx(0.0).margin(1e-15));
    REQUIRE(norm_ppf(0.975) == Approx(1.959963984540054).epsilon(1e-14));
    REQUIRE(norm_ppf(1e-10) == Approx(-6.361340902404056).epsilon(1e-12));
    for (double p : {1e-6, 0.01, 0.3, 0.77, 0.999}) REQUIRE(norm_cdf(norm_ppf(p)) == Approx(p).epsilon(1e-12));
}

TEST_CASE("Deflated Sharpe reproduces Bailey & Lopez de Prado (2014) worked example") {
    // Annualised SR 2.5 over 1250 days (250/yr), skew -3, kurtosis 10,
    // N = 100 trials whose annualised Sharpe ratios have variance 0.5.
    double sr = 2.5 / std::sqrt(250.0);
    double sr0 = expected_max_sharpe(0.5 / 250, 100);
    REQUIRE(sr0 == Approx(0.1132).margin(5e-5));
    REQUIRE(probabilistic_sharpe(sr, sr0, 1250, -3, 10) == Approx(0.9004).margin(5e-4));
}

TEST_CASE("moments of a normal sample") {
    std::mt19937 rng(1);
    std::normal_distribution<double> g(0.0, 2.0);
    std::vector<double> x(200000);
    for (auto& v : x) v = g(rng);
    Moments m = moments(x);
    REQUIRE(m.sd == Approx(2.0).epsilon(0.01));
    REQUIRE(m.skew == Approx(0.0).margin(0.02));
    REQUIRE(m.kurt == Approx(3.0).margin(0.05));
}

namespace {
std::vector<std::vector<double>> noise(std::size_t K, std::size_t T, unsigned seed, double drift0 = 0) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> g(0.0, 0.01);
    std::vector<std::vector<double>> r(K, std::vector<double>(T));
    for (std::size_t k = 0; k < K; ++k)
        for (auto& v : r[k]) v = g(rng) + (k == 0 ? drift0 : 0.0);
    return r;
}
}  // namespace

TEST_CASE("PBO: pure noise overfits about half the time, a real edge does not") {
    // One draw's PBO is noisy (its splits share data), so average several.
    double p = 0;
    for (unsigned s = 0; s < 10; ++s) p += pbo_cscv(noise(40, 1000, 5 + s), 10).pbo / 10;
    REQUIRE(p > 0.35);
    REQUIRE(p < 0.65);
    auto edge = noise(40, 1000, 5, 0.004);  // strategy 0 has Sharpe ~0.4 per day
    REQUIRE(pbo_cscv(edge, 10).pbo < 0.02);
    REQUIRE(pbo_cscv(edge, 10).logits.size() == 252);  // C(10, 5)
}

TEST_CASE("snooping tests: size under the null and power under an edge") {
    int rc_rej = 0, spa_rej = 0, sims = 120;
    for (int s = 0; s < sims; ++s) {
        auto r = noise(10, 300, 100 + s);
        SnoopTest t = snoop_tests(r, 200, 5, s + 1);
        rc_rej += t.reality_check_p < 0.05;
        spa_rej += t.spa_p < 0.05;
    }
    // Nominal 5%; allow sampling error (and RC is conservative).
    REQUIRE(rc_rej <= 15);
    REQUIRE(spa_rej <= 15);
    REQUIRE(spa_rej >= 1);

    auto edge = noise(10, 300, 7, 0.003);
    SnoopTest t = snoop_tests(edge, 500, 5, 3);
    REQUIRE(t.spa_p < 0.01);
    REQUIRE(t.reality_check_p < 0.01);
    REQUIRE(t.best == 0);
}

TEST_CASE("SPA ignores hopeless strategies, Reality Check does not") {
    // One modest edge among many terrible strategies: SPA keeps its power.
    auto r = noise(30, 400, 11, 0.0012);
    for (std::size_t k = 1; k < r.size(); ++k)
        for (auto& v : r[k]) v -= 0.003;
    SnoopTest t = snoop_tests(r, 1000, 5, 9);
    REQUIRE(t.spa_p <= t.reality_check_p);
}

TEST_CASE("stationary bootstrap is reproducible and in range") {
    std::uint64_t a = 42, b = 42;
    auto x = stationary_bootstrap(100, 5, a), y = stationary_bootstrap(100, 5, b);
    REQUIRE(x == y);
    for (auto i : x) REQUIRE(i < 100);
}

TEST_CASE("bootstrap tests reject empty or one-day series") {
    std::vector<std::vector<double>> empty(3), one(3, std::vector<double>(1, 0.01));
    REQUIRE_THROWS_AS(snoop_tests(empty, 10, 5, 1), std::invalid_argument);
    REQUIRE_THROWS_AS(snoop_tests(one, 10, 5, 1), std::invalid_argument);
    std::uint64_t s = 1;
    REQUIRE_THROWS_AS(stationary_bootstrap(0, 5, s), std::invalid_argument);
}
