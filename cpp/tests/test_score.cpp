#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "falsealpha/score.hpp"

using namespace fa;
using Catch::Approx;

TEST_CASE("long-short backtest on a toy panel") {
    // 4 assets, signal = asset index; with delay 0 day t's position earns returns(t+1).
    const std::size_t T = 4, N = 4;
    Data d;
    d.T = T; d.N = N;
    d.universe.assign(T * N, 1);
    Mat sig(T, N), ret(T, N, 0.0);
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t n = 0; n < N; ++n) sig(t, n) = static_cast<double>(n);
    ret(1, 0) = -0.01; ret(1, 3) = 0.02;
    ScoreConfig c;
    c.delay = 0; c.cost_bps = 10; c.min_assets = 2;
    Backtest b = backtest(sig, ret, d, c);
    // ranks 0..3, demeaned -1.5..1.5, gross 4 => weights -0.375, -0.125, 0.125, 0.375
    REQUIRE(b.gross[0] == Approx(-0.375 * -0.01 + 0.375 * 0.02));
    REQUIRE(b.turnover[0] == Approx(1.0));
    REQUIRE(b.cost[0] == Approx(0.001));
    REQUIRE(b.turnover[1] == Approx(0.0));
    REQUIRE(std::isnan(b.net[3]));  // target past the panel
    // signal ranks 0,1,2,3 vs return ranks 0,1.5,1.5,3 => 4.5 / sqrt(5 * 4.5)
    REQUIRE(b.ic[0] == Approx(4.5 / std::sqrt(22.5)));
}
