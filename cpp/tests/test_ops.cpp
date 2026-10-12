#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "falsealpha/eval.hpp"
#include "falsealpha/expr.hpp"
#include "helpers.hpp"

using namespace fa;
using Catch::Approx;

namespace {

bool same(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
    return std::fabs(a - b) <= 1e-9 * std::max(1.0, std::fabs(a));
}

std::vector<std::string> all_ops_formulas() {
    std::vector<std::string> out;
    for (const auto& o : op_table()) {
        if (o.kind == Kind::Leaf) continue;
        std::string f = std::string(o.name) + "(close";
        if (o.arity == 2) f += ", volume";
        if (o.window) f += ", " + std::to_string(std::max(o.min_window, 4));
        out.push_back(f + ")");
    }
    return out;
}

}  // namespace

TEST_CASE("no op can see the future") {
    // Perturb every input after day k; outputs on days <= k must not move.
    const std::size_t T = 80, N = 12, k = 50;
    Data a = test::random_data(T, N, 7);
    Data b = a;
    std::mt19937 rng(99);
    std::normal_distribution<double> g(0.0, 5.0);
    for (auto& [name, m] : b.fields)
        for (std::size_t t = k + 1; t < T; ++t)
            for (std::size_t n = 0; n < N; ++n) m(t, n) = g(rng);
    for (std::size_t t = k + 1; t < T; ++t)
        for (std::size_t n = 0; n < N; ++n) b.universe[t * N + n] = !b.universe[t * N + n];

    auto formulas = all_ops_formulas();
    formulas.push_back("rank(ts_corr(rank(close), decay_linear(volume, 6), 9)) - ts_rank(returns, 5)");
    for (const auto& f : formulas) {
        INFO(f);
        Mat x = evaluate(parse(f), a), y = evaluate(parse(f), b);
        for (std::size_t t = 0; t <= k; ++t)
            for (std::size_t n = 0; n < N; ++n) REQUIRE(same(x(t, n), y(t, n)));
    }
}

TEST_CASE("hand-checked values") {
    Data d;
    d.T = 6; d.N = 1;
    d.universe.assign(6, 1);
    Mat c(6, 1);
    c.v = {1, 2, 3, 4, 5, 10};
    d.fields["close"] = c;
    auto at = [&](const char* f, std::size_t t) { return evaluate(parse(f), d)(t, 0); };
    REQUIRE(std::isnan(at("ts_mean(close, 3)", 1)));
    REQUIRE(at("ts_mean(close, 3)", 2) == Approx(2));
    REQUIRE(at("ts_sum(close, 3)", 5) == Approx(19));
    REQUIRE(at("ts_std(close, 3)", 4) == Approx(1));
    REQUIRE(at("ts_delta(close, 2)", 5) == Approx(6));
    REQUIRE(at("ts_delay(close, 1)", 5) == Approx(5));
    REQUIRE(at("ts_max(close, 4)", 5) == Approx(10));
    REQUIRE(at("ts_min(close, 4)", 5) == Approx(3));
    REQUIRE(at("ts_argmin(close, 4)", 5) == Approx(3));
    REQUIRE(at("ts_argmax(close, 4)", 5) == Approx(0));
    REQUIRE(at("ts_rank(close, 3)", 5) == Approx(1));
    REQUIRE(at("decay_linear(close, 3)", 5) == Approx((3 * 10 + 2 * 5 + 1 * 4) / 6.0));
    REQUIRE(at("ts_corr(close, close * 2, 4)", 5) == Approx(1));
    REQUIRE(std::isnan(at("ts_corr(close, close * 0 + 1, 4)", 5)));  // constant leg
}

TEST_CASE("cross-sectional ops respect the universe and ties") {
    Data d;
    d.T = 1; d.N = 5;
    d.universe = {1, 1, 1, 0, 1};
    Mat c(1, 5);
    c.v = {3, 1, 3, 100, NaN};
    d.fields["close"] = c;
    Mat r = evaluate(parse("rank(close)"), d);
    REQUIRE(r(0, 0) == Approx(0.75));
    REQUIRE(r(0, 1) == Approx(0.0));
    REQUIRE(r(0, 2) == Approx(0.75));
    REQUIRE(std::isnan(r(0, 3)));  // not a member
    REQUIRE(std::isnan(r(0, 4)));  // no value
    Mat s = evaluate(parse("scale(close)"), d);
    REQUIRE(s(0, 0) == Approx(3.0 / 7));
}

TEST_CASE("rolling state survives long series (drift refresh)") {
    // Big level + small noise is the worst case for running sums.
    const std::size_t T = 3000;
    Data d;
    d.T = T; d.N = 2;
    d.universe.assign(T * 2, 1);
    Mat c(T, 2);
    std::mt19937 rng(3);
    std::normal_distribution<double> g(0.0, 1.0);
    for (auto& x : c.v) x = 1e6 + g(rng);
    d.fields["close"] = c;
    Mat sd = evaluate(parse("ts_std(close, 20)"), d);
    Mat dl = evaluate(parse("decay_linear(close, 20)"), d);
    for (std::size_t t = 19; t < T; t += 37) {
        double m = 0, ss = 0, w = 0;
        for (std::size_t k = 0; k < 20; ++k) { m += c(t - k, 0); w += (20.0 - k) * c(t - k, 0); }
        m /= 20;
        for (std::size_t k = 0; k < 20; ++k) ss += (c(t - k, 0) - m) * (c(t - k, 0) - m);
        REQUIRE(sd(t, 0) == Approx(std::sqrt(ss / 19)).epsilon(1e-6));
        REQUIRE(dl(t, 0) == Approx(w / 210).epsilon(1e-12));
    }
}

TEST_CASE("unknown fields are reported") {
    Data d = test::random_data(10, 3, 1);
    REQUIRE_THROWS_AS(evaluate(parse("rank(vwap)"), d), std::invalid_argument);
}
