#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <random>
#include <set>

#include "falsealpha/eval.hpp"
#include "falsealpha/gp.hpp"

using namespace fa;

namespace {

// Returns two days after day t are pushed by rank(ts_mean(volume, 5)) on day t,
// which is exactly what a delay-1 strategy trading that signal would earn.
Data planted(std::size_t T, std::size_t N, double beta, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    Data d;
    d.T = T; d.N = N;
    d.universe.assign(T * N, 1);
    Mat vol(T, N), ret(T, N), close(T, N);
    for (auto& x : vol.v) x = g(rng);
    d.fields["volume"] = vol;
    Mat sig = evaluate(parse("rank(ts_mean(volume, 5))"), d);
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t n = 0; n < N; ++n) {
            double push = t >= 2 && std::isfinite(sig(t - 2, n)) ? beta * (sig(t - 2, n) - 0.5) : 0.0;
            ret(t, n) = 0.01 * g(rng) + push;
            close(t, n) = (t ? close(t - 1, n) : 100.0) * (1 + ret(t, n));
        }
    d.fields["returns"] = ret;
    d.fields["close"] = close;
    return d;
}

SearchConfig small_config() {
    SearchConfig c;
    c.population = 120;
    c.generations = 8;
    c.fields = {"close", "volume", "returns"};
    c.train_begin = 0;
    c.train_end = 400;
    c.score.cost_bps = 0;
    c.threads = 4;
    return c;
}

}  // namespace

TEST_CASE("search recovers a planted alpha") {
    Data d = planted(500, 40, 0.01, 1);
    SearchConfig c = small_config();
    SearchResult r = search(d, d.fields.at("returns"), c);
    REQUIRE(!r.hall.empty());

    Backtest truth = backtest(evaluate(parse("rank(ts_mean(volume, 5))"), d), d.fields.at("returns"), d, c.score);
    double truth_sr = summarize(truth, 0, 400).sharpe;
    const Finalist& best = r.hall.front();
    INFO(best.formula << " sharpe " << best.train.sharpe << " vs planted " << truth_sr);
    REQUIRE(best.train.sharpe > 0.7 * truth_sr);
    REQUIRE(series_corr(best.net, truth.net, 0, 400) > 0.6);
}

TEST_CASE("ledger records each distinct formula once and is deterministic") {
    Data d = planted(200, 15, 0.0, 2);
    SearchConfig c = small_config();
    c.train_end = 150;
    c.generations = 4;
    SearchResult a = search(d, d.fields.at("returns"), c);
    c.threads = 1;
    SearchResult b = search(d, d.fields.at("returns"), c);

    std::set<std::string> uniq;
    for (const auto& t : a.ledger) REQUIRE(uniq.insert(t.formula).second);
    REQUIRE(a.ledger.size() > static_cast<std::size_t>(c.population));
    REQUIRE(a.ledger.size() == b.ledger.size());
    for (std::size_t i = 0; i < a.ledger.size(); ++i) REQUIRE(a.ledger[i].formula == b.ledger[i].formula);
}

TEST_CASE("hall of fame holds no near-copies") {
    Data d = planted(300, 20, 0.01, 3);
    SearchConfig c = small_config();
    c.train_end = 250;
    SearchResult r = search(d, d.fields.at("returns"), c);
    for (std::size_t i = 0; i < r.hall.size(); ++i)
        for (std::size_t j = i + 1; j < r.hall.size(); ++j)
            REQUIRE(std::fabs(series_corr(r.hall[i].net, r.hall[j].net, 0, 250)) <= c.max_corr + 1e-12);
}
