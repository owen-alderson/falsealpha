#include <catch2/catch_test_macros.hpp>

#include "falsealpha/expr.hpp"

using namespace fa;

TEST_CASE("op table is indexed by the Op enum") {
    for (std::size_t i = 0; i < op_table().size(); ++i)
        REQUIRE(static_cast<std::size_t>(op_table()[i].op) == i);
}

TEST_CASE("parse and print round-trip") {
    for (const char* s : {
             "rank(ts_delta(close, 5))",
             "-rank(close)",
             "(-1 * ts_corr(rank(volume), rank(close), 10))",
             "decay_linear((close - ts_mean(close, 20)), 5)",
             "ts_argmax(close, 7)",
         }) {
        Node n = parse(s);
        std::string once = to_string(n);
        REQUIRE(to_string(parse(once)) == once);
    }
}

TEST_CASE("commutative operands are canonicalised") {
    REQUIRE(to_string(parse("close + volume")) == to_string(parse("volume + close")));
    REQUIRE(to_string(parse("ts_corr(close, volume, 5)")) == to_string(parse("ts_corr(volume, close, 5)")));
    REQUIRE(to_string(parse("close - volume")) != to_string(parse("volume - close")));
}

TEST_CASE("precedence and unary minus") {
    REQUIRE(to_string(parse("close + volume * 2")) == "((2 * volume) + close)");
    REQUIRE(to_string(parse("-3")) == "-3");
    REQUIRE(to_string(parse("-close")) == "-close");
}

TEST_CASE("size and depth") {
    Node n = parse("rank(ts_delta(close, 5)) * volume");
    REQUIRE(n.size() == 5);
    REQUIRE(n.depth() == 4);
}

TEST_CASE("bad formulas are rejected") {
    for (const char* s : {"rank(", "nope(close)", "ts_mean(close)", "ts_mean(close, 2.5)",
                          "ts_corr(close, volume, 2)", "close +", "close close"})
        REQUIRE_THROWS_AS(parse(s), std::invalid_argument);
}

TEST_CASE("constants print exactly, so distinct formulas never share a ledger key") {
    REQUIRE(to_string(parse("close * 1.00000000001")) != to_string(parse("close * 1.00000000002")));
    REQUIRE(to_string(parse("close * 0.1")) == "(0.1 * close)");
    REQUIRE(to_string(parse("close + 0.001")) == "(0.001 + close)");
}

TEST_CASE("field names must survive a round trip through the parser") {
    REQUIRE(valid_field_name("close"));
    REQUIRE(valid_field_name("_x1"));
    REQUIRE_FALSE(valid_field_name("1d_ret"));
    REQUIRE_FALSE(valid_field_name("quote-volume"));
    REQUIRE_FALSE(valid_field_name("rank"));  // an operator name
    REQUIRE_FALSE(valid_field_name(""));
}
