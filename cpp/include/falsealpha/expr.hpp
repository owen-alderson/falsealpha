#pragma once
#include <string>
#include <vector>

namespace fa {

enum class Op {
    Field, Const,
    // elementwise
    Neg, Abs, Log, Sign, Add, Sub, Mul, Div, Max, Min,
    // cross-sectional (over universe members on one day)
    Rank, Zscore, Demean, Scale,
    // time-series (trailing window of d days, today included)
    Delay, Delta, TsSum, TsMean, TsStd, TsZscore, TsMin, TsMax, TsArgmin, TsArgmax,
    TsRank, DecayLinear, TsCorr, TsCov,
};

enum class Kind { Leaf, Elem, Cross, Series };

struct OpInfo {
    Op op;
    const char* name;
    int arity;       // series arguments (window not counted)
    bool window;     // takes a trailing integer window
    int min_window;
    Kind kind;
    bool commutative;
};

inline constexpr int kMaxWindow = 10000;  // largest window the parser accepts

const std::vector<OpInfo>& op_table();
const OpInfo& info(Op op);

struct Node {
    Op op = Op::Const;
    std::string field;  // Op::Field
    double value = 0;   // Op::Const
    int window = 0;     // windowed ops
    std::vector<Node> kids;

    std::size_t size() const;   // node count
    std::size_t depth() const;
};

// Field names must be identifiers that are not operator names, so that every
// formula's canonical text parses back to the same formula.
bool valid_field_name(const std::string& name);

// Parses e.g. "-rank(ts_delta(close, 5)) * ts_corr(volume, close, 10)".
// Infix + - * / and unary minus are sugar for add/sub/mul/div/neg.
Node parse(const std::string& text);

// Canonical text: fully parenthesised infix, commutative operands sorted,
// so formulas that are the same function print identically. The trial ledger
// keys on this string.
std::string to_string(const Node& n);

}  // namespace fa
