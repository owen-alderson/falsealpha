#include "falsealpha/expr.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace fa {

const std::vector<OpInfo>& op_table() {
    static const std::vector<OpInfo> t = {
        {Op::Field, "field", 0, false, 0, Kind::Leaf, false},
        {Op::Const, "const", 0, false, 0, Kind::Leaf, false},
        {Op::Neg, "neg", 1, false, 0, Kind::Elem, false},
        {Op::Abs, "abs", 1, false, 0, Kind::Elem, false},
        {Op::Log, "log", 1, false, 0, Kind::Elem, false},
        {Op::Sign, "sign", 1, false, 0, Kind::Elem, false},
        {Op::Add, "add", 2, false, 0, Kind::Elem, true},
        {Op::Sub, "sub", 2, false, 0, Kind::Elem, false},
        {Op::Mul, "mul", 2, false, 0, Kind::Elem, true},
        {Op::Div, "div", 2, false, 0, Kind::Elem, false},
        {Op::Max, "max", 2, false, 0, Kind::Elem, true},
        {Op::Min, "min", 2, false, 0, Kind::Elem, true},
        {Op::Rank, "rank", 1, false, 0, Kind::Cross, false},
        {Op::Zscore, "zscore", 1, false, 0, Kind::Cross, false},
        {Op::Demean, "demean", 1, false, 0, Kind::Cross, false},
        {Op::Scale, "scale", 1, false, 0, Kind::Cross, false},
        {Op::Delay, "ts_delay", 1, true, 1, Kind::Series, false},
        {Op::Delta, "ts_delta", 1, true, 1, Kind::Series, false},
        {Op::TsSum, "ts_sum", 1, true, 1, Kind::Series, false},
        {Op::TsMean, "ts_mean", 1, true, 1, Kind::Series, false},
        {Op::TsStd, "ts_std", 1, true, 2, Kind::Series, false},
        {Op::TsZscore, "ts_zscore", 1, true, 2, Kind::Series, false},
        {Op::TsMin, "ts_min", 1, true, 1, Kind::Series, false},
        {Op::TsMax, "ts_max", 1, true, 1, Kind::Series, false},
        {Op::TsArgmin, "ts_argmin", 1, true, 1, Kind::Series, false},
        {Op::TsArgmax, "ts_argmax", 1, true, 1, Kind::Series, false},
        {Op::TsRank, "ts_rank", 1, true, 2, Kind::Series, false},
        {Op::DecayLinear, "decay_linear", 1, true, 1, Kind::Series, false},
        {Op::TsCorr, "ts_corr", 2, true, 3, Kind::Series, true},
        {Op::TsCov, "ts_cov", 2, true, 2, Kind::Series, true},
    };
    return t;
}

const OpInfo& info(Op op) { return op_table()[static_cast<std::size_t>(op)]; }

std::size_t Node::size() const {
    std::size_t s = 1;
    for (const auto& k : kids) s += k.size();
    return s;
}

std::size_t Node::depth() const {
    std::size_t d = 0;
    for (const auto& k : kids) d = std::max(d, k.depth());
    return d + 1;
}

namespace {

Node make(Op op, std::vector<Node> kids, int window = 0) {
    Node n;
    n.op = op;
    n.kids = std::move(kids);
    n.window = window;
    return n;
}

struct Parser {
    const std::string& s;
    std::size_t i = 0;

    [[noreturn]] void fail(const std::string& msg) const {
        throw std::invalid_argument("parse error at " + std::to_string(i) + ": " + msg + " in '" + s + "'");
    }
    void skip() {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    }
    bool eat(char c) {
        skip();
        if (i < s.size() && s[i] == c) { ++i; return true; }
        return false;
    }
    void expect(char c) {
        if (!eat(c)) fail(std::string("expected '") + c + "'");
    }

    Node expr() {
        Node lhs = term();
        for (;;) {
            if (eat('+')) lhs = make(Op::Add, {std::move(lhs), term()});
            else if (eat('-')) lhs = make(Op::Sub, {std::move(lhs), term()});
            else return lhs;
        }
    }
    Node term() {
        Node lhs = unary();
        for (;;) {
            if (eat('*')) lhs = make(Op::Mul, {std::move(lhs), unary()});
            else if (eat('/')) lhs = make(Op::Div, {std::move(lhs), unary()});
            else return lhs;
        }
    }
    Node unary() {
        if (eat('-')) {
            Node x = unary();
            if (x.op == Op::Const) { x.value = -x.value; return x; }
            return make(Op::Neg, {std::move(x)});
        }
        return primary();
    }
    double number() {
        skip();
        std::size_t used = 0;
        double v;
        try { v = std::stod(s.substr(i), &used); } catch (...) { fail("expected a number"); }
        i += used;
        return v;
    }
    Node primary() {
        skip();
        if (i >= s.size()) fail("unexpected end");
        if (eat('(')) { Node x = expr(); expect(')'); return x; }
        if (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.') {
            Node n; n.op = Op::Const; n.value = number(); return n;
        }
        std::size_t start = i;
        while (i < s.size() && (std::isalnum(static_cast<unsigned char>(s[i])) || s[i] == '_')) ++i;
        if (start == i) fail("unexpected character");
        std::string name = s.substr(start, i - start);
        if (!eat('(')) { Node n; n.op = Op::Field; n.field = name; return n; }

        const OpInfo* op = nullptr;
        for (const auto& o : op_table())
            if (o.kind != Kind::Leaf && name == o.name) op = &o;
        if (!op) fail("unknown function '" + name + "'");
        Node n; n.op = op->op;
        for (int a = 0; a < op->arity; ++a) {
            if (a) expect(',');
            n.kids.push_back(expr());
        }
        if (op->window) {
            expect(',');
            double w = number();
            if (w != std::floor(w) || w < op->min_window || w > 10000)
                fail(std::string(op->name) + " window must be an integer >= " + std::to_string(op->min_window));
            n.window = static_cast<int>(w);
        }
        expect(')');
        return n;
    }
};

// Shortest text that parses back to exactly v (the ledger keys on it).
std::string fmt_number(double v) {
    char buf[32];
    for (int digits = 15; digits <= 17; ++digits) {
        std::snprintf(buf, sizeof buf, "%.*g", digits, v);
        if (std::strtod(buf, nullptr) == v) break;
    }
    return buf;
}

}  // namespace

Node parse(const std::string& text) {
    Parser p{text};
    Node n = p.expr();
    p.skip();
    if (p.i != text.size()) p.fail("trailing input");
    return n;
}

std::string to_string(const Node& n) {
    const OpInfo& o = info(n.op);
    if (n.op == Op::Field) return n.field;
    if (n.op == Op::Const) return fmt_number(n.value);
    std::vector<std::string> args;
    for (const auto& k : n.kids) args.push_back(to_string(k));
    if (o.commutative) std::sort(args.begin(), args.end());
    static const char* infix[] = {"+", "-", "*", "/"};
    switch (n.op) {
        case Op::Neg: return "-" + args[0];
        case Op::Add: case Op::Sub: case Op::Mul: case Op::Div:
            return "(" + args[0] + " " + infix[static_cast<int>(n.op) - static_cast<int>(Op::Add)] + " " + args[1] + ")";
        default: break;
    }
    std::string out = std::string(o.name) + "(";
    for (std::size_t a = 0; a < args.size(); ++a) out += (a ? ", " : "") + args[a];
    if (o.window) out += ", " + std::to_string(n.window);
    return out + ")";
}

}  // namespace fa
