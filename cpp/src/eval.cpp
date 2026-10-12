#include "falsealpha/eval.hpp"

#include <memory>
#include <unordered_map>

#include "falsealpha/ops.hpp"

namespace fa {

namespace {

// Only subtrees that occur more than once are kept, so memory stays at about
// one panel per tree level instead of one per node.
struct Cache {
    std::unordered_map<std::string, int> uses;
    std::unordered_map<std::string, std::shared_ptr<const Mat>> kept;
};

void count_uses(const Node& n, Cache& c) {
    if (n.op == Op::Field) return;
    if (++c.uses[to_string(n)] > 1) return;  // children already counted
    for (const auto& k : n.kids) count_uses(k, c);
}

std::shared_ptr<const Mat> eval_rec(const Node& n, const Data& d, Cache& cache) {
    if (n.op == Op::Field) return std::shared_ptr<const Mat>(&d.field(n.field), [](const Mat*) {});
    std::string key = to_string(n);
    if (auto it = cache.kept.find(key); it != cache.kept.end()) return it->second;

    Mat out;
    const OpInfo& o = info(n.op);
    if (n.op == Op::Const) {
        out = broadcast(n.value, d.T, d.N);
    } else {
        auto a = eval_rec(n.kids[0], d, cache);
        auto b = o.arity == 2 ? eval_rec(n.kids[1], d, cache) : nullptr;
        switch (o.kind) {
            case Kind::Elem: out = apply_elem(n.op, *a, b.get()); break;
            case Kind::Cross: out = apply_cross(n.op, *a, d); break;
            case Kind::Series: out = apply_series(n.op, *a, b.get(), n.window); break;
            case Kind::Leaf: break;
        }
    }
    auto p = std::make_shared<const Mat>(std::move(out));
    if (cache.uses[key] > 1) cache.kept.emplace(std::move(key), p);
    return p;
}

}  // namespace

void check_fields(const Node& n, const Data& d) {
    if (n.op == Op::Field) d.field(n.field);
    for (const auto& k : n.kids) check_fields(k, d);
}

Mat evaluate(const Node& n, const Data& d) {
    check_fields(n, d);
    Cache cache;
    count_uses(n, cache);
    Mat out = *eval_rec(n, d, cache);
    for (std::size_t i = 0; i < out.v.size(); ++i)
        if (!d.universe[i]) out.v[i] = NaN;
    return out;
}

}  // namespace fa
