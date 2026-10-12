#include "falsealpha/gp.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "falsealpha/eval.hpp"

namespace fa {

namespace {

constexpr double kNegInf = -std::numeric_limits<double>::infinity();

void parallel_for(std::size_t n, int threads, const std::function<void(std::size_t)>& fn) {
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    std::size_t workers = std::min<std::size_t>(n, threads > 0 ? threads : hw);
    if (workers <= 1) {
        for (std::size_t i = 0; i < n; ++i) fn(i);
        return;
    }
    std::atomic<std::size_t> next{0};
    std::exception_ptr error;
    std::mutex error_mu;
    std::vector<std::thread> pool;
    for (std::size_t w = 0; w < workers; ++w)
        pool.emplace_back([&] {
            for (std::size_t i; (i = next++) < n;) {
                try { fn(i); } catch (...) {
                    std::lock_guard lock(error_mu);
                    if (!error) error = std::current_exception();
                }
            }
        });
    for (auto& t : pool) t.join();
    if (error) std::rethrow_exception(error);
}

struct Rng {
    std::mt19937_64 g;
    explicit Rng(std::uint64_t seed) : g(seed) {}
    double uniform() { return std::uniform_real_distribution<double>(0, 1)(g); }
    std::size_t below(std::size_t n) { return std::uniform_int_distribution<std::size_t>(0, n - 1)(g); }
    template <class T> const T& pick(const std::vector<T>& v) { return v[below(v.size())]; }
};

class Builder {
public:
    Builder(const SearchConfig& c, Rng& rng) : c_(c), rng_(rng) {
        for (const auto& o : op_table())
            if (o.kind != Kind::Leaf) ops_.push_back(&o);
    }

    Node tree(int depth, bool full) {
        if (depth <= 1 || (!full && rng_.uniform() < 0.3)) return leaf();
        const OpInfo* o = rng_.pick(ops_);
        Node n;
        n.op = o->op;
        for (int a = 0; a < o->arity; ++a) n.kids.push_back(tree(depth - 1, full));
        if (o->window) n.window = window(*o);
        return n;
    }

    Node leaf() {
        Node n;
        n.op = Op::Field;
        n.field = rng_.pick(c_.fields);
        return n;
    }

    int window(const OpInfo& o) {
        std::vector<int> ok;
        for (int w : c_.windows) if (w >= o.min_window) ok.push_back(w);
        return ok.empty() ? o.min_window : rng_.pick(ok);
    }

    // Same arity and window-ness, so the children still fit.
    void swap_op(Node& n) {
        const OpInfo& cur = info(n.op);
        std::vector<const OpInfo*> alt;
        for (auto* o : ops_)
            if (o->arity == cur.arity && o->window == cur.window && o->op != n.op) alt.push_back(o);
        if (alt.empty()) return;
        const OpInfo* o = rng_.pick(alt);
        n.op = o->op;
        if (o->window) n.window = std::max(n.window, o->min_window);
    }

private:
    const SearchConfig& c_;
    Rng& rng_;
    std::vector<const OpInfo*> ops_;
};

void collect(Node& n, std::vector<Node*>& out) {
    out.push_back(&n);
    for (auto& k : n.kids) collect(k, out);
}

Node* random_node(Node& root, Rng& rng) {
    std::vector<Node*> all;
    collect(root, all);
    return all[rng.below(all.size())];
}

struct Scored {
    std::string formula;
    double fitness = kNegInf;  // penalised; what selection sees
    double raw = kNegInf;      // Sharpe minus parsimony, before the novelty penalty
    Summary train;
    std::vector<double> net;
};

}  // namespace

double series_corr(const std::vector<double>& a, const std::vector<double>& b, std::size_t begin, std::size_t end) {
    double n = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    end = std::min({end, a.size(), b.size()});
    for (std::size_t t = begin; t < end; ++t) {
        if (!std::isfinite(a[t]) || !std::isfinite(b[t])) continue;
        n += 1; sa += a[t]; sb += b[t]; saa += a[t] * a[t]; sbb += b[t] * b[t]; sab += a[t] * b[t];
    }
    if (n < 3) return 0;
    double va = saa - sa * sa / n, vb = sbb - sb * sb / n;
    return va > 0 && vb > 0 ? (sab - sa * sb / n) / std::sqrt(va * vb) : 0;
}

std::vector<Backtest> backtest_many(const std::vector<std::string>& formulas, const Data& d, const Mat& returns,
                                    const ScoreConfig& c, int threads) {
    std::vector<Node> nodes;
    for (const auto& f : formulas) nodes.push_back(parse(f));
    std::vector<Backtest> out(formulas.size());
    parallel_for(formulas.size(), threads, [&](std::size_t i) { out[i] = backtest(evaluate(nodes[i], d), returns, d, c); });
    return out;
}

SearchResult search(const Data& d, const Mat& returns, const SearchConfig& c, const Progress& progress) {
    if (c.fields.empty()) throw std::invalid_argument("search needs at least one field");
    if (c.train_end <= c.train_begin || c.train_end > d.T) throw std::invalid_argument("bad train window");
    if (c.population < 2 || c.generations < 1) throw std::invalid_argument("population >= 2 and generations >= 1");
    for (const auto& f : c.fields) d.field(f);

    Rng rng(c.seed);
    Builder build(c, rng);
    SearchResult result;
    std::unordered_map<std::string, double> fitness_cache;  // formula -> penalised fitness

    auto fits = [&](const Node& n) {
        return n.size() <= static_cast<std::size_t>(c.max_size) && n.depth() <= static_cast<std::size_t>(c.max_depth);
    };

    // Ramped half-and-half initial population, distinct formulas where possible.
    std::vector<Node> pop;
    std::unordered_set<std::string> seen;
    for (int tries = 0; static_cast<int>(pop.size()) < c.population && tries < c.population * 20; ++tries) {
        int depth = 2 + static_cast<int>(pop.size()) % std::max(1, c.init_depth - 1);
        Node n = build.tree(depth, pop.size() % 2 == 0);
        if (fits(n) && seen.insert(to_string(n)).second) pop.push_back(std::move(n));
    }
    while (static_cast<int>(pop.size()) < c.population) pop.push_back(build.leaf());

    std::vector<double> fit(pop.size());
    for (int gen = 0; gen < c.generations; ++gen) {
        // Score the formulas this search has not seen yet, in parallel.
        std::vector<std::string> keys(pop.size());
        std::vector<std::size_t> todo;
        std::unordered_set<std::string> queued;
        for (std::size_t i = 0; i < pop.size(); ++i) {
            keys[i] = to_string(pop[i]);
            if (!fitness_cache.count(keys[i]) && queued.insert(keys[i]).second) todo.push_back(i);
        }
        std::vector<Scored> scored(todo.size());
        const auto& hall = result.hall;
        parallel_for(todo.size(), c.threads, [&](std::size_t j) {
            const Node& n = pop[todo[j]];
            Scored& s = scored[j];
            s.formula = keys[todo[j]];
            Backtest b = backtest(evaluate(n, d), returns, d, c.score);
            s.train = summarize(b, c.train_begin, c.train_end);
            if (s.train.n < 2 || s.train.sd <= 0) return;  // degenerate: never trades
            s.raw = s.train.sharpe - c.parsimony * static_cast<double>(n.size());
            s.fitness = s.raw;
            double worst = 0;
            for (const auto& h : hall)
                if (h.formula != s.formula && h.fitness >= s.raw)
                    worst = std::max(worst, std::fabs(series_corr(b.net, h.net, c.train_begin, c.train_end)));
            if (worst > c.max_corr && s.fitness > 0)
                s.fitness -= (worst - c.max_corr) / (1 - c.max_corr) * s.fitness;
            s.net = std::move(b.net);
        });

        // Ledger, cache and hall of fame (sequential, in population order: deterministic).
        for (auto& s : scored) {
            Trial tr;
            tr.formula = s.formula;
            tr.size = static_cast<int>(pop[todo[&s - scored.data()]].size());
            tr.generation = gen;
            bool ok = std::isfinite(s.raw);
            tr.sharpe = ok ? s.train.sharpe : NaN;
            tr.mean_ic = ok ? s.train.mean_ic : NaN;
            tr.turnover = ok ? s.train.turnover : NaN;
            result.ledger.push_back(tr);
            fitness_cache[s.formula] = s.fitness;
            if (!ok) continue;

            auto& h = result.hall;
            bool beaten = false;
            std::vector<std::size_t> twins;
            for (std::size_t i = 0; i < h.size(); ++i)
                if (std::fabs(series_corr(s.net, h[i].net, c.train_begin, c.train_end)) > c.max_corr) {
                    if (h[i].fitness >= s.raw) beaten = true;
                    twins.push_back(i);
                }
            if (beaten) continue;
            for (auto it = twins.rbegin(); it != twins.rend(); ++it) h.erase(h.begin() + static_cast<long>(*it));
            Finalist f{s.formula, s.raw, s.train, std::move(s.net)};
            h.insert(std::upper_bound(h.begin(), h.end(), f, [](const Finalist& a, const Finalist& b) {
                return a.fitness > b.fitness;
            }), std::move(f));
            if (static_cast<int>(h.size()) > c.hall_size) h.pop_back();
        }
        for (std::size_t i = 0; i < pop.size(); ++i) fit[i] = fitness_cache[keys[i]];
        if (progress) progress(gen, result);
        if (gen + 1 == c.generations) break;

        // Next generation: elites, then tournament-selected variation.
        std::vector<std::size_t> order(pop.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return fit[a] > fit[b]; });
        auto select = [&]() -> const Node& {
            std::size_t best = rng.below(pop.size());
            for (int k = 1; k < c.tournament; ++k) {
                std::size_t i = rng.below(pop.size());
                if (fit[i] > fit[best]) best = i;
            }
            return pop[best];
        };

        std::vector<Node> next;
        std::unordered_set<std::string> taken;
        for (std::size_t i = 0; i < order.size() && static_cast<int>(next.size()) < c.elite; ++i)
            if (taken.insert(keys[order[i]]).second) next.push_back(pop[order[i]]);
        while (static_cast<int>(next.size()) < c.population) {
            Node child = select();
            double r = rng.uniform();
            if (r < c.p_crossover) {
                Node parent = select();
                Node donor = *random_node(parent, rng);
                *random_node(child, rng) = std::move(donor);
            } else if ((r -= c.p_crossover) < c.p_subtree) {
                *random_node(child, rng) = build.tree(1 + static_cast<int>(rng.below(3)), false);
            } else if ((r -= c.p_subtree) < c.p_point) {
                Node* n = random_node(child, rng);
                if (n->op == Op::Field) n->field = rng.pick(c.fields);
                else if (info(n->op).window && rng.uniform() < 0.5) n->window = build.window(info(n->op));
                else build.swap_op(*n);
            } else if ((r -= c.p_point) < c.p_hoist) {
                Node sub = *random_node(child, rng);
                child = std::move(sub);
            }
            if (!fits(child)) continue;
            // Prefer unexplored formulas; repeats are allowed but rarer.
            std::string key = to_string(child);
            if (!taken.insert(key).second && rng.uniform() < 0.9) continue;
            next.push_back(std::move(child));
        }
        pop = std::move(next);
    }
    return result;
}

}  // namespace fa
