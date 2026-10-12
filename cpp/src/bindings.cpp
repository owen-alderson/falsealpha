#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "falsealpha/eval.hpp"
#include "falsealpha/gp.hpp"
#include "falsealpha/score.hpp"
#include "falsealpha/stats.hpp"

namespace py = pybind11;
using namespace fa;

namespace {

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

Mat to_mat(const Array& a) {
    if (a.ndim() != 2) throw std::invalid_argument("expected a 2-D (days x assets) array");
    Mat m(a.shape(0), a.shape(1));
    std::copy(a.data(), a.data() + a.size(), m.v.begin());
    return m;
}

py::array_t<double> to_array(const Mat& m) {
    py::array_t<double> out({m.T, m.N});
    std::copy(m.v.begin(), m.v.end(), out.mutable_data());
    return out;
}

py::array_t<double> to_array(const std::vector<double>& v) {
    py::array_t<double> out(v.size());
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

std::vector<std::vector<double>> to_rows(const Array& a) {
    if (a.ndim() != 2) throw std::invalid_argument("expected a 2-D (strategies x days) array");
    std::vector<std::vector<double>> rows(a.shape(0));
    for (py::ssize_t k = 0; k < a.shape(0); ++k) rows[k].assign(a.data(k, 0), a.data(k, 0) + a.shape(1));
    return rows;
}

py::dict to_dict(const Backtest& b) {
    py::dict d;
    d["ic"] = to_array(b.ic);
    d["gross"] = to_array(b.gross);
    d["cost"] = to_array(b.cost);
    d["net"] = to_array(b.net);
    d["turnover"] = to_array(b.turnover);
    return d;
}

py::dict to_dict(const Summary& s) {
    py::dict d;
    d["mean"] = s.mean; d["sd"] = s.sd; d["sharpe"] = s.sharpe;
    d["mean_ic"] = s.mean_ic; d["ic_ir"] = s.ic_ir; d["turnover"] = s.turnover; d["n"] = s.n;
    return d;
}

ScoreConfig score_config(int delay, double cost_bps, int min_assets) {
    if (delay < 0) throw std::invalid_argument("delay must be >= 0");
    ScoreConfig c;
    c.delay = delay; c.cost_bps = cost_bps; c.min_assets = min_assets;
    return c;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "falsealpha C++ core: formula evaluation, backtests, GP search and overfitting statistics";
    py::register_exception<std::invalid_argument>(m, "FormulaError", PyExc_ValueError);

    py::class_<Data>(m, "Data", "Panel of named (days x assets) fields plus a boolean universe mask.")
        .def(py::init([](const std::map<std::string, Array>& fields,
                         py::array_t<bool, py::array::c_style | py::array::forcecast> universe) {
                 if (universe.ndim() != 2) throw std::invalid_argument("universe must be 2-D");
                 Data d;
                 d.T = universe.shape(0); d.N = universe.shape(1);
                 d.universe.assign(universe.data(), universe.data() + universe.size());
                 for (const auto& [name, a] : fields) {
                     if (!valid_field_name(name))
                         throw std::invalid_argument("field name '" + name +
                                                     "' must be an identifier that is not an operator name");
                     Mat mat = to_mat(a);
                     if (mat.T != d.T || mat.N != d.N)
                         throw std::invalid_argument("field '" + name + "' shape does not match the universe");
                     d.fields.emplace(name, std::move(mat));
                 }
                 return d;
             }),
             py::arg("fields"), py::arg("universe"))
        .def_property_readonly("shape", [](const Data& d) { return py::make_tuple(d.T, d.N); })
        .def_property_readonly("fields", [](const Data& d) {
            std::vector<std::string> names;
            for (const auto& kv : d.fields) names.push_back(kv.first);
            return names;
        });

    m.def("canonical", [](const std::string& f) { return to_string(parse(f)); }, py::arg("formula"),
          "Parse a formula and return its canonical text (raises ValueError if invalid).");
    m.def("valid_field_name", &valid_field_name, py::arg("name"));
    m.def("formula_size", [](const std::string& f) { return parse(f).size(); }, py::arg("formula"));
    m.def("operators", [] {
        py::list out;
        for (const auto& o : op_table()) {
            if (o.kind == Kind::Leaf) continue;
            static const char* kinds[] = {"leaf", "elementwise", "cross-sectional", "time-series"};
            out.append(py::dict(py::arg("name") = o.name, py::arg("arity") = o.arity, py::arg("window") = o.window,
                                py::arg("min_window") = o.min_window, py::arg("kind") = kinds[static_cast<int>(o.kind)]));
        }
        return out;
    });

    m.def("evaluate", [](const std::string& f, const Data& d) {
        Node n = parse(f);
        Mat out;
        { py::gil_scoped_release release; out = evaluate(n, d); }
        return to_array(out);
    }, py::arg("formula"), py::arg("data"));

    m.def("backtest", [](const Array& signal, const std::string& returns, const Data& d, int delay, double cost_bps,
                         int min_assets) {
        Mat s = to_mat(signal);
        if (s.T != d.T || s.N != d.N) throw std::invalid_argument("signal shape does not match the data");
        Backtest b;
        { py::gil_scoped_release release; b = backtest(s, d.field(returns), d, score_config(delay, cost_bps, min_assets)); }
        return to_dict(b);
    }, py::arg("signal"), py::arg("returns"), py::arg("data"), py::arg("delay") = 1, py::arg("cost_bps") = 10.0,
       py::arg("min_assets") = 5);

    m.def("backtest_many", [](const std::vector<std::string>& formulas, const std::string& returns, const Data& d,
                              int delay, double cost_bps, int min_assets, int threads) {
        std::vector<Backtest> out;
        ScoreConfig c = score_config(delay, cost_bps, min_assets);
        { py::gil_scoped_release release; out = backtest_many(formulas, d, d.field(returns), c, threads); }
        py::list l;
        for (const auto& b : out) l.append(to_dict(b));
        return l;
    }, py::arg("formulas"), py::arg("returns"), py::arg("data"), py::arg("delay") = 1, py::arg("cost_bps") = 10.0,
       py::arg("min_assets") = 5, py::arg("threads") = 0);

    m.def("summarize", [](py::dict bt, std::size_t begin, std::size_t end) {
        Backtest b;
        auto vec = [&](const char* k) { auto a = bt[k].cast<Array>(); return std::vector<double>(a.data(), a.data() + a.size()); };
        b.net = vec("net"); b.ic = vec("ic"); b.turnover = vec("turnover");
        return to_dict(summarize(b, begin, end));
    }, py::arg("backtest"), py::arg("begin"), py::arg("end"));

    m.def("search", [](const Data& d, const std::string& returns, const std::vector<std::string>& fields,
                       std::size_t train_begin, std::size_t train_end, int population, int generations,
                       int max_size, int max_depth, std::vector<int> windows, double parsimony, double max_corr,
                       int hall_size, int delay, double cost_bps, int min_assets, std::uint64_t seed, int threads,
                       std::function<void(int, std::size_t, double, std::string)> progress) {
        SearchConfig c;
        c.fields = fields; c.train_begin = train_begin; c.train_end = train_end;
        c.population = population; c.generations = generations; c.max_size = max_size; c.max_depth = max_depth;
        c.windows = std::move(windows); c.parsimony = parsimony; c.max_corr = max_corr; c.hall_size = hall_size;
        c.score = score_config(delay, cost_bps, min_assets);
        c.seed = seed; c.threads = threads;
        Progress cb;
        if (progress)
            cb = [&](int gen, const SearchResult& r) {
                py::gil_scoped_acquire acquire;
                double best = r.hall.empty() ? NaN : r.hall.front().train.sharpe;
                progress(gen, r.ledger.size(), best, r.hall.empty() ? "" : r.hall.front().formula);
            };
        SearchResult r;
        { py::gil_scoped_release release; r = search(d, d.field(returns), c, cb); }

        py::dict ledger;
        std::vector<std::string> formula;
        std::vector<double> sharpe, ic, turnover;
        std::vector<int> size, gen;
        for (const auto& t : r.ledger) {
            formula.push_back(t.formula); sharpe.push_back(t.sharpe); ic.push_back(t.mean_ic);
            turnover.push_back(t.turnover); size.push_back(t.size); gen.push_back(t.generation);
        }
        ledger["formula"] = formula; ledger["sharpe"] = to_array(sharpe); ledger["mean_ic"] = to_array(ic);
        ledger["turnover"] = to_array(turnover); ledger["size"] = size; ledger["generation"] = gen;
        py::list hall;
        for (const auto& f : r.hall)
            hall.append(py::dict(py::arg("formula") = f.formula, py::arg("fitness") = f.fitness,
                                 py::arg("train") = to_dict(f.train), py::arg("net") = to_array(f.net)));
        return py::dict(py::arg("ledger") = ledger, py::arg("hall") = hall);
    }, py::arg("data"), py::arg("returns"), py::arg("fields"), py::arg("train_begin"), py::arg("train_end"),
       py::arg("population") = 400, py::arg("generations") = 25, py::arg("max_size") = 20, py::arg("max_depth") = 6,
       py::arg("windows") = std::vector<int>{3, 5, 10, 20, 40}, py::arg("parsimony") = 0.0005,
       py::arg("max_corr") = 0.7, py::arg("hall_size") = 50, py::arg("delay") = 1, py::arg("cost_bps") = 10.0,
       py::arg("min_assets") = 5, py::arg("seed") = 1, py::arg("threads") = 0, py::arg("progress") = nullptr);

    m.def("norm_ppf", &norm_ppf);
    m.def("expected_max_sharpe", &expected_max_sharpe, py::arg("var_sr"), py::arg("n_trials"));
    m.def("probabilistic_sharpe", &probabilistic_sharpe, py::arg("sr"), py::arg("sr0"), py::arg("T"),
          py::arg("skew"), py::arg("kurt"));
    m.def("moments", [](const Array& x) {
        Moments mo = moments(std::vector<double>(x.data(), x.data() + x.size()));
        return py::dict(py::arg("mean") = mo.mean, py::arg("sd") = mo.sd, py::arg("skew") = mo.skew,
                        py::arg("kurt") = mo.kurt, py::arg("n") = mo.n);
    }, py::arg("x"));
    m.def("pbo", [](const Array& returns, int S) {
        auto rows = to_rows(returns);
        Pbo p;
        { py::gil_scoped_release release; p = pbo_cscv(rows, S); }
        return py::make_tuple(p.pbo, to_array(p.logits));
    }, py::arg("returns"), py::arg("S") = 16, "PBO via CSCV on a (strategies x days) matrix -> (pbo, logits).");
    m.def("snoop_tests", [](const Array& excess, int bootstraps, double mean_block, std::uint64_t seed) {
        auto rows = to_rows(excess);
        SnoopTest t;
        { py::gil_scoped_release release; t = snoop_tests(rows, bootstraps, mean_block, seed); }
        return py::dict(py::arg("reality_check_p") = t.reality_check_p, py::arg("spa_p") = t.spa_p,
                        py::arg("best") = t.best);
    }, py::arg("excess"), py::arg("bootstraps") = 1000, py::arg("mean_block") = 10.0, py::arg("seed") = 1);
}
