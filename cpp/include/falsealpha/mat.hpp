#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace fa {

inline constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

// A T x N panel stored time-major: one contiguous row of N assets per day.
// Time-series ops sweep rows keeping per-asset state (vectorises across assets);
// cross-sectional ops work on one contiguous row.
struct Mat {
    std::size_t T = 0, N = 0;
    std::vector<double> v;

    Mat() = default;
    Mat(std::size_t t, std::size_t n, double fill = NaN) : T(t), N(n), v(t * n, fill) {}

    double* row(std::size_t t) { return v.data() + t * N; }
    const double* row(std::size_t t) const { return v.data() + t * N; }
    double& operator()(std::size_t t, std::size_t n) { return v[t * N + n]; }
    double operator()(std::size_t t, std::size_t n) const { return v[t * N + n]; }
};

// Everything a formula can see: named input fields plus the point-in-time
// universe (1 = tradable member on that day). Cross-sectional ops and scoring
// only look at universe members; time-series ops use the full history.
struct Data {
    std::size_t T = 0, N = 0;
    std::map<std::string, Mat> fields;
    std::vector<std::uint8_t> universe;  // T x N, time-major

    const Mat& field(const std::string& name) const {
        auto it = fields.find(name);
        if (it == fields.end()) throw std::invalid_argument("unknown field: " + name);
        return it->second;
    }
    bool in_universe(std::size_t t, std::size_t n) const { return universe[t * N + n] != 0; }
};

}  // namespace fa
