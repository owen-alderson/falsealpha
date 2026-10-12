#pragma once
#include <random>

#include "falsealpha/mat.hpp"

namespace fa::test {

// Random panel with every field filled, a few NaN holes and a random universe.
inline Data random_data(std::size_t T, std::size_t N, unsigned seed, double hole_rate = 0.02) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    Data d;
    d.T = T; d.N = N;
    for (const char* f : {"close", "volume", "returns"}) {
        Mat m(T, N);
        for (auto& x : m.v) x = u(rng) < hole_rate ? NaN : g(rng);
        d.fields[f] = m;
    }
    d.universe.resize(T * N);
    for (auto& x : d.universe) x = u(rng) < 0.85;
    return d;
}

}  // namespace fa::test
