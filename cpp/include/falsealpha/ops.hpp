#pragma once
#include <vector>

#include "falsealpha/expr.hpp"
#include "falsealpha/mat.hpp"

namespace fa {

// NaN rules (shared with the Python reference implementation):
// - any non-finite value in a time-series window makes that output NaN, so a
//   window must be full of real data;
// - cross-sectional ops only rank/average universe members with finite values,
//   and return NaN for everyone else;
// - a variance at or below 1e-20 of the raw mean square counts as zero (ts_std 0,
//   ts_zscore / ts_corr NaN).
// 0-based ranks of xs; ties share their average rank. The one ranking rule
// used by the rank() operator, backtest weights and the IC.
std::vector<double> average_ranks(const std::vector<double>& xs);

Mat apply_elem(Op op, const Mat& a, const Mat* b);
Mat apply_cross(Op op, const Mat& a, const Data& d);
Mat apply_series(Op op, const Mat& a, const Mat* b, int window);
Mat broadcast(double value, std::size_t T, std::size_t N);

}  // namespace fa
