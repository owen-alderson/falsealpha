#pragma once
#include "falsealpha/expr.hpp"
#include "falsealpha/mat.hpp"

namespace fa {

// Evaluates a formula over the whole panel. Repeated subtrees are computed once.
// The result is masked to the universe: NaN wherever the asset is not a member.
Mat evaluate(const Node& n, const Data& d);

// Throws std::invalid_argument if the formula uses a field the data lacks.
void check_fields(const Node& n, const Data& d);

}  // namespace fa
