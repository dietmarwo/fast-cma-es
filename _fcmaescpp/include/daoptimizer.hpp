#pragma once

#include <Eigen/Core>

namespace dual_annealing {

using callback_type = double (*)(int, const double *);
using vec = Eigen::Matrix<double, Eigen::Dynamic, 1>;

struct DaResult {
    vec x;
    double y;
    int evaluations;
    int iterations;
    int stop;
};

DaResult optimize_da(long runid, callback_type func, int dim, int seed,
        const double *init, const double *lower, const double *upper,
        int maxEvals, bool use_local_search);

}  // namespace dual_annealing
