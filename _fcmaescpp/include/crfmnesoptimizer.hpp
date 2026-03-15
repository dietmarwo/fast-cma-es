#pragma once

#include <Eigen/Core>

#include <cstdint>
#include <memory>

namespace crmfnes {

using vec = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using callback_parallel = void (*)(int, int, double *, double *);

struct CrfmnesResult {
    vec x;
    double y;
    int evaluations;
    int iterations;
    int stop;
};

CrfmnesResult optimize_crfmnes(
    int64_t runid, callback_parallel func_par, int dim, const double *init,
    const double *lower, const double *upper, double sigma, int max_evals,
    double stop_fitness, int popsize, int64_t seed, double penalty_coef,
    bool use_constraint_violation, bool normalize
);

class CrfmnesState {
public:
    CrfmnesState(
        int64_t runid, int dim, const double *init, const double *lower,
        const double *upper, double sigma, int popsize, int64_t seed,
        double penalty_coef, bool use_constraint_violation, bool normalize
    );
    ~CrfmnesState();

    CrfmnesState(const CrfmnesState &) = delete;
    CrfmnesState &operator=(const CrfmnesState &) = delete;

    mat ask();
    int tell(const vec &ys);
    mat population() const;
    CrfmnesResult result() const;
    int dim() const;
    int popsize() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace crmfnes
