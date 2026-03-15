#pragma once

#include <Eigen/Core>

#include <cstdint>
#include <memory>

namespace pgpe {

using vec = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using callback_parallel = void (*)(int, int, double *, double *);

struct PgpeResult {
    vec x;
    double y;
    int evaluations;
    int iterations;
    int stop;
};

PgpeResult optimize_pgpe(
    int64_t runid, callback_parallel func_par, int dim, const double *init,
    const double *lower, const double *upper, const double *sigma,
    int max_evals, double stop_fitness, int popsize, int64_t seed,
    int lr_decay_steps, bool use_ranking, double center_learning_rate,
    double stdev_learning_rate, double stdev_max_change, double b1,
    double b2, double eps, double decay_coef, bool normalize
);

class PgpeState {
public:
    PgpeState(
        int64_t runid, int dim, const double *init, const double *lower,
        const double *upper, const double *sigma, int popsize, int64_t seed,
        int lr_decay_steps, bool use_ranking, double center_learning_rate,
        double stdev_learning_rate, double stdev_max_change, double b1,
        double b2, double eps, double decay_coef, bool normalize
    );
    ~PgpeState();

    PgpeState(const PgpeState &) = delete;
    PgpeState &operator=(const PgpeState &) = delete;

    mat ask();
    int tell(const vec &ys);
    mat population() const;
    PgpeResult result() const;
    int dim() const;
    int popsize() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pgpe
