#pragma once

#include <Eigen/Core>

#include <cstdint>
#include <memory>

namespace acmaes {

using vec = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using callback_type = bool (*)(int, const double *, double *);
using callback_parallel = void (*)(int, int, double *, double *);

struct AcmaResult {
    vec x;
    double y;
    int evaluations;
    int iterations;
    int stop;
};

AcmaResult optimize_acma(
    long runid, callback_type func, callback_parallel func_par, int dim,
    const double *init, const double *lower, const double *upper,
    const double *sigma, int max_evals, double stop_fitness,
    double stop_tol_hist_fun, int mu, int popsize, double accuracy, long seed,
    bool normalize, bool use_delayed_update, int update_gap, int workers
);

class AcmaState {
public:
    AcmaState(
        long runid, int dim, const double *init, const double *lower,
        const double *upper, const double *sigma, int max_evals,
        double stop_fitness, double stop_tol_hist_fun, int mu, int popsize,
        double accuracy, long seed, bool normalize, bool use_delayed_update,
        int update_gap
    );
    ~AcmaState();

    AcmaState(const AcmaState &) = delete;
    AcmaState &operator=(const AcmaState &) = delete;

    mat ask();
    int tell(const vec &ys);
    int tell_x(const vec &ys, const mat &xs);
    mat population() const;
    AcmaResult result() const;
    int dim() const;
    int popsize() const;
    int stop() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace acmaes
