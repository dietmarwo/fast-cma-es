#pragma once

#include <Eigen/Core>

#include <cstdint>
#include <memory>

namespace differential_evolution {

using vec = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using callback_type = bool (*)(int, const double *, double *);

struct DeResult {
    vec x;
    double y;
    int evaluations;
    int iterations;
    int stop;
};

DeResult optimize_de(
    long runid, callback_type func, int dim, int seed, const double *lower,
    const double *upper, const double *init, const double *sigma,
    double min_sigma, bool *ints, int max_evals, double keep,
    double stop_fitness, int popsize, double F, double CR,
    double min_mutate, double max_mutate, int workers
);

class DeState {
public:
    DeState(
        long runid, int dim, int seed, const double *lower, const double *upper,
        const double *init, const double *sigma, double min_sigma, bool *ints,
        double keep, int popsize, double F, double CR, double min_mutate,
        double max_mutate
    );
    ~DeState();

    DeState(const DeState &) = delete;
    DeState &operator=(const DeState &) = delete;

    mat ask();
    int tell(const vec &ys);
    mat population() const;
    DeResult result() const;
    int dim() const;
    int popsize() const;
    int stop() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace differential_evolution
