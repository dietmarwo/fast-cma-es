#pragma once

#include <Eigen/Core>

#include <cstdint>
#include <memory>

#include "evaluator.h"

namespace biteopt {

using vec = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using callback_type = bool (*)(int, const double *, double *);

struct BiteResult {
    vec x;
    double y;
    int evaluations;
    int iterations;
    int stop;
};

BiteResult optimize_bite(long runid, callback_type func, int dim, int seed,
        const double *init, const double *lower, const double *upper,
        int maxEvals, double stopfitness, int M, int popsize,
        int stall_iterations);

class BiteState {
public:
    BiteState(
        long runid, int dim, const double *init, const double *lower,
        const double *upper, int seed, int M, int popsize, int batch_size,
        int max_evals, double stop_fitness, int stall_criterion
    );
    ~BiteState();

    BiteState(const BiteState &) = delete;
    BiteState &operator=(const BiteState &) = delete;

    mat ask();
    int tell(const vec &ys);
    BiteResult result() const;
    int dim() const;
    int popsize() const;
    int population_size() const;
    int current_batch_size() const;
    int stop() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace biteopt
