#pragma once

#include "evaluator.h"

namespace biteopt {

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

}  // namespace biteopt
