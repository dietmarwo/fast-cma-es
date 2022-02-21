// Copyright (c) Dietmar Wolz.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory.

#include <Eigen/Core>
#include <iostream>
#include <float.h>
#include <ctime>
#include <random>
#include "pcg_random.hpp"
#include "biteopt.h"
#include "evaluator.h"

using namespace std;

namespace biteopt {

class BiteOptimizer: public CBiteOptDeep {

public:

    BiteOptimizer(long runid_, Fitness *fitfun_, int dim_, double *init_,
            int seed_, int M_, int stall_iterations_, int maxEvaluations_, double stopfitness_) {
        // runid used to identify a specific run
        runid = runid_;
        // fitness function to minimize
        fitfun = fitfun_;
        // Number of objective variables/problem dimension
        dim = dim_;
        // Depth to use, 1 for plain CBiteOpt algorithm, >1 for CBiteOptDeep. Expected range is [1; 36].
        M = M_ > 0 ? M_ : 1;
        // terminate if stall_iterations iterations stalled, if <= 0 not used
        stall_iterations = stall_iterations_;
        // maximal number of evaluations allowed.
        maxEvaluations = maxEvaluations_ > 0 ? maxEvaluations_ : 50000;
        // Number of iterations already performed.
        // Limit for fitness value.
        stopfitness = stopfitness_;
        //std::random_device rd;
        rs = new pcg64(seed_);
        // stop criteria
        stop = 0;

        iterations = 0;
        bestY = DBL_MAX;
        rnd.init(seed_);
        updateDims(dim_, M);
        init(rnd, init_);
    }

    ~BiteOptimizer() {
        delete rs;
    }

    virtual void getMinValues(double *const p) const {
        fitfun->getMinValues(p);
    }

    virtual void getMaxValues(double *const p) const {
        fitfun->getMaxValues(p);
    }

    virtual double optcost(const double *const p) {
        return fitfun->eval(p)(0);
    }

    vec getBestX() {
        vec bestX = vec(dim);
        const double *bx = getBestParams();
        for (int i = 0; i < dim; i++)
            bestX[i] = bx[i];
        return bestX;
    }

    double getBestValue() {
        return getBestCost();
    }

    double getIterations() {
        return iterations;
    }

    double getStop() {
        return stop;
    }

    void doOptimize() {

        // -------------------- Generation Loop --------------------------------
        for (iterations = 1; fitfun->evaluations() < maxEvaluations;
                iterations++) {
            int stallCount = optimize(rnd);
            if (getBestCost() < stopfitness) {
                stop = 1;
                break;
            }
            if (stall_iterations > 0 && stallCount > stall_iterations*dim) {
                stop = 2;
                break;
            }
        }
    }

private:
    long runid;
    Fitness *fitfun;
    int M; // deepness
    int stall_iterations; // terminate if stall_iterations stalled, if <= 0 not used
    int dim;
    int maxEvaluations;
    double stopfitness;
    int iterations;
    double bestY;
    int stop;
    vec bestX;
    pcg64 *rs;
    CBiteRnd rnd;
};

}

using namespace biteopt;

extern "C" {
void optimizeBite_C(long runid, callback_type func, int dim, int seed,
        double *init, double *lower, double *upper, int maxEvals,
        double stopfitness, int M, int stall_iterations, double* res) {
    int n = dim;
    vec lower_limit(n), upper_limit(n);
    bool useLimit = false;
    for (int i = 0; i < n; i++) {
        lower_limit[i] = lower[i];
        upper_limit[i] = upper[i];
        useLimit |= (lower[i] != 0);
        useLimit |= (upper[i] != 0);
    }
    if (useLimit == false) {
        lower_limit.resize(0);
        upper_limit.resize(0);
    }
    Fitness fitfun(func, n, 1, lower_limit, upper_limit);
    BiteOptimizer opt(runid, &fitfun, dim, init, seed, M, stall_iterations, maxEvals,
            stopfitness);

    try {
        opt.doOptimize();
        vec bestX = opt.getBestX();
        double bestY = opt.getBestValue();
        for (int i = 0; i < n; i++)
            res[i] = bestX[i];
        res[n] = bestY;
        res[n + 1] = fitfun.evaluations();
        res[n + 2] = opt.getIterations();
        res[n + 3] = opt.getStop();
    } catch (std::exception &e) {
        cout << e.what() << endl;
    }
}
}

