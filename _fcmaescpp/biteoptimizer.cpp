// Copyright (c) Dietmar Wolz.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory.

#include <Eigen/Core>
#include <iostream>
#include <float.h>
#include <ctime>
#include <random>
#include "biteopt.h"
#include "evaluator.h"

using namespace std;

namespace biteopt {

class BiteOptimizer: public CBiteOptDeep {

public:

    BiteOptimizer(long runid_, Fitness *fitfun_, int dim_, double *init_,
            int seed_, int M_, int popsize, int stallCriterion_, int maxEvaluations_, double stopfitness_) {
        // runid used to identify a specific run
        runid = runid_;
        // fitness function to minimize
        fitfun = fitfun_;
        // Number of objective variables/problem dimension
        dim = dim_;
        // Depth to use, 1 for plain CBiteOpt algorithm, >1 for CBiteOptDeep. Expected range is [1; 36].
        M = M_ > 0 ? M_ : 1;
        // terminate if stallCriterion_*128*evaluations stalled, if <= 0 not used
        stallCriterion = stallCriterion_ > 0 ? stallCriterion_ : 0;
        // maximal number of evaluations allowed.
        maxEvaluations = maxEvaluations_ > 0 ? maxEvaluations_ : 50000;
        // Number of iterations already performed.
        // Limit for fitness value.
        stopfitness = stopfitness_;
        //std::random_device rd;
        //rs = new Eigen::Rand::P8_mt19937_64(seed_);
        rs = new pcg64(seed_);
        // stop criteria
        stop = 0;

        iterations = 0;
        bestY = DBL_MAX;
        rnd.init(seed_);
        updateDims(dim_, M, popsize);
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
            if (stallCriterion > 0 && stallCount > stallCriterion*128*dim) {
                stop = 2;
                break;
            }
        }
    }

private:
    long runid;
    Fitness *fitfun;
    int M; // deepness
    int stallCriterion; // terminate if f stallCriterion*128*evaluations stalled, if <= 0 not used
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
        double stopfitness, int M, int popsize, int stall_iterations, double* res) {

	vec lower_limit(dim), upper_limit(dim);
    if (lower != NULL && upper != NULL) {
		for (int i = 0; i < dim; i++) {
			lower_limit[i] = lower[i];
			upper_limit[i] = upper[i];
		}
    } else {
        lower_limit.resize(0);
        upper_limit.resize(0);
    }

    Fitness fitfun(func, noop_callback_par,  dim, 1, lower_limit, upper_limit);
    BiteOptimizer opt(runid, &fitfun, dim, init, seed, M, popsize, stall_iterations, maxEvals,
            stopfitness);

    try {
        opt.doOptimize();
        vec bestX = opt.getBestX();
        double bestY = opt.getBestValue();
        for (int i = 0; i < dim; i++)
            res[i] = bestX[i];
        res[dim] = bestY;
        res[dim + 1] = fitfun.evaluations();
        res[dim + 2] = opt.getIterations();
        res[dim + 3] = opt.getStop();
    } catch (std::exception &e) {
        cout << e.what() << endl;
    }
}
}

