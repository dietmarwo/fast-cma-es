// Copyright (c) Dietmar Wolz.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory.

// Eigen based implementation of differential evolution using on the DE/best/1 strategy.
// Uses two deviations from the standard DE algorithm:
// a) temporal locality introduced in 
// https://www.researchgate.net/publication/309179699_Differential_evolution_for_protein_folding_optimization_based_on_a_three-dimensional_AB_off-lattice_model
// b) reinitialization of individuals based on their age. 
// requires https://github.com/imneme/pcg-cpp
//
// Supports parallel fitness function evaluation. 
// 
// You may keep parameters F and CR at their defaults since this implementation works well with the given settings for most problems,
// since the algorithm oscillates between different F and CR settings.
//
// For expensive objective functions (e.g. machine learning parameter optimization) use the workers
// parameter to parallelize objective function evaluation. The workers parameter is limited by the
// population size.
//
// The ints parameter is a boolean array indicating which parameters are discrete integer values. This
// parameter was introduced after observing non optimal DE-results for the ESP2 benchmark problem:
// https://github.com/AlgTUDelft/ExpensiveOptimBenchmark/blob/master/expensiveoptimbenchmark/problems/DockerCFDBenchmark.py
// If defined it causes a "special treatment" for discrete variables: They are rounded to the next integer value and
// there is an additional mutation to avoid getting stuck at local minima.

#include <Eigen/Core>
#include <iostream>
#include <float.h>
#include <stdint.h>
#include <ctime>
#include <random>
#include <queue>
#include <tuple>
#include "pcg_random.hpp"
#include "evaluator.h"

using namespace std;

namespace differential_evolution {

class DeOptimizer {

public:

    DeOptimizer(long runid_, Fitness *fitfun_, int dim_, int seed_,
            int popsize_, int maxEvaluations_, double keep_,
            double stopfitness_, double F_, double CR_, bool *isInt_) {
        // runid used to identify a specific run
        runid = runid_;
        // fitness function to minimize
        fitfun = fitfun_;
        // Number of objective variables/problem dimension
        dim = dim_;
        // Population size
        popsize = popsize_ > 0 ? popsize_ : 15 * dim;
        // maximal number of evaluations allowed.
        maxEvaluations = maxEvaluations_ > 0 ? maxEvaluations_ : 50000;
        // keep best young after each iteration.
        keep = keep_ > 0 ? keep_ : 30;
        // Limit for fitness value.
        stopfitness = stopfitness_;
        F = F0 = F_ > 0 ? F_ : 0.5;
        CR = CR0 = CR_ > 0 ? CR_ : 0.9;
        // Number of iterations already performed.
        iterations = 0;
        bestY = DBL_MAX;
        // stop criteria
        stop = 0;
        pos = 0;
        //std::random_device rd;
        rs = new pcg64(seed_);
        // Indicating which parameters are discrete integer values. If defined these parameters will be
        // rounded to the next integer and some additional mutation of discrete parameters are performed.
        isInt = isInt_;
        init();
    }

    ~DeOptimizer() {
        delete rs;
    }

    double rnd01() {
        return distr_01(*rs);
    }

    int rndInt(int max) {
        return (int) (max * distr_01(*rs));
    }

    vec nextX(int p, const vec &xp, const vec &xb) {
        if (p == 0) {
            iterations++;
            CR = iterations % 2 == 0 ? 0.5 * CR0 : CR0;
            F = iterations % 2 == 0 ? 0.5 * F0 : F0;
        }
        int r1, r2;
        do {
            r1 = rndInt(popsize);
        } while (r1 == p || r1 == bestI);
        do {
            r2 = rndInt(popsize);
        } while (r2 == p || r2 == bestI || r2 == r1);
        vec x1 = popX.col(r1);
        vec x2 = popX.col(r2);
        vec x = xb + (x1 - x2) * F;
        int r = rndInt(dim);
        for (int j = 0; j < dim; j++)
            if (j != r && rnd01() > CR)
                x[j] = xp[j];
        vec nextx = fitfun->getClosestFeasible(x);
        modify(nextx);
        return nextx;
    }

    vec next_improve(const vec &xb, const vec &x, const vec &xi) {
        vec nextx = fitfun->getClosestFeasible(xb + ((x - xi) * 0.5));
        modify(nextx);
        return nextx;
    }

    void modify(vec &x) {
        if (isInt == NULL)
            return;
        double n_ints = 0;
        for (int i = 0; i < dim; i++)
            if (isInt[i]) n_ints++;
        double min_mutate = 0.5;
        double max_mutate = std::max(1.0, n_ints/20.0);
        double to_mutate = min_mutate + rnd01()*(max_mutate - min_mutate);
        for (int i = 0; i < dim; i++) {
            if (isInt[i]) {
                if (rnd01() < to_mutate/n_ints)
                    x[i] = fitfun->sample_i(i, *rs); // resample
                x[i] = std::round(x[i]);
            }
        }
    }

    vec ask(int &p) {
        // ask for one new argument vector.
        if (improvesX.empty()) {
            p = pos;
            vec x = nextX(p, popX.col(p), popX.col(bestI));
            pos = (pos + 1) % popsize;
            return x;
        } else {
            p = improvesP.front();
            vec x = improvesX.front();
            improvesP.pop();
            improvesX.pop();
            return x;
        }
    }

    int tell(double y, const vec &x, int p) {
        //tell function value for a argument list retrieved by ask_one().
        if (isfinite(y) && y < popY[p]) {
            if (iterations > 1) {
                // temporal locality
                improvesP.push(p);
                improvesX.push(next_improve(popX.col(bestI), x, popX0.col(p)));
            }
            popX0.col(p) = popX.col(p);
            popX.col(p) = x;
            popY[p] = y;
            popIter[p] = iterations;
            if (y < popY[bestI]) {
                bestI = p;
                if (y < bestY) {
                    bestY = y;
                    bestX = x;
                    if (isfinite(stopfitness) && bestY < stopfitness)
                        stop = 1;
                }
            }
        } else {
            // reinitialize individual
            if (keep * rnd01() < iterations - popIter[p]) {
                popX.col(p) = fitfun->sample(*rs);
                popY[p] = DBL_MAX;
            }
        }
        return stop;
    }

    void doOptimize() {

        // -------------------- Generation Loop --------------------------------
        for (iterations = 1; fitfun->evaluations() < maxEvaluations
        		&& !fitfun->terminate(); iterations++) {

            CR = iterations % 2 == 0 ? 0.5 * CR0 : CR0;
            F = iterations % 2 == 0 ? 0.5 * F0 : F0;

            for (int p = 0; p < popsize; p++) {
                vec xp = popX.col(p);
                vec xb = popX.col(bestI);
                int r1, r2;
                do {
                    r1 = rndInt(popsize);
                } while (r1 == p || r1 == bestI);
                do {
                    r2 = rndInt(popsize);
                } while (r2 == p || r2 == bestI || r2 == r1);
                vec x1 = popX.col(r1);
                vec x2 = popX.col(r2);
                int r = rndInt(dim);
                vec x = vec(xp);
                for (int j = 0; j < dim; j++) {
                    if (j == r || rnd01() < CR) {
                        x[j] = xb[j] + F * (x1[j] - x2[j]);
                        if (!fitfun->feasible(j, x[j]))
                            x[j] = fitfun->sample_i(j, *rs);
                    }
                }
                modify(x);
                double y = fitfun->eval(x)(0);
                if (isfinite(y) && y < popY[p]) {
                    // temporal locality
                    vec x2 = next_improve(xb, x, xp);
                    double y2 = fitfun->eval(x2)(0);
                    if (isfinite(y2) && y2 < y) {
                        y = y2;
                        x = x2;
                    }
                    popX.col(p) = x;
                    popY(p) = y;
                    popIter[p] = iterations;
                    if (y < popY[bestI]) {
                        bestI = p;
                        if (y < bestY) {
                            bestY = y;
                            bestX = x;
                            if (isfinite(stopfitness) && bestY < stopfitness) {
                                stop = 1;
                                return;
                            }
                        }
                    }
                } else {
                    // reinitialize individual
                    if (keep * rnd01() < iterations - popIter[p]) {
                        popX.col(p) = fitfun->sample(*rs);
                        popY[p] = DBL_MAX;
                    }
                }
            }
        }
    }

    void do_optimize_delayed_update(int workers) {
    	 iterations = 0;
    	 fitfun->resetEvaluations();
         workers = std::min(workers, popsize); // workers <= popsize
    	 evaluator eval(fitfun, 1, workers);
         int evals_size = popsize*10;
    	 vec evals_x[evals_size];
   	     int evals_p[evals_size];
         int cp = 0; 
         
	     // fill eval queue with initial population
    	 for (int i = 0; i < workers; i++) {
    		 int p;
    		 vec x = ask(p);
    		 eval.evaluate(x, cp);
    		 evals_x[cp] = x;
    		 evals_p[cp] = p;
             cp = (cp + 1) % evals_size;             
    	 }
    	 while (fitfun->evaluations() < maxEvaluations && !fitfun->terminate()) {
    		 vec_id* vid = eval.result();
    		 vec y = vec(vid->_v);
    		 int id = vid->_id;
    		 delete vid;
    		 vec x = evals_x[id];
             int p = evals_p[id];
    		 tell(y(0), x, p); // tell evaluated x
    		 if (fitfun->evaluations() >= maxEvaluations)
    			 break;
    		 x = ask(p);
    		 eval.evaluate(x, cp);
    		 evals_x[cp] = x;
    		 evals_p[cp] = p;
             cp = (cp + 1) % evals_size; 
    	 }
	}

    void init() {
        popX = mat(dim, popsize);
        popX0 = mat(dim, popsize);
        popY = vec(popsize);
        for (int p = 0; p < popsize; p++) {
            popX0.col(p) = popX.col(p) = fitfun->sample(*rs);
            popY[p] = DBL_MAX; // compute fitness
        }
        bestI = 0;
        bestX = popX.col(bestI);
        popIter = zeros(popsize);
    }

    vec getBestX() {
        return bestX;
    }

    double getBestValue() {
        return bestY;
    }

    double getIterations() {
        return iterations;
    }

    double getStop() {
        return stop;
    }

    Fitness* getFitfun() {
        return fitfun;
    }

    int getDim() {
        return dim;
    }

private:
    long runid;
    Fitness *fitfun;
    int popsize; // population size
    int dim;
    int maxEvaluations;
    double keep;
    double stopfitness;
    int iterations;
    double bestY;
    vec bestX;
    int bestI;
    int stop;
    double F0;
    double CR0;
    double F;
    double CR;
    pcg64 *rs;
    mat popX;
    mat popX0;
    vec popY;
    vec popIter;
    queue<vec> improvesX;
    queue<int> improvesP;
    int pos;
    bool *isInt;
};

}

using namespace differential_evolution;

extern "C" {
void optimizeDE_C(long runid, callback_type func, int dim, int seed,
        double *lower, double *upper, bool *ints,
        int maxEvals, double keep,
        double stopfitness, int popsize, double F, double CR, int workers, double* res) {
    vec lower_limit(dim), upper_limit(dim);
    bool isInt[dim];
    bool useIsInt = false;
    for (int i = 0; i < dim; i++) {
        lower_limit[i] = lower[i];
        upper_limit[i] = upper[i];
        isInt[i] = ints[i];
        useIsInt |= ints[i];
        if (isInt[i]) {
            // adjust bounds because ints are rounded
            lower_limit[i] -= .499999999;
            upper_limit[i] += .499999999;
        }
    }
    Fitness fitfun(func, dim, 1, lower_limit, upper_limit);
    DeOptimizer opt(runid, &fitfun, dim, seed, popsize, maxEvals, keep,
            stopfitness, F, CR, useIsInt ? isInt : NULL);
    try {
        if (workers <= 1)
            opt.doOptimize();
        else
            opt.do_optimize_delayed_update(workers);
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

