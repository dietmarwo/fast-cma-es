// Copyright (c) Dietmar Wolz.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory.
//
// Eigen based implementation of PGPE see http://mediatum.ub.tum.de/doc/1099128/631352.pdf .
// Derived from https://github.com/google/evojax/blob/main/evojax/algo/pgpe.py .
//
// Requires Eigen version >= 3.4 because new slicing capabilities are used, see
// https://eigen.tuxfamily.org/dox-devel/group__TutorialSlicingIndexing.html
// requires https://github.com/bab2min/EigenRand for random number generation.
//
// Supports only ADAM based mean/baseline update.

#include <Eigen/Core>
#include <iostream>
#include <fstream>
#include <float.h>
#include <stdint.h>
#include <ctime>
#include <random>
#include <queue>
#include <tuple>
#include "evaluator.h"

using namespace std;

namespace pgpe {

double sdev(vec v) {
    return sqrt((v.array() - v.mean()).square().sum() / (v.size() - 1));
}

class ADAM {

public:

    ADAM(const vec &x0, double b1_, double b2_, double eps_,
            double center_learning_rate_, double decay_coef_) {
        dim = x0.size();
        x = x0;
        m = zeros(dim);
        v = zeros(dim);
        b1 = b1_;
        b2 = b2_;
        eps = constant(dim, eps_);
        center_lr = center_learning_rate_;
        decay_coef = decay_coef_;
    }

    void update(int i, const vec &g) {

        m = (1 - b1) * g + b1 * m;  // First moment estimate.
        v = (1 - b2) * g.array().square().matrix() + b2 * v; // Second moment estimate.
        vec mhat = m.array()
                * constant(dim, 1. / (1. - pow(b1, i + 1))).array(); // Bias correction.
        vec vhat = v.array()
                * constant(dim, 1. / (1. - pow(b2, i + 1))).array();
        vec delta = step_size(i)
                * (mhat.array()
                        * (vhat.cwiseSqrt() + eps).cwiseInverse().array());
        x -= delta;
//        cout << i << " " << step_size(i) << " "
//                << x.mean() << " " << sdev(x) << " "
//                << m.mean() << " " << sdev(m) << " "
//                << v.mean() << " " << sdev(v) << endl;
    }

    double step_size(int i) {
        return center_lr * pow(decay_coef, i);
    }

    vec x;
    vec m;  // First moment estimate
    vec v;  // Second moment estimate.

private:
    int dim;
    double b1;
    double b2;
    vec eps;
    double center_lr;
    double decay_coef;
};

class PGPEOptimizer {

public:

    PGPEOptimizer(long runid_, Fitness *fitfun_, int dim_, int seed_,
            int popsize_, const vec &guess_, const vec &inputSigma_,
            int maxEvaluations_, double stopfitness_, int lr_decay_steps_,
            bool use_ranking_,

            double center_learning_rate_, double stdev_learning_rate_,
            double stdev_max_change_,

            double b1_, double b2_, double eps_, double decay_coef_) {
        // runid used to identify a specific run
        runid = runid_;
        // fitness function to minimize
        fitfun = fitfun_;
        // Number of objective variables/problem dimension
        dim = dim_;
        // Population size
        popsize = popsize_ > 0 ? popsize_ : 4 * dim;
        // maximal number of evaluations allowed.
        maxEvaluations = maxEvaluations_ > 0 ? maxEvaluations_ : 50000;
        // Limit for fitness value.
        stopfitness = stopfitness_;
        // Number of iterations already performed.
        iterations = 0;
        bestY = DBL_MAX;
        // stop criteria
        stop = 0;
        rs = new pcg64(seed_);
        optimizer = new ADAM(guess_, b1_, b2_, eps_, center_learning_rate_,
                decay_coef_);
        center = fitfun->encode(guess_);
        stdev = inputSigma_;
        lr_decay_steps = lr_decay_steps_;

        use_ranking = use_ranking_;
        center_learning_rate_ = abs(center_learning_rate_);
        stdev_learning_rate = abs(stdev_learning_rate_);
        stdev_max_change = abs(stdev_max_change_);
        init();
    }

    ~PGPEOptimizer() {
        delete rs;
        delete optimizer;
    }

    vec process_scores(const vec &ys) {
        //Convert fitness scores to rank if necessary.
        if (use_ranking) {
            vec ranks(ys.size());
            ivec yi = sort_index(ys);
            for (int i = 0; i < ys.size(); i++)
                ranks[yi[i]] = ((double) i) / ys.size() - 0.5;
            return ranks;
        } else
            return ys;
    }

    void compute_reinforce_update(const vec &fitness_scores,
            const mat &scaled_noises, const vec &stdev) {
        //Compute the updates for the center and the standard deviation.
        int n = fitness_scores.size() / 2;
        mat fit = fitness_scores.reshaped(2, n);
        vec fit1 = fit.row(0); // fitness for baseline + noise
        vec fit2 = fit.row(1); // fitness for baseline - noise
        mat baseline = constant(scaled_noises.rows(), scaled_noises.cols(),
                fitness_scores.mean());
        vec all_scores = fit1 - fit2;
        vec all_avg_scores = 0.5 * (fit1 + fit2);
        mat stdev_sq = stdev.array().square().matrix();
        mat total_mu =
                scaled_noises.array()
                        * all_scores.replicate(1, scaled_noises.rows()).transpose().array()
                        * 0.5;
        mat total_sigma =
                (all_avg_scores.replicate(1, scaled_noises.rows()).transpose()
                        - baseline).array()
                        * (scaled_noises.array().square().matrix()
                                - stdev_sq.replicate(1, scaled_noises.cols())).array()
                        * stdev.replicate(1, scaled_noises.cols()).array().inverse();
        grad_center = total_mu.rowwise().mean();
        grad_stdev = total_sigma.rowwise().mean();
    }

    vec update_stdev(const vec &stdev, double lr, const vec &grad,
            double max_change) {
        vec allowed_delta = stdev.cwiseAbs() * max_change;
        vec min_allowed = stdev - allowed_delta;
        vec max_allowed = stdev + allowed_delta;
        return (stdev + lr * grad).cwiseMax(min_allowed).cwiseMin(max_allowed);
    }

    mat ask(const vec &stdev, const vec &center) { // undecoded
        int n = popsize / 2;
        scaled_noises = normal(dim, n, *rs).array()
                * stdev.replicate(1, n).array();
        mat x = mat(dim, popsize);
        mat x1 = center.replicate(1, n) + scaled_noises;
        mat x2 = center.replicate(1, n) - scaled_noises;
        for (int p = 0; p < n; p++) {
            x.col(2 * p) = x1.col(p);
            x.col(2 * p + 1) = x2.col(p);
        }
        return x;
    }

    mat ask() { // undecoded
        return ask(stdev, center);
    }

    void init() {
        popX = mat(dim, popsize);
        popY = vec(popsize);
        for (int p = 0; p < popsize; p++) {
            popY[p] = DBL_MAX; // compute fitness
        }
    }

    mat ask_decode() {
        // generate popsize offspring.
        mat xs = ask(stdev, center);
        for (int p = 0; p < popsize; p++)
            popX.col(p) = fitfun->decode(
                    fitfun->getClosestFeasibleNormed(xs.col(p)));
        return popX;
    }

    int tell(vec ys) {
        popY = process_scores(-ys); // negate values since we minimize
        double bY = -popY.maxCoeff();
        if (bestY > bY) {
            bestY = bY;
            for (int p = 0; p < popsize; p++) {
                if (popY[p] == -bestY) {
                    bestX = popX.col(p);
                    break;
                }
            }
//            cout << popsize * iterations << ": " << bestY << " "
//                    << bestX.transpose() << endl;
            if (bestY < stopfitness)
                stop = 1;
        }
        compute_reinforce_update(popY, scaled_noises, stdev);
        optimizer->update(((int) iterations) / lr_decay_steps, -grad_center);
        iterations += 1;
        center = optimizer->x;
        stdev = update_stdev(stdev, stdev_learning_rate, grad_stdev,
                stdev_max_change);
        return stop;
    }

    mat getPopulation() {
        return popX;
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

    int getPopsize() {
        return popsize;
    }

private:
    long runid;
    Fitness *fitfun;
    ADAM *optimizer;
    int popsize; // population size
    int dim;
    int maxEvaluations;
    double keep;
    double stopfitness;
    int iterations;
    double bestY;
    vec bestX;
    int stop;
    pcg64 *rs;
    mat popX;
    mat scaled_noises;
    vec popY;
    vec center;
    vec stdev;
    vec grad_center;
    vec grad_stdev;
    bool use_ranking;
    int lr_decay_steps;
    double center_learning_rate;
    double stdev_learning_rate;
    double stdev_max_change;
};

}

using namespace pgpe;

extern "C" {
void optimizePGPE_C(int64_t runid, callback_parallel func_par, int dim,
        double *init, double *lower, double *upper, double *sigma, int maxEvals,
        double stopfitness, int popsize, int64_t seed, int lr_decay_steps,
        bool use_ranking, double center_learning_rate,
        double stdev_learning_rate, double stdev_max_change, double b1,
        double b2, double eps, double decay_coef, bool normalize, double *res) {

    vec guess(dim), lower_limit(dim), upper_limit(dim), inputSigma(dim);
    for (int i = 0; i < dim; i++) {// guess is mandatory
    	guess[i] = init[i];
    	inputSigma[i] = sigma[i];
    }
    if (lower != NULL && upper != NULL) {
		for (int i = 0; i < dim; i++) {
	        guess[i] = init[i];
			lower_limit[i] = lower[i];
			upper_limit[i] = upper[i];
		}
    } else {
        lower_limit.resize(0);
        upper_limit.resize(0);
        normalize = false;
    }

    Fitness fitfun(noop_callback, func_par, dim, 1, lower_limit, upper_limit);
    fitfun.setNormalize(normalize);

    PGPEOptimizer opt(runid, &fitfun, dim, seed, popsize, guess, inputSigma,
            maxEvals, stopfitness, lr_decay_steps, use_ranking,
            center_learning_rate, stdev_learning_rate, stdev_max_change, b1, b2,
            eps, decay_coef);
    try {
        while(fitfun.evaluations() < maxEvals
                && !fitfun.terminate() && opt.getStop() == 0) {
            mat xs = opt.ask();
            vec ys(popsize);
            fitfun.values(xs, ys);
            opt.tell(ys);
        }
    } catch (std::exception &e) {
        cout << e.what() << endl;
    }
    vec bestX = opt.getBestX();
    double bestY = opt.getBestValue();
    for (int i = 0; i < dim; i++)
        res[i] = bestX[i];
    res[dim] = bestY;
    res[dim + 1] = fitfun.evaluations();
    res[dim + 2] = opt.getIterations();
    res[dim + 3] = opt.getStop();
}

uintptr_t initPGPE_C(int64_t runid, int dim, double *init, double *lower,
        double *upper, double *sigma, int popsize, int64_t seed,
        int lr_decay_steps, bool use_ranking, double center_learning_rate,
        double stdev_learning_rate, double stdev_max_change, double b1,
        double b2, double eps, double decay_coef, bool normalize) {

    vec guess(dim), lower_limit(dim), upper_limit(dim), inputSigma(dim);
    for (int i = 0; i < dim; i++) {// guess is mandatory
    	guess[i] = init[i];
    	inputSigma[i] = sigma[i];
    }
    if (lower != NULL && upper != NULL) {
		for (int i = 0; i < dim; i++) {
	        guess[i] = init[i];
			lower_limit[i] = lower[i];
			upper_limit[i] = upper[i];
		}
    } else {
        lower_limit.resize(0);
        upper_limit.resize(0);
        normalize = false;
    }
    Fitness *fitfun = new Fitness(noop_callback, noop_callback_par, dim, 1,
            lower_limit, upper_limit);
    fitfun->setNormalize(normalize);
    PGPEOptimizer *opt = new PGPEOptimizer(runid, fitfun, dim, seed, popsize,
            guess, inputSigma, 0, -DBL_MAX, lr_decay_steps, use_ranking,
            center_learning_rate, stdev_learning_rate, stdev_max_change, b1, b2,
            eps, decay_coef);
    return (uintptr_t) opt;
}

void destroyPGPE_C(uintptr_t ptr) {
    PGPEOptimizer *opt = (PGPEOptimizer*) ptr;
    Fitness *fitfun = opt->getFitfun();
    delete fitfun;
    delete opt;
}

void askPGPE_C(uintptr_t ptr, double *xs) {
    PGPEOptimizer *opt = (PGPEOptimizer*) ptr;
    int n = opt->getDim();
    int popsize = opt->getPopsize();
    mat popX = opt->ask_decode();
    Fitness *fitfun = opt->getFitfun();
    for (int p = 0; p < popsize; p++) {
        vec x = popX.col(p);
        for (int i = 0; i < n; i++)
            xs[p * n + i] = x[i];
    }
}

int tellPGPE_C(uintptr_t ptr, double *ys) { //, double* xs) {
    PGPEOptimizer *opt = (PGPEOptimizer*) ptr;
    int popsize = opt->getPopsize();
    vec vals(popsize);
    for (int i = 0; i < popsize; i++)
        vals[i] = ys[i];
    opt->tell(vals);
    return opt->getStop();
}

int populationPGPE_C(uintptr_t ptr, double *xs) {
    PGPEOptimizer *opt = (PGPEOptimizer*) ptr;
    int dim = opt->getDim();
    int popsize = opt->getPopsize();
    mat popX = opt->getPopulation();
    for (int p = 0; p < popsize; p++) {
        vec x = popX.col(p);
        for (int i = 0; i < dim; i++)
            x[i] = xs[p * dim + i];
    }
    return opt->getStop();
}

int resultPGPE_C(uintptr_t ptr, double* res) {
    PGPEOptimizer *opt = (PGPEOptimizer*) ptr;
    vec bestX = opt->getBestX();
    double bestY = opt->getBestValue();
    int n = bestX.size();
    for (int i = 0; i < bestX.size(); i++)
        res[i] = bestX[i];
    res[n] = bestY;
    Fitness* fitfun = opt->getFitfun();
    res[n + 1] = fitfun->evaluations();
    res[n + 2] = opt->getIterations();
    res[n + 3] = opt->getStop();
    return opt->getStop();
}
}

//bool rosen(int n, const double *x, double *y) {
//    double f = 0;
//    for (int i = 0; i < n - 1; i++)
//        f += 1e2 * (x[i] * x[i] - x[i + 1]) * (x[i] * x[i] - x[i + 1])
//                + (x[i] - 1.) * (x[i] - 1.);
//    y[0] = -f; //negate !!!!!!
//    return false; // don't terminate
//}
//
//double* test_rosen(int dim, int popsize) {
//    int n = dim;
//    int max_iteration = 10000;
//    int maxEvals = max_iteration * popsize;
//    int runid = 0;
//    int seed = 4242;
//    double stopfitness = -1E99;
//    vec lower_limit = constant(dim, -100);
//    vec upper_limit = constant(dim, 100);
//    double *res = new double[dim + 4];
//    vec guess = zeros(dim);
//    vec inputSigma = constant(dim, 0.1);
//
//    double center_learning_rate = 0.15;
//    double stdev_learning_rate = 0.1;
//    double stdev_max_change = 0.2;
//    double b1 = 0.9;
//    double b2 = 0.999;
//    double eps = 1e-8;
//    double decay_coef = 1.0;
//    int lr_decay_steps = 1000;
//    bool use_ranking = false;
//
//    Fitness fitfun(rosen, noop_callback_par, dim, 1, lower_limit, upper_limit);
//    PGPEOptimizer opt(runid, &fitfun, dim, seed, popsize, guess, inputSigma,
//            maxEvals, stopfitness, lr_decay_steps, use_ranking,
//            center_learning_rate, stdev_learning_rate, stdev_max_change, b1, b2,
//            eps, decay_coef);
//    try {
//
//        for (int i = 0; i < max_iteration && opt.getStop() == 0; i++) {
//            mat xs = opt.ask_decode();
//            vec ys(popsize);
//            for (int p = 0; p < popsize; p++) {
//                vec y = fitfun.eval(xs.col(p));
//                ys[p] = y[0];
//            }
//            opt.tell(ys);
//        }
//        vec bestX = opt.getBestX();
//        double bestY = opt.getBestValue();
//        for (int i = 0; i < n; i++)
//            res[i] = bestX[i];
//        res[n] = bestY;
//        res[n + 1] = fitfun.evaluations();
//        res[n + 2] = opt.getIterations();
//        res[n + 3] = opt.getStop();
//        return res;
//    } catch (std::exception &e) {
//        cout << e.what() << endl;
//        return res;
//    }
//}
//
//int main() {
//
//    int dim = 13;
//    int popsize = 80;
//    double *res = test_rosen(dim, popsize);
//    for (int i = 0; i < dim + 4; i++)
//        cout << res[i] << endl;
//    return 0;
//}
