// Copyright (c) Dietmar Wolz.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory.
//
// Eigen based implementation of PGPE see
// http://mediatum.ub.tum.de/doc/1099128/631352.pdf .
// Derived from https://github.com/google/evojax/blob/main/evojax/algo/pgpe.py .
//
// Requires Eigen version >= 3.4 because new slicing capabilities are used, see
// https://eigen.tuxfamily.org/dox-devel/group__TutorialSlicingIndexing.html
// requires https://github.com/bab2min/EigenRand for random number generation.
//
// Supports only ADAM based mean/baseline update.

#include <Eigen/Core>
#include <fstream>
#include <float.h>
#include <iostream>
#include <queue>
#include <random>
#include <stdint.h>
#include <ctime>
#include <tuple>

#include "evaluator.h"
#include "pgpeoptimizer.hpp"

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
            double stdev_max_change_, double b1_, double b2_, double eps_,
            double decay_coef_) {
        runid = runid_;
        fitfun = fitfun_;
        dim = dim_;
        popsize = popsize_ > 0 ? popsize_ : 4 * dim;
        maxEvaluations = maxEvaluations_ > 0 ? maxEvaluations_ : 50000;
        stopfitness = stopfitness_;
        iterations = 0;
        bestY = DBL_MAX;
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
        int n = fitness_scores.size() / 2;
        mat fit = fitness_scores.reshaped(2, n);
        vec fit1 = fit.row(0);
        vec fit2 = fit.row(1);
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

    mat ask(const vec &stdev, const vec &center) {
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

    mat ask() {
        return ask(stdev, center);
    }

    void init() {
        popX = mat(dim, popsize);
        popY = vec(popsize);
        for (int p = 0; p < popsize; p++)
            popY[p] = DBL_MAX;
    }

    mat ask_decode() {
        mat xs = ask(stdev, center);
        for (int p = 0; p < popsize; p++)
            popX.col(p) = fitfun->decode(
                    fitfun->getClosestFeasibleNormed(xs.col(p)));
        return popX;
    }

    int tell(vec ys) {
        popY = process_scores(-ys);
        double bY = -popY.maxCoeff();
        if (bestY > bY) {
            bestY = bY;
            for (int p = 0; p < popsize; p++) {
                if (popY[p] == -bestY) {
                    bestX = popX.col(p);
                    break;
                }
            }
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

    mat getPopulation() const {
        return popX;
    }

    vec getBestX() const {
        return bestX;
    }

    double getBestValue() const {
        return bestY;
    }

    double getIterations() const {
        return iterations;
    }

    double getStop() const {
        return stop;
    }

    Fitness* getFitfun() const {
        return fitfun;
    }

    int getDim() const {
        return dim;
    }

    int getPopsize() const {
        return popsize;
    }

private:
    long runid;
    Fitness *fitfun;
    ADAM *optimizer;
    int popsize;
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

namespace {

void initialize_problem(
        int dim, const double *init, const double *lower, const double *upper,
        const double *sigma, bool &normalize, vec &guess, vec &lower_limit,
        vec &upper_limit, vec &input_sigma) {
    guess.resize(dim);
    input_sigma.resize(dim);
    for (int i = 0; i < dim; i++) {
        guess[i] = init[i];
        input_sigma[i] = sigma[i];
    }
    if (lower != NULL && upper != NULL) {
        lower_limit.resize(dim);
        upper_limit.resize(dim);
        for (int i = 0; i < dim; i++) {
            lower_limit[i] = lower[i];
            upper_limit[i] = upper[i];
        }
    } else {
        lower_limit.resize(0);
        upper_limit.resize(0);
        normalize = false;
    }
}

PgpeResult make_result(const PGPEOptimizer &opt, const Fitness &fitfun) {
    PgpeResult result;
    result.x = opt.getBestX();
    result.y = opt.getBestValue();
    result.evaluations = fitfun.evaluations();
    result.iterations = static_cast<int>(opt.getIterations());
    result.stop = static_cast<int>(opt.getStop());
    return result;
}

}  // namespace

class PgpeState::Impl {
public:
    Impl(int64_t runid, int dim, const double *init, const double *lower,
            const double *upper, const double *sigma, int popsize, int64_t seed,
            int lr_decay_steps, bool use_ranking,
            double center_learning_rate, double stdev_learning_rate,
            double stdev_max_change, double b1, double b2, double eps,
            double decay_coef, bool normalize) {
        initialize_problem(dim, init, lower, upper, sigma, normalize, guess,
                           lower_limit, upper_limit, input_sigma);
        fitfun = std::make_unique<Fitness>(
            noop_callback, noop_callback_par, dim, 1, lower_limit, upper_limit
        );
        fitfun->setNormalize(normalize);
        opt = std::make_unique<PGPEOptimizer>(
            runid, fitfun.get(), dim, seed, popsize, guess, input_sigma, 0,
            -DBL_MAX, lr_decay_steps, use_ranking, center_learning_rate,
            stdev_learning_rate, stdev_max_change, b1, b2, eps, decay_coef
        );
    }

    mat ask() {
        return opt->ask_decode();
    }

    int tell(const vec &ys) {
        vec values = ys;
        opt->tell(values);
        evaluations += static_cast<int>(values.size());
        iterations += 1;
        return static_cast<int>(opt->getStop());
    }

    mat population() const {
        return opt->getPopulation();
    }

    PgpeResult result() const {
        PgpeResult current = make_result(*opt, *fitfun);
        current.evaluations = evaluations;
        current.iterations = iterations;
        return current;
    }

    int dim() const {
        return opt->getDim();
    }

    int popsize() const {
        return opt->getPopsize();
    }

private:
    vec guess;
    vec lower_limit;
    vec upper_limit;
    vec input_sigma;
    std::unique_ptr<Fitness> fitfun;
    std::unique_ptr<PGPEOptimizer> opt;
    int evaluations = 0;
    int iterations = 0;
};

PgpeResult optimize_pgpe(
        int64_t runid, callback_parallel func_par, int dim, const double *init,
        const double *lower, const double *upper, const double *sigma,
        int maxEvals, double stopfitness, int popsize, int64_t seed,
        int lr_decay_steps, bool use_ranking, double center_learning_rate,
        double stdev_learning_rate, double stdev_max_change, double b1,
        double b2, double eps, double decay_coef, bool normalize) {

    vec guess, lower_limit, upper_limit, input_sigma;
    initialize_problem(dim, init, lower, upper, sigma, normalize, guess,
                       lower_limit, upper_limit, input_sigma);

    Fitness fitfun(noop_callback, func_par, dim, 1, lower_limit, upper_limit);
    fitfun.setNormalize(normalize);

    PGPEOptimizer opt(runid, &fitfun, dim, seed, popsize, guess, input_sigma,
            maxEvals, stopfitness, lr_decay_steps, use_ranking,
            center_learning_rate, stdev_learning_rate, stdev_max_change, b1, b2,
            eps, decay_coef);
    try {
        while (fitfun.evaluations() < maxEvals && !fitfun.terminate()
                && opt.getStop() == 0) {
            mat xs = opt.ask();
            vec ys(popsize);
            fitfun.values(xs, ys);
            opt.tell(ys);
        }
    } catch (std::exception &e) {
        cout << e.what() << endl;
    }
    return make_result(opt, fitfun);
}

PgpeState::PgpeState(
        int64_t runid, int dim, const double *init, const double *lower,
        const double *upper, const double *sigma, int popsize, int64_t seed,
        int lr_decay_steps, bool use_ranking, double center_learning_rate,
        double stdev_learning_rate, double stdev_max_change, double b1,
        double b2, double eps, double decay_coef, bool normalize)
        : impl_(std::make_unique<Impl>(runid, dim, init, lower, upper, sigma,
                  popsize, seed, lr_decay_steps, use_ranking,
                  center_learning_rate, stdev_learning_rate, stdev_max_change,
                  b1, b2, eps, decay_coef, normalize)) {
}

PgpeState::~PgpeState() = default;

mat PgpeState::ask() {
    return impl_->ask();
}

int PgpeState::tell(const vec &ys) {
    return impl_->tell(ys);
}

mat PgpeState::population() const {
    return impl_->population();
}

PgpeResult PgpeState::result() const {
    return impl_->result();
}

int PgpeState::dim() const {
    return impl_->dim();
}

int PgpeState::popsize() const {
    return impl_->popsize();
}

}  // namespace pgpe
