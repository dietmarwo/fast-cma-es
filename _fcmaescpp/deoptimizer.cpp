// Copyright (c) Dietmar Wolz.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory.

// Eigen based implementation of differential evolution using on the DE/best/1 strategy.
// Uses two deviations from the standard DE algorithm:
// a) temporal locality introduced in
// https://www.researchgate.net/publication/309179699_Differential_evolution_for_protein_folding_optimization_based_on_a_three-dimensional_AB_off-lattice_model
// b) reinitialization of individuals based on their age.
//
// Requires Eigen version >= 3.4 because new slicing capabilities are used, see
// https://eigen.tuxfamily.org/dox-devel/group__TutorialSlicingIndexing.html
// requires https://github.com/bab2min/EigenRand for random number generation.
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
#include <stdexcept>
#include "deoptimizer.hpp"
#include "evaluator.h"

using namespace std;

namespace differential_evolution {

class DeOptimizer {

public:

    DeOptimizer(long runid_, Fitness *fitfun_, int dim_, int seed_,
            int popsize_, int maxEvaluations_, double keep_,
            double stopfitness_, double F_, double CR_,
            double min_mutate_, double max_mutate_, bool *isInt_,
            const vec &guess_, const vec &inputSigma_, double minSigma_) {
        // runid used to identify a specific run
        runid = runid_;
        // fitness function to minimize
        fitfun = fitfun_;
        // Number of objective variables/problem dimension
        dim = dim_;
        // Population size
        popsize = popsize_ > 0 ? popsize_ : 15 * dim;
        if (popsize < 4)
            throw std::invalid_argument("DE requires at least four population members");
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
        rs = new pcg64(seed_);
        // Indicating which parameters are discrete integer values. If defined these parameters will be
        // rounded to the next integer and some additional mutation of discrete parameters are performed.
        isInt = isInt_;
        if (isInt)
            for (int i = 0; i < dim; ++i)
                if (isInt[i])
                    integer_indices.push_back(i);
        // DE population update parameter used in connection with isInt. Determines
        // the mutation rate for discrete parameters.
        min_mutate = min_mutate_ > 0 ? min_mutate_ : 0.1;
        max_mutate = max_mutate_ > 0 ? max_mutate_ : 0.5;

        useNormal = guess_.size() > 0;
        mean = guess_;
        sigma = inputSigma_;
        minSigmaVal = minSigma_;
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

    vec sample() {
        if (useNormal)
            return fitfun->getClosestFeasible(mean + (normalVec(dim, *rs, gauss_01).array() * sigma.array()).matrix());
        else
            return fitfun->sample(*rs);
    }

    double sample_i(int i) {
        if (useNormal)
            return fitfun->getClosestFeasible_i(i, normreal(*rs, gauss_01, mean[i], sigma[i]));
        else
            return fitfun->sample_i(i, *rs);
    }

    void update_mean() {
        if (useNormal) {
            meanHist.col(meanHistIndex) = popX.col(bestI);
            meanHistIndex = (meanHistIndex + 1) % meanHist.cols();
            vec delta = meanHist.rowwise().maxCoeff() - meanHist.rowwise().minCoeff();
            vec sigma_new = delta.cwiseMin(maxSigma).cwiseMax(minSigma);
            sigma = sigma_new.mean() > sigma.mean() ? sigma_new :  0.9 * sigma + 0.1 * sigma_new;
            mean = 0.9 * mean + 0.1 * popX.col(bestI);
        }
    }

    vec nextX(int p) {
        vec out(dim);
        nextXInto(p, out);
        return out;
    }

    void nextXInto(int p, Eigen::Ref<vec> nextx) {
        const auto xp = popX.col(p);
        const auto xb = popX.col(bestI);
        if (p == 0) {
            iterations++;
            CR = iterations % 2 == 0 ? 0.5 * CR0 : CR0;
            F = iterations % 2 == 0 ? 0.5 * F0 : F0;
            if (iterations > 2)
                update_mean();
        }
        int r1, r2;
        do {
            r1 = rndInt(popsize);
        } while (r1 == p || r1 == bestI);
        do {
            r2 = rndInt(popsize);
        } while (r2 == p || r2 == bestI || r2 == r1);
        const auto x1 = popX.col(r1);
        const auto x2 = popX.col(r2);
        const int r = rndInt(dim);
        for (int j = 0; j < dim; ++j) {
            const double proposed = (j != r && rnd01() > CR) ?
                xp[j] : xb[j] + (x1[j] - x2[j]) * F;
            nextx[j] = fitfun->getClosestFeasible_i(j, proposed);
        }
        modify(nextx);
    }

    vec next_improve(const vec &xb, const vec &x, const vec &xi) {
        vec nextx = fitfun->getClosestFeasible(xb + ((x - xi) * F0));
        modify(nextx);
        return nextx;
    }

    void modify(Eigen::Ref<vec> x) {
        if (isInt == NULL)
            return;
        const double to_mutate = min_mutate + rnd01() * (max_mutate - min_mutate);
        if (integer_indices.empty())
            return;
        const double probability = to_mutate / integer_indices.size();
        for (int i : integer_indices)
            if (rnd01() < probability)
                x[i] = (int)sample_i(i);
    }

    vec ask(int &p) {
        // ask for one new argument vector.
        if (improves.empty()) {
            p = pos;
            vec x = nextX(p);
            pos = (pos + 1) % popsize;
            return x;
        } else {
            auto item = std::move(improves.front());
            improves.pop();
            p = item.first;
            return std::move(item.second);
        }
    }

    void observe(double y, const Eigen::Ref<const vec> &x) {
        if (isfinite(y) && (!hasBest || y < bestY)) {
            hasBest = true;
            bestY = y;
            bestX = x;
            if (isfinite(stopfitness) && bestY < stopfitness)
                stop = 1;
        }
    }

    int tell(double y, const vec &x, int p) {
        observe(y, x);
        //tell function value for a argument list retrieved by ask_one().
        if (isfinite(y) && y < popY[p]) {
            if (iterations > 1) {
                // temporal locality
                improves.emplace(p, next_improve(popX.col(bestI), x, popX0.col(p)));
            }
            popX0.col(p) = popX.col(p);
            popX.col(p) = x;
            popY[p] = y;
            popIter[p] = iterations;
            if (y < popY[bestI])
                bestI = p;
        } else {
            // reinitialize individual
            if (keep * rnd01() < iterations - popIter[p]) {
                popX.col(p) = sample();
                popY[p] = DBL_MAX;
            }
        }
        return stop;
    }

    mat askAll() {
       for (int i = 0; i < popsize;) {
           if (improves.empty()) {
               askedP[i] = pos;
               nextXInto(pos, askedX.col(i));
               pos = (pos + 1) % popsize;
           } else {
               int p;
               askedX.col(i) = ask(p);
               askedP[i] = p;
           }
           i++;
       }
       return askedX;
    }

    int tellAll(vec &ys) {
       for (int i = 0; i < popsize; i++) {
           tell(ys[i], askedX.col(i), askedP[i]);
       }
       //std::cout << fitfun->evaluations() << " y " << ys.transpose() << std::endl;
       return stop;
    }

    void doOptimize(bool strict_budget) {

        // -------------------- Generation Loop --------------------------------
        for (iterations = 1; fitfun->evaluations() < maxEvaluations
                && !fitfun->terminate(); iterations++) {

            if (iterations > 2)
                update_mean();

            CR = iterations % 2 == 0 ? 0.5 * CR0 : CR0;
            F = iterations % 2 == 0 ? 0.5 * F0 : F0;

            for (int p = 0; p < popsize && !fitfun->terminate(); p++) {
                if (strict_budget && fitfun->evaluations() >= maxEvaluations)
                    return;
                const auto xp = popX.col(p);
                const auto xb = popX.col(bestI);
                int r1, r2;
                do {
                    r1 = rndInt(popsize);
                } while (r1 == p || r1 == bestI);
                do {
                    r2 = rndInt(popsize);
                } while (r2 == p || r2 == bestI || r2 == r1);
                const auto x1 = popX.col(r1);
                const auto x2 = popX.col(r2);
                int r = rndInt(dim);
                auto &x = trial_x;
                x = xp;
                for (int j = 0; j < dim; j++) {
                    if (j == r || rnd01() < CR) {
                        x[j] = xb[j] + F * (x1[j] - x2[j]);
                        if (!fitfun->feasible(j, x[j]))
                            x[j] = sample_i(j);
                    }
                }
                modify(x);
                double y;
                fitfun->evaluateScalar(x.data(), y);
                if (isfinite(y) && y < popY[p]) {
                    // temporal locality
                    if (!fitfun->terminate() &&
                            (!strict_budget || fitfun->evaluations() < maxEvaluations)) {
                        auto &trial = second_x;
                        trial = fitfun->getClosestFeasible(xb + ((x - xp) * F0));
                        modify(trial);
                        double trial_y;
                        fitfun->evaluateScalar(trial.data(), trial_y);
                        if (isfinite(trial_y) && trial_y < y) {
                            y = trial_y;
                            x.swap(trial);
                        }
                    }
                    popX.col(p) = x;
                    popY(p) = y;
                    popIter[p] = iterations;
                    if (y < popY[bestI])
                        bestI = p;
                    observe(y, x);
                    if (stop != 0)
                        return;
                } else {
                    observe(y, x);
                    if (stop != 0)
                        return;
                    // reinitialize individual
                    if (keep * rnd01() < iterations - popIter[p]) {
                        popX.col(p) = sample();
                        popY[p] = DBL_MAX;
                    }
                }
            }
        }
    }

    void do_optimize_delayed_update(int workers) {
        iterations = 0;
        fitfun->resetEvaluations();
        const int window = std::min({workers, popsize, maxEvaluations});
        ScalarEvaluator evaluator(*fitfun, window);
        int submitted = 0;
        int pending = 0;
        auto submit = [&](int id) {
            int target;
            vec x = ask(target);
            evaluator.submit(id, std::move(x), target);
            ++submitted;
            ++pending;
        };
        for (int id = 0; id < window; ++id)
            submit(id);

        while (pending > 0) {
            const int id = evaluator.take();
            const auto &finished = evaluator.slot(id);
            --pending;
            if (finished.evaluated) {
                if (stop == 0)
                    tell(finished.y, finished.x, finished.target);
                else
                    observe(finished.y, finished.x);
            }
            if (stop == 0 && !fitfun->terminate() && submitted < maxEvaluations)
                submit(id);
        }
    }

    void init() {
        popX = mat(dim, popsize);
        popX0 = mat(dim, popsize);
        popY = vec(popsize);
        meanHist = mean.replicate(1,10);
        meanHistIndex = 0;
        maxSigma = sigma / (.1 + minSigmaVal);
        minSigma = minSigmaVal * sigma;
        for (int p = 0; p < popsize; p++) {
            popX0.col(p) = popX.col(p) = sample();
            popY[p] = DBL_MAX;
        }
        bestI = 0;
        bestX = popX.col(bestI);
        popIter = zeros(popsize);
        askedX = mat(dim, popsize);
        askedP = ivec(popsize);
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

    Fitness* getFitfun() const {
        return fitfun;
    }

    int getDim() const {
        return dim;
    }

    mat getPopulation() const {
         return popX;
    }

    int getStop() const {
        return stop;
    }

    int getPopsize() const {
        return popsize;
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
    std::vector<int> integer_indices;
    vec trial_x, second_x;
    bool hasBest = false;
    vec bestX;
    int bestI;
    int stop;
    double F0;
    double CR0;
    double F;
    double CR;
    pcg64 *rs;
    // normal distribution of this instance (keeps its own cached value)
    std::normal_distribution<> gauss_01;
    mat popX;
    mat popX0;
    mat askedX;
    ivec askedP;
    vec popY;
    vec popIter;
    std::queue<std::pair<int, vec>> improves;
    int pos;
    double min_mutate;
    double max_mutate;
    bool *isInt;

    bool useNormal;
    vec sigma;
    vec mean;
    vec maxSigma;
    vec minSigma;
    double minSigmaVal;
    mat meanHist;
    int meanHistIndex;
};

}  // namespace differential_evolution

namespace differential_evolution {

namespace {

void initialize_problem(
        int dim, const double *init, const double *lower, const double *upper,
        const double *sigma, vec &guess, vec &lower_limit, vec &upper_limit,
        vec &input_sigma, double &min_sigma) {
    if (init != nullptr && sigma != nullptr) {
        guess.resize(dim);
        input_sigma.resize(dim);
        for (int i = 0; i < dim; i++) {
            guess[i] = init[i];
            input_sigma[i] = sigma[i];
        }
    } else {
        guess.resize(0);
        input_sigma.resize(0);
        min_sigma = 0;
    }
    if (lower != nullptr && upper != nullptr) {
        lower_limit.resize(dim);
        upper_limit.resize(dim);
        for (int i = 0; i < dim; i++) {
            lower_limit[i] = lower[i];
            upper_limit[i] = upper[i];
        }
    } else {
        lower_limit.resize(0);
        upper_limit.resize(0);
    }
}

DeResult make_result(const DeOptimizer &opt, const Fitness &fitfun) {
    DeResult result;
    result.x = opt.getBestX();
    result.y = opt.getBestValue();
    result.evaluations = fitfun.evaluations();
    result.iterations = static_cast<int>(opt.getIterations());
    result.stop = opt.getStop();
    return result;
}

}  // namespace

class DeState::Impl {
public:
    Impl(
            long runid, int dim, int seed, const double *lower,
            const double *upper, const double *init, const double *sigma,
            double min_sigma, bool *ints, double keep, int popsize, double F,
            double CR, double min_mutate, double max_mutate) {
        initialize_problem(dim, init, lower, upper, sigma, guess, lower_limit,
                           upper_limit, input_sigma, min_sigma);
        if (ints != nullptr) {
            ints_storage = std::make_unique<bool[]>(dim);
            for (int i = 0; i < dim; i++)
                ints_storage[i] = ints[i];
        }
        fitfun = std::make_unique<Fitness>(
            noop_callback, noop_callback_par, dim, 1, lower_limit, upper_limit
        );
        opt = std::make_unique<DeOptimizer>(
            runid, fitfun.get(), dim, seed, popsize, 0, keep, -DBL_MAX, F, CR,
            min_mutate, max_mutate, ints_storage.get(), guess, input_sigma,
            min_sigma
        );
    }

    mat ask() {
        return opt->askAll();
    }

    int tell(const vec &ys) {
        vec values = ys;
        opt->tellAll(values);
        evaluations += static_cast<int>(values.size());
        return opt->getStop();
    }

    mat population() const {
        return opt->getPopulation();
    }

    DeResult result() const {
        DeResult current = make_result(*opt, *fitfun);
        current.evaluations = evaluations;
        return current;
    }

    int dim() const {
        return opt->getDim();
    }

    int popsize() const {
        return opt->getPopsize();
    }

    int stop() const {
        return opt->getStop();
    }

private:
    vec guess;
    vec lower_limit;
    vec upper_limit;
    vec input_sigma;
    std::unique_ptr<bool[]> ints_storage;
    std::unique_ptr<Fitness> fitfun;
    std::unique_ptr<DeOptimizer> opt;
    int evaluations = 0;
};

DeResult optimize_de(
        long runid, callback_type func, int dim, int seed, const double *lower,
        const double *upper, const double *init, const double *sigma,
        double min_sigma, bool *ints, int max_evals, double keep,
        double stopfitness, int popsize, double F, double CR,
        double min_mutate, double max_mutate, int workers, bool strict_budget) {
    vec guess, lower_limit, upper_limit, input_sigma;
    initialize_problem(dim, init, lower, upper, sigma, guess, lower_limit,
                       upper_limit, input_sigma, min_sigma);

    Fitness fitfun(func, noop_callback_par, dim, 1, lower_limit, upper_limit);
    DeOptimizer opt(runid, &fitfun, dim, seed, popsize, max_evals, keep,
            stopfitness, F, CR, min_mutate, max_mutate, ints, guess,
            input_sigma, min_sigma);
    try {
        if (workers <= 1)
            opt.doOptimize(strict_budget);
        else
            opt.do_optimize_delayed_update(workers);
    } catch (std::exception &e) {
        cout << e.what() << endl;
    }
    return make_result(opt, fitfun);
}

DeState::DeState(
        long runid, int dim, int seed, const double *lower,
        const double *upper, const double *init, const double *sigma,
        double min_sigma, bool *ints, double keep, int popsize, double F,
        double CR, double min_mutate, double max_mutate)
        : impl_(std::make_unique<Impl>(runid, dim, seed, lower, upper, init,
                  sigma, min_sigma, ints, keep, popsize, F, CR, min_mutate,
                  max_mutate)) {
}

DeState::~DeState() = default;

mat DeState::ask() {
    return impl_->ask();
}

int DeState::tell(const vec &ys) {
    return impl_->tell(ys);
}

mat DeState::population() const {
    return impl_->population();
}

DeResult DeState::result() const {
    return impl_->result();
}

int DeState::dim() const {
    return impl_->dim();
}

int DeState::popsize() const {
    return impl_->popsize();
}

int DeState::stop() const {
    return impl_->stop();
}

}  // namespace differential_evolution
