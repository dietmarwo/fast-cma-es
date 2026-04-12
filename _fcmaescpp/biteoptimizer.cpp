// Copyright (c) Dietmar Wolz.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory.

#include <Eigen/Core>

#include <algorithm>
#include <cfloat>
#include <memory>
#include <stdexcept>
#include <vector>

#include "biteoptimizer.hpp"
#include "biteopt_asktell.h"
#include "evaluator.h"

using namespace std;

namespace biteopt {

namespace {

void initialize_problem(
        int dim, const double *init, const double *lower, const double *upper,
        vec &guess, vec &lower_limit, vec &upper_limit) {
    if (init != nullptr) {
        guess.resize(dim);
        for (int i = 0; i < dim; i++)
            guess[i] = init[i];
    } else {
        guess.resize(0);
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

vec copy_best_params(const double *values, int dim) {
    vec best_x(dim);
    for (int i = 0; i < dim; i++)
        best_x[i] = values[i];
    return best_x;
}

}  // namespace

class BiteOptimizer : public CBiteOptDeepAT {

public:
    BiteOptimizer(
            long runid_, Fitness *fitfun_, int dim_, const double *init_,
            int seed_, int M_, int popsize_, int batchSize_,
            int stallCriterion_, int maxEvaluations_, double stopfitness_)
            : runid(runid_), fitfun(fitfun_), M(M_ > 0 ? M_ : 1),
              stallCriterion(stallCriterion_ > 0 ? stallCriterion_ : 0),
              dim(dim_),
              maxEvaluations(maxEvaluations_ > 0 ? maxEvaluations_ : 50000),
              stopfitness(stopfitness_),
              batchSize(batchSize_ > 0 ? batchSize_ : 1),
              populationSize(popsize_),
              lowerValues(dim_),
              upperValues(dim_),
              iterations(0),
              evaluations(0),
              stop(0),
              currentBatchSize(0) {
        rnd.init(seed_);
        updateDims(dim, M, popsize_);
        fitfun->getMinValues(lowerValues.data());
        fitfun->getMaxValues(upperValues.data());
        init(rnd, lowerValues.data(), upperValues.data(), init_);
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

    mat ask() {
        if (currentBatchSize != 0)
            throw std::logic_error("Bite ask() called twice without tell()");

        if (stop != 0 || evaluations >= maxEvaluations)
            return mat(dim, 0);

        const int requested = std::min(batchSize, maxEvaluations - evaluations);
        currentBatchSize = CBiteOptDeepAT::ask(rnd, requested);

        mat xs(dim, currentBatchSize);
        for (int i = 0; i < currentBatchSize; i++) {
            xs.col(i) = Eigen::Map<const vec, Eigen::Unaligned>(
                getAskValues(i), dim
            );
        }
        return xs;
    }

    int tell(const vec &ys) {
        if (currentBatchSize == 0)
            throw std::logic_error("Bite tell() called before ask()");
        if (ys.size() != currentBatchSize)
            throw std::invalid_argument("ys must match the current batch size");

        std::vector<double> costs(currentBatchSize);
        for (int i = 0; i < currentBatchSize; i++)
            costs[i] = ys[i];

        CBiteOptDeepAT::tell(rnd, costs.data());
        iterations += currentBatchSize;
        evaluations += currentBatchSize;
        currentBatchSize = 0;
        update_stop();
        return stop;
    }

    void doOptimize() {
        while (evaluations < maxEvaluations && stop == 0) {
            const int asked = CBiteOptDeepAT::ask(rnd, 1);
            if (asked <= 0)
                break;

            double cost = fitfun->eval(getAskValues(0))(0);
            CBiteOptDeepAT::tell(rnd, &cost);

            iterations++;
            evaluations++;
            update_stop();
        }
    }

    BiteResult result() const {
        BiteResult result;
        result.x = copy_best_params(getBestParams(), dim);
        result.y = getBestCost();
        result.evaluations = evaluations;
        result.iterations = iterations;
        result.stop = stop;
        return result;
    }

    int getDim() const {
        return dim;
    }

    int getBatchSize() const {
        return batchSize;
    }

    int getPopulationSize() const {
        return populationSize > 0 ? populationSize : 9 + dim * 3;
    }

    int getCurrentBatchSize() const {
        return currentBatchSize;
    }

    int getStop() const {
        return stop;
    }

private:
    void update_stop() {
        if (getBestCost() < stopfitness) {
            stop = 1;
        } else if (stallCriterion > 0 &&
                getStallCount() > stallCriterion * 128 * dim) {
            stop = 2;
        }
    }

    long runid;
    Fitness *fitfun;
    int M;
    int stallCriterion;
    int dim;
    int maxEvaluations;
    double stopfitness;
    int batchSize;
    int populationSize;
    vec lowerValues;
    vec upperValues;
    int iterations;
    int evaluations;
    int stop;
    int currentBatchSize;
    CBiteRnd rnd;
};

BiteResult optimize_bite(
        long runid, callback_type func, int dim, int seed, const double *init,
        const double *lower, const double *upper, int maxEvals,
        double stopfitness, int M, int popsize, int stall_iterations) {
    vec guess, lower_limit, upper_limit;
    initialize_problem(dim, init, lower, upper, guess, lower_limit, upper_limit);

    Fitness fitfun(func, noop_callback_par, dim, 1, lower_limit, upper_limit);
    BiteOptimizer opt(
        runid,
        &fitfun,
        dim,
        guess.size() == 0 ? nullptr : guess.data(),
        seed,
        M,
        popsize,
        1,
        stall_iterations,
        maxEvals,
        stopfitness
    );
    opt.doOptimize();
    return opt.result();
}

class BiteState::Impl {
public:
    Impl(
            long runid, int dim, const double *init, const double *lower,
            const double *upper, int seed, int M, int popsize, int batch_size,
            int max_evals, double stop_fitness, int stall_criterion) {
        initialize_problem(dim, init, lower, upper, guess, lower_limit,
                           upper_limit);
        fitfun = std::make_unique<Fitness>(
            noop_callback, noop_callback_par, dim, 1, lower_limit, upper_limit
        );
        opt = std::make_unique<BiteOptimizer>(
            runid,
            fitfun.get(),
            dim,
            guess.size() == 0 ? nullptr : guess.data(),
            seed,
            M,
            popsize,
            batch_size,
            stall_criterion,
            max_evals,
            stop_fitness
        );
    }

    mat ask() {
        return opt->ask();
    }

    int tell(const vec &ys) {
        return opt->tell(ys);
    }

    BiteResult result() const {
        return opt->result();
    }

    int dim() const {
        return opt->getDim();
    }

    int popsize() const {
        return opt->getBatchSize();
    }

    int population_size() const {
        return opt->getPopulationSize();
    }

    int current_batch_size() const {
        return opt->getCurrentBatchSize();
    }

    int stop() const {
        return opt->getStop();
    }

private:
    vec guess;
    vec lower_limit;
    vec upper_limit;
    std::unique_ptr<Fitness> fitfun;
    std::unique_ptr<BiteOptimizer> opt;
};

BiteState::BiteState(
        long runid, int dim, const double *init, const double *lower,
        const double *upper, int seed, int M, int popsize, int batch_size,
        int max_evals, double stop_fitness, int stall_criterion)
        : impl_(std::make_unique<Impl>(runid, dim, init, lower, upper, seed,
                  M, popsize, batch_size, max_evals, stop_fitness,
                  stall_criterion)) {
}

BiteState::~BiteState() = default;

mat BiteState::ask() {
    return impl_->ask();
}

int BiteState::tell(const vec &ys) {
    return impl_->tell(ys);
}

BiteResult BiteState::result() const {
    return impl_->result();
}

int BiteState::dim() const {
    return impl_->dim();
}

int BiteState::popsize() const {
    return impl_->popsize();
}

int BiteState::population_size() const {
    return impl_->population_size();
}

int BiteState::current_batch_size() const {
    return impl_->current_batch_size();
}

int BiteState::stop() const {
    return impl_->stop();
}

}  // namespace biteopt
