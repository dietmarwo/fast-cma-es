// Copyright (c) Dietmar Wolz.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory.

// Eigen based implementation of active CMA-ES

// Supports parallel fitness function evaluation. 
// 
// For expensive objective functions (e.g. machine learning parameter optimization) use the workers
// parameter to parallelize objective function evaluation. The workers parameter should be limited
// the population size because otherwize poulation update is delayed. 

// Derived from http://cma.gforge.inria.fr/cmaes.m which follows
// https://www.researchgate.net/publication/227050324_The_CMA_Evolution_Strategy_A_Comparing_Review

// Requires Eigen version >= 3.4 because new slicing capabilities are used, see
// https://eigen.tuxfamily.org/dox-devel/group__TutorialSlicingIndexing.html
// requires https://github.com/bab2min/EigenRand for random number generation.

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <random>
#include <float.h>
#include <stdint.h>
#include <ctime>
#include <EigenRand/EigenRand>
#include "evaluator.h"

using namespace std;

namespace acmaes {

static ivec inverse(const ivec &indices) {
    ivec inverse = ivec(indices.size());
    for (int i = 0; i < indices.size(); i++)
        inverse(indices(i)) = i;
    return inverse;
}

static vec sequence(double start, double end, double step) {
    int size = (int) ((end - start) / step + 1);
    vec d(size);
    double value = start;
    for (int r = 0; r < size; r++) {
        d(r) = value;
        value += step;
    }
    return d;
}

class AcmaesOptimizer {

public:

    AcmaesOptimizer(long runid_, Fitness *fitfun_, int popsize_, int mu_,
            const vec &guess_, const vec &inputSigma_, int maxEvaluations_,
			double accuracy_, double stopfitness_, double stopTolHistFun_,
            int update_gap_, long seed) {
        // runid used for debugging / logging
        runid = runid_;
        // fitness function to minimize
        fitfun = fitfun_;
        // initial guess for the arguments of the fitness function
        guess = guess_;
        // accuracy = 1.0 is default, > 1.0 reduces accuracy
        accuracy = accuracy_;
        // number of objective variables/problem dimension
        dim = guess_.size();
        // population size, offspring number. The primary strategy parameter to play
        // with, which can be increased from its default value. Increasing the
        // population size improves global search properties in exchange to speed.
        // Speed decreases, as a rule, at most linearly with increasing population
        // size. It is advisable to begin with the default small population size.
        if (popsize_ > 0)
            popsize = popsize_;
        else
            popsize = 4 + int(3. * log(dim));
        // individual sigma values - initial search volume. inputSigma determines
        // the initial coordinate wise standard deviations for the search. Setting
        // SIGMA one third of the initial search region is appropriate.
        if (inputSigma_.size() == 1)
            inputSigma = vec::Constant(dim, inputSigma_[0]);
        else
            inputSigma = inputSigma_;
        // overall standard deviation - search volume.
        sigma = inputSigma.maxCoeff();
        // termination criteria
        // maximal number of evaluations allowed.
        maxEvaluations = maxEvaluations_;
        // limit for fitness value.
        stopfitness = stopfitness_;
        // stop if x-changes larger stopTolUpX.
        stopTolUpX = 1e3 * sigma;
        // stop if x-change smaller stopTolX.
        stopTolX = 1e-11 * sigma * accuracy;
        // stop if fun-changes smaller stopTolFun.
        stopTolFun = 1e-12 * accuracy;
        // stop if back fun-changes smaller stopTolHistFun.
        stopTolHistFun = stopTolHistFun_ < 0 ? 1e-13 * accuracy : stopTolHistFun_;
        // selection strategy parameters
        // number of parents/points for recombination.
        mu = mu_ > 0 ? mu_ : popsize / 2;
        // array for weighted recombination.
        weights = (log(sequence(1, mu, 1).array()) * -1.) + log(mu + 0.5);
        double sumw = weights.sum();
        double sumwq = weights.squaredNorm();
        weights *= 1. / sumw;
        // variance-effectiveness of sum w_i x_i.
        mueff = sumw * sumw / sumwq;

        // dynamic strategy parameters and constants
        // cumulation constant.
        cc = (4. + mueff / dim) / (dim + 4. + 2. * mueff / dim);
        // cumulation constant for step-size.
        cs = (mueff + 2.) / (dim + mueff + 3.);
        // damping for step-size.
        damps = (1. + 2. * std::max(0., sqrt((mueff - 1.) / (dim + 1.)) - 1.))
                * max(0.3,
                        1. - // modification for short runs
                                dim / (1e-6 + (maxEvaluations/popsize)))
                + cs; // minor increment
        // learning rate for rank-one update.
        ccov1 = 2. / ((dim + 1.3) * (dim + 1.3) + mueff);
        // learning rate for rank-mu update'
        ccovmu = min(1. - ccov1,
                2. * (mueff - 2. + 1. / mueff)
                        / ((dim + 2.) * (dim + 2.) + mueff));
        // expectation of ||N(0,I)|| == norm(randn(N,1)).
        chiN = sqrt(dim) * (1. - 1. / (4. * dim) + 1 / (21. * dim * dim));
        ccov1Sep = min(1., ccov1 * (dim + 1.5) / 3.);
        ccovmuSep = min(1. - ccov1, ccovmu * (dim + 1.5) / 3.);
        // lazy covariance update gap
        lazy_update_gap =
                update_gap_ >= 0 ?
                        update_gap_ :
                        1.0 / (ccov1 + ccovmu + 1e-23) / dim / 10.0;
        // CMA internal values - updated each generation
        // objective variables.
        xmean = fitfun->encode(guess);
        // evolution path.
        pc = zeros(dim);
        // evolution path for sigma.
        ps = zeros(dim);
        // norm of ps, stored for efficiency.
        normps = ps.norm();
        // coordinate system.
        B = Eigen::MatrixXd::Identity(dim, dim);
        // diagonal of sqrt(D), stored for efficiency.
        diagD = inputSigma / sigma;
        diagC = diagD.cwiseProduct(diagD);
        // B*D, stored for efficiency.
        BD = B.cwiseProduct(diagD.transpose().replicate(dim, 1));
        // covariance matrix.
        C = B * (Eigen::MatrixXd::Identity(dim, dim) * B.transpose());
        // number of iterations.
        iterations = 1;
        // size of history queue of best values.
        historySize = 10 + int(3. * 10. * dim / popsize);
        // stop criteria
        stop = 0;
        // best value so far
        bestValue = DBL_MAX;
        // best parameters so far
        bestX = guess;
        // history queue of best values.
        fitnessHistory = vec::Constant(historySize, DBL_MAX);
        fitnessHistory(0) = bestValue;
        rs = new Eigen::Rand::P8_mt19937_64(seed);
    }

    ~AcmaesOptimizer() {
        delete rs;
    }

    // param zmean weighted row matrix of the gaussian random numbers generating the current offspring
    // param xold xmean matrix of the previous generation
    // return hsig flag indicating a small correction

    bool updateEvolutionPaths(const vec &zmean, const vec &xold) {
        ps = ps * (1. - cs) + ((B * zmean) * sqrt(cs * (2. - cs) * mueff));
        normps = ps.norm();
        bool hsig = normps / sqrt(1. - pow(1. - cs, 2. * iterations)) / chiN
                < 1.4 + 2. / (dim + 1.);
        pc *= (1. - cc);
        if (hsig)
            pc += (xmean - xold) * (sqrt(cc * (2. - cc) * mueff) / sigma);
        return hsig;
    }

    // param hsig flag indicating a small correction
    // param bestArx fitness-sorted matrix of the argument vectors producing the current offspring
    // param arz unsorted matrix containing the gaussian random values of the current offspring
    // param arindex indices indicating the fitness-order of the current offspring
    // param xold xmean matrix of the previous generation

    double updateCovariance(bool hsig, const mat &bestArx, const mat &arz,
            const ivec &arindex, const mat &xold) {
        double negccov = 0;
        if (ccov1 + ccovmu > 0) {
            mat arpos = (bestArx - xold.replicate(1, mu)) * (1. / sigma); // mu difference vectors
            mat roneu = pc * pc.transpose() * ccov1;
            // minor correction if hsig==false
            double oldFac = hsig ? 0 : ccov1 * cc * (2. - cc);
            oldFac += 1. - ccov1 - ccovmu;
            // Adapt covariance matrix C active CMA
            negccov = (1. - ccovmu) * 0.25 * mueff
                    / (pow(dim + 2., 1.5) + 2. * mueff);
            double negminresidualvariance = 0.66;
            // keep at least 0.66 in all directions, small popsize are most critical
            double negalphaold = 0.5; // where to make up for the variance loss,
            // prepare vectors, compute negative updating matrix Cneg
            ivec arReverseIndex = arindex.reverse();
            mat arzneg = arz(Eigen::indexing::all, arReverseIndex.head(mu));
            vec arnorms = arzneg.colwise().norm();
            ivec idxnorms = sort_index(arnorms);
            vec arnormsSorted = arnorms(idxnorms);
            ivec idxReverse = idxnorms.reverse();
            vec arnormsReverse = arnorms(idxReverse);
            arnorms = arnormsReverse.cwiseQuotient(arnormsSorted);
            vec arnormsInv = arnorms(inverse(idxnorms));
            mat sqarnw = arnormsInv.cwiseProduct(arnormsInv).transpose()
                    * weights;
            double negcovMax = (1. - negminresidualvariance) / sqarnw(0);
            if (negccov > negcovMax)
                negccov = negcovMax;
            arzneg = arzneg.cwiseProduct(
                    arnormsInv.transpose().replicate(dim, 1));
            mat artmp = BD * arzneg;
            mat Cneg = artmp * weights.asDiagonal() * artmp.transpose();
            oldFac += negalphaold * negccov;
            C = (C * oldFac) + roneu
                    + (arpos * (ccovmu + (1. - negalphaold) * negccov)
                            * weights.replicate(1, dim).cwiseProduct(
                                    arpos.transpose())) - (Cneg * negccov);
        }
        return negccov;
    }

    // Update B and diagD from C
    // param negccov Negative covariance factor.

    void updateBD(double negccov) {

        if (ccov1 + ccovmu + negccov > 0
                && (std::fmod(iterations,
                        1. / (ccov1 + ccovmu + negccov) / dim / 10.)) < 1.) {
            // to achieve O(N^2) enforce symmetry to prevent complex numbers
            mat triC = C.triangularView<Eigen::Upper>();
            mat triC1 = C.triangularView<Eigen::StrictlyUpper>();
            C = triC + triC1.transpose();
            Eigen::SelfAdjointEigenSolver<mat> sades;
            sades.compute(C);
            // diagD defines the scaling
            diagD = sades.eigenvalues();
            B = sades.eigenvectors();
            if (diagD.minCoeff() <= 0) {
                for (int i = 0; i < dim; i++)
                    if (diagD(i, 0) < 0)
                        diagD(i, 0) = 0.;
                double tfac = diagD.maxCoeff() / 1e14;
                C += Eigen::MatrixXd::Identity(dim, dim) * tfac;
                diagD += vec::Constant(dim, 1.0) * tfac;
            }
            if (diagD.maxCoeff() > 1e14 * diagD.minCoeff()) {
                double tfac = diagD.maxCoeff() / 1e14 - diagD.minCoeff();
                C += Eigen::MatrixXd::Identity(dim, dim) * tfac;
                diagD += vec::Constant(dim, 1.0) * tfac;
            }
            diagC = C.diagonal();
            diagD = diagD.cwiseSqrt(); // D contains standard deviations now
            BD = B.cwiseProduct(diagD.transpose().replicate(dim, 1));
        }
    }

    mat ask_all() { // undecoded
        // generate popsize offspring.
        mat xz = normal(dim, popsize, *rs);
        mat xs(dim, popsize);
        for (int k = 0; k < popsize; k++) {
            vec delta = (BD * xz.col(k)) * sigma;
            xs.col(k) = fitfun->getClosestFeasibleNormed(xmean + delta);
        }
        return xs;
    }

    int tell_all(mat ys, mat xs) {
       told = 0;
       for (int p = 0; p < popsize; p++)
           tell(ys(p), xs.col(p));
       return stop;
    }

    mat getPopulation() {
        mat pop(dim, popsize);
        for (int p = 0; p < popsize; p++)
            pop.col(p) = fitfun->decode(fitfun->getClosestFeasibleNormed(popX.col(p)));
        return pop;
    }

    vec ask() {
        // ask for one new argument vector.
        vec arz1 = normalVec(dim, *rs);
        vec delta = (BD * arz1) * sigma;
        vec arx1 = fitfun->getClosestFeasibleNormed(xmean + delta);
        return arx1;
    }

    int tell(double y, const vec &x) {
        //tell function value for a argument list retrieved by ask_one().
        if (told == 0) {
            fitness = vec(popsize);
            arx = mat(dim, popsize);
            arz = mat(dim, popsize);
        }
        fitness[told] = isfinite(y) ? y : DBL_MAX;
        arx.col(told) = x;
        told++;

        if (told >= popsize) {
            xmean = fitfun->getClosestFeasibleNormed(xmean);
            try {
                arz = (BD.inverse()
                        * ((arx - xmean.replicate(1, popsize)) / sigma));
            } catch (std::exception &e) {
                arz = normal(dim, popsize, *rs);
            }
            updateCMA();
            told = 0;
            iterations += 1;
        }
        return stop;
    }

    void updateCMA() {
        // sort by fitness and compute weighted mean into xmean
        ivec arindex = sort_index(fitness);
        // calculate new xmean, this is selection and recombination
        vec xold = xmean; // for speed up of Eq. (2) and (3)
        ivec bestIndex = arindex.head(mu);
        mat bestArx = arx(Eigen::indexing::all, bestIndex);
        xmean = bestArx * weights;
        mat bestArz = arz(Eigen::indexing::all, bestIndex);
        mat zmean = bestArz * weights;
        bool hsig = updateEvolutionPaths(zmean, xold);
        // adapt step size sigma
        sigma *= exp(min(1.0, (normps / chiN - 1.) * cs / damps));
        double bestFitness = fitness(arindex(0));
        double worstFitness = fitness(arindex(arindex.size() - 1));
        if (bestValue > bestFitness) {
            bestValue = bestFitness;
            bestX = fitfun->decode(bestArx.col(0));
            if (isfinite(stopfitness) && bestFitness < stopfitness) {
                stop = 1;
                return;
            }
        }
        if (iterations >= last_update + lazy_update_gap) {
            last_update = iterations;
            double negccov = updateCovariance(hsig, bestArx, arz, arindex,
                    xold);
            updateBD(negccov);
            // handle termination criteria
            vec sqrtDiagC = diagC.cwiseSqrt();
            vec pcCol = pc;
            for (int i = 0; i < dim; i++) {
                if (sigma * (max(abs(pcCol[i]), sqrtDiagC[i])) > stopTolX)
                    break;
                if (i >= dim - 1)
                    stop = 2;
            }
            if (stop > 0)
                return;
            for (int i = 0; i < dim; i++)
                if (sigma * sqrtDiagC[i] > stopTolUpX)
                    stop = 3;
            if (stop > 0)
                return;
        }
        double historyBest = fitnessHistory.minCoeff();
        double historyWorst = fitnessHistory.maxCoeff();
        if (iterations > 2
                && max(historyWorst, worstFitness)
                        - min(historyBest, bestFitness) < stopTolFun) {
            stop = 4;
            return;
        }
        if (iterations > fitnessHistory.size()
                && historyWorst - historyBest < stopTolHistFun) {
            stop = 5;
            return;
        }
        // condition number of the covariance matrix exceeds 1e14
        if (diagD.maxCoeff() / diagD.minCoeff() > 1e7 * 1.0 / sqrt(accuracy)) {
            stop = 6;
            return;
        }
        // adjust step size in case of equal function values (flat fitness)
        if (bestValue == fitness[arindex[(int) (0.1 + popsize / 4.)]]) {
            sigma *= exp(0.2 + cs / damps);
        }
        if (iterations > 2
                && max(historyWorst, bestFitness)
                        - std::min(historyBest, bestFitness) == 0) {
            sigma *= ::exp(0.2 + cs / damps);
        }
        // store best in history
        for (int i = 1; i < fitnessHistory.size(); i++)
            fitnessHistory[i] = fitnessHistory[i - 1];
        fitnessHistory[0] = bestFitness;
    }

    int doOptimize() {

        // -------------------- Generation Loop --------------------------------
        iterations = 0;
        fitfun->resetEvaluations();
        while (fitfun->evaluations() < maxEvaluations && !fitfun->terminate()) {
            // generate and evaluate popsize offspring
            mat xs = ask_all();
            vec ys(popsize);
            fitfun->values(xs, ys); // decodes
            for (int k = 0; k < popsize; k++)
                tell(ys(k), xs.col(k)); // tell encoded
            if (stop != 0)
                return fitfun->evaluations();
        }
        return fitfun->evaluations();
    }

    int do_optimize_delayed_update(int workers) {
    	 iterations = 0;
    	 fitfun->resetEvaluations();
    	 evaluator eval(fitfun, 1, workers);
    	 vec evals_x[workers];
	     // fill eval queue with initial population
    	 for (int i = 0; i < workers; i++) {
    		 vec x = ask();
    		 vec xdec = fitfun->decode(x);
    		 eval.evaluate(xdec, i);
    		 evals_x[i] = x; // encoded
    	 }
    	 while (fitfun->evaluations() < maxEvaluations) {
    		 vec_id* vid = eval.result();
    		 vec y = vec(vid->_v);
    		 int p = vid->_id;
    		 delete vid;
    		 vec x = evals_x[p];
    		 tell(y(0), x); // tell evaluated encoded x
    		 if (fitfun->evaluations() >= maxEvaluations || stop != 0)
    			 break;
    		 x = ask();
    		 eval.evaluate(x, p);
    		 evals_x[p] = x;
    	 }
    	 return fitfun->evaluations();
	}

    vec getBestX() {
        return bestX;
    }

    double getBestValue() {
        return bestValue;
    }

    double getIterations() {
        return iterations;
    }

    int getStop() {
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

    Fitness* getFitfunPar() {
        return fitfun;
    }

    mat popX;

private:
    long runid;
    Fitness *fitfun;
    vec guess;
    double accuracy;
    int popsize; // population size
    vec inputSigma;
    int dim;
    int maxEvaluations;
    double stopfitness;
    double stopTolUpX;
    double stopTolX;
    double stopTolFun;
    double stopTolHistFun;
    int mu; //
    vec weights;
    double mueff; //
    double sigma;
    double cc;
    double cs;
    double damps;
    double ccov1;
    double ccovmu;
    double chiN;
    double ccov1Sep;
    double ccovmuSep;
    double lazy_update_gap = 0;
    vec xmean;
    vec pc;
    vec ps;
    double normps;
    mat B;
    mat BD;
    mat diagD;
    mat C;
    vec diagC;
    mat arz;
    mat arx;
    vec fitness;
    int iterations = 0;
    int last_update = 0;
    vec fitnessHistory;
    int historySize;
    double bestValue;
    vec bestX;
    int stop;
    int told = 0;
    Eigen::Rand::P8_mt19937_64 *rs;
};
}

using namespace acmaes;

extern "C" {
void optimizeACMA_C(long runid, callback_type func, callback_parallel func_par, int dim,
        double *init, double *lower, double *upper, double *sigma,
        int maxEvals, double stopfitness, double stopTolHistFun, int mu, int popsize, double accuracy,
        long seed, bool normalize, bool use_delayed_update, int update_gap, int workers, double* res) {

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

    Fitness fitfun(func, func_par, dim, 1, lower_limit, upper_limit);
    fitfun.setNormalize(normalize);

    AcmaesOptimizer opt(runid, &fitfun, popsize, mu, guess, inputSigma,
            maxEvals, accuracy, stopfitness, stopTolHistFun, update_gap, seed);
    try {
        int evals = 0;
        if (workers > 1 && use_delayed_update)
            evals = opt.do_optimize_delayed_update(workers);
        else
            evals = opt.doOptimize();
        vec bestX = opt.getBestX();
        double bestY = opt.getBestValue();
        for (int i = 0; i < dim; i++)
            res[i] = bestX[i];
        res[dim] = bestY;
        res[dim + 1] = evals;
        res[dim + 2] = opt.getIterations();
        res[dim + 3] = opt.getStop();
    } catch (std::exception &e) {
        cout << e.what() << endl;
    }
}

uintptr_t initACMA_C(long runid, int dim,
        double *init, double *lower, double *upper, double *sigma,
        int maxEvals, double stopfitness, double stopTolHistFun, int mu, int popsize, double accuracy,
        long seed, bool normalize, bool use_delayed_update, int update_gap) {

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

    Fitness* fitfun = new Fitness(noop_callback, noop_callback_par, dim, 1, lower_limit, upper_limit); // never used here
    fitfun->setNormalize(normalize);

    AcmaesOptimizer* opt = new AcmaesOptimizer(runid, fitfun, popsize, mu, guess, inputSigma,
            maxEvals, accuracy, stopfitness, stopTolHistFun, update_gap, seed);
    return (uintptr_t) opt;
}

void destroyACMA_C(uintptr_t ptr) {
    AcmaesOptimizer* opt = (AcmaesOptimizer*)ptr;
    Fitness* fitfun = opt->getFitfun();
    delete fitfun;
    delete opt;
}

void askACMA_C(uintptr_t ptr, double* xs) {
    AcmaesOptimizer *opt = (AcmaesOptimizer*) ptr;
    int n = opt->getDim();
    int popsize = opt->getPopsize();
    opt->popX = opt->ask_all();
    Fitness* fitfun = opt->getFitfun();
    for (int p = 0; p < popsize; p++) {
        vec x = fitfun->decode(opt->popX.col(p));
        for (int i = 0; i < n; i++)
            xs[p * n + i] = x[i];
    }
}

int tellACMA_C(uintptr_t ptr, double* ys) {
    AcmaesOptimizer *opt = (AcmaesOptimizer*) ptr;
    int popsize = opt->getPopsize();
    vec vals(popsize);
    for (int i = 0; i < popsize; i++)
        vals[i] = ys[i];
    opt->tell_all(vals, opt->popX);
    return opt->getStop();
}

int tellXACMA_C(uintptr_t ptr, double* ys, double* xs) {
    AcmaesOptimizer *opt = (AcmaesOptimizer*) ptr;
    int popsize = opt->getPopsize();
    int dim = opt->getDim();
    Fitness* fitfun = opt->getFitfun();
    opt->popX = mat(dim, popsize);
    for (int p = 0; p < popsize; p++) {
        vec x(dim);
        for (int i = 0; i < dim; i++)
            x[i] = xs[p * dim + i];
        opt->popX.col(p) = fitfun->encode(x);
    }
    vec vals(popsize);
    for (int i = 0; i < popsize; i++)
        vals[i] = ys[i];
    opt->tell_all(vals, opt->popX);
    return opt->getStop();
}

int populationACMA_C(uintptr_t ptr, double* xs) {
    AcmaesOptimizer *opt = (AcmaesOptimizer*) ptr;
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

int resultACMA_C(uintptr_t ptr, double* res) {
    AcmaesOptimizer *opt = (AcmaesOptimizer*) ptr;
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

int testACMA_C(int n, double* res) {
    for (int i = 0; i < n; i++) {
        cout << i << ": " << res[i] << endl;
        res[i] = -res[i];
    }
    return 7;
}

}
