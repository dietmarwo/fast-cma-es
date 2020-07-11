// Copyright (c) Dietmar Wolz.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory.

// Eigen based implementation of active CMA-ES
// derived from http://cma.gforge.inria.fr/cmaes.m which follows
// https://www.researchgate.net/publication/227050324_The_CMA_Evolution_Strategy_A_Comparing_Review
// Requires Eigen version >= 3.3.90 because new slicing capabilities are used, see
// https://eigen.tuxfamily.org/dox-devel/group__TutorialSlicingIndexing.html
// requires https://github.com/imneme/pcg-cpp

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <random>
#include <float.h>
#include <ctime>
#include "pcg_random.hpp"

using namespace std;

typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vec;
typedef Eigen::Matrix<int, Eigen::Dynamic, 1> ivec;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat;

typedef void (*callback_parallel)(int, int, double[], double[]);
typedef bool (*is_terminate_type)(long, int, double); // runid, iter, value -> isTerminate

namespace acmaes {

static vec zeros(int n) {
	return Eigen::MatrixXd::Zero(n, 1);
}

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

struct IndexVal {
	int index;
	double val;
};

static bool compareIndexVal(IndexVal i1, IndexVal i2) {
	return (i1.val < i2.val);
}

static ivec sort_index(const vec &x) {
	int size = x.size();
	IndexVal ivals[size];
	for (int i = 0; i < size; i++) {
		ivals[i].index = i;
		ivals[i].val = x[i];
	}
	std::sort(ivals, ivals + size, compareIndexVal);
	return Eigen::MatrixXi::NullaryExpr(size, 1, [&ivals](int i) {
		return ivals[i].index;
	});
}

static normal_distribution<> gauss_01 = std::normal_distribution<>(0, 1);

static Eigen::MatrixXd normal(int dx, int dy, pcg64 &rs) {
	return Eigen::MatrixXd::NullaryExpr(dx, dy, [&]() {
		return gauss_01(rs);
	});
}

// wrapper around the fittness function, scales according to boundaries

class Fittness {

public:

	Fittness(callback_parallel func_par_, const vec &lower_limit,
			const vec &upper_limit) {
		func_par = func_par_;
		lower = lower_limit;
		upper = upper_limit;
		evaluationCounter = 0;
		if (lower.size() > 0) { // bounds defined
			scale = 0.5 * (upper - lower);
			typx = 0.5 * (upper + lower);
		}
	}

	vec getClosestFeasible(const vec &X) const {
		if (lower.size() > 0) {
			return X.cwiseMin(1.0).cwiseMax(-1.0);
		}
		return X;
	}

	void values(const mat &popX, vec &ys) {
		int popsize = popX.cols();
		int n = popX.rows();
		double pargs[popsize*n];
		double res[popsize];
		for (int p = 0; p < popX.cols(); p++) {
			vec x = decode(getClosestFeasible(popX.col(p)));
			for (int i = 0; i < n; i++)
				pargs[p * n + i] = x[i];
		}
		func_par(popsize, n, pargs, res);
		for (int p = 0; p < popX.cols(); p++)
			ys[p] = res[p];
		evaluationCounter += popsize;
	}

	void values(const mat &popX, int popsize, vec &ys) {
		for (int p = 0; p < popsize; p++)
			ys[p] = value(popX.col(p));
	}

	double value(const vec &X) {
		if (lower.size() > 0)
			return eval(decode(getClosestFeasible(X)));
		else
			return eval(X);
	}

	vec encode(const vec &X) const {
		if (lower.size() > 0)
			return (X - typx).array() / scale.array();
		else
			return X;
	}

	vec decode(const vec &X) const {
		if (lower.size() > 0)
			return (X.array() * scale.array()).matrix() + typx;
		else
			return X;
	}

	int getEvaluations() {
		return evaluationCounter;
	}

private:
	callback_parallel func_par;
	vec lower;
	vec upper;
	long evaluationCounter;
	vec scale;
	vec typx;
};

class AcmaesOptimizer {

public:

	AcmaesOptimizer(long runid_, Fittness *fitfun_, int popsize_, int mu_,
			const vec &guess_, const vec &inputSigma_, int maxIterations_,
			int maxEvaluations_, double accuracy_, double stopfitness_,
			is_terminate_type isTerminate_, long seed) {
		// runid used in isTerminate callback to identify a specific run at different iteration
		runid = runid_;
		// fitness function to minimize
		fitfun = fitfun_;
		// initial guess for the arguments of the fitness function
		guess = guess_;
		// accuracy = 1.0 is default, > 1.0 reduces accuracy
		accuracy = accuracy_;
		// callback to check if to terminate
		isTerminate = isTerminate_;
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
		// maximal number of iterations allowed.
		maxIterations = maxIterations_;
		// limit for fitness value.
		stopfitness = stopfitness_;
		// stop if x-changes larger stopTolUpX.
		stopTolUpX = 1e3 * sigma;
		// stop if x-change smaller stopTolX.
		stopTolX = 1e-11 * sigma * accuracy;
		// stop if fun-changes smaller stopTolFun.
		stopTolFun = 1e-12 * accuracy;
		// stop if back fun-changes smaller stopTolHistFun.
		stopTolHistFun = 1e-13 * accuracy;
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
								dim
										/ (1e-6
												+ min(maxIterations,
														maxEvaluations
																/ popsize)))
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
		// number of iterations already performed.
		iterations = 0;
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
		rs = new pcg64(seed);
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
			mat arzneg = arz(Eigen::all, arReverseIndex.head(mu));
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

	void doOptimize() {

		// -------------------- Generation Loop --------------------------------
		outer: for (iterations = 1;
				iterations <= maxIterations
						&& fitfun->getEvaluations() < maxEvaluations;
				iterations++) {
			// generate and evaluate popsize offspring
			mat arz = normal(dim, popsize, *rs);
			mat arx = mat(dim, popsize);
			// generate random offspring
			xmean = fitfun->getClosestFeasible(xmean);
			for (int k = 0; k < popsize; k++) {
				vec delta = (BD * arz.col(k)) * sigma;
				arx.col(k) = fitfun->getClosestFeasible(xmean + delta);
			}
			vec fitness = vec(popsize);
			fitfun->values(arx, fitness);
			for (int k = 0; k < popsize; k++) {
				if (!isfinite(fitness[k]))
					fitness[k] = DBL_MAX;
			}
			// sort by fitness and compute weighted mean into xmean
			ivec arindex = sort_index(fitness);
			// calculate new xmean, this is selection and recombination
			vec xold = xmean; // for speed up of Eq. (2) and (3)
			ivec bestIndex = arindex.head(mu);
			mat bestArx = arx(Eigen::all, bestIndex);
			xmean = bestArx * weights;
			mat bestArz = arz(Eigen::all, bestIndex);
			mat zmean = bestArz * weights;
			bool hsig = updateEvolutionPaths(zmean, xold);
			double negccov = updateCovariance(hsig, bestArx, arz, arindex,
					xold);
			updateBD(negccov);
			// adapt step size sigma
			sigma *= exp(min(1.0, (normps / chiN - 1.) * cs / damps));
			double bestFitness = fitness(arindex(0));
			double worstFitness = fitness(arindex(arindex.size() - 1));
			if (bestValue > bestFitness) {
				bestValue = bestFitness;
				bestX = fitfun->decode(bestArx.col(0));
			}
			// handle termination criteria
			if (isfinite(stopfitness) && bestFitness < stopfitness) {
				stop = 1;
				break;
			}
			vec sqrtDiagC = diagC.cwiseSqrt();
			vec pcCol = pc;
			for (int i = 0; i < dim; i++) {
				if (sigma * (max(abs(pcCol[i]), sqrtDiagC[i])) > stopTolX)
					break;
				if (i >= dim - 1)
					stop = 2;
			}
			for (int i = 0; i < dim; i++)
				if (sigma * sqrtDiagC[i] > stopTolUpX)
					stop = 3;
			if (stop > 0)
				break;
			double historyBest = fitnessHistory.minCoeff();
			double historyWorst = fitnessHistory.maxCoeff();
			if (iterations > 2
					&& max(historyWorst, worstFitness)
							- min(historyBest, bestFitness) < stopTolFun) {
				stop = 4;
				break;
			}
			if (iterations > fitnessHistory.size()
					&& historyWorst - historyBest < stopTolHistFun) {
				stop = 5;
				break;
			}
			// condition number of the covariance matrix exceeds 1e14
			if (diagD.maxCoeff() / diagD.minCoeff()
					> 1e7 * 1.0 / sqrt(accuracy)) {
				stop = 6;
				break;
			}
			if (isTerminate != NULL
					&& isTerminate(runid, iterations, bestValue)) {
				stop = 7;
				break;
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

	double getStop() {
		return stop;
	}

private:
	long runid;
	Fittness *fitfun;
	vec guess;
	double accuracy;
	is_terminate_type isTerminate;
	int popsize; // population size
	vec inputSigma;
	int dim;
	int maxIterations;
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
	vec xmean;
	vec pc;
	vec ps;
	double normps;
	mat B;
	mat BD;
	mat diagD;
	mat C;
	vec diagC;
	int iterations;
	vec fitnessHistory;
	int historySize;
	double bestValue;
	vec bestX;
	int stop;
	pcg64 *rs;
};
}

using namespace acmaes;

extern "C" {
double* optimizeACMA_C(long runid, callback_parallel func_par, int dim, double *init,
		double *lower, double *upper, double *sigma, int maxIter, int maxEvals,
		double stopfitness, int mu, int popsize, double accuracy,
		bool useTerminate, is_terminate_type isTerminate, long seed, bool isParallel) {
	int n = dim;
	double *res = new double[n + 4];
	vec guess(n), lower_limit(n), upper_limit(n), inputSigma(n);
	bool useLimit = false;
	for (int i = 0; i < n; i++) {
		guess[i] = init[i];
		inputSigma[i] = sigma[i];
		lower_limit[i] = lower[i];
		upper_limit[i] = upper[i];
		useLimit |= (lower[i] != 0);
		useLimit |= (upper[i] != 0);
	}
	if (useLimit == false) {
		lower_limit.resize(0);
		upper_limit.resize(0);
	}
	Fittness fitfun(func_par, lower_limit, upper_limit);
	AcmaesOptimizer opt(runid, &fitfun, popsize, mu, guess, inputSigma, maxIter,
			maxEvals, accuracy, stopfitness, useTerminate ? isTerminate : NULL,
			seed);
	try {
		opt.doOptimize();
		vec bestX = opt.getBestX();
		double bestY = opt.getBestValue();
		for (int i = 0; i < n; i++)
			res[i] = bestX[i];
		res[n] = bestY;
		res[n + 1] = fitfun.getEvaluations();
		res[n + 2] = opt.getIterations();
		res[n + 3] = opt.getStop();
		return res;
	} catch (std::exception &e) {
		cout << e.what() << endl;
		return res;
	}
}
}

extern "C" {
void free_mem(double *a) {
	delete[] a;
}
}
