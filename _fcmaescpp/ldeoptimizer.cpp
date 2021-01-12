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
// To be used to further optimize a given solution. Initial population is created using a normal distribition
// with mean=init and sdev=sigma (normalized over the bounds, defined separately for each variable).

#include <Eigen/Core>
#include <iostream>
#include <float.h>
#include <ctime>
#include <random>
#include "pcg_random.hpp"

using namespace std;

typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vec;
typedef Eigen::Matrix<int, Eigen::Dynamic, 1> ivec;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat;

typedef double (*callback_type)(int, double[]);

namespace l_differential_evolution {

static uniform_real_distribution<> distr_01 = std::uniform_real_distribution<>(
		0, 1);
static normal_distribution<> gauss_01 = std::normal_distribution<>(0, 1);

static vec zeros(int n) {
	return Eigen::MatrixXd::Zero(n, 1);
}

static vec constant(int n, double val) {
	return vec::Constant(n, val);
}

static Eigen::MatrixXd uniform(int dx, int dy, pcg64 &rs) {
	return Eigen::MatrixXd::NullaryExpr(dx, dy, [&]() {
		return distr_01(rs);
	});
}

static Eigen::MatrixXd uniformVec(int dim, pcg64 &rs) {
	return Eigen::MatrixXd::NullaryExpr(dim, 1, [&]() {
		return distr_01(rs);
	});
}

static double normreal(double mean, double sdev, pcg64 &rs) {
	return gauss_01(rs) * sdev + mean;
}

static vec normalVec(const vec& mean, const vec& sdev, int dim, pcg64 &rs) {
	vec nv =  Eigen::MatrixXd::NullaryExpr(dim, 1, [&]() {
		return gauss_01(rs);
	});
	return (nv.array() * sdev.array()).matrix() + mean;
}


static int index_min(vec &v) {
	double minv = DBL_MAX;
	int mi = -1;
	for (int i = 0; i < v.size(); i++) {
		if (v[i] < minv) {
			mi = i;
			minv = v[i];
		}
	}
	return mi;
}

struct IndexVal {
	int index;
	double val;
};

// wrapper around the fitness function, scales according to boundaries

class Fitness {

public:

	Fitness(callback_type func_, int dim_, const vec &lower_limit,
			const vec &upper_limit, const vec &guess_, const vec &sigma_, pcg64 &rs_) {
		func = func_;
		dim = dim_;
		lower = lower_limit;
		upper = upper_limit;
		// initial guess for the arguments of the fitness function
		guess = guess_;
		xmean = vec(guess);
		rs = rs_;
		evaluationCounter = 0;
		if (lower.size() > 0) // bounds defined
			scale = (upper - lower);
		else
			scale = constant(dim, 1.0);
		invScale = scale.cwiseInverse();
		maxSigma = 0.25*scale;
		// individual sigma values - initial search volume. inputSigma determines
		// the initial coordinate wise standard deviations for the search.
		if (sigma_.size() == 1)
			sigma0 = 0.5 * (scale.array() * (vec::Constant(dim, sigma_[0])).array()).matrix();
		else
			sigma0 = 0.5 * (scale.array() * sigma_.array()).matrix();
		sigma = vec(sigma0);
	}

	void updateSigma(const vec &X) {
		vec delta = (xmean - X).cwiseAbs() * 0.5;
		sigma = delta.cwiseMin(maxSigma);
		xmean = X;
	}

	vec normX() {
		return distr_01(rs) < 0.5 ? 
            getClosestFeasible(normalVec(xmean, sigma0, dim, rs)) :
            getClosestFeasible(normalVec(xmean, sigma, dim, rs));
	}

	double normXi(int i) {
		double nx;
        if (distr_01(rs) < 0.5) {
		    do {
			    nx = normreal(xmean[i], sigma0[i], rs);
		    } while (!feasible(i, nx));
        } else {
		    do {
			    nx = normreal(xmean[i], sigma[i], rs);
		    } while (!feasible(i, nx));
        }
		return nx;
	}

	bool feasible(int i, double x) {
		return lower.size() == 0 || (x >= lower[i] && x <= upper[i]);
	}

	vec sample() {
		if (lower.size() > 0) {
			vec rv = uniformVec(dim, rs);
			return (rv.array() * scale.array()).matrix() + lower;
		} else
			return normX();
	}

	double sample_i(int i) {
		if (lower.size() > 0)
			return lower[i] + scale[i] * distr_01(rs);
		else
			return normXi(i);
	}

	vec getClosestFeasible(const vec &X) const {
		if (lower.size() > 0) {
			return X.cwiseMin(upper).cwiseMax(lower);
		}
		return X;
	}

	double eval(const vec &X) {
		int n = X.size();
		double parg[n];
		for (int i = 0; i < n; i++)
			parg[i] = X(i);
		double res = func(n, parg);
		evaluationCounter++;
		return res;
	}

	int getEvaluations() {
		return evaluationCounter;
	}

	vec guess;

private:
	callback_type func;
	int dim;
	vec lower;
	vec upper;
	vec xmean;
	vec sigma0;
	vec sigma;
	vec maxSigma;
	pcg64 rs;
	long evaluationCounter;
	vec scale;
	vec invScale;
};

class LDeOptimizer {

public:

	LDeOptimizer(long runid_, Fitness *fitfun_, int dim_, pcg64* rs_,
			int popsize_, int maxEvaluations_, double keep_,
			double stopfitness_, double F_, double CR_) {
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
		F = F_ > 0 ? F_ : 0.5;
		CR = CR_ > 0 ? CR_ : 0.9;
		// Number of iterations already performed.
		iterations = 0;
		bestY = DBL_MAX;
		// stop criteria
		stop = 0;
		rs = rs_;
		init();
	}

	~LDeOptimizer() {
		delete rs;
	}

	double rnd01() {
		return distr_01(*rs);
	}

	double rnd02() {
		double rnd = distr_01(*rs);
		return rnd * rnd;
	}

	int rndInt(int max) {
		return (int) (max * distr_01(*rs));
	}

	void doOptimize() {

		// -------------------- Generation Loop --------------------------------
		for (iterations = 1; fitfun->getEvaluations() < maxEvaluations;
				iterations++) {

			double CRu = iterations % 2 == 0 ? 0.5*CR : CR;
			double Fu = iterations % 2 == 0 ? 0.5*F : F;

			for (int p = 0; p < popsize; p++) {
				vec xi = popX.col(p);
				vec xb = popX.col(bestI);

				int r1, r2;
				do {
					r1 = rndInt(popsize);
				} while (r1 == p || r1 == bestI);
				do {
					r2 = rndInt(popsize);
				} while (r2 == p || r2 == bestI || r2 == r1);

				int jr = rndInt(dim);
				vec x = vec(xi);

				for (int j = 0; j < dim; j++) {
					if (j == jr || rnd01() < CRu) {
						x[j] = xb[j] + Fu * (popX(j, r1) - popX(j, r2));
                        if (!fitfun->feasible(j, x[j]))
						    x[j] = fitfun->normXi(j);
                    }
				}
				double y = fitfun->eval(x);
				if (isfinite(y) && y < popY[p]) {
					// temporal locality
					vec x2 = fitfun->getClosestFeasible(
							xb + ((x - xi) * 0.5));
					double y2 = fitfun->eval(x2);
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
							fitfun->updateSigma(x);
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
						popX.col(p) = fitfun->normX();
						popY[p] = fitfun->eval(popX.col(p)); // compute fitness
					}
				}
			}
		}
	}

	void init() {
		popX = mat(dim, popsize);
		popY = vec(popsize);
		for (int p = 0; p < popsize; p++) {
			popX.col(p) = fitfun->guess;
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
	double F;
	double CR;
	pcg64 *rs;
	mat popX;
	vec popY;
	vec popIter;
};

// see https://cvstuff.wordpress.com/2014/11/27/wraping-c-code-with-python-ctypes-memory-and-pointers/

}

using namespace l_differential_evolution;

extern "C" {
double* optimizeLDE_C(long runid, callback_type func, int dim,
		double *init, double *sigma, int seed,
		double *lower, double *upper, int maxEvals, double keep,
		double stopfitness, int popsize, double F, double CR) {
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
	pcg64* rs = new pcg64(seed);
	Fitness fitfun(func, dim, lower_limit, upper_limit, guess, inputSigma, *rs);
	LDeOptimizer opt(runid, &fitfun, dim, rs, popsize, maxEvals, keep,
			stopfitness, F, CR);
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
