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

namespace differential_evolution {

static uniform_real_distribution<> distr_01 = std::uniform_real_distribution<>(
		0, 1);

static vec zeros(int n) {
	return Eigen::MatrixXd::Zero(n, 1);
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

// wrapper around the fittness function, scales according to boundaries

class Fittness {

public:

	Fittness(callback_type pfunc, const vec &lower_limit,
			const vec &upper_limit) {
		func = pfunc;
		lower = lower_limit;
		upper = upper_limit;
		evaluationCounter = 0;
		if (lower.size() > 0) { // bounds defined
			scale = (upper - lower);
			invScale = scale.cwiseInverse();
		}
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

	void values(const mat &popX, int popsize, vec &ys) {
		for (int p = 0; p < popsize; p++)
			ys[p] = eval(popX.col(p));
	}

	double distance(const vec &x1, const vec &x2) {
		return ((x1 - x2).array() * invScale.array()).matrix().squaredNorm();
	}

	vec getClosestFeasible(const vec &X) const {
		if (lower.size() > 0) {
			return X.cwiseMin(upper).cwiseMax(lower);
		}
		return X;
	}

	bool feasible(int i, double x) {
		return x >= lower[i] && x <= upper[i];
	}

	bool feasible(const vec &x) {
		return (x.array() >= lower.array()).all()
				&& (x.array() <= upper.array()).all();
	}

	vec uniformX(pcg64 &rs) {
		vec rv = uniformVec(lower.size(), rs);
		return (rv.array() * scale.array()).matrix() + lower;
	}

	double uniformXi(int i, pcg64 &rs) {
		return lower[i] + scale[i] * distr_01(rs);
	}

	int getEvaluations() {
		return evaluationCounter;
	}

private:
	callback_type func;
	vec lower;
	vec upper;
	long evaluationCounter;
	vec scale;
	vec invScale;
};

class DeOptimizer {

public:

	DeOptimizer(long runid_, Fittness *fitfun_, int dim_, int seed_,
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
		//std::random_device rd;
		rs = new pcg64(seed_);
		init();
	}

	~DeOptimizer() {
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
				vec ui = vec(xi);

				for (int j = 0; j < dim; j++) {
					if (j == jr || rnd01() < CRu) {
						ui[j] = xb[j] + Fu * (popX(j, r1) - popX(j, r2));
					} if (!fitfun->feasible(j, ui[j]))
						ui[j] = fitfun->uniformXi(j, *rs);
				}
				double eu = fitfun->eval(ui);
				if (isfinite(eu) && eu < popY[p]) {
					// temporal locality
					vec uis = fitfun->getClosestFeasible(
							xb + ((ui - xi) * 0.5));
					double eus = fitfun->eval(uis);
					if (isfinite(eus) && eus < eu) {
						eu = eus;
						ui = uis;
					}
					popX.col(p) = ui;
					popY(p) = eu;
					popIter[p] = iterations;
					if (eu < popY[bestI]) {
						bestI = p;
						if (eu < bestY) {
							bestY = eu;
							bestX = ui;
							if (isfinite(stopfitness) && bestY < stopfitness) {
								stop = 1;
								return;
							}
						}
					}
				} else {
					// reinitialize individual
					if (keep * rnd01() < iterations - popIter[p]) {
						popX.col(p) = fitfun->uniformX(*rs);
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
			popX.col(p) = fitfun->uniformX(*rs);
			popY[p] = fitfun->eval(popX.col(p)); // compute fitness
		}
		bestI = index_min(popY);
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
	Fittness *fitfun;
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

using namespace differential_evolution;

extern "C" {
double* optimizeDE_C(long runid, callback_type func, int dim, int seed,
		double *lower, double *upper, int maxEvals, double keep,
		double stopfitness, int popsize, double F, double CR) {
	int n = dim;
	double *res = new double[n + 4];
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
	Fittness fitfun(func, lower_limit, upper_limit);
	DeOptimizer opt(runid, &fitfun, dim, seed, popsize, maxEvals, keep,
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

