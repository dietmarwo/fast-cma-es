// Copyright (c)  Mingcheng Zuo, Dietmar Wolz.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory.

// Eigen based implementation of differential evolution (LCL-DE) derived from
// "A case learning-based differential evolution algorithm for global optimization of interplanetary trajectory design,
// Mingcheng Zuo, Guangming Dai, Lei Peng, Maocai Wang, Zhengquan Liu", https://doi.org/10.1016/j.asoc.2020.106451
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

typedef void (*callback_parallel)(int, int, double[], double[]);

namespace lcl_differential_evolution {

static uniform_real_distribution<> distr_01 = std::uniform_real_distribution<>(
		0, 1);
static normal_distribution<> gauss_01 = std::normal_distribution<>(0, 1);

static vec zeros(int n) {
	return Eigen::MatrixXd::Zero(n, 1);
}

static vec constant(int n, double val) {
	return vec::Constant(n, val);
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

	Fittness(callback_parallel func_par_, int dim_, const vec &lower_limit,
			const vec &upper_limit, const vec &guess_, const vec &sigma_, pcg64 &rs_) {
		func_par = func_par_;
		dim = dim_;
		lower = lower_limit;
		upper = upper_limit;
		// initial guess for the arguments of the fitness function
		guess = guess_;
		xmean = vec(guess);
		rs = rs_;
		evaluationCounter = 0;
		scale = (upper - lower);
		invScale = scale.cwiseInverse();
		maxSigma = 0.25*scale;
		// individual sigma values - initial search volume. inputSigma determines
		// the initial coordinate wise standard deviations for the search. Setting
		// SIGMA one third of the initial search region is appropriate.
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

	vec getClosestFeasible(const vec &X) const {
		if (lower.size() > 0) {
			return X.cwiseMin(upper).cwiseMax(lower);
		}
		return X;
	}

	void values(const mat &popX, vec &ys) {
		int popsize = popX.cols();
		int n = popX.rows();
		double pargs[popsize*n];
		double res[popsize];
		for (int p = 0; p < popX.cols(); p++) {
			for (int i = 0; i < n; i++)
				pargs[p * n + i] = popX(i, p);
		}
		func_par(popsize, n, pargs, res);
		for (int p = 0; p < popX.cols(); p++)
			ys[p] = res[p];
		evaluationCounter += popsize;
	}

	double distance(const vec &x1, const vec &x2) {
		return ((x1 - x2).array() * invScale.array()).matrix().squaredNorm();
	}

	bool feasible(int i, double x) {
		return x >= lower[i] && x <= upper[i];
	}

	vec uniformX() {
		vec rv = uniformVec(lower.size(), rs);
		return (rv.array() * scale.array()).matrix() + lower;
	}

	double uniformXi(int i) {
		return lower[i] + scale[i] * distr_01(rs);
	}

	int getEvaluations() {
		return evaluationCounter;
	}

private:
	callback_parallel func_par;
	int dim;
	vec lower;
	vec upper;
	vec guess;
	vec xmean;
	vec sigma0;
	vec sigma;
	vec maxSigma;
	pcg64 rs;
	long evaluationCounter;
	vec scale;
	vec invScale;
};

class LclDeOptimizer {

public:

	LclDeOptimizer(long runid_, Fittness *fitfun_, int dim_, pcg64* rs_,
			int popsize_, int maxEvaluations_, double pbest_,
			double stopfitness_, double F0_, double CR0_) {
		// runid used to identify a specific run
		runid = runid_;
		// fitness function to minimize
		fitfun = fitfun_;
		// Number of objective variables/problem dimension
		dim = dim_;
		// Population size
		popsize = popsize_ > 0 ? popsize_ : int(dim*8.5+150);
		// maximal number of evaluations allowed.
		maxEvaluations = maxEvaluations_;
		// use low value 0 < pbest <= 1 to narrow search.
		pbest = pbest_;
		// Limit for fitness value.
		stopfitness = stopfitness_;
		F0 = F0_;
		CR0 = CR0_;
		// stop criteria
		stop = 0;
		rs = rs_;
		init();
	}

	~LclDeOptimizer() {
		delete rs;
	}

	double rnd01() {
		return distr_01(*rs);
	}

	int rndInt(int max) {
		return (int) (max * distr_01(*rs));
	}

	int rndInt2(int max) {
		double u = distr_01(*rs);
		return (int) (max * u*u);
	}

	void doOptimize() {
		int gen_stuck = 0;
		vector<vec> sp;

		int maxIter = maxEvaluations / popsize + 1;
		double previous_best = DBL_MAX;
		double CR, F;

		// -------------------- Generation Loop --------------------------------

		for (iterations = 1;; iterations++) {
			// sort population
			ivec sindex = sort_index(nextY);
			popY = nextY(sindex, Eigen::all);
			popX = nextX(Eigen::all, sindex);

			bestX = popX.col(0);
			bestY = popY[0];

			if (isfinite(stopfitness) && bestY < stopfitness) {
				stop = 1;
				return;
			}
			if (bestY >= previous_best)
				gen_stuck++;
			else {
				gen_stuck = 0;
				if (iterations > 1) {
					fitfun->updateSigma(bestX);
				}
			}
			previous_best = bestY;

			if (fitfun->getEvaluations() >= maxEvaluations)
				return;
			for (int p = 0; p < popsize; p++) {
				int r1, r2, r3;
				do {
					r1 = rndInt(popsize);
				} while (r1 == p);
				do {
					r2 = rndInt(int(popsize * pbest));
				} while (r2 == p || r2 == r1);
				do {
					r3 = rndInt(popsize + sp.size());
				} while (r3 == p || r3 == r2 || r3 == r1);
				int jr = rndInt(dim);
				//Produce the CR and F
				double mu = 1
						- sqrt(float(iterations / maxIter))
								* exp(float(-gen_stuck / iterations));
				if (iterations % 2 == 1) {
					CR = normreal(0.95, 0.01, *rs);
					F = normreal(mu, 1, *rs);
					if (F < 0 || F > 1)
						F = rnd01();
				} else {
					CR = abs(normreal(CR0, 0.01, *rs));
					F = F0;
				}
				vec ui = popX.col(p);
				for (int j = 0; j < dim; j++) {
					if (j == jr || rnd01() < CR) {
						if (r3 < popsize)
							ui[j] = popX(j, r1)
									+ F * (popX(j, r2) - popX(j, r3));
						else
							ui[j] = popX(j, r1)
									+ F * ((popX)(j, r2) - sp[r3 - popsize][j]);
						if (!fitfun->feasible(j, ui[j]))
							ui[j] = fitfun->normXi(j);
					}
				}
				nextX.col(p) = ui;
			}
			fitfun->values(nextX, nextY);
			for (int p = 0; p < popsize; p++) {
				if (nextY[p] < popY[p]) {
					if (sp.size() < popsize)
						sp.push_back(popX.col(p));
					else
						sp[rndInt(popsize)] = popX.col(p);
				} else {        // no improvement, copy from parent
					nextX.col(p) = popX.col(p);
					nextY[p] = popY[p];
				}
			}
		}
	}

	void init() {
		popCR = zeros(popsize);
		popF = zeros(popsize);
		nextX = mat(dim, popsize);
		for (int p = 0; p < popsize; p++)
			nextX.col(p) = fitfun->normX();
		nextY = vec(popsize);
		fitfun->values(nextX, nextY);
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
	double pbest;
	double stopfitness;
	int iterations;
	double bestY;
	vec bestX;
	int stop;
	double F0;
	double CR0;
	pcg64 *rs;
	mat popX;
	vec popY;
	mat nextX;
	vec nextY;
	vec popCR;
	vec popF;
};
}

using namespace lcl_differential_evolution;

extern "C" {
double* optimizeLCLDE_C(long runid, callback_parallel func_par, int dim,
		double *init, double *sigma, int seed,
		double *lower, double *upper, int maxEvals, double pbest,
		double stopfitness, int popsize, double F0, double CR0) {
	int n = dim;
	double *res = new double[n + 4];
	vec guess(n), lower_limit(n), upper_limit(n), inputSigma(n);;
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
	Fittness fitfun(func_par, dim, lower_limit, upper_limit, guess, inputSigma, *rs);
	LclDeOptimizer opt(runid, &fitfun, dim, rs, popsize,
			maxEvals, pbest,
			stopfitness, F0, CR0);
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
