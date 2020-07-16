// Copyright (c) Dietmar Wolz.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory.

// Eigen based implementation of the Harris hawks optimization, see
// Harris hawks optimization: Algorithm and applications
// Ali Asghar Heidari, Seyedali Mirjalili, Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, Huiling Chen
// Future Generation Computer Systems, 
// DOI: https://doi.org/10.1016/j.future.2019.02.028

// derived from https://github.com/7ossam81/EvoloPy/blob/master/optimizers/HHO.py

#include <Eigen/Core>
#include <iostream>
#include <float.h>
#include <math.h>
#include <ctime>
#include <random>
#include "pcg_random.hpp"

using namespace std;

typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vec;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat;

typedef double (*callback_type)(int, double[]);

namespace harris_hawks {

static uniform_real_distribution<> distr_01 = std::uniform_real_distribution<>(
		0, 1);

static uniform_real_distribution<> distr_0x = std::uniform_real_distribution<>(
		0, 0.3);

static normal_distribution<> gauss_01 = std::normal_distribution<>(0, 1);

static vec zeros(int n) {
	return Eigen::MatrixXd::Zero(n, 1);
}

static Eigen::MatrixXd uniform(int dx, int dy, pcg64 &rs) {
	return Eigen::MatrixXd::NullaryExpr(dx, dy, [&]() {
		return distr_01(rs);
	});
}

static Eigen::MatrixXd normalVec(int dim, pcg64 &rs) {
	return Eigen::MatrixXd::NullaryExpr(dim, 1, [&]() {
		return gauss_01(rs);
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
		if (lower.size() > 0) // bounds defined
			scale = (upper - lower);
	}

	vec getClosestFeasible(const vec &X) const {
		if (lower.size() > 0) {
			return X.cwiseMin(1.0).cwiseMax(0.0);
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

	double value(const vec &X) {
		if (lower.size() > 0)
			return eval(decode(X));
		else
			return eval(X);
	}

	vec decode(const vec &X) const {
		if (lower.size() > 0)
			return (X.array() * scale.array()).matrix() + lower;
		else
			return X;
	}

	vec encode(const vec &X) const {
		if (lower.size() > 0)
			return (X - lower).array() / scale.array();
		else
			return X;
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
};

class HHOptimizer {

public:

	HHOptimizer(long runid_, Fittness *fitfun_, int dim_, int seed_,
			int popsize_, int maxEvaluations_, double stopfitness_) {
		// runid used to identify a specific run
		runid = runid_;
		// fitness function to minimize
		fitfun = fitfun_;
		// Number of objective variables/problem dimension
		dim = dim_;
		// Population size
		if (popsize_ > 0)
			popsize = popsize_;
		else
			popsize = 31;
		// termination criteria
		// maximal number of evaluations allowed.
		maxEvaluations = maxEvaluations_;
		// Limit for fitness value.
		stopfitness = stopfitness_;
		// Number of iterations already performed.
		iterations = 0;
		// stop criteria
		stop = 0;
		//std::random_device rd;
		rs = new pcg64(seed_);
		init();
	}

	~HHOptimizer() {
		delete rs;
	}

	double rnd01() {
		return distr_01(*rs);
	}

	double rnd0x() {
		return distr_0x(*rs);
	}

	vec levy(int dim) {
		double beta = 1.5;
		double sigma = pow(
				(tgamma(1 + beta) * sin(M_PI * beta / 2)
						/ (tgamma((1 + beta) / 2) * beta
								* pow(2, ((beta - 1) / 2)))), (1 / beta));
		vec u = 0.01 * normalVec(dim, *rs) * sigma;
		vec v = normalVec(dim, *rs).cwiseAbs();
		vec zz = v.array().pow(vec::Constant(dim, 1.0 / beta).array());
		vec step = u.cwiseProduct(zz.cwiseInverse());
		return step;
	}

	void doOptimize() {

		// -------------------- Generation Loop --------------------------------

		for (iterations = 1; fitfun->getEvaluations() < maxEvaluations;
				iterations++) {

			// fitness of locations
			for (int i = 0; i < popsize; i++) {
				popX.col(i) = fitfun->getClosestFeasible(popX.col(i));
				double y = fitfun->value(popX.col(i)); // compute fitness
				if (!isfinite(y)) {
					stop = -1;
					return;
				}
				if (y < bestY) {
					// update the location of Rabbit
					bestY = y;
					bestX = popX.col(i);
				}
			}
			double e1 = 2 * (1 - ((iterations - 1) / maxIter)); // factor to decrease the energy of the rabbit

			// Update the location of the harris hawks
			for (int i = 0; i < popsize; i++) {
				double e0 = 2 * rnd01() - 1;  // -1<e0<1
				vec xi = popX.col(i);
				double escapingEnergy = e1 * e0; // escaping energy of rabbit Eq. (3) in the paper

				// -------- Exploration phase Eq. (1) in paper -------------------

				if (abs(escapingEnergy) >= 1) {
					// harris hawks perch randomly based on 2 strategy:
					double q = rnd01();
					int randHawkIndex = (int) (popsize * rnd01());
					vec xr = popX.col(randHawkIndex);
					if (q < 0.5)
						// perch based on other family members
						popX.col(i) = xr
								- rnd01() * (xr - 2 * rnd01() * xi).cwiseAbs();

					else {
						// perch on a random tall tree (random site inside group's home range)
						vec xmean = popX.rowwise().mean();
						vec rvec = vec::Constant(dim, rnd01());
						popX.col(i) = (bestX - xmean) - rnd01() * rvec;
					}
				}
				// -------- Exploitation phase -------------------
				else {
					//Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

					//phase 1: ----- surprise pounce (seven kills) ----------
					// multiple, short rapid dives by different hawks

					double r = rnd01(); // probability of each event

					if (r >= 0.5 && abs(escapingEnergy) < 0.5) // Hard besiege Eq. (6) in paper
						popX.col(i) = bestX
								- escapingEnergy * (bestX - xi).cwiseAbs();

					if (r >= 0.5 && abs(escapingEnergy) >= 0.5) { // Soft besiege Eq. (4) in paper
						double jumpStrength = 2 * (1 - rnd01()); // random jump strength of the rabbit
						popX.col(i) =
								(bestX - popX.col(i))
										- escapingEnergy
												* (jumpStrength * bestX - xi).cwiseAbs();
					}
					// phase 2: --------performing team rapid dives (leapfrog movements)----------

					if (r < 0.5 && abs(escapingEnergy) >= 0.5) { // Soft besiege Eq. (10) in paper
					// rabbit try to escape by many zigzag deceptive motions
						double jumpStrength = 2 * (1 - rnd01());
						vec x1 =
								fitfun->getClosestFeasible(
										bestX
												- escapingEnergy
														* (jumpStrength * bestX
																- xi).cwiseAbs());
						double y1 = fitfun->value(x1);
						if (y1 < bestY) // improved move?
							popX.col(i) = x1;
						else { // hawks perform levy-based short rapid dives around the rabbit
							vec x2 =
									fitfun->getClosestFeasible(
											bestX
													- escapingEnergy
															* (jumpStrength
																	* bestX - xi).cwiseAbs()
													+ normalVec(dim, *rs).cwiseProduct(
															levy(dim)));
							double y2 = fitfun->value(x2);
							if (y2 < bestY)
								popX.col(i) = x2;
						}
					}
					if (r < 0.5 && abs(escapingEnergy) < 0.5) { // Hard besiege Eq. (11) in paper
						double jumpStrength = 2 * (1 - rnd01());
						vec xmean = popX.rowwise().mean();
						vec x1 =
								fitfun->getClosestFeasible(
										bestX
												- escapingEnergy
														* (jumpStrength * bestX
																- xmean).cwiseAbs());

						if (fitfun->value(x1) < bestY) // improved move?
							popX.col(i) = x1;
						else { // Perform levy-based short rapid dives around the rabbit
							vec x2 = fitfun->getClosestFeasible(
									bestX
											- escapingEnergy
													* (jumpStrength * bestX
															- xmean).cwiseAbs()
											+ normalVec(dim, *rs).cwiseProduct(
													levy(dim)));
							double y2 = fitfun->value(x2);
							if (y2 < bestY)
								popX.col(i) = x2;
						}
					}
				}
				if (isfinite(stopfitness) && bestY < stopfitness) {
					stop = 1;
					return;
				}
			}
		}
	}

	void init() {
		// initialize the locations of the harris hawks
		popX = uniform(dim, popsize, *rs);
		// initialize the location and energy of the rabbit
		bestX = zeros(popsize);
		bestY = DBL_MAX;
		maxIter = maxEvaluations / popsize;
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
	int maxIter;
	double stopfitness;
	int iterations;
	double guessValue;
	vec guess;
	double bestY;
	vec bestX;
	int stop;
	pcg64 *rs;
	mat popX;
	vec popY;
};

}

using namespace harris_hawks;

extern "C" {
double* optimizeHH_C(long runid, callback_type func, int dim, int seed,
		double *lower, double *upper, int maxEvals, double stopfitness,
		int popsize) {
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
	HHOptimizer opt(runid, &fitfun, dim, seed, popsize, maxEvals, stopfitness);
	try {
		opt.doOptimize();
		vec bestX = fitfun.decode(opt.getBestX());
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
