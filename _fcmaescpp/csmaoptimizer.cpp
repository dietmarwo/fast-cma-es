// Copyright (c) Dietmar Wolz.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory.

#include <Eigen/Core>
#include <iostream>
#include <float.h>
#include <ctime>
#include <random>
#include "pcg_random.hpp"
#include "smaesopt.h"

using namespace std;

typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vec;
typedef Eigen::Matrix<int, Eigen::Dynamic, 1> ivec;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat;

typedef double (*callback_type)(int, double[]);

namespace csmaopt {


// wrapper around the Fitness function, scales according to boundaries

class Fitness {

public:

	Fitness(callback_type pfunc, const vec &lower_limit,
			const vec &upper_limit) {
		func = pfunc;
		lower = lower_limit;
		upper = upper_limit;
		evaluationCounter = 0;
		if (lower.size() > 0) // bounds defined
			scale = (upper - lower);
	}

	double eval(const double* const p) {
		int n = lower.size();
		double parg[n];
		for (int i = 0; i < n; i++)
			parg[i] = p[i];
		double res = func(n, parg);
		evaluationCounter++;
		return res;
	}

	int getEvaluations() {
		return evaluationCounter;
	}

    void getMinValues( double* const p) const {
		for (int i = 0; i < lower.size(); i++)
			p[i] = lower[i];
	}

    void getMaxValues( double* const p) const {
		for (int i = 0; i < upper.size(); i++)
			p[i] = upper[i];
	}

private:
	callback_type func;
	vec lower;
	vec upper;
	long evaluationCounter;
	vec scale;
};

class CsmaOptimizer : public CSMAESOpt {

public:

	CsmaOptimizer(long runid_, Fitness *fitfun_, int dim_, double* init_, double* sdev_,
			int seed_, int popsize_, int maxEvaluations_, double stopfitness_) {
		// runid used to identify a specific run
		runid = runid_;
		// fitness function to minimize
		fitfun = fitfun_;
		// Number of objective variables/problem dimension
		dim = dim_;
		// Population size
		popsize = popsize_;
		// maximal number of evaluations allowed.
		maxEvaluations = maxEvaluations_ > 0 ? maxEvaluations_ : 50000;
		// Number of iterations already performed.
		// Limit for fitness value.
		stopfitness = stopfitness_;
		//std::random_device rd;
		rs = new pcg64(seed_);
		// stop criteria
		stop = 0;

		iterations = 0;
		bestY = DBL_MAX;
        rnd.init(seed_);
        updateDims( dim_,  popsize);
        init( rnd, init_, sdev_ );
	}

	~CsmaOptimizer() {
		delete rs;
	}

	virtual void getMinValues( double* const p ) const {
        fitfun->getMinValues(p);
	}

	virtual void getMaxValues( double* const p ) const {
        fitfun->getMaxValues(p);
	}

	virtual double optcost( const double* const p ) {
        return fitfun->eval(p);
    }

	vec getBestX() {
        vec bestX = vec(dim);
        const double* bx = getBestParams();
        for (int i = 0; i < dim; i++)
            bestX[i] = bx[i];
		return bestX;
	}

	double getBestValue() {
		return getBestCost();
	}

	double getIterations() {
		return iterations;
	}

	double getStop() {
		return stop;
	}

	void doOptimize() {

		// -------------------- Generation Loop --------------------------------
		for (iterations = 1; fitfun->getEvaluations() < maxEvaluations;
				iterations++) {
            optimize( rnd );
            if ( getBestCost() < stopfitness ) {
                stop = 1;
		        break;
            }
        }
    }

private:
	long runid;
	Fitness *fitfun;
	int popsize; // population size
	int dim;
	int maxEvaluations;
	double stopfitness;
	int iterations;
	double bestY;
	int stop;
	vec bestX;
	pcg64 *rs;
    CBiteRnd rnd;
};

}

using namespace csmaopt;

extern "C" {
double* optimizeCsma_C(long runid, callback_type func, int dim, int seed,
		double *init, double *lower, double *upper, double *sigma, int maxEvals,
		double stopfitness, int popsize) {
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
	Fitness fitfun(func, lower_limit, upper_limit);
	CsmaOptimizer opt(runid, &fitfun, dim, init, sigma, seed, popsize, maxEvals, stopfitness);

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


