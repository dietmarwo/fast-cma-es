// Copyright (c) Dietmar Wolz.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory.

// Eigen based implementation of dual annealing 
// derived from https://github.com/scipy/scipy/blob/master/scipy/optimize/_dual_annealing.py
// Implementation only differs regarding boundary handling - this implementattion 
// uses boundary-normalized X values. Local search is fixed to LBFGS-B, see
// https://github.com/yixuan/LBFGSpp/tree/master/include 
// requires https://github.com/imneme/pcg-cpp

#include <Eigen/Core>
#include <iostream>
#include <float.h>
#include <math.h>
#include <ctime>
#include <random>
#include "pcg_random.hpp"
#include <LBFGSB.h>

using namespace LBFGSpp;
using namespace std;

typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vec;
typedef Eigen::Matrix<int, Eigen::Dynamic, 1> ivec;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat;

namespace dual_annealing {

typedef double (*callback_type)(int, double[]);

// wrapper around the fitness function, scales according to boundaries

class Fitness;

static uniform_real_distribution<> distr_01 = std::uniform_real_distribution<>(
		0, 1);
static normal_distribution<> gauss_01 = std::normal_distribution<>(0, 1);

static vec zeros(int n) {
	return Eigen::MatrixXd::Zero(n, 1);
}

static Eigen::MatrixXd normalVec(int dim, pcg64 &rs) {
	return Eigen::MatrixXd::NullaryExpr(dim, 1, [&]() {
		return gauss_01(rs);
	});
}

static Eigen::MatrixXd uniformVec(int dim, pcg64 &rs) {
	return Eigen::MatrixXd::NullaryExpr(dim, 1, [&]() {
		return distr_01(rs);
	});
}

static vec emptyVec = { };

static vec logv(vec v) {
	return v.unaryExpr([](double x) {
		return log(x);
	});
}

static vec expv(vec v) {
	return v.unaryExpr([](double x) {
		return exp(x);
	});
}

double minLBFGS(Fitness *fitfun, vec &X0_, int maxIterations);

class Fitness {

public:

	vec lower;
	vec upper;

	Fitness(callback_type pfunc, vec *lower_limit, vec *upper_limit,
			long maxEvals_) {
		func = pfunc;
		lower = *lower_limit;
		upper = *upper_limit;
		if (lower.size() > 0) // bounds defined
			scale = (upper - lower);
		maxEvals = maxEvals_;
	}

	vec getClosestFeasible(const vec &X) const {
		if (lower.size() > 0) {
			return X.cwiseMin(1.0).cwiseMax(-1.0);
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
		double res = DBL_MAX;
		if (lower.size() > 0)
			res = eval(decode(getClosestFeasible(X)));
		else
			res = eval(X);
		if (res < bestY) {
			bestY = res;
			bestX = vec(X);
		}
		return res;
	}

	const double LS_MAXITER_RATIO = 6;
	const double LS_MAXITER_MIN = 100;
	const double LS_MAXITER_MAX = 1000;

	double local_search(const vec &x0, double currval, vec &res) {
		vec init = getClosestFeasible(x0);
		bestY = DBL_MAX;
		int maxIter = LS_MAXITER_RATIO * x0.size();
		if (maxIter > LS_MAXITER_MAX)
			maxIter = LS_MAXITER_MAX;
		if (maxIter < LS_MAXITER_MIN)
			maxIter = LS_MAXITER_MIN;
		minLBFGS(this, init, maxIter);
		if (bestY < DBL_MAX) {
			for (int i = 0; i < res.size(); i++)
				res[i] = bestX(i);
		}
		return bestY;
	}

	vec encode(const vec &X) const {
		if (lower.size() > 0)
			return (X - lower).array() / scale.array();
		else
			return X;
	}

	vec decode(const vec &X) const {
		if (lower.size() > 0)
			return X.cwiseProduct(scale) + lower;
		else
			return X;
	}

	int getEvaluations() {
		return evaluationCounter;
	}

	bool maxEvalReached() {
		return evaluationCounter >= maxEvals;
	}

private:
	callback_type func;
	long evaluationCounter = 0;
	long maxEvals;
	vec scale;
	double bestY = DBL_MAX;
	vec bestX;
};

class LBFGSFunc {
private:
	Fitness *func;
	int dim;

public:

	LBFGSFunc(Fitness *Fitness_, int dim_) {
		func = Fitness_;
		dim = dim_;
	}

	double operator()(const vec &x, vec &grad) {
		if (!x.allFinite())
			return DBL_MAX;
		double eps = 1E-6;
		vec arg = vec(dim);
		for (int i = 0; i < dim; i++)
			arg[i] = x(i);
		for (int i = 0; i < dim; i++) {
			vec x1 = vec(arg);
			vec x2 = vec(arg);
			double e1 = eps;
			double e2 = eps;
			x1[i] += eps;
			if (x1[i] > 1) {
				x1[i] = 1;
				e1 = 1 - arg[i];
			}
			x2[i] -= eps;
			if (x2[i] < 0) {
				x2[i] = 0;
				e2 = arg[i];
			}
			double f1 = func->value(x1);
			double f2 = func->value(x2);
			grad[i] = (f1 - f2) / (e1 + e2);
		}
		double f = func->value(arg);
		return f;
	}
};

double minLBFGS(Fitness *fitfun, vec &X0, int maxIterations) {
	int dim = X0.size();
	LBFGSFunc fun = LBFGSFunc(fitfun, dim);

	LBFGSBParam<double> param;
	param.max_iterations = maxIterations;
	LBFGSBSolver<double> solver(param);
	vec lb = vec::Constant(dim, 0.0);
	vec ub = vec::Constant(dim, 1.0);
	// Initial values
	vec x = vec::Constant(dim, 0);
	for (int i = 0; i < dim; i++)
		x[i] = X0[i];
	double fx;
	int niter;
	try {
		niter = solver.minimize(fun, x, fx, lb, ub);
	} catch (std::exception &e) {
		//cout << e.what() << endl;
		return DBL_MAX;
	}
	return fx;
}

class VisitingDistribution {

	//Class used to generate new coordinates based on the distorted
	//Cauchy-Lorentz distribution. Depending on the steps within the Markov
	//chain, the class implements the strategy for generating new location
	//changes.

public:

	VisitingDistribution(int dim, double visiting_param_, pcg64 *rs_) {
		_visiting_param = visiting_param_;
		rs = rs_;

		// these are invariant numbers unless visiting_param changes
		double factor2 = exp(
				(4.0 - _visiting_param) * log(_visiting_param - 1.0));
		double factor3 = exp(
				(2.0 - _visiting_param) * log(2.0) / (_visiting_param - 1.0));
		_factor4_p = sqrt(M_PI) * factor2 / (factor3 * (3.0 - _visiting_param));

		double factor5 = 1.0 / (_visiting_param - 1.0) - 0.5;
		double d1 = 2.0 - factor5;
		_factor6 = M_PI * (1.0 - factor5) / sin(M_PI * (1.0 - factor5))
				/ exp(lgamma(d1));
	}

	vec visiting(const vec &x, int step, double temperature) {
		//Based on the step in the strategy chain, new coordinated are
		//generated by changing all components is the same time or only
		//one of them, the new values are computed with visit_fn method

		int dim = x.size();
		if (step < dim) {
			// Changing all coordinates with a new visiting value
			double upper_sample = distr_01(*rs);
			double lower_sample = distr_01(*rs);
			vec visits = visit_fn(temperature, dim);
			for (int i = 0; i < dim; i++) {
				if (visits[i] > TAIL_LIMIT)
					visits[i] = TAIL_LIMIT * upper_sample;
				else if (visits[i] < -TAIL_LIMIT)
					visits[i] = -TAIL_LIMIT * lower_sample;
			}
			vec x_visit = visits + x;
			vec a = x_visit;
			vec b = vec(dim);
			for (int i = 0; i < dim; i++) {
				b[i] = fmod(a[i], 1) + 1;
				x_visit[i] = fmod(b[i], 1);
				if (abs(x_visit[i]) < MIN_VISIT_BOUND)
					x_visit[i] += 1.e-10;
			}
			//cerr << step << " " << temperature <<  endl;// << x_visit << endl;
			return x_visit;
		} else {
			// Changing only one coordinate at a time based on strategy
			// chain step
			vec x_visit = vec(x);
			double visit = visit_fn(temperature, 1)[0];
			if (visit > TAIL_LIMIT)
				visit = TAIL_LIMIT * distr_01(*rs);
			else if (visit < -TAIL_LIMIT)
				visit = -TAIL_LIMIT * distr_01(*rs);
			int index = step - dim;
			x_visit[index] = visit + x[index];
			double a = x_visit[index];
			double b = fmod(a, 1) + 1;
			x_visit[index] = fmod(b, 1);
			if (abs(x_visit[index]) < MIN_VISIT_BOUND)
				x_visit[index] += MIN_VISIT_BOUND;
			//cerr << step << " " << temperature <<  endl;// << x_visit << endl;
			return x_visit;
		}
	}

	vec visit_fn(double temperature, int dim) {

		//Formula Visita from p. 405 of reference [2]
		vec x = normalVec(dim, *rs);
		vec y = normalVec(dim, *rs);
		;

		double factor1 = exp(log(temperature) / (_visiting_param - 1.0));
		double factor4 = _factor4_p * factor1;

		// sigmax
		x = x
				* exp(
						-(_visiting_param - 1.0) * log(_factor6 / factor4)
								/ (3.0 - _visiting_param));

		vec den = expv(
				logv(y.cwiseAbs() * (_visiting_param - 1.0))
						/ (3.0 - _visiting_param));
		return x.cwiseQuotient(den);
	}

private:

	pcg64 *rs;
	double _visiting_param;
	double _factor4_p;
	double _factor6;

	const double TAIL_LIMIT = 1.e8;
	const double MIN_VISIT_BOUND = 1.e-10;
};

class nanexception: public exception {
	virtual const char* what() const throw () {
		return "Objective function is returning nan";
	}
} naneexc;

const double BIG_VALUE = 1e16;

class EnergyState {

	//Class used to record the energy state-> At any time, it knows what is the
	//currently used coordinates and the most recent best location
public:

	double ebest;
	vec xbest;
	double current_energy;
	vec current_location;

	EnergyState(int dim_) {
		dim = dim_;
		ebest = DBL_MAX;
		xbest = { };
		current_energy = DBL_MAX;
		current_location = { };
	}

	void reset(Fitness *owf, pcg64 *rs, const vec &x0) {
		if (x0.size() == 0)
			current_location = normalVec(dim, *rs);
		else
			current_location = vec(x0);
		bool init_error = true;
		int reinit_counter = 0;
		while (init_error) {
			current_energy = owf->value(current_location);
			if (current_energy >= BIG_VALUE || isnan(current_energy)) {
				if (reinit_counter >= MAX_REINIT_COUNT) {
					init_error = false;
					throw naneexc;
				}
				current_location = uniformVec(dim, *rs);
				reinit_counter++;
			} else
				init_error = false;
			// If first time reset, initialize ebest and xbest
			if (ebest == DBL_MAX && xbest.size() == 0) {
				ebest = current_energy;
				xbest = vec(current_location);
			}
			// Otherwise, keep them in case of reannealing reset
		}
	}

	void update_best(double e, const vec &x) {
		ebest = e;
		xbest = vec(x);
	}

	void update_current(double e, const vec &x) {
		current_energy = e;
		current_location = vec(x);
	}

private:
	// Maximimum number of trials for generating a valid starting point
	int MAX_REINIT_COUNT = 1000;
	int dim;
};

class StrategyChain {
	// Class used for the Markov chain and related strategy for local search
	// decision
public:

	StrategyChain(double acceptance_param_, VisitingDistribution *vd_,
			Fitness *ofw_, pcg64 *rs_, EnergyState *state_) {
		// Global optimizer state
		state = state_;
		// Local markov chain minimum energy and location
		emin = state->current_energy;
		xmin = vec(state->current_location);
		// Acceptance parameter
		acceptance_param = acceptance_param_;
		// Visiting distribution instance
		vd = vd_;
		// Wrapper to objective function and related local minimizer
		ofw = ofw_;
		not_improved_idx = 0;
		not_improved_max_idx = 1000;
		rs = rs_;
		temperature_step = 0;
		K = 100 * (state->current_location).size();
	}

	void accept_reject(int j, double e, const vec &x_visit) {
		double r = distr_01(*rs);
		double pqv_temp = (acceptance_param - 1.0) * (e - state->current_energy)
				/ (temperature_step + 1.);
		double pqv = 0;
		if (pqv_temp < 0.)
			pqv = 0.;
		else
			pqv = exp(log(pqv_temp) / (1. - acceptance_param));
		if (r <= pqv) {
			// We accept the new location and update state
			state->update_current(e, x_visit);
			xmin = vec(state->current_location);
		}
		// No improvement since long time
		if (not_improved_idx >= not_improved_max_idx) {
			if (j == 0 || state->current_energy < emin) {
				emin = state->current_energy;
				xmin = vec(state->current_location);
			}
		}
	}

	void run(int step, double temperature) {
		temperature_step = temperature / (double) (step + 1);
		not_improved_idx += 1;
		for (unsigned int j = 0; j < (state->current_location).size() * 2;
				j++) {
			if (j == 0)
				state_improved = false;
			if (step == 0 && j == 0)
				state_improved = true;
			vec x_visit = vd->visiting(state->current_location, j, temperature);
			// Calling the objective function
			double e = ofw->value(x_visit);
			if (e < state->current_energy) {
				// We have got a better energy value
				state->update_current(e, x_visit);
				if (e < state->ebest) {
					state->update_best(e, x_visit);
					state_improved = true;
					not_improved_idx = 0;
				}
			} else {
				// We have not improved but do we accept the new location?
				accept_reject(j, e, x_visit);
			}
			if (ofw->maxEvalReached())
				return;
		}	// End of StrategyChain loop
	}

	void local_search() {
		// Decision making for performing a local search
		// based on Markov chain results
		// If energy has been improved or no improvement since too long,
		// performing a local search with the best Markov chain location
		int dim = state->xbest.size();
		if (state_improved) {
			// Global energy has improved, let's see if LS improved further
			vec x = vec(dim);
			double e = ofw->local_search(state->xbest, state->ebest, x);
			if (e < state->ebest) {
				not_improved_idx = 0;
				state->update_best(e, x);
				state->update_current(e, x);
				if (ofw->maxEvalReached())
					return;
			}
		}
		// Check probability of a need to perform a LS even if no improvment
		bool do_ls = false;
		if (K < 90 * state->current_location.size()) {
			double pls = exp(
					K * (state->ebest - state->current_energy)
							/ temperature_step);
			if (pls >= distr_01(*rs))
				do_ls = true;
		}
		// Global energy not improved, let's see what LS gives
		// on the best strategy chain location
		if (not_improved_idx >= not_improved_max_idx)
			do_ls = true;
		if (do_ls) {
			vec x = vec(dim);
			double e = ofw->local_search(xmin, state->ebest, x);
			xmin = vec(x);
			emin = e;
			not_improved_idx = 0;
			not_improved_max_idx = state->current_location.size();
			if (e < state->ebest) {
				state->update_best(emin, xmin);
				state->update_current(e, x);
			}
		}
	}

private:

	double emin;
	vec xmin;
	EnergyState *state;
	double acceptance_param;
	VisitingDistribution *vd;
	int not_improved_idx;
	int not_improved_max_idx;
	pcg64 *rs;
	Fitness *ofw;
	double temperature_step;
	double K;
	bool state_improved = false;
};

class sizeexception: public exception {
	virtual const char* what() const throw () {
		return "Bounds size does not match x0";
	}
} sizeeexc;

class DARunner {

public:

	DARunner(Fitness *fun_, vec &x0_, long seed_, bool use_local_search_) {
		owf = fun_;
		if (x0_.size() > 0 && x0_.size() != owf->lower.size())
			throw sizeeexc;
		//Initialization of RandomState for reproducible runs if seed provided
		rs = new pcg64(seed_);
		use_local_search = use_local_search_;
		// Initialization of the energy state
		es = new EnergyState(owf->lower.size());
		es->reset(owf, rs, x0_);
		// VisitingDistribution instance
		vd = new VisitingDistribution(owf->lower.size(), qv, rs);
		// Markov chain instance
		sc = new StrategyChain(qa, vd, owf, rs, es);
	}

	~DARunner() {
		delete rs;
		delete vd;
		delete sc;
		delete es;
	}

	void search() {
		iter = 0;
		double t1 = exp((qv - 1) * log(2.0)) - 1.0;
		for (;;) {
			for (int i = 0; i < maxsteps; i++) {
				// Compute temperature for this step
				double s = i + 2.0;
				double t2 = exp((qv - 1) * log(s)) - 1.0;
				double temperature = temperature_start * t1 / t2;
				if (iter++ >= maxsteps)
					return;
				// Need a re-annealing process?
				if (temperature < temperature_restart) {
					es->reset(owf, rs, emptyVec);
					break;
				}
				// starting strategy chain
				sc->run(i, temperature);
				if (owf->maxEvalReached())
					return;
				if (use_local_search) {
					sc->local_search();
					if (owf->maxEvalReached())
						return;
				}
			}
		}
	}

	vec bestX() {
		return es->xbest;
	}

	double bestY() {
		return es->ebest;
	}

private:

	int MAX_REINIT_COUNT = 1000;
	double temperature_start = 5230;
	double qv = 2.62;
	double qa = -5.0;
	bool use_local_search;
	// maximum number of step (main iteration)
	double maxsteps = 1000;
	// minimum value of annealing temperature reached to perform
	// re-annealing temperature_start
	double temperature_restart = 0.1;
	Fitness *owf;
	pcg64 *rs;
	EnergyState *es;
	StrategyChain *sc;
	VisitingDistribution *vd;
	int iter = 0;
};

double minimize(Fitness *fun, vec &x0, long seed, bool use_local_search,
		vec &X) {
	DARunner gr = DARunner(fun, x0, seed, use_local_search);
	gr.search();
	int dim = x0.size();
	vec bx = gr.bestX();
	for (int i = 0; i < dim; i++)
		X[i] = bx[i];
	return gr.bestY();
}
}

using namespace dual_annealing;

extern "C" {
double* optimizeDA_C(long runid, callback_type func, int dim, int seed,
		double *init, double *lower, double *upper, int maxEvals,
		bool use_local_search) {
	int n = dim;
	double *res = new double[n + 4];
	vec guess(n), lower_limit(n), upper_limit(n);
	bool useLimit = false;
	for (int i = 0; i < n; i++) {
		guess[i] = init[i];
		lower_limit[i] = lower[i];
		upper_limit[i] = upper[i];
		useLimit |= (lower[i] != 0);
		useLimit |= (upper[i] != 0);
	}
	if (useLimit == false) {
		lower_limit.resize(0);
		upper_limit.resize(0);
	}
	if (maxEvals <= 0)
		maxEvals = 1E7;
	Fitness fitfun(func, &lower_limit, &upper_limit, maxEvals);

	try {
		vec X = zeros(dim);
		vec enc = fitfun.encode(guess);
		double bestY = minimize(&fitfun, enc, seed, use_local_search, X);
		vec bestX = fitfun.decode(X);
		for (int i = 0; i < n; i++)
			res[i] = bestX[i];
		res[n] = bestY;
		res[n + 1] = fitfun.getEvaluations();
		res[n + 2] = 0;
		res[n + 3] = 0;
		return res;
	} catch (std::exception &e) {
		cerr << e.what() << endl;
		return res;
	}
}
}

