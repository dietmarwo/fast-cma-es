// Copyright (c) Dietmar Wolz.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory.

// Eigen based implementation of Fast Moving Natural Evolution Strategy
//    for High-Dimensional Problems (CR-FM-NES), see https://arxiv.org/abs/2201.11422 .
//    Derived from https://github.com/nomuramasahir0/crfmnes .
//
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
#include <inttypes.h>
#include "evaluator.h"

using namespace std;

namespace crmfnes {

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

class CrfmnesOptimizer {

public:

    CrfmnesOptimizer(int64_t  runid_, Fitness* fitfun_, int dim_, vec m_, double sigma_, int lamb_,
            int maxEvaluations_, double stopfitness_,
            double penalty_coef_, bool use_constraint_violation_, int64_t  seed) {

        runid = runid_;
        fitfun = fitfun_;
        dim = dim_;
        m = fitfun->encode(m_);
        sigma = sigma_;
        lamb = lamb_;
        mu = lamb_/2;
        maxEvaluations = maxEvaluations_;
        stopfitness = stopfitness_;
        penalty_coef = penalty_coef_ > 0 ? penalty_coef_ : 1e5;
        use_constraint_violation = use_constraint_violation_;
        rs = new pcg64(seed);

        stop = 0;
        v = normalVec(dim, *rs) / sqrt(dim);
        D = constant(dim, 1);
        w_rank_hat = ((log(sequence(1, lamb, 1).array()) * -1.) + log(mu + 1)).cwiseMax(0);
        w_rank = (w_rank_hat / w_rank_hat.sum()).array() - (1. / lamb);
        vec wlamb = w_rank.array() + (1. / lamb);
        mueff = 1. / (wlamb.transpose() * wlamb)(0,0);
        cs = (mueff + 2.) / (dim + mueff + 5.);
        cc = (4. + mueff / dim) / (dim + 4. + 2. * mueff / dim);
        c1_cma = 2. / (pow(dim + 1.3, 2) + mueff);
        // initialization
        chiN = sqrt(dim) * (1. - 1. / (4. * dim) + 1. / (21. * dim * dim));
        pc = zeros(dim);
        ps = zeros(dim);
        // distance weight parameter
        h_inv = get_h_inv(dim);
        // learning rate
        eta_m = 1.0;
        eta_move_sigma = 1.;

        g = 0;
        no_of_evals = 0;
        z = mat(dim, lamb);
        f_best = INFINITY;
        x_best = vec(dim);
    };

    virtual ~CrfmnesOptimizer() {
        delete rs;
    }

    void doOptimize() {
        // -------------------- Generation Loop --------------------------------
        for (iterations = 1; fitfun->evaluations() < maxEvaluations && !fitfun->terminate();
                iterations++) {
            // generate and evaluate lamb offspring
            try {
                mat xs = ask();
                vec evs(lamb);
                fitfun->values(xs, evs);
                tell(evs);
            } catch (std::exception &e) {
                 stop = -1;
            }
            if (stop != 0)
                return;
        }
    }

    mat getPopulation() {
         return xs_no_sort;
    }

    vec getBestX() {
        return x_best;
    }

    double getBestValue() {
        return f_best;
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
        return lamb;
    }

    mat ask() {
        mat zhalf = normal(dim, mu, *rs);
        for (int i = 0; i < lamb; i++) {
            if (i < mu) z.col(i) = zhalf.col(i);
            else z.col(i) = -zhalf.col(i-mu);
        }
        normv = v.norm();
        normv2 = normv*normv;
        vbar = v / normv;
        y = z + ((sqrt(1 + normv2) - 1.) * (vbar  * (vbar.transpose() * z)));
        x = ((sigma * y.array()).colwise() * D.array()).colwise() + m.array();
        return x;
    }

    void tell(vec &evs) {
        evals_no_sort = vec(evs);
        for (int k = 0; k < lamb; k++) {
            if (!isfinite(evals_no_sort[k]))
                evals_no_sort[k] = DBL_MAX;
        }
        xs_no_sort = mat(x);
        ivec sorted_indices;
        if (use_constraint_violation) {
            vec violations = fitfun->violations(x, penalty_coef);
            sorted_indices = sort_indices_by(evals_no_sort + violations, z);
        } else
            sorted_indices = sort_indices_by(evals_no_sort, z);
        int best_eval_id = sorted_indices[0];
        double f_best_ = evals_no_sort[best_eval_id];

        z = mat(z)(Eigen::indexing::all, sorted_indices);
        y = mat(y)(Eigen::indexing::all, sorted_indices);
        x = mat(x)(Eigen::indexing::all, sorted_indices);
        no_of_evals += lamb;
        g += 1;
        if (f_best_ < f_best) {
            f_best = f_best_;
            x_best = fitfun->decode(xs_no_sort.col(best_eval_id));
            if (f_best < stopfitness)
                stop = 1;
            //cout << f_best << endl;
        }
        // This operation assumes that if the solution is infeasible, infinity comes in as input.
        double lambF = 0;
        for (int k = 0; k < lamb; k++)
            if (evals_no_sort[k] < DBL_MAX) lambF++;
        // evolution path p_sigma
        ps = (1 - cs) * ps + sqrt(cs * (2. - cs) * mueff) * (z * w_rank);
        double ps_norm = ps.norm();
        // distance weight
        vec w_tmp(lamb);
        for (int k = 0; k < lamb; k++)
            w_tmp[k] = w_rank_hat[k] * w_dist_hat(z.col(k), lambF);
        vec weights_dist = (w_tmp / w_tmp.sum()).array() - 1. / lamb;
        // switching weights and learning rate
        vec weights = ps_norm >= chiN ? weights_dist : w_rank;
        double eta_sigma = ps_norm >= chiN ? eta_move_sigma :
                (ps_norm >= 0.1 * chiN ? eta_stag_sigma(lambF) : eta_conv_sigma(lambF));
        // update pc, m
        vec wxm = (x.array().colwise() - m.array()).matrix() * weights;
        pc = (1. - cc) * pc + sqrt(cc * (2. - cc) * mueff) * wxm / sigma;
        m += eta_m * wxm;
        // calculate s, t
        // step1
        double normv4 = normv2 * normv2;
        mat exY(dim, lamb+1);
        for (int k = 0; k < lamb; k++)
            exY.col(k) = y.col(k);
        exY.col(lamb) = pc.array() / D.array();
        mat yy = exY.array() * exY.array();  // dim x lamb+1
        vec ip_yvbar = vbar.transpose() * exY;
        mat yvbar = exY.array().colwise() * vbar.array(); // dim x lamb+1. exYのそれぞれの列にvbarがかかる
        double gammav = 1. + normv2;
        vec vbarbar = vbar.array() * vbar.array();
        double alphavd = min(1., sqrt(normv4 + (2 * gammav - sqrt(gammav)) / vbarbar.maxCoeff()) / (2. + normv2));  // scalar
        mat vbar_bc = zeros(dim, lamb+1).colwise() + vbar; // broadcasting vbar
        vec ibg = (ip_yvbar.array()*ip_yvbar.array()) + gammav;
        mat t = (exY.array().rowwise() * ip_yvbar.transpose().array()) -
            (vbar_bc.array().rowwise() * ibg.transpose().array()) / 2.;
        double b = -(1 - alphavd * alphavd) * normv4 / gammav + 2 * alphavd * alphavd;
        vec H = constant(dim, 2.) - (b + 2 * alphavd * alphavd) * vbarbar;  // dim x 1
        vec invH = 1. / H.array();
        mat s_step1 = yy.array() - normv2 / gammav * (yvbar.array().rowwise() * ip_yvbar.transpose().array()).array()
                - constant(dim, lamb+1, 1.).array(); // dim x lamb+1
        vec ip_vbart = vbar.transpose() * t;  // 1 x lamb+1
        mat s_step2 = s_step1.array()
                - (alphavd / gammav * ((2 + normv2) * (t.array().colwise() * vbar.array()).array()
                        - (normv2 * (vbarbar * ip_vbart.transpose())).array()));  // dim x lamb+1
        vec invHvbarbar = invH.array() * vbarbar.array();
        vec ip_s_step2invHvbarbar = invHvbarbar.transpose() * s_step2;  // 1 x lamb+1

        double div = 1 + b * (vbarbar.transpose() * invHvbarbar)(0,0);
        if (div == 0)
//            div = 1E-13;int
            throw std::invalid_argument( "division by 0" );
        mat s = (s_step2.array().colwise() * invH.array()).array()
                - ((b / div) * (invHvbarbar * ip_s_step2invHvbarbar.transpose())).array();  // dim x lamb+1

        vec ip_svbarbar = vbarbar.transpose() * s;  // 1 x lamb+1
        t = t.array() - alphavd * ((2 + normv2) * (s.array().colwise() * vbar.array()).array()
                                    - (vbar * ip_svbarbar.transpose()).array());  // dim x lamb+1
        // update v, D
        vec exw(lamb+1);
        for (int k = 0; k < lamb; k++)
            exw[k] = eta_B(lambF) * weights[k];
        exw[lamb] = c1(lambF);

        v = v.array() + (t * exw).array() / normv;
        D = D.array() + (s * exw).array() * D.array();

        // calculate detA
        if (D.minCoeff() < 0) {
            //throw std::invalid_argument( "D < 0" );
            stop = -1;
            return;
        }
        double nthrootdetA = cexp(D.array().log().sum() / dim + log(1 + (v.transpose() * v)(0,0)) / (2 * dim));
        D = D.array() / nthrootdetA;
        // update sigma
        double G_s = (((z.array() * z.array()).array() - constant(dim, lamb, 1.).array()).matrix() * weights).sum() / dim;
        sigma = sigma * cexp(eta_sigma / 2 * G_s);
    }

private:

    double cexp(double a) { return exp(min(a, 100.0)); } // avoid overflow
    double c1(double lambF) { return c1_cma * (dim - 5) / 6 * (lambF / lamb); }
    double eta_B(double lambF) { return tanh((min(0.02 * lambF, 3 * log(dim)) + 5) / (0.23 * dim + 25)); }
    double alpha_dist(double lambF) { return h_inv * min(1., sqrt(((double)lamb) / dim)) * sqrt(((double)lambF) / lamb); }
    double w_dist_hat(mat z, double lambF) { return cexp(alpha_dist(lambF) * z.norm()); }
    double eta_stag_sigma(double lambF) { return tanh((0.024 * lambF + 0.7 * dim + 20.) / (dim + 12.)); }
    double eta_conv_sigma(double lambF) { return 2. * tanh((0.025 * lambF + 0.75 * dim + 10.) / (dim + 4.)); }
    double f(double a) { return  ((1. + a * a) * cexp(a * a / 2.) / 0.24) - 10. - dim; }
    double f_prime(double a) { return  (1. / 0.24) * a * cexp(a * a / 2.) * (3. + a * a); }

    double get_h_inv(int dim) {
        double h_inv = 1.0;
        while (abs(f(h_inv)) > 1e-10)
            h_inv = h_inv - 0.5 * (f(h_inv) / f_prime(h_inv));
        return h_inv;
    }

    int num_feasible(const vec &evals) {
        int n = 0;
        for (int i = 0; i < evals.size(); i++)
            if (evals[i] != INFINITY) n++;
        return n;
    }

    ivec sort_indices_by(const vec &evals, mat z) {
        int lam = evals.size();
        ivec sorted_indices = sort_index(evals);
        vec sorted_evals = evals(sorted_indices);
        int no_of_feasible_solutions = num_feasible(sorted_evals);
        if (no_of_feasible_solutions != lam) {
            vec distances(lam - no_of_feasible_solutions);
            int n = 0;
            for (int i = 0; i < lam; i++)
                if (evals[i] == INFINITY) {
                    distances[n] = z.col(i).squaredNorm();
                    n++;
                }
            ivec indices_sorted_by_distance = sort_index(distances);
            for (int i = no_of_feasible_solutions; i < lam; i++)
                sorted_indices[i] = sorted_indices[no_of_feasible_solutions
                           + indices_sorted_by_distance[i-no_of_feasible_solutions]];
        }
        return sorted_indices;
    }

    int64_t  runid;
    Fitness *fitfun;
    int dim;
    vec m;
    double sigma;
    int lamb;
    int mu;
    bool use_constraint_violation;
    pcg64 *rs;
    vec v;
    vec D;
    double penalty_coef;
    vec w_rank_hat;
    vec  w_rank;
    double mueff;
    double cs;
    double cc;
    double c1_cma;
    // initialization
    double chiN;
    vec pc;
    vec ps;
    // distance weight parameter
    double h_inv;
    // learning rate
    double eta_m;
    double eta_move_sigma;

    double g = 0;
    int no_of_evals;
    mat z;

    double f_best;
    vec x_best;
    mat xs_no_sort;
    vec evals_no_sort;

    int iterations;
    int maxEvaluations;
    double stopfitness;
    int stop;

    double normv;
    double normv2;
    vec vbar;
    mat y;
    mat x;

};
}

using namespace crmfnes;

extern "C" {
void optimizeCRFMNES_C(int64_t  runid, callback_parallel func_par, int dim,
        double *init, double *lower, double *upper, double sigma,
        int maxEvals, double stopfitness, int popsize,
        int64_t  seed, double penalty_coef, bool use_constraint_violation, bool normalize, double* res) {

	vec guess(dim), lower_limit(dim), upper_limit(dim);
    for (int i = 0; i < dim; i++) // guess is mandatory
   	 guess[i] = init[i];
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

    Fitness fitfun(noop_callback, func_par, dim, 1, lower_limit, upper_limit);
    fitfun.setNormalize(normalize);

    CrfmnesOptimizer opt(runid, &fitfun, dim, guess, sigma, popsize,
            maxEvals, stopfitness, penalty_coef, use_constraint_violation, seed);
    try {
        opt.doOptimize();
    } catch (std::exception &e) {
         cout << e.what() << endl;
    }
    vec bestX = opt.getBestX();
    double bestY = opt.getBestValue();
    for (int i = 0; i < dim; i++)
        res[i] = bestX[i];
    res[dim] = bestY;
    res[dim + 1] = fitfun.evaluations();
    res[dim + 2] = opt.getIterations();
    res[dim + 3] = opt.getStop();
}

uintptr_t initCRFMNES_C(int64_t  runid, int dim,
        double *init, double *lower, double *upper, double sigma,
        int popsize, int64_t  seed, double penalty_coef, bool use_constraint_violation, bool normalize) {

     vec guess(dim), lower_limit(dim), upper_limit(dim);
     for (int i = 0; i < dim; i++) // guess is mandatory
    	 guess[i] = init[i];
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
     Fitness* fitfun = new Fitness(noop_callback, noop_callback_par, dim, 1, lower_limit, upper_limit);
     fitfun->setNormalize(normalize);

     CrfmnesOptimizer* opt = new CrfmnesOptimizer(runid, fitfun, dim, guess, sigma, popsize,
             0, -DBL_MAX, penalty_coef, use_constraint_violation, seed);

     return (uintptr_t) opt;
}

void destroyCRFMNES_C(uintptr_t ptr) {
    CrfmnesOptimizer* opt = (CrfmnesOptimizer*)ptr;
    Fitness* fitfun = opt->getFitfun();
    delete fitfun;
    delete opt;
}

void askCRFMNES_C(uintptr_t ptr, double* xs) {
    CrfmnesOptimizer *opt = (CrfmnesOptimizer*) ptr;
    int n = opt->getDim();
    int lamb = opt->getPopsize();
    mat popX = opt->ask();
    Fitness* fitfun = opt->getFitfun();
    for (int p = 0; p < lamb; p++) {
        vec x = fitfun->getClosestFeasible(fitfun->decode(popX.col(p)));
        for (int i = 0; i < n; i++)
            xs[p * n + i] = x[i];
    }
}

int tellCRFMNES_C(uintptr_t ptr, double* ys) {//, double* xs) {
    CrfmnesOptimizer *opt = (CrfmnesOptimizer*) ptr;
    int lamb = opt->getPopsize();
//    int dim = opt->getDim();
//    Fitness* fitfun = opt->getFitfun();
//    mat popX(dim, lamb);
//    for (int p = 0; p < lamb; p++) {
//        vec x(dim);
//        for (int i = 0; i < dim; i++)
//            x[i] = xs[p * dim + i];
//        popX.col(p) = fitfun->decode(x);
//    }
    vec vals(lamb);
    for (int i = 0; i < lamb; i++)
        vals[i] = ys[i];
    opt->tell(vals);
    return opt->getStop();
}

int populationCRFMNES_C(uintptr_t ptr, double* xs) {
    CrfmnesOptimizer *opt = (CrfmnesOptimizer*) ptr;
    int dim = opt->getDim();
    int lamb = opt->getPopsize();
    mat popX = opt->getPopulation();
    for (int p = 0; p < lamb; p++) {
        vec x = popX.col(p);
        for (int i = 0; i < dim; i++)
            x[i] = xs[p * dim + i];
    }
    return opt->getStop();
}

int resultCRFMNES_C(uintptr_t ptr, double* res) {
    CrfmnesOptimizer *opt = (CrfmnesOptimizer*) ptr;
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

}
