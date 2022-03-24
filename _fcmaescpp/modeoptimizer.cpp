// Copyright (c) Dietmar Wolz.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory.

// Eigen based implementation of multi objective
// Differential Evolution using the DE/all/1 strategy.
//
// Can switch to NSGA-II like population update via parameter 'nsga_update'.
// Then it works essentially like NSGA-II but instead of the tournament selection
// the whole population is sorted and the best individuals survive. To do this
// efficiently the crowd distance ordering is slightly inaccurate.
//
// Supports parallel fitness function evaluation.
//
// Features enhanced multiple constraint ranking (https://www.jstage.jst.go.jp/article/tjpnsec/11/2/11_18/_article/-char/en/)
// improving its performance in handling constraints for engineering design optimization.
//
// Enables the comparison of DE and NSGA-II population update mechanism with everything else
// kept completely identical.
//
// Uses the following deviation from the standard DE algorithm:
// a) oscillating CR/F parameters.
//
// You may keep parameters F and CR at their defaults since this implementation works well with the given settings for most problems,
// since the algorithm oscillates between different F and CR settings.
//
// For expensive objective functions (e.g. machine learning parameter optimization) use the workers
// parameter to parallelize objective function evaluation. The workers parameter is limited by the
// population size.
//
// The ints parameter is a boolean array indicating which parameters are discrete integer values. This
// parameter was introduced after observing non optimal DE-results for the ESP2 benchmark problem:
// https://github.com/AlgTUDelft/ExpensiveOptimBenchmark/blob/master/expensiveoptimbenchmark/problems/DockerCFDBenchmark.py
// If defined it causes a "special treatment" for discrete variables: They are rounded to the next integer value and
// there is an additional mutation to avoid getting stuck at local minima.

#include <Eigen/Core>
#include <iostream>
#include <float.h>
#include <stdint.h>
#include <ctime>
#include <random>
#include <queue>
#include <tuple>
#include "pcg_random.hpp"
#include "evaluator.h"

namespace mode_optimizer {

class MoDeOptimizer {

public:

    MoDeOptimizer(long runid_, Fitness *fitfun_, callback_type log_, int dim_,
            int nobj_, int ncon_, int seed_, int popsize_, int maxEvaluations_,
            double F_, double CR_, double pro_c_, double dis_c_, double pro_m_,
            double dis_m_, bool nsga_update_, double pareto_update_,
            int log_period_, bool *isInt_) {
        // runid used to identify a specific run
        runid = runid_;
        // fitness function to minimize
        fitfun = fitfun_;
        // callback to log progress
        log = log_;
        // Number of objective variables/problem dimension
        dim = dim_;
        // Number of objectives
        nobj = nobj_;
        // Number of constraints
        ncon = ncon_;
        // Population size
        popsize = popsize_ > 0 ? popsize_ : 128;
        // maximal number of evaluations allowed.
        maxEvaluations = maxEvaluations_ > 0 ? maxEvaluations_ : 500000;
        // DE population update parameters, ignored if nsga_update == true
        F = F0 = F_ > 0 ? F_ : 0.5;
        CR = CR0 = CR_ > 0 ? CR_ : 0.9;
        // Number of iterations already performed.
        iterations = 0;
        // Number of evaluations already performed.
        n_evals = 0;
        // position of current x/y
        pos = 0;
        //std::random_device rd;
        rs = new pcg64(seed_);
        // NSGA population update parameters, ignored if nsga_update == false
        // usually use pro_c = 1.0, dis_c = 20.0, pro_m = 1.0, dis_m = 20.0.
        pro_c = pro_c_;
        dis_c = dis_c_;
        pro_m = pro_m_;
        dis_m = dis_m_;
        // if true, use NSGA population update, if false, use DE population update
        // Use DE update to diversify your results.
        nsga_update = nsga_update_;
        // DE population update parameter. Only applied if nsga_update = false.
        // Favor better solutions for sample generation. Default 0 -
        // use all population members with the same probability.
        pareto_update = pareto_update_;
        // The log callback is called each log_period iterations
        log_period = log_period_;
        if (log_period <= 0)
            log_period = 1000;
        // Indicating which parameters are discrete integer values. If defined these parameters will be
        // rounded to the next integer and some additional mutation of discrete parameters are performed.
        isInt = isInt_;
        init();
    }

    ~MoDeOptimizer() {
        delete rs;
    }

    double rnd01() {
        return distr_01(*rs);
    }

    int rndInt(int max) {
        return (int) (max * distr_01(*rs));
    }

    vec nextX(int p) {
        if (p == 0) {
            iterations++;
            if (iterations % log_period == 0) {
                if (log(popX.cols(), popX.data(), popY.data()))
                    fitfun->setTerminate();
            }
        }
        if (nsga_update) {
            vec x = vX.col(vp);
            vp = (vp + 1) % popsize;
            return x;
        }
        // use DE update strategy.
        if (p == 0) {
            CR = iterations % 2 == 0 ? 0.5 * CR0 : CR0;
            F = iterations % 2 == 0 ? 0.5 * F0 : F0;
        }
        int r3;
        if (pareto_update > 0) {
            // sample elite solutions
            do {
                r3 = (int) (pow(rnd01(), 1.0 + pareto_update) * popsize);
            } while (r3 == p);
        } else {
            // sample from whole population
            do {
                r3 = rndInt(popsize);
            } while (r3 == p);
        }
        vec xp = popX.col(p);
        vec x3 = popX.col(r3);
        int r1, r2;
        do {
            r1 = rndInt(popsize);
        } while (r1 == p || r1 == r3);
        do {
            r2 = rndInt(popsize);
        } while (r2 == p || r2 == r3 || r2 == r1);
        vec x1 = popX.col(r1);
        vec x2 = popX.col(r2);
        vec x = x3 + (x1 - x2) * F;
        int r = rndInt(dim);
        for (int j = 0; j < dim; j++)
            if (j != r && rnd01() > CR)
                x[j] = xp[j];
        x = fitfun->getClosestFeasible(x);
        modify(x);
        return x;
    }

    void modify(vec &x) {
        if (isInt == NULL)
            return;
        double n_ints = 0;
        for (int i = 0; i < dim; i++)
            if (isInt[i])
                n_ints++;
        double min_mutate = 0.5;
        double max_mutate = std::max(1.0, n_ints / 20.0);
        double to_mutate = min_mutate + rnd01() * (max_mutate - min_mutate);
        for (int i = 0; i < dim; i++) {
            if (isInt[i]) {
                if (rnd01() < to_mutate / n_ints)
                    x[i] = fitfun->sample_i(i, *rs); // resample
                x[i] = std::round(x[i]);
            }
        }
    }

    vec crowd_dist(mat &y) { // crowd distance for 1st objective
        int n = y.cols();
        vec y0 = y.row(0);
        ivec si = sort_index(y0); // sort 1st objective
        vec y0s = y0(si); // sorted y0
        vec d(n - 1);
        for (int i = 0; i < n - 1; i++)
            d(i) = y0s[i + 1] - y0s[i]; // neighbor distance
        if (d.maxCoeff() == 0)
            return zeros(n);
        vec dsum = zeros(n);
        for (int i = 0; i < n; i++) {
            if (i > 0)
                dsum(i) += d(i - 1); // distance to left
            if (i < n - 1)
                dsum(i) += d(i); //  distance to right
        }
        dsum(0) = DBL_MAX; // keep borders
        dsum(n - 1) = DBL_MAX;
        vec ds(n);
        ds(si) = dsum;  // inverse order
        return ds;
    }

    bool is_dominated(const mat &y, int i, int index) {
        for (int j = 0; j < y.rows(); j++)
            if (y(j, i) < y(j, index))
                return false;
        return true;
    }

    vec pareto_levels(const mat &y) {
        int n = y.cols();
        ivec pareto(n);
        for (int i = 0; i < n; i++)
            pareto(i) = i;
        vec domination = zeros(n);
        bool mask[n];
        for (int i = 0; i < n; i++)
            mask[i] = true;
        for (int index = 0; index < n;) {
            for (int i = 0; i < n; i++) {
                if (i != index && mask[i] && is_dominated(y, i, index))
                    mask[i] = false;
            }
            for (int i = 0; i < n; i++) {
                if (mask[i])
                    domination[i] += 1;
            }
            index++;
            while (!mask[index] && index < n)
                index++;
        }
        return domination;
    }

    vec objranks(mat objs) {
        imat ci(objs.cols(), objs.rows());
        for (int i = 0; i < objs.rows(); i++)
            ci.col(i) = sort_index(objs.row(i).transpose());
        mat rank(objs.rows(), objs.cols());
        for (int j = 0; j < objs.rows(); j++)
            for (int i = 0; i < objs.cols(); i++) {
                rank(j, ci(i, j)) = i;
            }
        return rank.colwise().sum();
    }

    vec ranks(mat cons) {
        imat ci(cons.cols(), cons.rows());
        for (int i = 0; i < cons.rows(); i++)
            ci.col(i) = sort_index(cons.row(i).transpose());
        mat rank(cons.rows(), cons.cols());
        vec alpha = zeros(cons.rows());
        for (int j = 0; j < cons.rows(); j++) {
            for (int i = 0; i < cons.cols(); i++) {
                if (cons(j, i) <= 0) {
                    rank(j, ci(i, j)) = 0;
                } else {
                    rank(j, ci(i, j)) = i;
                    alpha[j]++;
                }
            }
        }
        for (int j = 0; j < cons.rows(); j++) {
            for (int i = 0; i < cons.cols(); i++)
                rank(j, ci(i, j)) *= alpha[j] / cons.rows();
        }
        return rank.colwise().sum();
    }

    vec pareto(const mat &ys) {
        if (ncon == 0)
            return pareto_levels(ys);
        int popn = ys.cols();
        mat yobj = ys(Eigen::seqN(0, nobj), Eigen::all);
        mat ycon = ys(Eigen::lastN(ncon), Eigen::all);
        vec csum = ranks(ycon);
        bool feasible[ys.cols()];
        bool hasFeasible = false;
        for (int i = 0; i < ys.cols(); i++) {
            feasible[i] = ycon.col(i).maxCoeff() <= 0;
            if (feasible[i])
                hasFeasible = true;
        }
        if (hasFeasible)
            csum += objranks(yobj);
        // first pareto front of feasible solutions
        vec domination = zeros(popn);
        std::vector<int> cyv;
        for (int i = 0; i < ys.cols(); i++) // collect feasibles
            if (feasible[i])
                cyv.push_back(i);
        ivec cy = Eigen::Map<ivec, Eigen::Unaligned>(cyv.data(), cyv.size());
        if (hasFeasible) { // compute pareto levels only for feasible
            vec ypar = pareto_levels(yobj(Eigen::all, cy));
            domination(cy) += ypar;
        }
        // then constraint violations
        ivec ci = sort_index(csum);
        std::vector<int> civ;
        for (int i = 0; i < ci.size(); i++)
            if (!feasible[ci(i)])
                civ.push_back(ci(i));
        if (civ.size() > 0) {
            ivec ci = Eigen::Map<ivec, Eigen::Unaligned>(civ.data(),
                    civ.size());
            int maxcdom = ci.size();
            // higher constraint violation level gets lower domination level assigned
            for (int i = 0; i < ci.size(); i++)
                domination(ci(i)) += maxcdom - i;
            if (cy.size() > 0) { // priorize feasible solutions
                for (int i = 0; i < cy.size(); i++)
                    domination(cy(i)) += maxcdom + 1;
            }
        } // higher dominates lower
        return domination;
    }

    mat variation(const mat &x) {
        int n2 = x.cols() / 2;
        int n = 2 * n2;
        mat parent1 = x(Eigen::all, Eigen::seq(0, n2 - 1));
        mat parent2 = x(Eigen::all, Eigen::seq(n2, n - 1));
        mat beta = mat(dim, n2);
        vec to1;
        if (pro_c < 1.0) {
            to1 = uniformVec(dim, *rs);
        }
        for (int p = 0; p < n2; p++) {
            for (int i = 0; i < dim; i++) {
                if (rnd01() > 0.5 || (pro_c < 1.0 && to1(i) < pro_c))
                    beta(i, p) = 1.0;
                else {
                    double r = rnd01();
                    if (r <= 0.5)
                        beta(i, p) = pow(2 * r, 1.0 / (dis_c + 1.0));
                    else
                        beta(i, p) = pow(2 * r, -1.0 / (dis_c + 1.0));
                    if (rnd01() > 0.5)
                        beta(i, p) = -beta(i, p);
                }
            }
        }
        mat offspring1 = ((parent1 + parent2) * 0.5);
        mat offspring2 = mat(offspring1);
        mat delta = (beta.array() * (parent1 - parent2).array()).matrix() * 0.5;
        offspring1 += delta;
        offspring2 -= delta;
        mat offspring = mat(dim, n);
        offspring << offspring1, offspring2;

        double limit = pro_m / dim;
        vec scale = fitfun->scale();
        for (int p = 0; p < n; p++) {
            for (int i = 0; i < dim; i++) {
                if (rnd01() < limit) { // site
                    double mu = rnd01();
                    double norm = fitfun->norm_i(i, offspring(i, p));
                    if (mu <= 0.5) // temp
                        offspring(i, p) += scale(i) *
                        (pow(2. * mu + (1. - 2. * mu) * pow(1. - norm, dis_m + 1.),
                                1. / (dis_m + 1.)) - 1.);
                    else
                        offspring(i, p) += scale(i) *
                        (1. - pow(2. * (1. - mu) + 2. * (mu - 0.5) * pow(1. - norm, dis_m + 1.),
                                1. / (dis_m + 1.)));
                }
            }
        }
        fitfun->setClosestFeasible(offspring);
        return offspring;
    }

    ivec random_int_vector(int size) {
        std::vector<int> v;
        for (int i = 0; i < size; i++)
            v.push_back(i);
        std::random_shuffle(v.begin(), v.end());
        return Eigen::Map<ivec, Eigen::Unaligned>(v.data(),
                v.size());
    }

    void pop_update() {
        mat x0 = popX;
        mat y0 = popY;
        if (nobj == 1) {
            ivec yi = sort_index(popY.row(0)).reverse();
            x0 = popX(Eigen::all, yi);
            y0 = popY(Eigen::all, yi);
        }
        vec domination = pareto(y0);
        std::vector<vec> x;
        std::vector<vec> y;
        int maxdom = (int) domination.maxCoeff();
        for (int dom = maxdom; dom >= 0; dom--) {
            std::vector<int> level;
            for (int i = 0; i < domination.size(); i++)
                if (domination(i) == dom)
                    level.push_back(i);
            ivec domlevel = Eigen::Map<ivec, Eigen::Unaligned>(level.data(),
                    level.size());
            mat domx = x0(Eigen::all, domlevel);
            mat domy = y0(Eigen::all, domlevel);
            if ((int) (x.size() + domlevel.size()) <= popsize) {
                // whole level fits
                for (int i = 0; i < domy.cols(); i++) {
                    x.push_back(domx.col(i));
                    y.push_back(domy.col(i));
                }
            } else {
                std::vector<int> si;
                si.push_back(0);
                if (domy.cols() > 1) {
                    vec cd = crowd_dist(domy);
                    ivec si = sort_index(cd).reverse();
                    for (int i = 0; i < si.size(); i++) {
                        if (((int) x.size()) >= popsize)
                            break;
                        x.push_back(domx.col(si(i)));
                        y.push_back(domy.col(si(i)));
                    }
                }
                break;
            }
        }
        for (int i = 0; i < popsize; i++) {
            popX.col(i) = x[i];
            popY.col(i) = y[i];
        }
        if (nsga_update)
            vX = variation(popX(Eigen::all, Eigen::seqN(0, popsize)));
    }

    vec ask(int &p) {
        p = pos;
        vec x = nextX(p);
        pos = (pos + 1) % popsize;
        return x;
    }

    int tell(const vec &y, const vec &x, int p) {
        long unsigned int dp = 0;
        for (; dp < vdone.size(); dp++)
            if (!vdone[dp])
                break;
        nX.col(dp) = x;
        nY.col(dp) = y;
        vdone[dp] = true;
        int ndone = 0;
        for (long unsigned int i = 0; i < vdone.size(); i++)
            if (vdone[i])
                ndone++;
        if (ndone >= popsize) {
            int p = popsize;
            for (dp = 0; dp < vdone.size(); dp++) {
                if (vdone[dp]) {
                    popX.col(p) = nX.col(dp);
                    popY.col(p) = nY.col(dp);
                    vdone[dp] = false;
                    if (p >= popY.cols())
                        break;
                    p++;
                }
            }
            pop_update();
        }
        n_evals += 1;
        //        if (n_evals % 1000 == 999)
        //        	std::cout << popY << std::endl;
        return stop;
    }

    void doOptimize() {
        iterations = 0;
        fitfun->resetEvaluations();
        while (fitfun->evaluations() < maxEvaluations && !fitfun->terminate()) {
            for (int p = 0; p < popsize; p++) {
                vec x = nextX(p);
                popX.col(popsize + p) = x;
                popY.col(popsize + p) = fitfun->eval(x);
            }
            pop_update();
        }
    }

    void do_optimize_delayed_update(int workers) {
        iterations = 0;
        fitfun->resetEvaluations();
        workers = std::min(workers, popsize); // workers <= popsize
        evaluator eval(fitfun, nobj, workers);
        vec evals_x[popsize];
        // fill eval queue with initial population
        for (int i = 0; i < workers; i++) {
            int p;
            vec x = ask(p);
            eval.evaluate(x, p);
            evals_x[p] = x;
        }
        while (fitfun->evaluations() < maxEvaluations && !fitfun->terminate()) {
            vec_id *vid = eval.result();
            vec y = vec(vid->_v);
            int p = vid->_id;
            delete vid;
            vec x = evals_x[p];
            tell(y, x, p); // tell evaluated x
            if (fitfun->evaluations() >= maxEvaluations)
                break;
            x = ask(p);
            eval.evaluate(x, p);
            evals_x[p] = x;
        }
    }

    void init() {
        popX = mat(dim, 2 * popsize);
        popY = mat(nobj + ncon, 2 * popsize);
        for (int p = 0; p < popsize; p++) {
            popX.col(p) = fitfun->sample(*rs);
            popY.col(p) = constant(nobj + ncon, DBL_MAX);
        }
        next_size = 2 * popsize;
        vdone = std::vector<bool>(next_size, false);
        nX = mat(dim, next_size);
        nY = mat(nobj + ncon, next_size);
        vX = mat(popX);
        vp = 0;
    }

    mat getX() {
        return popX;
    }

    mat getY() {
        return popY;
    }

    double getIterations() {
        return iterations;
    }

    double getStop() {
        return stop;
    }

    Fitness* getFitfun() {
        return fitfun;
    }

    int getDim() {
        return dim;
    }

    int getNobj() {
        return nobj;
    }

    int getNcon() {
        return ncon;
    }

private:
    long runid;
    Fitness *fitfun;
    callback_type log;
    int popsize; // population size
    int dim;
    int nobj;
    int ncon;
    int maxEvaluations;
    double keep;
    double stopfitness;
    int iterations;
    int n_evals;
    int stop;
    double F0;
    double CR0;
    double F;
    double CR;
    double pro_c;
    double dis_c;
    double pro_m;
    double dis_m;
    pcg64 *rs;
    mat popX;
    mat popY;
    mat nX;
    mat nY;
    mat vX;
    int vp;
    int next_size;
    std::vector<bool> vdone;
    int pos;
    bool nsga_update;
    double pareto_update;
    int log_period;
    bool *isInt;
};
}

using namespace mode_optimizer;

extern "C" {
void optimizeMODE_C(long runid, callback_type func, callback_type log, int dim,
        int nobj, int ncon, int seed, double *lower, double *upper, bool *ints,
        int maxEvals, int popsize, int workers, double F, double CR,
        double pro_c, double dis_c, double pro_m, double dis_m,
        bool nsga_update, double pareto_update, int log_period, double *res) {
    vec lower_limit(dim), upper_limit(dim);
    bool isInt[dim];
    bool useIsInt = false;
    for (int i = 0; i < dim; i++) {
        lower_limit[i] = lower[i];
        upper_limit[i] = upper[i];
        isInt[i] = ints[i];
        useIsInt |= ints[i];
        if (isInt[i]) {
            // adjust bounds because ints are rounded
            lower_limit[i] -= .499999999;
            upper_limit[i] += .499999999;
        }
    }
    Fitness fitfun(func, dim, nobj + ncon, lower_limit, upper_limit);
    MoDeOptimizer opt(runid, &fitfun, log, dim, nobj, ncon, seed, popsize,
            maxEvals, F, CR, pro_c, dis_c, pro_m, dis_m, nsga_update,
            pareto_update, log_period, useIsInt ? isInt : NULL);
    try {
        if (workers <= 1)
            opt.doOptimize();
        else
            opt.do_optimize_delayed_update(workers);
        double *xdata = opt.getX().data();
        memcpy(res, xdata, sizeof(double) * opt.getX().size());
    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
    }
}
}
