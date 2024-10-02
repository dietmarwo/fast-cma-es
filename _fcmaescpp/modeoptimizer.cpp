// Copyright (c) Dietmar Wolz.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory.

// Eigen based implementation of multi objective
// Differential Evolution using the DE/all/1 strategy.
//
// Requires Eigen version >= 3.4 because new slicing capabilities are used, see
// https://eigen.tuxfamily.org/dox-devel/group__TutorialSlicingIndexing.html
// requires https://github.com/bab2min/EigenRand for random number generation.
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
#include "evaluator.h"

namespace mode_optimizer {

class MoDeOptimizer {

public:

    MoDeOptimizer(long runid_, Fitness *fitfun_, int dim_,
            int nobj_, int ncon_, int seed_, int popsize_,
            double F_, double CR_, double pro_c_, double dis_c_, double pro_m_,
            double dis_m_, bool nsga_update_, double pareto_update_,
            double min_mutate_, double max_mutate_,
            bool *isInt_) {
        // runid used to identify a specific run
        runid = runid_;
        // fitness function to minimize
        fitfun = fitfun_;
        // Number of objective variables/problem dimension
        dim = dim_;
        // Number of objectives
        nobj = nobj_;
        // Number of constraints
        ncon = ncon_;
        // Population size
        popsize = popsize_ > 0 ? popsize_ : 128;
        // DE population update parameters, ignored if nsga_update == true
        F = F0 = F_ > 0 ? F_ : 0.5;
        CR = CR0 = CR_ > 0 ? CR_ : 0.9;
        // Number of iterations already performed.
        iterations = 0;
        // Number of evaluations already performed.
        n_evals = 0;
        // position of current x/y
        pos = 0;
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
        // DE population update parameter used in connection with isInt. Determines
        // the mutation rate for discrete parameters.
        min_mutate = min_mutate_ > 0 ? min_mutate_ : 0.1;
        max_mutate = max_mutate_ > 0 ? max_mutate_ : 0.5;
        // Indicating which parameters are discrete integer values. If defined these parameters will be
        // rounded to the next integer and some additional mutation of discrete parameters are performed.
        isInt = isInt_;
        stop = 0;
        init();
        //std::cout << popX.leftCols(popsize) << std::endl;
    }

    ~MoDeOptimizer() {
        delete rs;
    }

    mat variation(const mat &x) {
        double dis_c_ = (0.5 * rand01(*rs) + 0.5) * dis_c;
        double dis_m_ = (0.5 * rand01(*rs) + 0.5) * dis_m;
        int n2 = x.cols() / 2;
        int n = 2 * n2;
        mat parent1 = x(Eigen::indexing::all, Eigen::seq(0, n2 - 1));
        mat parent2 = x(Eigen::indexing::all, Eigen::seq(n2, n - 1));
        mat beta = mat(dim, n2);
        vec to1;
        if (pro_c < 1.0) {
            to1 = uniformVec(dim, *rs);
        }
        for (int p = 0; p < n2; p++) {
            for (int i = 0; i < dim; i++) {
                if (rand01(*rs) > 0.5 || (pro_c < 1.0 && to1(i) < pro_c))
                    beta(i, p) = 1.0;
                else {
                    double r = rand01(*rs);
                    if (r <= 0.5)
                        beta(i, p) = pow(2 * r, 1.0 / (dis_c_ + 1.0));
                    else
                        beta(i, p) = pow(2 * r, -1.0 / (dis_c_ + 1.0));
                    if (rand01(*rs) > 0.5)
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
                if (rand01(*rs) < limit) { // site
                    double mu = rand01(*rs);
                    double norm = fitfun->norm_i(i, offspring(i, p));
                    if (mu <= 0.5) // temp
                        offspring(i, p) += scale(i) *
                        		(pow(2. * mu + (1. - 2. * mu) * pow(1. - norm, dis_m_ + 1.),
                        					1. / (dis_m_ + 1.)) - 1.);
                    else
                        offspring(i, p) += scale(i) *
                        		(1. - pow(2. * (1. - mu) + 2. * (mu - 0.5) * pow(1. - norm, dis_m_ + 1.),
                        					1. / (dis_m_ + 1.)));
                }
            }
    	}
        fitfun->setClosestFeasible(offspring);
        return offspring;
    }

    vec nextX(int p) {
        if (p == 0) {
            iterations++;
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
        vec xp = popX.col(p);
        int r1, r2, r3;
        do {
            r1 = randInt(*rs,popsize);
            r2 = randInt(*rs,popsize);
            if (pareto_update > 0)
                // sample elite solutions
                 r3 = (int) (pow(rand01(*rs), 1.0 + pareto_update) * popsize);
            else
                // sample from whole population
                 r3 = randInt(*rs,popsize);
        } while (r3 == p || r3 == r1 || r3 == r2 || r2 == p || r2 == r1 || r1 == p);
        vec x1 = popX.col(r1);
        vec x2 = popX.col(r2);
        vec x3 = popX.col(r3);
        vec x = x3 + (x1 - x2) * F;
        int r = randInt(*rs,dim);
        for (int j = 0; j < dim; j++)
            if (j != r && rand01(*rs) > CR)
                x[j] = xp[j];
        modify(x);
        x = fitfun->getClosestFeasible(x);
        return x;
    }

    void modify(vec &x) {
        if (isInt == NULL)
            return;
        double n_ints = 0;
        for (int i = 0; i < dim; i++)
            if (isInt[i])
                n_ints++;
        double to_mutate = min_mutate + rand01(*rs) * (max_mutate - min_mutate);
        for (int i = 0; i < dim; i++) {
            if (isInt[i]) {
                if (rand01(*rs) < to_mutate / n_ints)
                    x[i] = (int)fitfun->sample_i(i, *rs); // resample
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

    bool is_dominated(const vec &y, int p) {
        for (int j = 0; j < y.rows(); j++)
            if (y(j) < popY(j, p))
                return false;
        return true;
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
        //std::cout << "ypar " << domination.transpose() << std::endl;
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

    vec ranks(mat cons, vec eps) {
        imat ci(cons.cols(), cons.rows());
        for (int i = 0; i < cons.rows(); i++)
            ci.col(i) = sort_index(cons.row(i).transpose());
        mat rank(cons.rows(), cons.cols());
        vec alpha = zeros(cons.cols());
        for (int j = 0; j < cons.rows(); j++) {
            for (int i = 0; i < cons.cols(); i++) {
                int ci_ = ci(i, j);
                if (cons(j, ci_) <= eps[j]) {
                    rank(j, ci_) = 0;
                } else {
                    rank(j, ci_) = i;
                    alpha[ci_]++;
                }
            }
        }
        for (int j = 0; j < cons.rows(); j++) {
            for (int i = 0; i < cons.cols(); i++)
                rank(j, i) *= alpha[i] / cons.rows();
        }
        return rank.colwise().sum();
    }

    vec pareto(const mat &ys) {
        if (ncon == 0)
            return pareto_levels(ys);

        int popn = ys.cols();
        mat yobj = ys(Eigen::seqN(0, nobj), Eigen::indexing::all);
        mat ycon = ys(Eigen::indexing::lastN(ncon), Eigen::indexing::all).cwiseMax(0);

        vec eps = zeros(ncon);
        if (iterations > 1 && lastCon.maxCoeff() < 1E90) {
        	vec eps_mean = 0.5*(lastEps + 0.5*lastCon.rowwise().mean());
        	if (eps_mean.maxCoeff() > 1E-8) {
        		eps = eps_mean;
        		//std::cout << "eps = " << eps.transpose() << std::endl;
        	}
        }
        lastCon = ycon;
        lastEps = eps;

        bool feasible[ys.cols()];
        bool hasFeasable = false;
        bool hasInfeasable = false;
        for (int i = 0; i < popn; i++) {
            feasible[i] = (ycon.col(i).array() <= eps.array()).all();
            if (feasible[i])
            	hasFeasable = true;
            else
                hasInfeasable = true;
        }
        vec csum = ranks(ycon, eps);
        if (hasFeasable)
            csum += objranks(yobj);

//        std::cout << "csum " << csum.transpose() << std::endl;
//        std::cout << "ranks " << ranks(ycon).transpose() << std::endl;
//        std::cout << "objranks " << objranks(yobj).transpose() << std::endl;

        ivec ci = sort_index(csum);
        std::vector<int> fiv;
        std::vector<int> viv;
        for (int i = 0; i < ci.size(); i++) // collect feasibles
            if (feasible[ci[i]])
            	fiv.push_back(ci[i]);
            else
            	viv.push_back(ci[i]);
        vec domination = zeros(popn);
        if (hasFeasable) { // compute pareto levels only for feasible
            ivec fi = Eigen::Map<ivec, Eigen::Unaligned>(fiv.data(), fiv.size());
            vec ypar = pareto_levels(yobj(Eigen::indexing::all, fi));
            domination(fi) += ypar;
        }
        // then constraint violations
        if (hasInfeasable) {
             // higher constraint violation level gets lower domination level assigned
            for (int i = 0; i < viv.size(); i++)
                domination(viv[i]) += viv.size() - i;
            for (int i = 0; i < fiv.size(); i++) // feasible first
                domination(fiv[i]) += viv.size() + 1;
        } // higher dominates lower
        return domination;
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
            x0 = popX(Eigen::indexing::all, yi);
            y0 = popY(Eigen::indexing::all, yi);
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
            if (level.size() == 0)
            	continue;
            ivec domlevel = Eigen::Map<ivec, Eigen::Unaligned>(level.data(),
                    level.size());
            mat domx = x0(Eigen::indexing::all, domlevel);
            mat domy = y0(Eigen::indexing::all, domlevel);
            if ((int) (x.size() + domlevel.size()) <= popsize) {
                // whole level fits
                for (int i = 0; i < domy.cols(); i++) {
                    x.push_back(domx.col(i));
                    y.push_back(domy.col(i));
                }
            } else {
                if (domy.cols() > 1) {
                    vec cd = crowd_dist(domy);
                    ivec si = sort_index(cd).reverse();
                    for (int i = 0; i < si.size(); i++) {
                        if (((int) x.size()) >= popsize)
                            break;
                        x.push_back(domx.col(si(i)));
                        y.push_back(domy.col(si(i)));
                    }
                } else {
                    x.push_back(domx.col(0));
                    y.push_back(domy.col(0));
//                    std::cerr << "XXXXXXXX " << level.size()  << " " << domy.col(0) << std::endl;
                }
                break; // we have filled popsize members
            }
        }
        for (int i = 0; i < popsize; i++) {
            popX.col(i) = x[i];
            popY.col(i) = y[i];
        }
        if (nsga_update)
        	vX = variation(popX.leftCols(popsize));

    }

    mat ask() {
       for (int p = 0; p < popsize; p++) {
           vec x = nextX(p);
           popX.col(popsize + p) = x;
       }
       return popX.rightCols(popsize);
    }

    void setX(mat xs) {
       for (int p = 0; p < popsize; p++) {
           vec x = xs.col(p);
           popX.col(popsize + p) = x;
       }
    }

    int tell(mat ys) {
       for (int p = 0; p < popsize; p++)
            popY.col(popsize + p) = ys.col(p);
//            std::cout << p << " x " << popX.col(popsize + p).transpose() << std::endl;
//            std::cout << p << " y " << ys.col(p).transpose() << std::endl;
       pop_update();
       return stop;
    }

    int tell(mat ys, bool nsga_update_, double pareto_update_) {
        nsga_update = nsga_update_;
        pareto_update = pareto_update_;
        return tell(ys);
    }

    mat getPopulation() {
         return popX.leftCols(popsize);
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

    int getPopsize() {
        return popsize;
    }

    void setPopsize(int size) {
    	popsize = size;
    	init();
    }

private:
    long runid;
    Fitness *fitfun;
    int popsize; // population size
    int dim;
    int nobj;
    int ncon;
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
    mat vX;
    int vp;
    mat lastCon;
    vec lastEps;
    int next_size;
    std::vector<bool> vdone;
    int pos;
    bool nsga_update;
    double pareto_update;
    double min_mutate;
    double max_mutate;
    bool *isInt;
};
}

using namespace mode_optimizer;

extern "C" {

uintptr_t initMODE_C(int64_t  runid, int dim,
       int nobj, int ncon, int seed, double *lower, double *upper, bool *ints,
       int popsize, double F, double CR,
       double pro_c, double dis_c, double pro_m, double dis_m,
       bool nsga_update, double pareto_update,
       double min_mutate, double max_mutate) {

    vec lower_limit(dim), upper_limit(dim);
    bool isInt[dim];
    bool useIsInt = false;
    for (int i = 0; i < dim; i++) {
        lower_limit[i] = lower[i];
        upper_limit[i] = upper[i];
        isInt[i] = ints[i];
        useIsInt |= ints[i];
    }
    Fitness* fitfun = new Fitness(noop_callback, noop_callback_par, dim, nobj + ncon, lower_limit, upper_limit);
    MoDeOptimizer* opt = new MoDeOptimizer(runid, fitfun, dim, nobj, ncon, seed, popsize,
            F, CR, pro_c, dis_c, pro_m, dis_m, nsga_update,
            pareto_update, min_mutate, max_mutate, useIsInt ? isInt : NULL);
    return (uintptr_t) opt;
}

void destroyMODE_C(uintptr_t ptr) {
    MoDeOptimizer* opt = (MoDeOptimizer*)ptr;
    Fitness* fitfun = opt->getFitfun();
    delete fitfun;
    delete opt;
}

void askMODE_C(uintptr_t ptr, double* xs) {
    MoDeOptimizer *opt = (MoDeOptimizer*) ptr;
    int dim = opt->getDim();
    int popsize = opt->getPopsize();
    mat pop = opt->ask();
    for (int p = 0; p < popsize; p++) {
        vec x = pop.col(p);
        for (int i = 0; i < dim; i++)
            xs[p * dim + i] = x[i];
    }
}

int tellMODE_C(uintptr_t ptr, double* ys) {
    MoDeOptimizer *opt = (MoDeOptimizer*) ptr;
    int popsize = opt->getPopsize();
    int nobj = opt->getNobj() + opt->getNcon();
    mat vals(nobj, popsize);
    for (int p = 0; p < popsize; p++) {
        vec y(nobj);
        for (int i = 0; i < nobj; i++)
            y[i] = ys[p * nobj + i];
        vals.col(p) = y;
    }
    return opt->tell(vals);
}

int setPopulationMODE_C(uintptr_t ptr, int size, double* xs, double* ys) {
    MoDeOptimizer *opt = (MoDeOptimizer*) ptr;
    int popsize = opt->getPopsize();
    if (size != popsize) {
    	opt->setPopsize(size);
    	popsize = size;
    }
    int nobj = opt->getNobj() + opt->getNcon();
    int dim = opt->getDim();
    mat pop(dim, popsize);
    for (int p = 0; p < popsize; p++) {
        vec x(dim);
        for (int i = 0; i < dim; i++)
            x[i] = xs[p * dim + i];
        pop.col(p) = x;
    }
    //std::cout << pop << std::endl;
    opt->setX(pop);
    mat vals(nobj, popsize);
    for (int p = 0; p < popsize; p++) {
        vec y(nobj);
        for (int i = 0; i < nobj; i++)
            y[i] = ys[p * nobj + i];
        vals.col(p) = y;
    }
    return opt->tell(vals);
}

int tellMODE_switchC(uintptr_t ptr, double* ys, bool nsga_update, double pareto_update) {
    MoDeOptimizer *opt = (MoDeOptimizer*) ptr;
    int popsize = opt->getPopsize();
    int nobj = opt->getNobj() + opt->getNcon();
    mat vals(nobj, popsize);
    for (int p = 0; p < popsize; p++) {
        vec y(nobj);
        for (int i = 0; i < nobj; i++)
            y[i] = ys[p * nobj + i];
        vals.col(p) = y;
    }
    return opt->tell(vals, nsga_update, pareto_update);
}

int populationMODE_C(uintptr_t ptr, double* xs) {
    MoDeOptimizer *opt = (MoDeOptimizer*) ptr;
    int n = opt->getDim();
    int popsize = opt->getPopsize();
    mat pop = opt->getPopulation();
    for (int p = 0; p < popsize; p++) {
        vec x = pop.col(p);
        for (int i = 0; i < n; i++)
            xs[p * n + i] = x[i];
    }
    return opt->getStop();
}
}
