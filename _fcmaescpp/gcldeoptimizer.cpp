// Copyright (c)  Mingcheng Zuo, Dietmar Wolz.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory.

// Eigen based implementation of differential evolution (GCL-DE) derived from
// "A case learning-based differential evolution algorithm for global optimization of interplanetary trajectory design,
//  Mingcheng Zuo, Guangming Dai, Lei Peng, Maocai Wang, Zhengquan Liu", https://doi.org/10.1016/j.asoc.2020.106451

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

namespace gcl_differential_evolution {

static uniform_real_distribution<> distr_01 = std::uniform_real_distribution<>(0, 1);
static normal_distribution<> gauss_01 = std::normal_distribution<>(0, 1);

static double normreal(pcg64* rs, double mu, double sdev) {
	return gauss_01(*rs)*sdev+mu;
}

static vec zeros(int n) {
	return  Eigen::MatrixXd::Zero(n, 1);
}

static vec constant(int n, double val) {
	return  Eigen::MatrixXd::Constant(n, 1, val);
}

static Eigen::MatrixXd uniform(int dx, int dy, pcg64& rs) {
	return Eigen::MatrixXd::NullaryExpr( dx, dy, [&](){return distr_01(rs);});
}

struct IndexVal {
    int index;
    double val;
};

static bool compareIndexVal(IndexVal i1, IndexVal i2) {
    return (i1.val < i2.val);
}

static ivec sort_index(const vec& x) {
	int size = x.size();
	IndexVal ivals[size];
	for (int i = 0; i < size; i++) {
		ivals[i].index = i;
		ivals[i].val = x[i];
	}
	std::sort(ivals, ivals+size, compareIndexVal);
	return Eigen::MatrixXi::NullaryExpr( size, 1, [&ivals](int i){return ivals[i].index;});
}

// wrapper around the fittness function, scales according to boundaries

class Fittness {

public:

    Fittness(callback_type pfunc, const vec& lower_limit,
            const vec& upper_limit) {
        func = pfunc;
        lower = lower_limit;
        upper = upper_limit;
        evaluationCounter = 0;
        if (lower.size() > 0) // bounds defined
            scale = (upper - lower);
    }

    vec getClosestFeasible(const vec& X) const {
        if (lower.size() > 0) {
        	return X.cwiseMin(1.0).cwiseMax(0.0);
        }
        return X;
    }

    double eval(const vec& X) {
        int n = X.size();
        double parg[n];
        for (int i = 0; i < n; i++)
            parg[i] = X(i);
        double res = func(n, parg);
        evaluationCounter++;
        return res;
    }

    void values(const mat& popX, int popsize, vec& ys) {
        for (int p = 0; p < popsize; p++)
            ys[p] = value(popX.col(p));
    }

    double value(const vec& X) {
        if (lower.size() > 0)
            return eval(decode(X));
        else
            return eval(X);
    }

	vec decode(const vec& X) const {
		if (lower.size() > 0)
			return (X.array() * scale.array()).matrix() + lower;
		else
			return X;
	}

    vec encode(const vec& X) const {
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

typedef struct
{
	double F;
	double CR;
	vec x;
} InfoStore;

class GclDeOptimizer {

public:

    GclDeOptimizer(long runid_, Fittness* fitfun_, int dim_, int seed_, int popsize_,
            int maxEvaluations_, double pbest_,
            double stopfitness_, double F0_, double CR0_) {
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
            popsize = 15*dim;
        // termination criteria
        // maximal number of evaluations allowed.
        maxEvaluations = maxEvaluations_;
        // use low value 0 < pbest <= 1 to narrow search.
        pbest = pbest_;
        // Limit for fitness value.
        stopfitness = stopfitness_;
        F0 = F0_;
        CR0 = CR0_;
        // Number of iterations already performed.
        iterations = 0;
        bestValue = DBL_MAX;
        // stop criteria
        stop = 0;
        //std::random_device rd;
        rs = new pcg64(seed_);
        init();
    }

    ~GclDeOptimizer() {
    	delete rs;
    }

    double rnd01() {
        return distr_01(*rs);
    }

    int rndInt(int max) {
        return (int) (max*distr_01(*rs));
    } 

    int findNear(vec x, vector<InfoStore> storage) {
     	int index = 0;
    	double minDist = DBL_MAX;
    	for (int j = 0; j < storage.size(); j++) {
    		double distance = (x - storage[j].x).squaredNorm();
    		if (distance < minDist) {
    			index = j;
    			minDist = distance;
    		}
    	}
    	return index;
    }

    void doOptimize() {
    	int Gr = 2;
     	int gen_stuck = 0;
    	vector<vec> sp;
    	vector<InfoStore> storage;

    	int maxIter = maxEvaluations / popsize + 1;
    	double previous_best = DBL_MAX;
       	double CR = CR0;
        double F = F0;

        // -------------------- Generation Loop --------------------------------

        for (iterations = 1; fitfun->getEvaluations() < maxEvaluations; iterations++) {
        	mat nextX = mat(popX);
        	vec nextY = vec(popY);
         	vec popCR = vec(popsize);
        	vec popF = vec(popsize);
            vec xb = popX.col(0);
            for (int p = 0; p < popsize; p++) {
                vec xi = popX.col(p);
                int r1, r2, r3;
                do { r1 = rndInt(popsize); } while (r1 == p);
                do { r2 = rndInt(int(popsize*pbest)); } while (r2 == p || r2 == r1);
                do { r3 = rndInt(popsize + sp.size()); } while (r3 == p || r3 == r2 || r3 == r1);
                int jr = rndInt(dim);
    			//Produce the CR and F
    			float mu = 1 - sqrt(float(iterations/maxIter))*exp(float(-gen_stuck/iterations));
    			if (storage.empty() || iterations < Gr) {
    				CR = normreal(rs, 0.95, 0.01);
    				F  = normreal(rs, mu, 1);
    				if (F < 0 || F > 1)
    				    F = rnd01();
    			}
    			else {
    				int i = findNear(popX.col(p), storage);
    				CR = storage[i].CR;
    				F  = storage[i].F;
    			}
    			popCR[p] = CR;
       			popF[p] = F;
                vec ui = vec(xi);
                for (int j = 0; j < dim; j++) {
                    if (j == jr || rnd01() < CR) {
    					if (r3 < popsize)
    						ui[j] = popX(j,r1) + F*(popX(j,r2) - popX(j,r3));
    					else
    						ui[j] = popX(j,r1) + F*(popX(j,r2) - sp[r3-popsize][j]);
    					if (ui[j] > 1 || ui[j] < 0)
    						ui[j] = rnd01();
                    }
    			}
                nextX.col(p) = ui;
            }
            vec fitness = vec(popsize);
            fitfun->values(nextX, popsize, fitness);
            for (int p = 0; p < popsize; p++) {
//            	fitness[p] = fitfun->value(nextX.col(p));
                if (!isfinite(fitness[p]))
                	fitness[p] = DBL_MAX;
            }
            for (int p = 0; p < popsize; p++) {
                double y = fitness[p];
                if (y < popY[p]) {
					nextY[p] = y;
                    if (y < bestValue) {
						bestValue = y;
						bestX = nextX.col(p);
						if (isfinite(stopfitness) && bestValue < stopfitness) {
							stop = 1;
							return;
						}
					}
                 } else // no improvement
                	 nextX.col(p) = popX.col(p);
            }
    		for (int p = 0; p < popsize; p++) {
    			if (nextY[p] < popY[p]) { // improvement
    				if (sp.size() < popsize)
    				    sp.push_back(popX.col(p));
    				else
    					sp[rndInt(popsize)] = popX.col(p);
    				InfoStore entry;
    				entry.CR = popCR[p];
    				entry.F = popF[p];
    				entry.x = nextX.col(p);
    				storage.push_back(entry);
    			}
    		}
            // sort population
    		ivec sindex = sort_index(nextY);
    		popY = nextY(sindex, Eigen::all);
    		popX = nextX(Eigen::all, sindex);
    		if (iterations % Gr == 0)
    			storage.clear();
    		if (bestValue >= previous_best)
    			gen_stuck++;
			else
				gen_stuck = 0;
			previous_best = bestValue;
        }
    }
 
    void init() {
        popX = uniform(dim, popsize, *rs);
        popY = vec(popsize);
        for (int i = 0; i < popsize; i++)
            popY[i] = fitfun->value(popX.col(i)); // compute fitness
        bestX = popX.col(0);
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
      Fittness* fitfun;
      int popsize; // population size
      int dim;
      int maxEvaluations;
      double pbest;
      double stopfitness;
      int iterations;
      double bestValue;
      vec bestX;
      int stop;
      double F0;
      double CR0;
      pcg64* rs;
      mat popX;
      vec popY;
};

// see https://cvstuff.wordpress.com/2014/11/27/wraping-c-code-with-python-ctypes-memory-and-pointers/

}

using namespace gcl_differential_evolution;

extern "C" {
    double* optimizeGCLDE_C(long runid, callback_type func, int dim, int seed,
            double *lower, double *upper, int maxEvals, double pbest,
            double stopfitness, int popsize, double F0, double CR0) {
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
        GclDeOptimizer opt(
            runid,
            &fitfun,
            dim,
            seed,
            popsize,
            maxEvals,
            pbest,
            stopfitness,
            F0,
            CR0);
        try {
            opt.doOptimize();
            vec bestX = fitfun.decode(opt.getBestX());
            double bestY = opt.getBestValue();
            for (int i = 0; i < n; i++)
                res[i] = bestX[i];
            res[n] = bestY;
            res[n+1] = fitfun.getEvaluations();
            res[n+2] = opt.getIterations();
            res[n+3] = opt.getStop();
            return res;
        } catch (std::exception& e) {
            cout << e.what() << endl;
            return res;
        }
    }
}
