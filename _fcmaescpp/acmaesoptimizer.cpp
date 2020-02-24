//#define ARMA_NO_DEBUG

#include <iostream>
#include <armadillo>
#include <float.h>
#include <ctime>

using namespace std;
using namespace arma;

typedef double (*callback_type)(int, double[]);
typedef bool (*is_terminate_type)(long, int, double); // runid, iter, value -> isTerminate

static uvec inverse(const uvec& indices) {
    uvec inverse = uvec(indices.size());
    for (int i = 0; i < indices.size(); i++)
        inverse(indices(i)) = i;
    return inverse;
}

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
            scale = 0.5 * (upper - lower);
            typx = 0.5 * (upper + lower);
        }
    }

    void closestFeasible(vec &X) const { // in place
        if (lower.size() > 0)
            X.for_each([](double &val) {
                val = max(min(val, 1.0), -1.0);
            });
    }

    vec getClosestFeasible(const vec &col) const {
        if (lower.size() > 0) {
            vec X(col);
            X.for_each([](double &val) {
                val = max(min(val, 1.0), -1.0);
            });
            return X;
        }
        return col;
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
        double value;
        if (lower.size() > 0) {
            return eval(decode(getClosestFeasible(X)));
        } else
            return eval(X);
    }

    vec encode(const vec &X) const {
        if (lower.size() > 0)
            return (X - typx) / scale;
        else
            return X;
    }

    vec decode(const vec &X) const {
        if (lower.size() > 0)
            return (X % scale) + typx;
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
   vec typx;
};

class AcmaesOptimizer {

public:

    AcmaesOptimizer(long runid_, Fittness* fitfun_, int popsize_, int mu_, const vec &guess_, const vec &inputSigma_,
            int maxIterations_, int maxEvaluations_, double accuracy_, double stopfitness_, is_terminate_type isTerminate_) {
// runid used in isTerminate callback to identify a specific run at different iteration
        runid = runid_;
// fitness function to minimize
        fitfun = fitfun_;
// initial guess for the arguments of the fitness function
        guess = guess_;
// accuracy = 1.0 is default, > 1.0 reduces accuracy
        accuracy = accuracy_;
// callback to check if to terminate
        isTerminate = isTerminate_;
// Number of objective variables/problem dimension
        dim = guess_.size();
//     Population size, offspring number. The primary strategy parameter to play
//     with, which can be increased from its default value. Increasing the
//     population size improves global search properties in exchange to speed.
//     Speed decreases, as a rule, at most linearly with increasing population
//     size. It is advisable to begin with the default small population size.
        if (popsize_ > 0)
            popsize = popsize_;
        else
            popsize = 4 + int(3. * log(dim));
//     Individual sigma values - initial search volume. inputSigma determines
//     the initial coordinate wise standard deviations for the search. Setting
//     SIGMA one third of the initial search region is appropriate.
        if (inputSigma_.size() == 1)
            inputSigma = vec(dim).fill(inputSigma_(0));
        else
            inputSigma = inputSigma_;
// Overall standard deviation - search volume.
        sigma = max(inputSigma);
// termination criteria
// maximal number of evaluations allowed.
        maxEvaluations = maxEvaluations_;
// maximal number of iterations allowed.
        maxIterations = maxIterations_;
// Limit for fitness value.
        stopfitness = stopfitness_;
// Stop if x-changes larger stopTolUpX.
        stopTolUpX = 1e3 * sigma;
// Stop if x-change smaller stopTolX.
        stopTolX = 1e-11 * sigma * accuracy;
// Stop if fun-changes smaller stopTolFun.
        stopTolFun = 1e-12 * accuracy;
// Stop if back fun-changes smaller stopTolHistFun.
        stopTolHistFun = 1e-13 * accuracy;
// selection strategy parameters
// Number of parents/points for recombination.
        mu = mu_ > 0 ? mu_ : popsize / 2;
// Array for weighted recombination.
        weights = (log(sequence(1, mu, 1)) * -1.) + log(mu + 0.5);
        double sumw = sum(weights);
        double sumwq = sum(weights % weights);
        weights *= 1. / sumw;
// Variance-effectiveness of sum w_i x_i.
        mueff = sumw * sumw / sumwq;

// dynamic strategy parameters and constants
// Cumulation constant.
        cc = (4. + mueff / dim) / (dim + 4. + 2. * mueff / dim);
// Cumulation constant for step-size.
        cs = (mueff + 2.) / (dim + mueff + 3.);
// Damping for step-size.
           damps = (1. + 2. * ::max(0., ::sqrt((mueff - 1.)
                    / (dim + 1.)) - 1.))
                    * max(0.3, 1.
                            - // modification for short runs
                            dim / (1e-6 + min(maxIterations, maxEvaluations
                                    / popsize))) + cs; // minor increment
// Learning rate for rank-one update.
        ccov1 = 2. / ((dim + 1.3) * (dim + 1.3) + mueff);
// Learning rate for rank-mu update'
        ccovmu = min(1. - ccov1,
                2. * (mueff - 2. + 1. / mueff)
                        / ((dim + 2.) * (dim + 2.) + mueff));
// Expectation of ||N(0,I)|| == norm(randn(N,1)).
        chiN = sqrt(dim) * (1. - 1. / (4. * dim) + 1 / (21. * dim * dim));
        ccov1Sep = min(1., ccov1 * (dim + 1.5) / 3.);
        ccovmuSep = min(1. - ccov1, ccovmu * (dim + 1.5) / 3.);

// CMA internal values - updated each generation
// Objective variables.
        xmean = fitfun->encode(guess);
// Evolution path.
        pc = zeros<vec>(dim);
// Evolution path for sigma.
        ps = zeros<vec>(dim);
// Norm of ps, stored for efficiency.
        normps = norm(ps);
// Coordinate system.
        B = mat(dim,dim).eye();
// Diagonal of sqrt(D), stored for efficiency.
        diagD = inputSigma / sigma;
        diagC = square(diagD);
// B*D, stored for efficiency.
        BD = B % repmat(diagD.t(), dim, 1);
// Covariance matrix.
        C = B * (mat(dim,dim).eye() * B.t());
// Number of iterations already performed.
        iterations = 0;
// Size of history queue of best values.
        historySize = 10 + int(3. * 10. * dim / popsize);
// stop criteria
        stop = 0;
// best value so far
        bestValue = fitfun->value(xmean);
// best parameters so far
        bestX = guess;
// History queue of best values.
        fitnessHistory = vec(historySize).fill(DBL_MAX);
        fitnessHistory(0) = bestValue;
    }

    // param zmean weighted row matrix of the gaussian random numbers generating the current offspring
    // param xold xmean matrix of the previous generation
    // return hsig flag indicating a small correction

    bool updateEvolutionPaths(const vec& zmean, const vec& xold) {
        ps = ps * (1. - cs) + ((B * zmean) * sqrt(cs * (2. - cs) * mueff));
        normps = norm(ps);
        bool hsig = normps / sqrt(1. - pow(1. - cs, 2. * iterations)) / chiN < 1.4 + 2. / (dim + 1.);
        pc *= (1. - cc);
        if (hsig)
            pc += (xmean - xold) * (sqrt(cc * (2. - cc) * mueff) / sigma);
        return hsig;
    }

    // param hsig flag indicating a small correction
    // param bestArx fitness-sorted matrix of the argument vectors producing the current offspring
    // param arz unsorted matrix containing the gaussian random values of the current offspring
    // param arindex indices indicating the fitness-order of the current offspring
    // param xold xmean matrix of the previous generation

    double updateCovariance(bool hsig, const mat& bestArx, const mat& arz,
            const uvec& arindex, const mat& xold) {
        double negccov = 0;
        if (ccov1 + ccovmu > 0) {
            mat arpos = (bestArx - repmat(xold, 1, mu)) * (1. / sigma); // mu difference vectors
            mat roneu = pc * pc.t() * ccov1;
            // minor correction if hsig==false
            double oldFac = hsig ? 0 : ccov1 * cc * (2. - cc);
            oldFac += 1. - ccov1 - ccovmu;
            // Adapt covariance matrix C active CMA
            negccov = (1. - ccovmu) * 0.25 * mueff
                    / (pow(dim + 2., 1.5) + 2. * mueff);
            double negminresidualvariance = 0.66;
            // keep at least 0.66 in all directions, small popsize are most critical
            double negalphaold = 0.5; // where to make up for the variance loss,
            // prepare vectors, compute negative updating matrix Cneg
            uvec arReverseIndex = reverse(arindex);
            mat arzneg = arz.cols(arReverseIndex.head(mu));
            vec arnorms = sqrt(sum(square(arzneg.t()), 1)); // rowsum
            uvec idxnorms = sort_index(arnorms);
            vec arnormsSorted = arnorms(idxnorms);
            uvec idxReverse = reverse(idxnorms);
            vec arnormsReverse = arnorms(idxReverse);
            arnorms = arnormsReverse / arnormsSorted;
            vec arnormsInv = arnorms(inverse(idxnorms));
            mat sqarnw = square(arnormsInv).t() * weights;
            double negcovMax = (1. - negminresidualvariance) / sqarnw(0);
            if (negccov > negcovMax)
                negccov = negcovMax;
            arzneg = arzneg % repmat(arnormsInv.t(), dim, 1);
            mat artmp = BD * arzneg;
            mat Cneg = artmp * diagmat(weights) * artmp.t();
            oldFac += negalphaold * negccov;
            C = (C * oldFac) + roneu
                    + ( arpos * (ccovmu + (1. - negalphaold) * negccov) *
                               (repmat(weights, 1, dim) % arpos.t()))
                    - (Cneg * negccov);
        }
        return negccov;
    }

    // Update B and diagD from C
    // param negccov Negative covariance factor.

    void updateBD(double negccov) {

        if (ccov1 + ccovmu + negccov > 0
                && (std::fmod(iterations, 1. / (ccov1 + ccovmu + negccov) / dim / 10.)) < 1.) {
            // to achieve O(N^2) enforce symmetry to prevent complex numbers
            C = trimatu(C) + trimatu(C, 1).t();
            // diagD defines the scaling
            vec eigval;
            mat eigvec;
            eig_sym(eigval, eigvec, C);

            diagD = reverse(eigval); // descending order of eigenvalues
            for (int i = 0; i < dim; i++)
                B.col(i) = eigvec.col(dim - 1 - i);

            if (diagD.min() <= 0) {
                for (int i = 0; i < dim; i++)
                    if (diagD(i, 0) < 0)
                        diagD(i, 0) = 0.;
                double tfac = diagD.max() / 1e14;
                C += mat(dim,dim).eye() * tfac;
                diagD += ones(dim, 1) * tfac;
            }
            if (diagD.max() > 1e14 * diagD.min()) {
                double tfac = diagD.max() / 1e14 - diagD.min();
                C += mat(dim,dim).eye() * tfac;
                diagD += ones(dim, 1) * tfac;
            }
            diagC = C.diag();
            diagD = sqrt(diagD); // D contains standard deviations now
            BD = B % repmat(diagD.t(), dim, 1);
        }
    }

    void doOptimize() {

        // -------------------- Generation Loop --------------------------------

        for (iterations = 1; iterations <= maxIterations &&
                    fitfun->getEvaluations() < maxEvaluations; iterations++) {
            // Generate and evaluate popsize offspring
            mat arz = mat(dim, popsize).randn();
            mat arx = mat(dim, popsize);
            vec fitness = vec(popsize).fill(DBL_MAX);
            // generate random offspring
            fitfun->closestFeasible(xmean);
            for (int k = 0; k < popsize; k++) {
                vec delta = (BD * arz.col(k)) * sigma;
                arx.col(k) = fitfun->getClosestFeasible(xmean + delta);
                fitness[k] = fitfun->value(arx.col(k)); // compute fitness
                if (!isfinite(fitness[k])) {
                    stop = -1;
                    break;
                }
            }
            if (stop != 0)
                break;
            // Sort by fitness and compute weighted mean into xmean
            uvec arindex = sort_index(fitness);

            // Calculate new xmean, this is selection and recombination
            vec xold = xmean; // for speed up of Eq. (2) and (3)
            uvec bestIndex = arindex.head(mu);
            mat bestArx = arx.cols(bestIndex);
            xmean = bestArx * weights;
            mat bestArz = arz.cols(bestIndex);
            mat zmean = bestArz * weights;

            bool hsig = updateEvolutionPaths(zmean, xold);
            double negccov = updateCovariance(hsig, bestArx, arz, arindex, xold);
            updateBD(negccov);
            // Adapt step size sigma - Eq. (5)
            sigma *= exp(min(1.0, (normps / chiN - 1.) * cs / damps));
            double bestFitness = fitness(arindex(0));
            double worstFitness = fitness(arindex(arindex.size()-1));
            if (bestValue > bestFitness) {
                bestValue = bestFitness;
                bestX = fitfun->decode(bestArx.col(0));
            }

            // handle termination criteria
            if (isfinite(stopfitness) && bestFitness < stopfitness) {
                stop = 1;
                break;
            }
            vec sqrtDiagC = sqrt(diagC);
            vec pcCol = pc;

            for (int i = 0; i < dim; i++) {
                if (sigma * (max(abs(pcCol[i]), sqrtDiagC[i])) > stopTolX)
                    break;
                if (i >= dim - 1)
                    stop = 2;
            }
            for (int i = 0; i < dim; i++)
                if (sigma * sqrtDiagC[i] > stopTolUpX)
                    stop = 3;
            if (stop > 0)
                break;
            double historyBest = min(fitnessHistory);
            double historyWorst = max(fitnessHistory);
            if (iterations > 2
                    && max(historyWorst, worstFitness)
                            - min(historyBest, bestFitness) < stopTolFun) {
                stop = 4;
                break;
            }
            if (iterations > fitnessHistory.size()
                    && historyWorst - historyBest < stopTolHistFun) {
                stop = 5;
                break;
            }
            // condition number of the covariance matrix exceeds 1e14
            if ( diagD.max() / diagD.min() > 1e7 * 1.0 / sqrt(accuracy)) {
                stop = 6;
                break;
            }
            if (isTerminate != NULL && isTerminate(runid, iterations, bestValue)) {
                stop = 7;
                break;
            }
            // Adjust step size in case of equal function values (flat fitness)
            if (bestValue == fitness[arindex[(int) (0.1 + popsize / 4.)]]) {
                sigma *= exp(0.2 + cs / damps);
            }
            if (iterations > 2
                    && max(historyWorst, bestFitness)
                            - ::min(historyBest, bestFitness) == 0) {
                sigma *= ::exp(0.2 + cs / damps);
            }
            // store best in history
            for (int i = 1; i < fitnessHistory.size(); i++)
                fitnessHistory[i] = fitnessHistory[i - 1];
            fitnessHistory[0] = bestFitness;
        }
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
      vec guess;
      double accuracy;
      is_terminate_type isTerminate;
      int popsize; // population size
      vec inputSigma;
      int dim;
      int maxIterations;
      int maxEvaluations;
      double stopfitness;
      double stopTolUpX;
      double stopTolX;
      double stopTolFun;
      double stopTolHistFun;
      int mu; //
      vec weights;
      double mueff; //
      double sigma;
      double cc;
      double cs;
      double damps;
      double ccov1;
      double ccovmu;
      double chiN;
      double ccov1Sep;
      double ccovmuSep;
      vec xmean;
      vec pc;
      vec ps;
      double normps;
      mat B;
      mat BD;
      mat diagD;
      mat C;
      vec diagC;
      int iterations;
      vec fitnessHistory;
      int historySize;
      double bestValue;
      vec bestX;
      int stop;
};

// see https://cvstuff.wordpress.com/2014/11/27/wraping-c-code-with-python-ctypes-memory-and-pointers/

extern "C" {
    void seed(int s) {
        arma_rng::set_seed(s);
    }
}

extern "C" {
    void seedRandom() {
        arma_rng::set_seed_random();
    }
}

extern "C" {
    void free_mem(double* a) {
        delete[] a;
    }
}

extern "C" {
    double* optimizeACMA_C(long runid, callback_type func, int dim, double *init,
            double *lower, double *upper, double *sigma, int maxIter, int maxEvals,
            double stopfitness, int mu, int popsize, double accuracy,
            bool useTerminate, is_terminate_type isTerminate) {
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
        Fittness fitfun(func, lower_limit, upper_limit);
        AcmaesOptimizer opt(
            runid,
            &fitfun,
            popsize,
            mu,
            guess,
            inputSigma,
            maxIter,
            maxEvals,
            accuracy,
            stopfitness,
            useTerminate ? isTerminate : NULL);
        try {
            opt.doOptimize();
            vec bestX = opt.getBestX();
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
