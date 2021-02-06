# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

""" Numpy based implementation of active CMA-ES.
    Derived from http://cma.gforge.inria.fr/cmaes.m which follows
    https://www.researchgate.net/publication/227050324_The_CMA_Evolution_Strategy_A_Comparing_Review
"""

import sys
import os
import math
import numpy as np
import multiprocessing as mp
from scipy import linalg
from scipy.optimize import OptimizeResult
from numpy.random import MT19937, Generator
from fcmaes.evaluator import Evaluator, eval_parallel

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun, 
             bounds=None, 
             x0=None, 
             input_sigma = 0.3, 
             popsize = 31, 
             max_evaluations = 100000, 
             max_iterations = 100000,  
             workers = None,
             accuracy = 1.0, 
             stop_fitness = np.nan, 
             is_terminate = None, 
             rg = Generator(MT19937()),
             runid=0,
             delayed_update = False,
             normalize = True,
             update_gap = None):    
    """Minimization of a scalar function of one or more variables using CMA-ES.
     
    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
            ``fun(x, *args) -> float``
        where ``x`` is an 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    bounds : sequence or `Bounds`, optional
        Bounds on variables. There are two ways to specify the bounds:
            1. Instance of the `scipy.Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where 'n' is the number of independent variables.  
    input_sigma : ndarray, shape (n,) or scalar
        Initial step size for each dimension.
    popsize = int, optional
        CMA-ES population size.
    max_evaluations : int, optional
        Forced termination after ``max_evaluations`` function evaluations.
    max_iterations : int, optional
        Forced termination after ``max_iterations`` iterations.
    workers : int or None, optional
        If not workers is None, function evaluation is performed in parallel for the whole population. 
        Useful for costly objective functions but is deactivated for parallel retry.      
    accuracy : float, optional
        values > 1.0 reduce the accuracy.
    stop_fitness : float, optional 
         Limit for fitness value. If reached minimize terminates.
    is_terminate : callable, optional
        Callback to be used if the caller of minimize wants to 
        decide when to terminate. 
    rg = numpy.random.Generator, optional
        Random generator for creating random guesses.
    runid : int, optional
        id used by the is_terminate callback to identify the CMA-ES run. 
    delayed_update : boolean, optional
        delayed_update if workers > 1. 
    normalize : boolean, optional
        pheno -> if true geno transformation maps arguments to interval [-1,1] 
    update_gap : int, optional
        number of iterations without distribution update
   
    Returns
    -------
    res : scipy.OptimizeResult
        The optimization result is represented as an ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, 
        ``fun`` the best function value, ``nfev`` the number of function evaluations,
        ``nit`` the number of CMA-ES iterations, ``status`` the stopping critera and
        ``success`` a Boolean flag indicating if the optimizer exited successfully. """
        
    if workers and workers > 1 and delayed_update:
        cmaes = Cmaes(bounds, x0, 
                      input_sigma, popsize, 
                      max_evaluations, max_iterations, 
                      accuracy, stop_fitness, 
                      is_terminate, rg, np.random.randn, runid, normalize, 
                      update_gap, fun)
        x, val, evals, iterations, stop = cmaes.do_optimize_delayed_update(fun, workers)
    else:      
        fun = serial(fun) if workers is None else parallel(fun, workers)
        cmaes = Cmaes(bounds, x0, 
                      input_sigma, popsize, 
                      max_evaluations, max_iterations, 
                      accuracy, stop_fitness, 
                      is_terminate, rg, np.random.randn, runid, normalize, 
                      update_gap, fun)
        x, val, evals, iterations, stop = cmaes.doOptimize()
        if not workers is None:
            fun.stop() # stop all parallel evaluation processes
    return OptimizeResult(x=x, fun=val, nfev=evals, nit=iterations, status=stop, 
                          success=True)

class Cmaes(object):
    """Implements the cma-es ask/tell interactive interface."""
    
    def __init__(self, bounds=None, 
                        x0=None, 
                        input_sigma = 0.3, 
                        popsize = 31, 
                        max_evaluations = 100000, 
                        max_iterations = 100000,  
                        accuracy = 1.0, 
                        stop_fitness = np.nan, 
                        is_terminate = None, 
                        rg = Generator(MT19937()), # used if x0 is undefined
                        randn = np.random.randn, # used for random offspring 
                        runid=0, 
                        normalize = False,
                        update_gap = None,
                        fun = None
                        ):
                        
    # runid used in is_terminate callback to identify a specific run at different iteration
        self.runid = runid
    # bounds and guess
        lower, upper, guess = _check_bounds(bounds, x0, rg)   
        self.fitfun = _fitness(fun, lower, upper, normalize)
    # initial guess for the arguments of the fitness function
        self.guess = self.fitfun.encode(guess)
    # random generators    
        self.rg = rg # used if x0 is undefined
        self.randn = randn # used for random offspring 
    # accuracy = 1.0 is default, > 1.0 reduces accuracy
        self.accuracy = accuracy
    # callback to check if to terminate
        self.is_terminate = is_terminate
    # Number of objective variables/problem dimension
        self.dim = guess.size
    #     Population size, offspring number. The primary strategy parameter to play
    #     with, which can be increased from its default value. Increasing the
    #     population size improves global search properties in exchange to speed.
    #     Speed decreases, as a rule, at most linearly with increasing population
    #     size. It is advisable to begin with the default small population size.
        if popsize:
            self.popsize = popsize #population size
        else:
            self.popsize = 4 + int(3. * math.log(self.dim))
    #     Individual sigma values - initial search volume. input_sigma determines
    #     the initial coordinate wise standard deviations for the search. Setting
    #     SIGMA one third of the initial search region is appropriate.   
        if isinstance(input_sigma, list):
            self.insigma = np.asarray(input_sigma)
        elif np.isscalar(input_sigma):
            self.insigma = np.full(self.dim, input_sigma)    
        else:
            self.insigma = input_sigma
    # Overall standard deviation - search volume.
        self.sigma = max(self.insigma)
    # termination criteria
    # Maximal number of iterations allowed.
        self.max_evaluations = max_evaluations
        self.max_iterations = max_iterations
    # Limit for fitness value.
        self.stop_fitness = stop_fitness
    # Stop if x-changes larger stopTolUpX.
        self.stopTolUpX = 1e3 * self.sigma
    # Stop if x-change smaller stopTolX.
        self.stopTolX = 1e-11 * self.sigma * accuracy
    # Stop if fun-changes smaller stopTolFun.
        self.stopTolFun = 1e-12 * accuracy
    # Stop if back fun-changes smaller stopTolHistFun.
        self.stopTolHistFun = 1e-13 * accuracy
    # selection strategy parameters
    # Number of parents/points for recombination.
        self.mu = int(self.popsize/2)
    # Array for weighted recombination.    
        self.weights = (np.log(np.arange(1, self.mu+1, 1)) * -1) + math.log(self.mu + 0.5)
        sumw = np.sum(self.weights)
        sumwq = np.einsum('i,i->', self.weights, self.weights)
        self.weights *= 1./sumw        
    # Variance-effectiveness of sum w_i x_i.
        self.mueff = sumw * sumw / sumwq 

    # dynamic strategy parameters and constants
    # Cumulation constant.
        self.cc = (4. + self.mueff / self.dim) / (self.dim + 4. + 2. * self.mueff / self.dim)
    # Cumulation constant for step-size.
        self.cs = (self.mueff + 2.) / (self.dim + self.mueff + 3.)
    # Damping for step-size.    
        self.damps = (1. + 2. * max(0., math.sqrt((self.mueff - 1.) / (self.dim + 1.)) - 1.)) * \
            max(0.3, 1. - self.dim / (1e-6 + min(self.max_iterations, self.max_evaluations/self.popsize))) + self.cs
    # Learning rate for rank-one update.
        self.ccov1 = 2. / ((self.dim + 1.3) * (self.dim + 1.3) + self.mueff)
    # Learning rate for rank-mu update'
        self.ccovmu = min(1. - self.ccov1, 2. * (self.mueff - 2. + 1. / self.mueff) \
                / ((self.dim + 2.) * (self.dim + 2.) + self.mueff))
    # Expectation of ||N(0,I)|| == norm(randn(N,1)).
        self.chiN = math.sqrt(self.dim) * (1. - 1. / (4. * self.dim) + 1 / (21. * self.dim * self.dim))
        self.ccov1Sep = min(1., self.ccov1 * (self.dim + 1.5) / 3.)
        self.ccovmuSep = min(1. - self.ccov1, self.ccovmu * (self.dim + 1.5) / 3.)        
    # lazy covariance update gap
        self.lazy_update_gap = 1. / (self.ccov1 + self.ccovmu + 1e-23) / self.dim / 10 \
                                    if update_gap is None else update_gap

    # CMA internal values - updated each generation
    # Objective variables.
        self.xmean = self.guess
    # Evolution path.
        self.pc = np.zeros(self.dim)
    # Evolution path for sigma.
        self.ps = np.zeros(self.dim)
    # Norm of ps, stored for efficiency.
        self.normps = math.sqrt(self.ps @ self.ps)
    # Coordinate system.
        self.B = np.eye(self.dim)        
    # Diagonal of sqrt(D), stored for efficiency.
        self.diagD = self.insigma / self.sigma
        self.diagC = self.diagD * self.diagD
    # B*D, stored for efficiency.
        self.BD = self.B * self.diagD
    # Covariance matrix.
        self.C = self.B @ (np.diag(np.ones(self.dim)) @ self.B)
    # Number of iterations already performed.
        self.iterations = 0
    # Size of history queue of best values.
        self.historySize = 10 + int(3. * 10. * self.dim / popsize)    
        
        self.iterations = 0
        self.last_update = 0
        self.stop = 0
        self.best_value = sys.float_info.max
        self.best_x = None    
        # History queue of best values.
        self.fitness_history = np.full(self.historySize, sys.float_info.max)
        self.fitness_history[0] = self.best_value    
        self.arz = None
        self.fitness = None

    def ask(self):
        """ask for popsize new argument vectors.
            
        Returns
        -------
        xs : popsize sized list of dim sized argument lists."""

        self.newArgs()
        return [self.fitfun.decode(x) for x in self.arx]
 
    def tell(self, ys, xs = None):      
        """tell function values for the argument lists retrieved by ask().
    
        Parameters
        ----------
        ys : popsize sized list of function values
        xs : popsize sized list of dim sized argument lists, optional
            use only if you want to submit values for arguments not from ask()
            Needs either to be defined, or ask needs to be called before. 
 
        Returns
        -------
        stop : int termination criteria, if != 0 loop should stop."""

        if xs is None:
            if self.arz is None:
                raise ValueError('either call ask before or define xs')
        else:
            self.arx = np.array([self.fitfun.encode(x) for x in xs])
            try:
                self.arz = (linalg.inv(self.BD) @ \
                            ((self.arx - self.xmean).transpose() / self.sigma)).transpose()   
            except Exception:
                if self.arz is None: 
                    self.arz = self.randn(self.popsize, self.dim)
        self.fitness = np.asarray(ys)
        self.iterations += 1
        self.updateCMA()
        self.arz = None
        return self.stop
 
    def ask_one(self):
        """ask for one new argument vector.
        
        Returns
        -------
        x : dim sized argument ."""
        arz = self.randn(self.dim) 
        delta = (self.BD @ arz.transpose()) * self.sigma
        arx = self.fitfun.closestFeasible(self.xmean + delta.transpose())  
        return self.fitfun.decode(arx)

    def tell_one(self, y, x):      
        """tell function value for a argument list retrieved by ask_one().
    
        Parameters
        ----------
        y : function value
        x : dim sized argument list
 
        Returns
        -------
        stop : int termination criteria, if != 0 loop should stop."""

        if self.fitness is None or not type(self.fitness) is list:
            self.arx = []
            self.fitness = []
        self.fitness.append(y)
        self.arx.append(x)
        if len(self.fitness) >= self.popsize:
            self.fitness = np.asarray(self.fitness)
            self.arx = np.array([self.fitfun.encode(x) for x in self.arx])
            try:
                self.arz = (linalg.inv(self.BD) @ \
                            ((self.arx - self.xmean).transpose() / self.sigma)).transpose()   
            except Exception:
                if self.arz is None: 
                    self.arz = self.randn(self.popsize, self.dim)
            self.iterations += 1
            self.updateCMA()
            self.arz = None
            self.arx = []
            self.fitness = []
        return self.stop                
           
    def newArgs(self):
        self.xmean = self.fitfun.closestFeasible(self.xmean)
        self.fitness = np.full(self.popsize, math.inf)
        # generate random offspring
        self.arz = self.randn(self.popsize, self.dim)    
        delta = (self.BD @ self.arz.transpose()) * self.sigma
        self.arx = self.fitfun.closestFeasible(self.xmean + delta.transpose())  
    
    def do_optimize_delayed_update(self, fun, workers=mp.cpu_count()):
        evaluator = Evaluator(fun)
        evaluator.start(workers)
        evals_x = {}
        self.evals = 0;
        for _ in range(workers): # fill queue
            x = self.ask_one()
            evaluator.pipe[0].send((self.evals, x))
            evals_x[self.evals] = x # store x
            self.evals += 1
            
        while True: # read from pipe, tell es and create new x
            evals, y = evaluator.pipe[0].recv()
            
            x = evals_x[evals] # retrieve evaluated x
            del evals_x[evals]
            stop = self.tell_one(y, x) # tell evaluated x
            if stop != 0 or self.evals >= self.max_evaluations:
                break # shutdown worker if stop criteria met
            
            x = self.ask_one() # create new x
            evaluator.pipe[0].send((self.evals, x))       
            evals_x[self.evals] = x  # store x
            self.evals += 1            
        evaluator.stop()
        return self.best_x, self.best_value, evals, self.iterations, self.stop 
         
    def doOptimize(self):
        # -------------------- Generation Loop --------------------------------
        while True:
            if self.iterations > self.max_iterations:
                break
            self.iterations += 1
            if self.fitfun.evaluation_counter > self.max_evaluations:
                break
            # Generate and evaluate popsize offspring
            self.newArgs()            
            self.fitness = self.fitfun.values(self.arx)
            self.updateCMA()
            if self.stop != 0:
                break
        return self.best_x, self.best_value, self.fitfun.evaluation_counter, self.iterations, self.stop 
        
    def updateCMA(self):
        # Stop for Nan / infinite fitness values 
        if np.isfinite(self.fitness).sum() < self.popsize:
            return -1
        # Sort by fitness and compute weighted mean into xmean
        arindex = self.fitness.argsort()        
        best_fitness = self.fitness[arindex[0]]
        worstFitness = self.fitness[arindex[-1]]                        
        if self.best_value > best_fitness:
            self.best_value = best_fitness
            self.best_x = self.fitfun.decode(self.arx[arindex[0]])    
            if self.stop_fitness != None: # only if stop_fitness is defined
                if best_fitness < self.stop_fitness:
                    self.stop = 1
                    return 

        # Calculate new xmean, this is selection and recombination
        xold = self.xmean # for speed up of Eq. (2) and (3)
        bestIndex = arindex[:self.mu] 
        bestArx = self.arx[bestIndex]
        self.xmean = np.transpose(bestArx) @ self.weights
        bestArz = self.arz[bestIndex]
        zmean = np.transpose(bestArz) @ self.weights
        hsig = self.updateEvolutionPaths(zmean, xold)            
        # Adapt step size sigma - Eq. (5)
        self.sigma *= math.exp(min(1.0, (self.normps / self.chiN - 1.) * self.cs / self.damps))            
        
        if self.iterations >= self.last_update + self.lazy_update_gap:
            self.last_update = self.iterations
            negccov = self.updateCovariance(hsig, bestArx, self.arz, arindex, xold)
            self.updateBD(negccov)                        
            # handle termination criteria
            sqrtDiagC = np.sqrt(np.abs(self.diagC))
            pcCol = self.pc
            for i in range(self.dim):
                if self.sigma * max(abs(pcCol[i]), sqrtDiagC[i]) > self.stopTolX:
                    break
                if i == self.dim - 1:
                    self.stop = 2
            if self.stop != 0:
                return            
            for i in range(self.dim):
                if self.sigma * sqrtDiagC[i] > self.stopTolUpX:
                    self.stop = 3
                    break
            if self.stop != 0:
                return 
        history_best = min(self.fitness_history)
        history_worst = max(self.fitness_history)
        if self.iterations > 2 and max(history_worst, worstFitness) - min(history_best, best_fitness) < self.stopTolFun:
            self.stop = 4
            return 
        if self.iterations > self.fitness_history.size and history_worst - history_best < self.stopTolHistFun:
            self.stop = 5
            return 
        # condition number of the covariance matrix exceeds 1e14
        if min(self.diagD) != 0 and \
                max(self.diagD) / min(self.diagD) > 1e7 * 1.0 / math.sqrt(self.accuracy):
            self.stop = 6
            return 
        # call callback
        if (not self.is_terminate is None) and \
                       self.is_terminate(self.runid, self.iterations, self.best_value):
            self.stop = 7
            return 
        # Adjust step size in case of equal function values (flat fitness)
        if self.best_value == self.fitness[arindex[int(0.1 + self.popsize / 4.)]]:
            self.sigma *= math.exp(0.2 + self.cs / self.damps)
        if self.iterations > 2 and max(history_worst, best_fitness) - min(history_best, best_fitness) == 0:
            self.sigma *= math.exp(0.2 + self.cs / self.damps)
        # store best in history
        self.fitness_history[1:] = self.fitness_history[:-1]
        self.fitness_history[0] = best_fitness
        return       
    
    def updateEvolutionPaths(self, zmean, xold):
        """update evolution paths.
    
        Parameters
        ----------
        zmean: weighted row matrix of the gaussian random numbers generating the current offspring
        xold: xmean matrix of the previous generation
        
        Returns
        -------
        hsig flag indicating a small correction."""

        self.ps = self.ps * (1. - self.cs) + \
            ((self.B @ zmean) * math.sqrt(self.cs * (2. - self.cs) * self.mueff))
        self.normps = math.sqrt(self.ps @ self.ps) 
        hsig = self.normps / math.sqrt(1. - math.pow(1. - self.cs, 2. * self.iterations)) / \
            self.chiN < 1.4 + 2. / (self.dim + 1.)
        self.pc *= 1. - self.cc        
        if hsig:
            self.pc += (self.xmean - xold) * (math.sqrt(self.cc * (2. - self.cc) * self.mueff) / self.sigma)
        return hsig
    
    def updateCovariance(self, hsig, bestArx, arz, arindex, xold):
        """update covariance matrix.
    
        Parameters
        ----------
        hsig: flag indicating a small correction.
        bestArx: fitness-sorted matrix of the argument vectors producing the current offspring.
        arz: unsorted matrix containing the gaussian random values of the current offspring.
        arindex: indices indicating the fitness-order of the current offspring.
        xold: xmean matrix of the previous generation.
        
        Returns
        -------
        negccov: Negative covariance factor."""
      
        negccov = 0.
        if self.ccov1 + self.ccovmu > 0:
            arpos = (bestArx - xold) * (1. / self.sigma) # mu difference vectors
            pc2d = self.pc[:, np.newaxis] 
            roneu = (pc2d @ np.transpose(pc2d)) * self.ccov1
            # minor correction if hsig==false
            oldFac = 0 if hsig else self.ccov1 * self.cc * (2. - self.cc)
            oldFac += 1. - self.ccov1 - self.ccovmu
            # Adapt covariance matrix C active CMA
            negccov = (1. - self.ccovmu) * 0.25 * self.mueff \
                    / (math.pow(self.dim + 2., 1.5) + 2. * self.mueff)
            negminresidualvariance = 0.66
            # keep at least 0.66 in all directions, small popsize are most critical
            negalphaold = 0.5 # where to make up for the variance loss,
            # prepare vectors, compute negative updating matrix Cneg
            arReverseIndex = arindex[::-1]
            arzneg = arz[arReverseIndex[:self.mu]]
            arnorms = np.sqrt(np.einsum('ij->i', arzneg * arzneg))
            idxnorms = arnorms.argsort()
            arnormsSorted = arnorms[idxnorms]
            idxReverse = idxnorms[::-1]
            arnormsReverse = arnorms[idxReverse]
            arnorms = arnormsReverse / arnormsSorted
            arnormsInv = np.empty(arnorms.size)
            arnormsInv[idxnorms] = arnorms
            # check and set learning rate negccov
            negcovMax = (1. - negminresidualvariance) / ((arnormsInv*arnormsInv) @ self.weights)
            if negccov > negcovMax:
                negccov = negcovMax
            arzneg = np.transpose(arzneg) * arnormsInv
            artmp = self.BD @ arzneg
            Cneg = artmp @ (np.diagflat(self.weights) @ np.transpose(artmp))
            oldFac += negalphaold * negccov
            C = (self.C * oldFac) + roneu + \
                np.transpose(arpos * (self.ccovmu + (1. - negalphaold) * negccov)) @ \
                    np.transpose(self.weights * np.transpose(arpos))
            self.C = C - Cneg*negccov
        return negccov
    
    def updateBD(self, negccov):
        """update B and diagD from covariance matrix C.
    
        Parameters
        ----------
        negccov: Negative covariance factor."""
 
        if self.ccov1 + self.ccovmu + negccov > 0 and \
                self.iterations % (1. / (self.ccov1 + self.ccovmu + negccov) / self.dim / 10.) < 1.:
            # to achieve O(N^2) enforce symmetry to prevent complex numbers
            self.C = np.triu(self.C, 0) + np.transpose(np.triu(self.C, 1))
            
            # diagD defines the scaling
            eigen_values, eigen_vectors = linalg.eigh(self.C)

            idx = eigen_values.argsort()[::-1]   
            self.diagD = eigen_values[idx]
            self.B = eigen_vectors[:,idx]

            # Coordinate system
            if min(self.diagD) <= 0:    
                self.diagD = np.maximum(self.diagD, 0)
                tfac = max(self.diagD) / 1e14
                self.C += np.eye(self.dim) * tfac
                self.diagD += np.ones(self.dim) * tfac

            if max(self.diagD) > 1e14 * min(self.diagD):
                tfac = max(self.diagD) / 1e14 - min(self.diagD)
                self.C += np.eye(self.dim) * tfac
                self.diagD += np.ones(self.dim) * tfac
                
            self.diagC = np.diag(self.C)
            self.diagD = np.sqrt(self.diagD) # diagD contains standard deviations now
            
            self.BD = self.B * self.diagD # O(n^2)

def serial(fun):
    """Convert an objective function for serial execution for cmaes.minimize.
    
    Parameters
    ----------
    fun : objective function mapping a list of float arguments to a float value

    Returns
    -------
    out : function
        A function mapping a list of lists of float arguments to a list of float values
        by applying the input function in a loop."""
  
    return lambda xs : [_tryfun(fun, x) for x in xs]

class parallel(object):
    """Convert an objective function for parallel execution for cmaes.minimize.
    
    Parameters
    ----------
    fun : objective function mapping a list of float arguments to a float value.
   
    represents a function mapping a list of lists of float arguments to a list of float values
    by applying the input function using parallel processes. stop needs to be called to avoid
    a resource leak"""
        
    def __init__(self, fun, workers = mp.cpu_count()):
        self.evaluator = Evaluator(fun)
        self.evaluator.start(workers)
    
    def __call__(self, xs):
        return eval_parallel(xs, self.evaluator)

    def stop(self):
        self.evaluator.stop()
        
def _func_serial(fun, num, pid, xs, ys):
    for i in range(pid, len(xs), num):
        ys[i] = _tryfun(fun, xs[i])

def _tryfun(fun, x):
    try:
        fit = fun(x)
        return fit if math.isfinite(fit) else sys.float_info.max
    except Exception:
        return sys.float_info.max
                        
def _check_bounds(bounds, guess, rg):
    if bounds is None and guess is None:
        raise ValueError('either guess or bounds need to be defined')
    if bounds is None:
        return None, None, np.asarray(guess)
    if guess is None:
        guess = rg.uniform(bounds.lb, bounds.ub)
    return np.asarray(bounds.lb), np.asarray(bounds.ub), np.asarray(guess)

class _fitness(object):
    """wrapper around the objective function, scales relative to boundaries."""
     
    def __init__(self, fun, lower, upper, normalize = None):
        self.fun = fun
        self.evaluation_counter = 0
        self.lower = lower
        self.normalize = False
        if not (lower is None or normalize is None):
            self.normalize = normalize
        if not lower is None:
            self.upper = upper
            self.scale = 0.5 * (upper - lower)
            self.typx = 0.5 * (upper + lower)
                
    def values(self, Xs): #enables parallel evaluation
        values = self.fun([self.decode(X) for X in Xs])
        self.evaluation_counter += len(Xs)
        return np.array(values)

    def closestFeasible(self, X):
        if self.lower is None:
            return X    
        else:
            if self.normalize:
                return np.maximum(np.minimum(X, 1.0), -1.0)
            else:
                return np.maximum(np.minimum(X, self.upper), self.lower)

    def encode(self, X):
        if self.normalize:
            return (X - self.typx) / self.scale
        else:
            return X
   
    def decode(self, X):
        if self.normalize:
            return (X * self.scale) + self.typx
        else:
            return X
         
