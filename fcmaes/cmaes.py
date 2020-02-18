'''
Created on Jan 24, 2020

@author: Dietmar Wolz
'''

import sys
import os
import math
import ctypes as ct
import numpy as np
from scipy import linalg
from scipy.optimize import OptimizeResult, Bounds
from numpy.random import MT19937, Generator
from multiprocessing import Process
import multiprocessing as mp

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun, 
             bounds=None, 
             x0=None, 
             input_sigma = 1.0, 
             popsize = 32, 
             max_evaluations = 100000, 
             max_iterations = 100000,  
             is_parallel = False,
             accuracy = 1.0, 
             stop_fittness = np.nan, 
             is_terminate = None, 
             rg = Generator(MT19937()),
             runid=0):    
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
        Bounds on variables for L-BFGS-B, TNC, SLSQP and
        trust-constr methods. There are two ways to specify the bounds:
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
    is_parallel : bool, optional
        If True, function evaluation is performed in parallel for the
        whole population, if the CPU has enough threads. Otherwise
        mp.cpu_count() processes are used.  
    accuracy : float, optional
        values > 1.0 reduce the accuracy.
    stop_fittness : float, optional 
         Limit for fitness value. If reached minimize terminates.
    is_terminate : callable, optional
        Callback to be used if the caller of minimize wants to 
        decide when to terminate. 
    rg = numpy.random.Generator, optional
        Random generator for creating random guesses.
    runid : int, optional
        id used by the is_terminate callback to identify the CMA-ES run. 
    
    Returns
    -------
    res : scipy.OptimizeResult
        The optimization result is represented as an ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, 
        ``fun`` the best function value, ``nfev`` the number of function evaluations,
        ``nit`` the number of CMA-ES iterations, ``status`` the stopping critera and
        ``success`` a Boolean flag indicating if the optimizer exited successfully. """
    
    lower, upper, guess = checkBounds(bounds, x0, rg)   
    fun = parallel(fun) if is_parallel else serial(fun)
    fitfun = Fittness(fun, lower, upper)
    cmaes = Cmaes(runid, fitfun.encode(guess), rg, accuracy, max_evaluations, max_iterations, popsize, \
                input_sigma, stop_fittness, is_terminate)
    x, val, evals, iterations, stop = cmaes.doOptimize(fitfun)
    return OptimizeResult(x=x, fun=val, nfev=evals, nit=iterations, status=stop, success=True)

def checkBounds(bounds, guess, rg):
    if bounds is None and guess is None:
        raise ValueError('either guess or bounds need to be defined')
    if bounds is None:
        return None, None, np.asarray(guess)
    if guess is None:
        guess = rg.uniform(bounds.lb, bounds.ub)
    return np.asarray(bounds.lb), np.asarray(bounds.ub), np.asarray(guess)

class Cmaes(object):
    """manages a single cma-es retry."""
    
    def __init__(self, runid, guess, rg, accuracy, max_evaluations, max_iterations,  \
                popsize, input_sigma, stop_fittness, is_terminate):
    # runid used in is_terminate callback to identify a specific run at different iteration
        self.runid = runid
    # initial guess for the arguments of the fitness function
        self.guess = guess
    # random generator    
        self.rg = rg
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
        self.stop_fitness = stop_fittness
    # Stop if x-changes larger stopTolUpX.
        self.stopTolUpX = 1e3 * self.sigma
    # Stop if x-change smaller stopTolX.
        self.stopTolX = 1e-11 * self.sigma * accuracy;
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
        self.ccov1 = 2. / ((self.dim + 1.3) * (self.dim + 1.3) + self.mueff);
    # Learning rate for rank-mu update'
        self.ccovmu = min(1. - self.ccov1, 2. * (self.mueff - 2. + 1. / self.mueff) \
                / ((self.dim + 2.) * (self.dim + 2.) + self.mueff))
    # Expectation of ||N(0,I)|| == norm(randn(N,1)).
        self.chiN = math.sqrt(self.dim) * (1. - 1. / (4. * self.dim) + 1 / (21. * self.dim * self.dim))
        self.ccov1Sep = min(1., self.ccov1 * (self.dim + 1.5) / 3.)
        self.ccovmuSep = min(1. - self.ccov1, self.ccovmu * (self.dim + 1.5) / 3.)        

    # CMA internal values - updated each generation
    # Objective variables.
        self.xmean = guess
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
    
    def doOptimize(self, fitfun):
        self.iterations = 0
        stop = 0
        bestValue = sys.float_info.max
        bestX = None    
        # History queue of best values.
        fitness_history = np.full(self.historySize, sys.float_info.max)
        fitness_history[0] = bestValue        
        # -------------------- Generation Loop --------------------------------
        while True:
            if self.iterations > self.max_iterations:
                break
            self.iterations += 1
            if fitfun.evaluation_counter > self.max_evaluations:
                break
            # Generate and evaluate popsize offspring
            self.xmean = fitfun.closestFeasible(self.xmean)
            fitness = np.full(self.popsize, math.inf)

            arz = np.random.randn(self.popsize, self.dim)    
            arx = np.empty((self.popsize, self.dim))

            # generate random offspring
            delta = (self.BD @ arz.transpose()) * self.sigma
            arx = fitfun.closestFeasible(self.xmean + delta.transpose())
            
            fitness = fitfun.values(arx)
            # Stop for Nan / infinite fitness values 
            if np.isfinite(fitness).sum() < self.popsize:
                stop = -1
                break 
            # Sort by fitness and compute weighted mean into xmean
            arindex = fitness.argsort()        
            best_fitness = fitness[arindex[0]]
            worstFitness = fitness[arindex[-1]]                        
            if bestValue > best_fitness:
                bestValue = best_fitness
                bestX = fitfun.decode(arx[arindex[0]])    

            # Calculate new xmean, this is selection and recombination
            xold = self.xmean # for speed up of Eq. (2) and (3)
            bestIndex = arindex[:self.mu] 
            bestArx = arx[bestIndex]
            self.xmean = np.transpose(bestArx) @ self.weights
            bestArz = arz[bestIndex]
            zmean = np.transpose(bestArz) @ self.weights
            hsig = self.updateEvolutionPaths(zmean, xold)            
            negccov = self.updateCovariance(hsig, bestArx, arz, arindex, xold)
            self.updateBD(negccov)                        
            # Adapt step size sigma - Eq. (5)
            self.sigma *= math.exp(min(1.0, (self.normps / self.chiN - 1.) * self.cs / self.damps))            
            # handle termination criteria
            if self.stop_fitness != None: # only if stop_fitness is defined
                if best_fitness < self.stop_fitness:
                    stop = 1
                    break
            sqrtDiagC = np.sqrt(self.diagC)
            pcCol = self.pc
            for i in range(self.dim):
                if self.sigma * max(abs(pcCol[i]), sqrtDiagC[i]) > self.stopTolX:
                    break
                if i == self.dim - 1:
                    stop = 2
            if stop != 0:
                break            
            for i in range(self.dim):
                if self.sigma * sqrtDiagC[i] > self.stopTolUpX:
                    stop = 3
                    break
            if stop != 0:
                break
            history_best = min(fitness_history)
            history_worst = max(fitness_history)
            if self.iterations > 2 and max(history_worst, worstFitness) - min(history_best, best_fitness) < self.stopTolFun:
                stop = 4
                break
            if self.iterations > fitness_history.size and history_worst - history_best < self.stopTolHistFun:
                stop = 5
                break
            # condition number of the covariance matrix exceeds 1e14
            if min(self.diagD) != 0 and \
                    max(self.diagD) / min(self.diagD) > 1e7 * 1.0 / math.sqrt(self.accuracy):
                stop = 6
                break
            # compare to other CMA-ES retries and stop of value is bad 
            if (not self.is_terminate is None) and \
                       self.is_terminate(self.runid, self.iterations, bestValue):
                stop = 7
                break
            # Adjust step size in case of equal function values (flat fitness)
            if bestValue == fitness[arindex[int(0.1 + self.popsize / 4.)]]:
                self.sigma *= math.exp(0.2 + self.cs / self.damps)
            if self.iterations > 2 and max(history_worst, best_fitness) - min(history_best, best_fitness) == 0:
                self.sigma *= math.exp(0.2 + self.cs / self.damps)
            # store best in history
            fitness_history[1:] = fitness_history[:-1]
            fitness_history[0] = best_fitness

        return bestX, bestValue, fitfun.evaluation_counter, self.iterations, stop        
    
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

def serial(func):
    """Convert an objective function for serial execution for cmaes.minimize.
    
    Parameters
    ----------
    func : objective function mapping a list of float arguments to a float value

    Returns
    -------
    out : function
        A function mapping a list of lists of float arguments to a list of float values
        by applying the input function in a loop."""
  
    return lambda xs : [func(x) for x in xs]

def parallel(func):
    """Convert an objective function for parallel execution for cmaes.minimize.
    
    Parameters
    ----------
    func : objective function mapping a list of float arguments to a float value.

    Returns
    -------
    out : function
        A function mapping a list of lists of float arguments to a list of float values
        by applying the input function using parallel processes. """
 
    return lambda xs : func_parallel(func, xs)            
            
def func_parallel(func, xs):
    popsize = len(xs)
    num = min(popsize, mp.cpu_count())
    ys = mp.RawArray(ct.c_double, popsize) 
    proc=[Process(target=func_serial, args=(func, num, pid, xs, ys)) for pid in range(num)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    return [y for y in ys]

def func_serial(func, num, pid, xs, ys):
    for i in range(pid, len(xs), num):
        ys[i] = func(xs[i])
                       
class Fittness(object):
    """wrapper around the objective function, scales relative to boundaries."""
     
    def __init__(self, fun, lower, upper):
        self.func = fun
        self.evaluation_counter = 0
        self.lower = lower
        if not lower is None:
            self.upper = upper
            self.scale = 0.5 * (upper - lower)
            self.typx = 0.5 * (upper + lower)
        
    def values(self, Xs): #enables parallel evaluation
        values = self.func([self.decode(X) for X in Xs]);
        self.evaluation_counter += len(Xs)
        return np.array(values)
        
    def closestFeasible(self, X):
        if self.lower is None:
            return X    
        else:
            return np.maximum(np.minimum(X, 1.0), -1.0)

    def encode(self, X):
        if self.lower is None:
            return X
        else:
            return (X - self.typx) / self.scale
    
    def decode(self, X):
        if self.lower is None:
            return X
        else:
            return (X * self.scale) + self.typx
    