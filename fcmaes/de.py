# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

""" Numpy based implementation of Differential Evolution using on the DE/best/1 strategy.
    Derived from its C++ counterpart 
    https://github.com/dietmarwo/fast-cma-es/blob/master/_fcmaescpp/deoptimizer.cpp
    
    Uses two deviations from the standard DE algorithm:
    a) temporal locality introduced in 
    https://www.researchgate.net/publication/309179699_Differential_evolution_for_protein_folding_optimization_based_on_a_three-dimensional_AB_off-lattice_model
    b) reinitialization of individuals based on their age. 

    You may keep parameters F and Cr at their defaults since this implementation works well with the given settings for most problems,
    since the algorithm oscillates between different F and Cr settings. 
    
    Use the C++ implementation combined with parallel retry instead for objective functions which are fast to evaluate. 
    For expensive objective functions (e.g. machine learning parameter optimization) use the workers
    parameter to parallelize objective function evaluation. This causes delayed population update.
    It is usually preferrable if popsize > workers and workers = mp.cpu_count() to improve CPU utilization.  
"""

import numpy as np
import math, sys
from fcmaes.testfun import Wrapper, Rosen, Rastrigin, Eggholder
from numpy.random import Generator, MT19937
from scipy.optimize import OptimizeResult
from fcmaes.evaluator import Evaluator
import multiprocessing as mp
from collections import deque

def minimize(fun, 
             bounds = None, 
             popsize = None, 
             max_evaluations = 100000, 
             workers = None,
             stop_fittness = None, 
             keep = 200, 
             f = 0.5, 
             cr = 0.9, 
             rg = Generator(MT19937())):    
    """Minimization of a scalar function of one or more variables using
    Differential Evolution.
     
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
    popsize : int, optional
        Population size.
    max_evaluations : int, optional
        Forced termination after ``max_evaluations`` function evaluations.
    workers : int or None, optional
        If not workers is None, function evaluation is performed in parallel for the whole population. 
        Useful for costly objective functions but is deactivated for parallel retry.      
    stop_fittness : float, optional 
         Limit for fitness value. If reached minimize terminates.
    keep = float, optional
        changes the reinitialization probability of individuals based on their age. Higher value
        means lower probablity of reinitialization.
    f = float, optional
        The mutation constant. In the literature this is also known as differential weight, 
        being denoted by F. Should be in the range [0, 2].
    cr = float, optional
        The recombination constant. Should be in the range [0, 1]. 
        In the literature this is also known as the crossover probability.     
    rg = numpy.random.Generator, optional
        Random generator for creating random guesses.
            
    Returns
    -------
    res : scipy.OptimizeResult
        The optimization result is represented as an ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, 
        ``fun`` the best function value, 
        ``nfev`` the number of function evaluations,
        ``nit`` the number of iterations,
        ``success`` a Boolean flag indicating if the optimizer exited successfully. """

    if popsize is None and not bounds is None:
        popsize = len(bounds.lb)*15
    de = DE(bounds, popsize, stop_fittness, keep, f, cr, rg)
    try:
        if workers and workers > 1:
            x, val, evals, iterations, stop = de._do_optimize_delayed_update(fun, max_evaluations, workers)
        else:      
            x, val, evals, iterations, stop = de._do_optimize(fun, max_evaluations)
        return OptimizeResult(x=x, fun=val, nfev=evals, nit=iterations, status=stop, 
                              success=True)
    except Exception as ex:
        return OptimizeResult(x=None, fun=sys.float_info.max, nfev=0, nit=0, status=-1, success=False)  

class DE(object):
    
    def __init__(self, bounds, popsize, stop_fittness = None, keep = 200, 
                 F = 0.5, Cr = 0.9, rg = Generator(MT19937())):
        self.dim = len(bounds.ub)
        self.bounds = bounds
        self.lower = np.asarray(bounds.lb)      
        self.upper = np.asarray(bounds.ub)      
        self.popsize = 31 if popsize is None else popsize
        self.stop_fittness = stop_fittness
        self.keep = keep 
        self.rg = rg
        self.F0 = F
        self.Cr0 = Cr
        self.stop = 0
        self.iterations = 0
        self.evals = 0
        self.p = 0
        self.improves = deque()
        self._init()
         
    def ask(self):
        """ask for popsize new argument vectors.
            
        Returns
        -------
        xs : popsize sized array of dim sized argument lists."""
        
        xs = [None] * self.popsize
        for i in range(self.popsize):
            if self.improves:
                p, x = self.improves[0]
                if xs[p] is None:
                    xs[p] = x
                    self.improves.popleft()
                else:
                    break
            else:
                break
        for p in range(self.popsize):
            if xs[p] is None:
                _, _, xs[p] = self._next_x(p)      
        return xs
 
    def tell(self, ys, xs):      
        """tell function values for the argument lists retrieved by ask().
    
        Parameters
        ----------
        ys : popsize sized list of function values
        xs : popsize sized list of dim sized argument lists
 
        Returns
        -------
        stop : int termination criteria, if != 0 loop should stop."""

        self.evals += len(ys)
        for p in range(len(ys)):
            self.tell_one(p, ys[p], xs[p])
        return self.stop

    def ask_one(self):
        """ask for one new argument vector.
        
        Returns
        -------
        p : int population index 
        x : dim sized argument ."""
        
        if self.improves:
            p, x = self.improves.popleft()
        else:
            p = self.p
            _, _, x = self._next_x(p)
            self.p = (self.p + 1) % self.popsize
        return p, x

    def tell_one(self, p, y, x):      
        """tell function value for a argument list retrieved by ask_one().
    
        Parameters
        ----------
        p : int population index 
        y : function value
        x : dim sized argument list
 
        Returns
        -------
        stop : int termination criteria, if != 0 loop should stop."""
        
        if (self.y[p] > y):
            # temporal locality
            if self.iterations > 1:
                self.improves.append((p, self._next_improve(self.x[self.best_i], x, self.x0[p])))     
            self.x0[p] = self.x[p]
            self.x[p] = x
            self.y[p] = y
            if self.y[self.best_i] > y:
                self.best_i = p
                if self.best_value > y:
                    self.best_x = x
                    self.best_value = y
                    if not self.stop_fittness is None and self.stop_fittness > y:
                        self.stop = 1
            self.pop_iter[p] = self.iterations
        else:
            if self.rg.uniform(0, self.keep) < self.iterations - self.pop_iter[p]:
                self.x[p] = self.rg.uniform(self.bounds.lb, self.bounds.ub)
                self.y[p] = math.inf
        return self.stop 

    def _init(self):
        self.x = np.zeros((self.popsize, self.dim))
        self.x0 = np.zeros((self.popsize, self.dim))
        self.y = np.empty(self.popsize)
        for i in range(self.popsize):
            self.x[i] = self.x0[i] = self.rg.uniform(self.bounds.lb, self.bounds.ub)
            self.y[i] = math.inf
        self.best_x = self.x[0]
        self.best_value = math.inf
        self.best_i = 0
        self.pop_iter = np.zeros(self.popsize)
       
    def _do_optimize(self, fun, max_evals):
        self.fun = fun
        self.max_evals = max_evals    
        self.iterations = 0
        self.evals = 0
        while self.evals < self.max_evals:
            for p in range(self.popsize):
                xb, xi, x = self._next_x(p)
                y = self.fun(x)
                self.evals += 1
                if y < self.y[p]:
                    # temporal locality
                    if self.iterations > 1:
                        x2 = self._next_improve(xb, x, xi)
                        y2 = self.fun(x2)
                        self.evals += 1
                        if y2 < y:
                            y = y2
                            x = x2
                    self.x[p] = x
                    self.y[p] = y
                    self.pop_iter[p] = self.iterations
                    if y < self.y[self.best_i]:
                        self.best_i = p;
                        if y < self.best_value:
                            self.best_value = y;
                            self.best_x = x;
                            if not self.stop_fittness is None and self.stop_fittness > y:
                                self.stop = 1
                else:
                    # reinitialize individual
                    if self.rg.uniform(0, self.keep) < self.iterations - self.pop_iter[p]:
                        self.x[p] = self.rg.uniform(self.bounds.lb, self.bounds.ub)
                        self.y[p] = math.inf
                if self.evals >= self.max_evals:
                    break

        return self.best_x, self.best_value, self.evals, self.iterations, self.stop

    def _do_optimize_delayed_update(self, fun, max_evals, workers=mp.cpu_count()):
        self.fun = fun
        self.max_evals = max_evals    
        evaluator = Evaluator(self.fun)
        evaluator.start(workers)
        evals_x = {}
        self.iterations = 0
        self.evals = 0
        self.p = 0
        self.improves = deque()
        for _ in range(workers): # fill queue with initial population
            p, x = self.ask_one()
            evaluator.pipe[0].send((self.evals, x))
            evals_x[self.evals] = p, x # store x
            self.evals += 1
            
        while True: # read from pipe, tell de and create new x
            evals, y = evaluator.pipe[0].recv()
            
            p, x = evals_x[evals] # retrieve evaluated x
            del evals_x[evals]
            self.tell_one(p, y, x) # tell evaluated x
            if self.stop != 0 or self.evals >= self.max_evals:
                break # shutdown worker if stop criteria met
            
            p, x = self.ask_one() # create new x          
            evaluator.pipe[0].send((self.evals, x))       
            evals_x[self.evals] = p, x  # store x
            self.evals += 1
            
        evaluator.stop()
        return self.best_x, self.best_value, self.evals, self.iterations, self.stop
       
    def _next_x(self, p):
        if p == 0:
            self.iterations += 1
            self.Cr = 0.5*self.Cr0 if self.iterations % 2 == 0 else self.Cr0
            self.F = 0.5*self.F0 if self.iterations % 2 == 0 else self.F0
        while True:
            r = self.rg.integers(0, self.popsize, 2)
            if r[0] != p and r[0] != self.best_i and r[0] != r[1] \
                    and r[1] != p and r[1] != self.best_i:
                break
        xi = self.x[p]
        xb = self.x[self.best_i]
        xr0 = self.x[r[0]]
        xr1 = self.x[r[1]]
        jr = self.rg.integers(0, self.dim)  
        return xb, xi, np.asarray([self._next_x_j(xi, xb, xr0, xr1, j, jr) for j in range(self.dim)])

    def _next_improve(self, xb, x, xi):
        return self._feasible(xb + ((x - xi) * 0.5))
    
    def _next_x_j(self, xi, xb, xr0, xr1, j, jr):
        if j == jr or self.rg.uniform(0,1) < self.Cr:
            return self._feasible_j(j, xb[j] + self.F*(xr0[j] - xr1[j]))
        else:
            return xi[j]
    
    def _feasible(self, x):
        return np.maximum(np.minimum(x, self.upper), self.lower)
    
    def _feasible_j(self, j, xj):
        if xj >= self.lower[j] and xj <= self.upper[j]:
            return xj
        else:
            return self.rg.uniform(self.lower[j], self.upper[j])
            


