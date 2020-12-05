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
import math
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
             stop_fittness = -math.inf, 
             keep = 200, 
             f = 0.5, 
             cr = 0.9, 
             rg = Generator(MT19937())):    

    de = DE(fun, bounds, popsize, max_evaluations, stop_fittness, 
            keep, f, cr, rg)
    if workers and workers > 1:
        x, val, evals, iterations, stop = de.do_optimize_delayed_update(workers)
    else:      
        x, val, evals, iterations, stop = de.do_optimize()
    return OptimizeResult(x=x, fun=val, nfev=evals, nit=iterations, status=stop, 
                          success=True)

class DE(object):
    
    def __init__(self, fun, bounds, popsize, max_evals, stop_fittness = -math.inf, keep = 200, 
                 F = 0.5, Cr = 0.9, rg = Generator(MT19937())):
        self.fun = fun
        self.dim = len(bounds.ub)
        self.bounds = bounds
        self.lower = np.asarray(bounds.lb)      
        self.upper = np.asarray(bounds.ub)      
        self.popsize = 31 if popsize is None else popsize
        self.stop_fittness = stop_fittness
        self.max_evals = max_evals    
        self.keep = keep 
        self.rg = rg
        self.F0 = F
        self.Cr0 = Cr
        self.stop = 0
        self.init()
 
    def init(self):
        self.x = np.zeros((self.popsize, self.dim))
        self.y = np.empty(self.popsize)
        for i in range(self.popsize):
            self.x[i] = self.rg.uniform(self.bounds.lb, self.bounds.ub)
            self.y[i] = math.inf
        self.best_x = self.x[0]
        self.best_y = math.inf
        self.best_i = 0
        self.pop_iter = np.zeros(self.popsize)
       
    def do_optimize_delayed_update(self, workers=mp.cpu_count()):
        evaluator = Evaluator(self.fun)
        evaluator.start(workers)
        evals_x = {}
        self.iteration = 0
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
            self.tell_one(y, p, x) # tell evaluated x
            if self.stop != 0 or self.evals >= self.max_evals:
                break # shutdown worker if stop criteria met
            
            p, x = self.ask_one() # create new x          
            evaluator.pipe[0].send((self.evals, x))       
            evals_x[self.evals] = p, x  # store x
            self.evals += 1
            
        evaluator.stop()
        return self.best_x, self.best_y, self.evals, self.iteration, self.stop

    def do_optimize(self):
        self.iteration = 0
        self.evals = 0
        while self.evals < self.max_evals:
            self.iteration += 1
            self.Cr = 0.5*self.Cr0 if self.iteration % 2 == 0 else self.Cr0
            self.F = 0.5*self.F0 if self.iteration % 2 == 0 else self.F0
            for p in range(self.popsize):
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
                x = np.asarray([self._next_x_j(xi, xb, xr0, xr1, j, jr) for j in range(self.dim)])   
                y = self.fun(x)
                self.evals += 1
                if y < self.y[p]:
                    # temporal locality
                    x2 = self._feasible(xb + ((x - xi) * 0.5));
                    y2 = self.fun(x2)
                    self.evals += 1
                    if y2 < y:
                        y = y2
                        x = x2
                    self.x[p] = x
                    self.y[p] = y
                    self.pop_iter[p] = self.iteration
                    if y < self.y[self.best_i]:
                        self.best_i = p;
                        if y < self.best_y:
                            self.best_y = y;
                            self.best_x = x;
                            if self.stop_fittness > y:
                                self.stop = 1
                else:
                    # reinitialize individual
                    if self.rg.uniform(0, self.keep) < self.iteration - self.pop_iter[p]:
                        self.x[p] = self.rg.uniform(self.bounds.lb, self.bounds.ub)
                        self.y[p] = math.inf

        return self.best_x, self.best_y, self.evals, self.iteration, self.stop

    def ask_one(self):
        """ask for one new argument vector.
        
        Returns
        -------
        p : int population index 
        x : dim sized argument ."""
        
        if self.improves:
            p, x = self.improves.popleft()
        else:
            p, x = self._next_x()
        return p, x

    def tell_one(self, y, p, x):      
        """tell function value for a argument list retrieved by ask_one().
    
        Parameters
        ----------
        y : function value
        p : int population index 
        x : dim sized argument list
 
        Returns
        -------
        stop : int termination criteria, if != 0 loop should stop."""
        
        if (self.y[p] > y):
            self.improves.append((p, self._next_improve(x, self.x[p])))            
            self.x[p] = x
            self.y[p] = y
            if self.y[self.best_i] > y:
                self.best_i = p
                if self.best_y > y:
                    self.best_x = x
                    self.best_y = y
                    if self.stop_fittness > y:
                        self.stop = 1
            self.pop_iter[p] = self.iteration

        else:
            if self.rg.uniform(0, self.keep) < self.iteration - self.pop_iter[p]:
                self.x[p] = self.rg.uniform(self.bounds.lb, self.bounds.ub)
                self.y[p] = math.inf
        return self.stop 
    
    def _feasible(self, x):
        return np.maximum(np.minimum(x, self.upper), self.lower)
    
    def _feasible_j(self, j, xj):
        if xj >= self.lower[j] and xj <= self.upper[j]:
            return xj
        else:
            return self.rg.uniform(self.lower[j], self.upper[j])
                
    def _next_x_j(self, xi, xb, xr0, xr1, j, jr):
        if j == jr or self.rg.uniform(0,1) < self.Cr:
            return self._feasible_j(j, xb[j] + self.F*(xr0[j] - xr1[j]))
        else:
            return xi[j]
    
    def _next_x(self):
        p = self.p
        if p == 0:
            self.iteration += 1
            self.Cr = 0.5*self.Cr0 if self.iteration % 2 == 0 else self.Cr0
            self.F = 0.5*self.F0 if self.iteration % 2 == 0 else self.F0
        self.p = (self.p + 1) % self.popsize
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
        return p, np.asarray([self._next_x_j(xi, xb, xr0, xr1, j, jr) for j in range(self.dim)])

    def _next_improve(self, ui, xi):
        return self._feasible(self.best_x + ((ui - xi) * 0.5))


from fcmaes import decpp, cmaes, cmaescpp, bitecpp, csmacpp
              
def test_rosen():
    popsize = 16
    dim = 5
    testfun = Rastrigin(dim)
    #testfun = Rosen(dim)
    sdevs = [0.3]*dim
    
    wrapper = Wrapper(testfun.fun, dim)
    ret = minimize(wrapper.eval, testfun.bounds, popsize, 40000, workers = 16)
    #ret = minimize(wrapper.eval, testfun.bounds, popsize, 40000, keep = 200, workers = None)
    #ret = cmaes.minimize(wrapper.eval, testfun.bounds, popsize = popsize, max_evaluations = 40000, workers = None)
    #ret = cmaes.minimize(wrapper.eval, testfun.bounds, popsize = popsize, max_evaluations = 40000, workers = 32, delayed_update=True)
    #ret = cmaescpp.minimize(wrapper.eval, testfun.bounds, popsize = popsize, max_evaluations = 40000, workers = None)
    #ret = bitecpp.minimize(wrapper.eval, testfun.bounds, popsize = popsize, max_evaluations = 40000)
    #ret = csmacpp.minimize(wrapper.eval, testfun.bounds, max_evaluations = 40000)

    #ret = decpp.minimize(wrapper.eval, dim, testfun.bounds, popsize, 40000, keep=200)
    
 #  print(str(ret.nfev) + " " + str(ret.fun) + " " + str(ret.x))
    print(str(wrapper.get_count()) + " " + str(wrapper.get_best_y()) + " " + str(wrapper.get_best_x()))

def main():       
    test_rosen()

if __name__ == '__main__':
    main()
    pass
