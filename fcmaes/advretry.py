# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

import time
import os
import math
import random
import ctypes as ct
import numpy as np
from random import Random
import multiprocessing as mp
from multiprocessing import Process
from numpy.random import Generator, MT19937, SeedSequence
from scipy.optimize import OptimizeResult, Bounds

from fcmaes.retry import _convertBounds
from fcmaes.optimizer import dtime, fitting, de_cma

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

def minimize(fun, 
             bounds, 
             value_limit = math.inf,
             num_retries = 5000,
             logger = None,
             workers = mp.cpu_count(),
             popsize = 31, 
             min_evaluations = 1500, 
             max_eval_fac = 50, 
             check_interval = 100,
             stop_fittness = None,
             optimizer = None
             ):   
    """Minimization of a scalar function of one or more variables using 
    coordinated parallel CMA-ES retry.
     
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
    value_limit : float, optional
        Upper limit for optimized function values to be stored. 
        This limit needs to be carefully set to a value which is seldom
        found by optimization retry to keep the store free of bad runs.
        The crossover offspring of bad parents can
        cause the algorithm to get stuck at local minima.   
    num_retries : int, optional
        Number of optimization retries.    
    logger : logger, optional
        logger for log output of the retry mechanism. If None, logging
        is switched off. Default is a logger which logs both to stdout and
        appends to a file ``optimizer.log``.
    workers : int, optional
        number of parallel processes used. Default is mp.cpu_count()
    popsize = int, optional
        CMA-ES population size used for all CMA-ES runs. 
        Not used for differential evolution. 
        Ignored if parameter optimizer is defined. 
    min_evaluations : int, optional 
        Initial limit of the number of function evaluations. Only used if optimizer is undefined, 
        otherwise this setting is defined in the optimizer. 
    max_eval_fac : int, optional
        Final limit of the number of function evaluations = max_eval_fac*min_evaluations
    check_interval : int, optional
        After ``check_interval`` runs the store is sorted and the evaluation limit
        is incremented by ``evals_step_size``
    stop_fittness : float, optional 
         Limit for fitness value. optimization runs terminate if this value is reached. 
    optimizer : optimizer.Optimizer, optional
        optimizer to use. Default is a sequence of differential evolution and CMA-ES.
        Since advanced retry sets the initial step size it works best if CMA-ES is 
        used / in the sequence of optimizers. 
    
    Returns
    -------
    res : scipy.OptimizeResult
        The optimization result is represented as an ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, 
        ``fun`` the best function value, ``nfev`` the number of function evaluations,
        ``success`` a Boolean flag indicating if the optimizer exited successfully. """

    if optimizer is None:
        optimizer = de_cma(min_evaluations, popsize, stop_fittness)     
    store = Store(bounds, max_eval_fac = max_eval_fac, 
              check_interval = check_interval, logger = logger)
    return retry(fun, store, optimizer.minimize, num_retries, value_limit, workers)

def retry(fun, store, optimize, num_retries, value_limit = math.inf, workers=mp.cpu_count()):
    sg = SeedSequence()
    rgs = [Generator(MT19937(s)) for s in sg.spawn(workers)]
    proc=[Process(target=_retry_loop,
            args=(pid, rgs, fun, store, optimize, num_retries, value_limit)) for pid in range(workers)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    store.sort()
    store.dump()
    return OptimizeResult(x=store.get_x_best(), fun=store.get_y_best(), 
                          nfev=store.get_count_evals(), success=True)
 
class Store(object):
    """thread safe storage for optimization retry results; 
    delivers boundary and initial step size vectors for advanced retry crossover operation."""
         
    def __init__(self, 
                 bounds, # bounds of the objective function arguments
                 max_eval_fac = 50, # maximal number of evaluations
                 check_interval = 100, # sort evaluation store after check_interval iterations
                 capacity = 500, # capacity of the evaluation store
                 logger = None # if None logging is switched off
               ):
         
        self.lower, self.upper = _convertBounds(bounds)
        self.logger = logger        
        self.delta = self.upper - self.lower
        self.capacity = capacity
        self.max_eval_fac = max_eval_fac
        self.check_interval = check_interval       
        self.dim = len(self.lower)
        self.random = Random()
        self.t0 = time.perf_counter()
        
        #shared between processes
        self.add_mutex = mp.Lock()    
        self.check_mutex = mp.Lock()                     
        self.xs = mp.RawArray(ct.c_double, capacity * self.dim)
        self.lowers = mp.RawArray(ct.c_double, capacity * self.dim)
        self.uppers = mp.RawArray(ct.c_double, capacity * self.dim)
        self.ys = mp.RawArray(ct.c_double, capacity)                  
        self.eval_fac = mp.RawValue(ct.c_int, 1)
        self.count_evals = mp.RawValue(ct.c_long, 0)   
        self.count_runs = mp.RawValue(ct.c_int, 0) 
        self.num_stored = mp.RawValue(ct.c_int, 0) 
        self.num_sorted = mp.RawValue(ct.c_int, 0)  
        self.best_y = mp.RawValue(ct.c_double, math.inf) 
        self.worst_y = mp.RawValue(ct.c_double, math.inf)  
        self.best_x = mp.RawArray(ct.c_double, self.dim)
                                    
    def eval_num(self, max_evals):
        return self.eval_fac.value * max_evals
                                               
    def limits(self): 
        """guess, boundaries and initial step size for crossover operation."""
        diff_fac = self.random.uniform(0.5, 1.0);
        lim_fac =  self.random.uniform(2.0, 4.0) * diff_fac;
        with self.add_mutex:
            i, j = self.crossover()
            if i < 0:
                return math.inf, None, None, None, None
            x0 = np.asarray(self.get_x(i))
            x1 = np.asarray(self.get_x(j))
            y0 = np.asarray(self.get_y(i))
             
        deltax = np.abs(x1 - x0)
        delta_bound = np.maximum(0.0001, lim_fac * deltax)
        lower = np.maximum(self.lower, x0 - delta_bound)
        upper = np.minimum(self.upper, x0 + delta_bound)
        sdev = np.maximum(0.001, np.minimum(0.5, diff_fac * deltax / self.delta))        
        return y0, x1, lower, upper, sdev
                 
    def distance(self, yprev, y, xprev, x): 
        """mean quadratic X difference to neighbor entry."""
        diff = self.worst_y.value - self.best_y.value
        if diff != 0 and (y - yprev) / diff > 0.01: #enough y distance: accept 
            return 1        
        dx = (np.asarray(x) - np.asarray(xprev)) / self.delta
        return math.sqrt(sum(dx*dx)/self.dim)
  
    def replace(self, i, y, xs, lower, upper):
        """replace entry in store."""
        self.set_y(i, y)
        self.set_x(i, xs)
        self.set_lower(i, lower)
        self.set_upper(i, upper)
        
    def crossover(self): # Choose two good entries for recombination
        """indices of store entries to be used for crossover operation."""
        n = self.num_sorted.value
        if n < 2:
            return -1, -1
        lim = self.random.uniform(min(0.1*n, 1.5), 0.2*n)/n
        for c in range(100):
            i1 = -1
            i2 = -1
            for j in range(n):
                if self.random.random() < lim:
                    if i1 < 0:
                        i1 = j
                    else:
                        i2 = j
                        return i1, i2
        return -1, -1
             
    def sort(self): 
        """sorts all store entries, keep only the 90% best to make room for new ones;
        skip entries having similar x values than their neighbors to preserve diversity"""
        ns = self.num_stored.value
        if ns == 0:
            return
        ys = np.asarray(self.ys[:ns])
        yi = ys.argsort()
        sortRuns = []
        yprev = xprev = yprev2 = xprev2 = None
        for i in range(len(yi)):
            y = ys[yi[i]]
            x = self.get_x(yi[i])
            if (yprev is None or (self.distance(yprev, y, xprev, x) > 0.15) and 
                (yprev2 is None or self.distance(yprev2, y, xprev2, x) > 0.15)):
                sortRuns.append((y, x, self.get_lower(yi[i]), self.get_upper(yi[i])))
                yprev2, xprev2 = yprev, xprev
                yprev, xprev = y, x
        numStored = min(len(sortRuns),int(0.9*self.capacity)) # keep 90% best 
        for i in range(numStored):
            self.replace(i, sortRuns[i][0], sortRuns[i][1], sortRuns[i][2], sortRuns[i][3])
        self.num_sorted.value = numStored  
        self.num_stored.value = numStored     
        self.worst_y.value = self.get_y(numStored-1)
        return numStored        
     
    def add_result(self, y, xs, lower, upper, evals, limit=math.inf):
        """registers an optimization result at the store."""
        with self.add_mutex:
            self.incr_count_evals(evals)
            if y < limit:
                if y < self.best_y.value:
                    self.best_y.value = y
                    self.best_x[:] = xs[:]
                    self.dump()
                if self.num_stored.value >= self.capacity - 1:
                    self.sort()
                ns = self.num_stored.value
                self.num_stored.value = ns + 1
                self.replace(ns, y, xs, lower, upper)
                     
    def get_x(self, pid):
        return self.xs[pid*self.dim:(pid+1)*self.dim]

    def get_x_best(self):
        return self.best_x[:]
     
    def get_y(self, pid):
        return self.ys[pid]

    def get_y_best(self):
        return self.best_y.value
 
    def get_ys(self):
        return self.ys[:self.num_stored.value]
         
    def get_lower(self, pid):
        return self.lowers[pid*self.dim:(pid+1)*self.dim]
 
    def get_upper(self, pid):
        return self.uppers[pid*self.dim:(pid+1)*self.dim]
     
    def get_count_evals(self):
        return self.count_evals.value
  
    def get_count_runs(self):
        return self.count_runs.value
    
    def set_x(self, pid, xs):
        self.xs[pid*self.dim:(pid+1)*self.dim] = xs[:]
        
    def set_y(self, pid, y):
        self.ys[pid] = y            
 
    def set_lower(self, pid, lower):
        self.lowers[pid*self.dim:(pid+1)*self.dim] = lower[:]
 
    def set_upper(self, pid, upper):
        self.uppers[pid*self.dim:(pid+1)*self.dim] = upper[:]
 
    def get_runs_compare_incr(self, limit):
        with self.add_mutex:
            if self.count_runs.value < limit:
                self.count_runs.value += 1
                return True
            else:
                return False 

    def incr_count_evals(self, evals):
        """registers the number of evaluations of an optimization run; 
        trigger sorting after check_interval calls. """
        if self.count_runs.value % self.check_interval == self.check_interval-1:
            if self.eval_fac.value < self.max_eval_fac:
                self.eval_fac.value += 1
                #print(self.eval_fac.value)
            self.sort()
        self.count_evals.value += evals
         
    def dump(self):
        """logs the current status of the store if logger defined."""
        if self.logger is None:
            return
        Ys = self.get_ys()
        vals = []
        for i in range(min(20, len(Ys))):
            vals.append(round(Ys[i],2))     
        dt = dtime(self.t0)            
        message = '{0} {1} {2} {3} {4:.6f} {5:.2f} {6} {7} {8!s} {9!s}'.format(
            dt, int(self.count_evals.value / dt), self.count_runs.value, self.count_evals.value, 
            self.best_y.value, self.worst_y.value, self.num_stored.value, self.eval_fac.value, 
            vals, self.best_x[:])
        self.logger.info(message)
          
def _retry_loop(pid, rgs, fun, store, optimize, num_retries, value_limit):
    while store.get_runs_compare_incr(num_retries):               
        if _crossover(fun, store, optimize, rgs[pid]):
            continue
        try:
            dim = len(store.lower)
            sol, y, evals = optimize(fun, Bounds(store.lower, store.upper), None, 
                                     [random.uniform(0.05, 0.1)]*dim, rgs[pid], store)
            store.add_result(y, sol, store.lower, store.upper, evals, value_limit)
        except Exception as ex:
            continue
#         if pid == 0:
#             store.dump()
   
def _crossover(fun, store, optimize, rg):
    if random.random() < 0.5:
        return False
    y0, guess, lower, upper, sdev = store.limits()
    if guess is None:
        return False
    guess = fitting(guess, lower, upper) # take X from lower
    try:       
        sol, y, evals = optimize(fun, Bounds(lower, upper), guess, sdev, rg, store)
        store.add_result(y, sol, lower, upper, evals, y0) # limit to y0  
    except:
        return False   
    return True
