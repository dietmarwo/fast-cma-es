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
from fcmaes.optimizer import Optimizer, dtime, fitting, seed_random

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
             min_evaluations = 2000, 
             max_evaluations = 50000, 
             evals_step_size = 1000, 
             check_interval = 100,
             useCpp = False,
             stop_fittness = None,
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
        Upper limit for CMA-ES optimized function values to be stored. 
        This limit needs to be carefully set to a value which is seldom
        found by CMA-ES retry to keep the store free of bad runs.
        The crossover offspring of bad parents can
        cause the algorithm to get stuck at local minima.   
    num_retries : int, optional
        Number of CMA-ES retries.    
    logger : logger, optional
        logger for log output of the retry mechanism. If None, logging
        is switched off. Default is a logger which logs both to stdout and
        appends to a file ``optimizer.log``.
    workers : int, optional
        number of parallel processes used. Default is mp.cpu_count()
    popsize = int, optional
        CMA-ES population size used for all CMA-ES runs.
    min_evaluations : int, optional 
        Initial limit of the number of function evaluations.
    max_evaluations : int, optional
        Final limit of the number of function evaluations.
    evals_step_size : int, optional
        Delta the limit of the number of function evaluations is incremented after 
        ``check_interval`` runs
    check_interval : int, optional
        After ``check_interval`` runs the store is sorted and the evaluation limit
        is incremented by ``evals_step_size``
    useCpp : bool, optional
        Flag indicating use of the C++ CMA-ES implementation. Default is `False` - 
        use of the Python CMA-ES implementation
    stop_fittness : float, optional 
         Limit for fitness value. CMA-ES runs terminate if this value is reached. 
    
    Returns
    -------
    res : scipy.OptimizeResult
        The optimization result is represented as an ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, 
        ``fun`` the best function value, ``nfev`` the number of function evaluations,
        ``nit`` the number of CMA-ES iterations, ``status`` the stopping critera and
        ``success`` a Boolean flag indicating if the optimizer exited successfully. """

    store = Store(bounds, min_evaluations = min_evaluations, max_evaluations = max_evaluations, 
                  evals_step_size = evals_step_size, check_interval = check_interval, logger = logger)
    optimizer = Optimizer(store, popsize, stop_fittness)
    optimize = optimizer.cma_cpp if useCpp else optimizer.cma_python
    return retry(fun, store, optimize, num_retries, value_limit, workers)

def retry(fun, store, optimize, num_retries, value_limit = math.inf, workers=mp.cpu_count()):
    sg = SeedSequence()
    rgs = [Generator(MT19937(s)) for s in sg.spawn(workers)]
    proc=[Process(target=_retry_loop,
            args=(pid, rgs, fun, store, optimize, num_retries, value_limit)) for pid in range(workers)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    store.sort()
    store.dump()
    return OptimizeResult(x=store.get_x(0), fun=store.get_y(0), nfev=store.get_count_evals(), success=True)

class Store(object):
    """thread safe storage for optimization retry results; 
    delivers boundary and initial step size vectors for advanced retry crossover operation."""
        
    def __init__(self, 
                 bounds, # bounds of the objective function arguments
                 min_evaluations = 2000, # start with this number of evaluations
                 max_evaluations = 50000, # maximal number of evaluations
                 evals_step_size = 1000, # increase evaluation number by eval_step_size after sorting
                 check_interval = 100, # sort evaluation store after check_interval iterations
                 capacity = 500, # capacity of the evaluation store
                 logger = None # if None logging is switched off
               ):
        
        self.lower, self.upper = _convertBounds(bounds)
        self.logger = logger        
        self.delta = self.upper - self.lower
        self.capacity = capacity
        self.max_evaluations = max_evaluations
        self.evals_step_size = evals_step_size
        self.check_interval = check_interval       
        self.dim = len(self.lower)
        self.random = Random()
        self.t0 = time.perf_counter();
       
        #shared between processes
        self.add_mutex = mp.Lock()    
        self.check_mutex = mp.Lock()                     
        self.xs = mp.RawArray(ct.c_double, capacity * self.dim)
        self.lowers = mp.RawArray(ct.c_double, capacity * self.dim)
        self.uppers = mp.RawArray(ct.c_double, capacity * self.dim)
        self.ys = mp.RawArray(ct.c_double, capacity)                  
        self.max_evals = mp.RawValue(ct.c_int, min_evaluations)
        self.count_evals = mp.RawValue(ct.c_long, 0)   
        self.count_runs = mp.RawValue(ct.c_int, 0) 
        self.num_stored = mp.RawValue(ct.c_int, 0) 
        self.num_sorted = mp.RawValue(ct.c_int, 0)  
        self.best_y = mp.RawValue(ct.c_double, math.inf) 
        self.worst_y = mp.RawValue(ct.c_double, math.inf)  
                                    
    def eval_num(self):
        return self.max_evals.value
                                              
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
        ys = np.asarray(self.ys[:ns])
        yi = ys.argsort()
        sortRuns = []
        yprev = None
        xprev = None
        for i in range(len(yi)):
            y = ys[yi[i]]
            x = self.get_x(yi[i])
            if i == 0 or self.distance(yprev, y, xprev, x) > 0.15:
                sortRuns.append((y, x, self.get_lower(yi[i]), self.get_upper(yi[i])))
                yprev = y
                xprev = x
        numStored = min(len(sortRuns),int(0.9*self.capacity)) # keep 90% best 
        for i in range(numStored):
            self.replace(i, sortRuns[i][0], sortRuns[i][1], sortRuns[i][2], sortRuns[i][3])
        self.num_sorted.value = numStored  
        self.num_stored.value = numStored     
        self.best_y.value = self.get_y(0);
        self.worst_y.value = self.get_y(numStored-1);
        return numStored        
    
    def add_result(self, y, xs, lower, upper, evals, limit=math.inf):
        """registers an optimization result at the store."""
        with self.add_mutex:
            self.incr_count_evals(evals)
            if y < limit:
                if y < self.best_y.value:
                    self.best_y.value = y
                    self.dump()
                if self.num_stored.value >= self.capacity - 1:
                    self.sort()
                ns = self.num_stored.value
                self.num_stored.value = ns + 1
                self.replace(ns, y, xs, lower, upper)
                    
    def get_x(self, pid):
        return self.xs[pid*self.dim:(pid+1)*self.dim]
    
    def get_y(self, pid):
        return self.ys[pid]

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

    def incr_count_evals(self, evals):
        """registers the number of evaluations of an optimization run; 
        trigger sorting after check_interval calls. """
        self.count_runs.value += 1
        if self.count_runs.value % self.check_interval == self.check_interval-1:
            if self.max_evals.value < self.max_evaluations:
                self.max_evals.value += self.evals_step_size
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
        message = '{0} {1} {2} {3} {4:.6f} {5:.2f} {6} {7!s} {8!s}'.format(
            dt, int(self.count_evals.value / dt), self.count_runs.value, self.count_evals.value, 
            self.best_y.value, self.worst_y.value, self.num_stored.value, vals, self.get_x(0))
        self.logger.info(message)

        
def _retry_loop(pid, rgs, fun, store, optimize, num_retries, value_limit):
    seed_random() # make sure cpp random generator for this process is initialized properly
    while store.get_count_runs() < num_retries:
        if _crossover(fun, store, optimize, rgs[pid]):
            continue
        try:
            dim = len(store.lower)
            sol, y, evals = optimize(fun, None, Bounds(store.lower, store.upper), 
                                     [random.uniform(0.05, 0.1)]*dim, rgs[pid])
            store.add_result(y, sol, store.lower, store.upper, evals, value_limit)
        except Exception as ex:
            continue
 
def _crossover(fun, store, optimize, rg):
    if random.random() < 0.5:
        return False
    y0, guess, lower, upper, sdev = store.limits()
    if guess is None:
        return False
    guess = fitting(guess, lower, upper) # take X from lower
    try:       
        sol, y, evals = optimize(fun, guess, Bounds(lower, upper), sdev, rg)
        store.add_result(y, sol, lower, upper, evals, y0) # limit to y0  
    except:
        return False   
    return True
