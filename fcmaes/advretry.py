# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

import time
import os
import sys
import math
import random
import ctypes as ct
import numpy as np
from numpy.linalg import norm
from random import Random
import multiprocessing as mp
from multiprocessing import Process
from numpy.random import Generator, MT19937, SeedSequence
from scipy.optimize import OptimizeResult, Bounds

from fcmaes.retry import _convertBounds
from fcmaes.optimizer import dtime, fitting, de2_cma, logger

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
             max_eval_fac = None, 
             check_interval = 100,
             capacity = 500,
             stop_fittness = None,
             optimizer = None,
             statistic_num = 0
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
    capacity : int, optional
        capacity of the evaluation store. Higher value means broader search.
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
        optimizer = de2_cma(min_evaluations, popsize, stop_fittness)     
    if max_eval_fac is None:
        max_eval_fac = int(min(50, 1 + num_retries // check_interval))
    store = Store(bounds, max_eval_fac, check_interval, capacity, logger, num_retries, statistic_num)
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
                 max_eval_fac = None, # maximal number of evaluations factor
                 check_interval = 100, # sort evaluation store after check_interval iterations
                 capacity = 500, # capacity of the evaluation store
                 logger = None, # if None logging is switched off
                 num_retries = None,
                 statistic_num = 0
               ):

        self.lower, self.upper = _convertBounds(bounds)
        self.delta = self.upper - self.lower
        self.logger = logger        
        self.capacity = capacity
        if max_eval_fac is None:
            if num_retries is None:
                max_eval_fac = 50
            else:
                max_eval_fac = int(min(50, 1 + num_retries // check_interval))
        if num_retries == None:
            num_retries = max_eval_fac * check_interval
        # increment eval_fac so that max_eval_fac is reached at last retry
        self.eval_fac_incr = max_eval_fac / (num_retries/check_interval)
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
        self.eval_fac = mp.RawValue(ct.c_double, 1)
        self.count_evals = mp.RawValue(ct.c_long, 0)   
        self.count_runs = mp.RawValue(ct.c_int, 0) 
        self.num_stored = mp.RawValue(ct.c_int, 0) 
        self.num_sorted = mp.RawValue(ct.c_int, 0)  
        self.best_y = mp.RawValue(ct.c_double, math.inf) 
        self.worst_y = mp.RawValue(ct.c_double, math.inf)  
        self.best_x = mp.RawArray(ct.c_double, self.dim)
        self.statistic_num = statistic_num
 
        if statistic_num > 0:  # enable statistics                          
            self.statistic_num = 1000
            self.time = mp.RawArray(ct.c_double, self.statistic_num)
            self.val = mp.RawArray(ct.c_double, self.statistic_num)
            self.si = mp.RawValue(ct.c_int, 0)
 
    # store improvement - time and value
    def add_statistics(self):
        if self.statistic_num > 0:
            si = self.si.value
            if si < self.statistic_num - 1:
                self.si.value = si + 1
            self.time[si] = dtime(self.t0)
            self.val[si] = self.best_y.value  
        
    def get_improvements(self):
        return zip(self.time[:self.si.value], self.val[:self.si.value])
 
    # get num best values at evenly distributed times
    def get_statistics(self, num):
        ts = self.time[:self.si.value]
        vs = self.val[:self.si.value]
        mt = ts[-1]
        dt = 0.9999999 * mt / num
        stats = []
        ti = 0
        val = vs[0]
        for i in range(num):
            while ts[ti] < (i+1) * dt:
                ti += 1
                val = vs[ti]
            stats.append(val)
        return stats
                                    
    def eval_num(self, max_evals):
        return int(self.eval_fac.value * max_evals)
                                               
    def limits(self): 
        """guess, boundaries and initial step size for crossover operation."""
        diff_fac = self.random.uniform(0.5, 1.0)
        lim_fac =  self.random.uniform(2.0, 4.0) * diff_fac
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
                 
    def distance(self, xprev, x): 
        """distance between entries in store."""
        return norm((x - xprev) / self.delta) / math.sqrt(self.dim)
        
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
        lim = self.random.uniform(min(0.1*n, 1), 0.2*n)/n
        for _ in range(100):
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
        if ns < 2:
            return

        ys = np.asarray(self.ys[:ns])
        yi = ys.argsort()
        sortRuns = []

        xprev = xprev2 = None
        for i in range(ns):
            y = ys[yi[i]]
            x = np.asarray(self.get_x(yi[i]))
            if (xprev is None or self.distance(xprev, x) > 0.15) and \
                (xprev2 is None or self.distance(xprev2, x) > 0.15): 
                sortRuns.append( (y, x, self.get_lower(yi[i]), self.get_upper(yi[i])) )
                xprev2 = xprev
                xprev = x

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
                    self.add_statistics()
                    self.dump()
                if self.num_stored.value >= self.capacity - 1:
                    self.sort()
                ns = self.num_stored.value
                self.num_stored.value = ns + 1
                self.replace(ns, y, xs, lower, upper)
      
    def get_x(self, pid):
        return self.xs[pid*self.dim:(pid+1)*self.dim]

    def get_xs(self):
        return [self.get_x(i) for i in range(self.num_stored.value)]

    def get_x_best(self):
        return self.best_x[:]

    def get_y(self, pid):
        return self.ys[pid]

    def get_ys(self):
        return self.ys[:self.num_stored.value]

    def get_y_best(self):
        return self.best_y.value

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
                self.eval_fac.value += self.eval_fac_incr
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
            self.best_y.value, self.worst_y.value, self.num_stored.value, int(self.eval_fac.value), 
            vals, self.best_x[:])
        self.logger.info(message)
   
def _retry_loop(pid, rgs, fun, store, optimize, num_retries, value_limit):
    
    #reinitialize logging config for windows -  multi threading fix
    if 'win' in sys.platform and not store.logger is None:
        store.logger = logger()
        
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
