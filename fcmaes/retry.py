# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.
from __future__ import annotations

import time
import math
import os
import sys
import threadpoolctl
import ctypes as ct
from scipy import interpolate
import numpy as np
from numpy.random import Generator, MT19937, SeedSequence
from scipy.optimize._constraints import new_bounds_to_old
from scipy.optimize import OptimizeResult, Bounds
import multiprocessing as mp
from multiprocessing import Process
from fcmaes.optimizer import de_cma, dtime, Optimizer
from fcmaes.evaluator import is_debug_active
from loguru import logger
from typing import Optional, Callable, List
from numpy.typing import ArrayLike

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

def minimize(fun: Callable[[ArrayLike], float], 
             bounds: Bounds, 
             value_limit: Optional[float] = np.inf,
             num_retries: Optional[int] = 1024,
             workers: Optional[int] = mp.cpu_count(),
             popsize: Optional[int] = 31, 
             max_evaluations: Optional[int] = 50000, 
             capacity: Optional[int] = 500,
             stop_fitness: Optional[float] = -np.inf,
             optimizer: Optional[Optimizer] = None,
             statistic_num: Optional[int] = 0,
             plot_name:str = None
             ) -> OptimizeResult:   
    """Minimization of a scalar function of one or more variables using parallel 
     optimization retry.
     
    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
            ``fun(x) -> float``
        where ``x`` is an 1-D array with shape (n,)
    bounds : sequence or `Bounds`, optional
        Bounds on variables. There are two ways to specify the bounds:
            1. Instance of the `scipy.Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.
    value_limit : float, optional
        Upper limit for optimized function values to be stored. 
    num_retries : int, optional
        Number of optimization retries.    
    workers : int, optional
        number of parallel processes used. Default is mp.cpu_count()
    popsize = int, optional
        CMA-ES population size used for all CMA-ES runs. 
        Not used for differential evolution. 
        Ignored if parameter optimizer is defined. 
    max_evaluations : int, optional
        Forced termination of all optimization runs after ``max_evaluations`` 
        function evaluations. Only used if optimizer is undefined, otherwise
        this setting is defined in the optimizer. 
    capacity : int, optional
        capacity of the evaluation store.
    stop_fitness : float, optional 
         Limit for fitness value. optimization runs terminate if this value is reached. 
    optimizer : optimizer.Optimizer, optional
        optimizer to use. Default is a sequence of differential evolution and CMA-ES.
    statistic_num: int, optional
        if > 0 stores the progress of the optimization. Defines the size of this store. 
    plot_name : String, optional
        if defined plots are generated during the optimization to monitor progress.
        Requires statistic_num > 100.

     
    Returns
    -------
    res : scipy.OptimizeResult
        The optimization result is represented as an ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, 
        ``fun`` the best function value, ``nfev`` the number of function evaluations,
        ``success`` a Boolean flag indicating if the optimizer exited successfully. """

    if optimizer is None:
        optimizer = de_cma(max_evaluations, popsize, stop_fitness)        
    store = Store(fun, bounds, capacity = capacity, statistic_num = statistic_num, 
                  plot_name = plot_name)
    return retry(store, optimizer.minimize, num_retries, value_limit, workers, stop_fitness)

def retry(store: Store, 
          optimize: Callable, 
          num_retries: int, 
          value_limit: Optional[float] = np.inf, 
          workers: Optional[int] = mp.cpu_count(), 
          stop_fitness: Optional[float] = -np.inf) -> OptimizeResult:
    
    sg = SeedSequence()
    rgs = [Generator(MT19937(s)) for s in sg.spawn(workers)]
    proc=[Process(target=_retry_loop,
            args=(pid, rgs, store, optimize, num_retries, value_limit, stop_fitness)) for pid in range(workers)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    store.sort()
    store.dump()
    return OptimizeResult(x=store.get_x_best(), fun=store.get_y_best(), 
                          nfev=store.get_count_evals(), success=True)

def minimize_plot(name: str, 
                  optimizer: Optimizer, 
                  fun: Callable[[ArrayLike], float], 
                  bounds: Bounds, 
                  value_limit: Optional[float] = np.inf, 
                  plot_limit: Optional[float] = np.inf, 
                  num_retries: Optional[int] = 1024, 
                  workers: Optional[int] = mp.cpu_count(), 
                  stop_fitness: Optional[float] = -np.inf, 
                  statistic_num: Optional[int] = 5000, 
                  plot_name:str = None) -> OptimizeResult:
    
    time0 = time.perf_counter() # optimization start time
    name += '_' + optimizer.name
    logger.info('optimize ' + name)       
    store = Store(fun, bounds, capacity = 500,
                  statistic_num = statistic_num, plot_name = plot_name)
    ret = retry(store, optimizer.minimize, num_retries, value_limit, workers, stop_fitness)
    impr = store.get_improvements()
    np.savez_compressed(name, ys=impr)
    for _ in range(10):
        filtered = np.array([imp for imp in impr if imp[1] < plot_limit])
        if len(filtered) > 0: 
            impr = filtered
            break
        else:
            plot_limit *= 3
    logger.info(name + ' time ' + str(dtime(time0))) 
    plot(impr, 'progress_ret.' + name + '.png', label = name, 
         xlabel = 'time in sec', ylabel = r'$f$')
    return ret

def plot(front: ArrayLike, fname: str, interp: Optional[bool] = True, 
         label: Optional[str] = r'$\chi$', 
         xlabel: Optional[str] = r'$f_1$', ylabel:Optional[str] = r'$f_2$', 
         zlabel: Optional[str] = r'$f_3$', plot3d: Optional[bool] = False, 
         s = 1, dpi=300):
    if len(front[0]) == 3 and plot3d:
        plot3(front, fname, label, xlabel, ylabel, zlabel)
        return
    if len(front[0]) >= 3:
        for i in range(1, len(front[0])):
            plot(front.T[np.array([0,i])].T, str(i) + '_' + fname, 
                 interp=interp, ylabel = r'$f_{0}$'.format(i+1))     
        return   
    if len(front[0]) == 1:
        ys = np.array(list(zip(range(100), [front[0][0]]*100)))
        plot(ys, str(1) + '_' + fname, 
                 interp=interp, xlabel = '', ylabel = r'$f_{0}$'.format(1))     
        return      
    import matplotlib.pyplot as pl
    fig, ax = pl.subplots(1, 1)
    x = front[:, 0]; y = front[:, 1]
    if interp and len(x) > 2:
        xa = np.argsort(x)
        xs = x[xa]; ys = y[xa]
        x = []; y = []
        for i in range(len(xs)): # filter equal x values
            if i == 0 or xs[i] > xs[i-1] + 1E-5:
                x.append(xs[i]); y.append(ys[i])
        tck = interpolate.InterpolatedUnivariateSpline(x,y,k=1)
        x = np.linspace(min(x),max(x),1000)
        y = [tck(xi) for xi in x]
    ax.scatter(x, y, label=label, s=s)
    ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.savefig(fname, dpi=dpi)
    pl.close('all')

def plot3(front: ArrayLike, fname: str, label: Optional[str] =r'$\chi$', 
         xlabel: Optional[str] = r'$f_1$', ylabel: Optional[str] = r'$f_2$', 
         zlabel: Optional[str] = r'$f_3$'):
    import matplotlib.pyplot as pl
    fig = pl.figure()
    ax = fig.add_subplot(projection='3d')
    x = front[:, 0]; y = front[:, 1]; z = front[:, 2]
    ax.scatter(x, y, z, label=label, s=1)
    ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.legend()
    #pl.show()
    fig.savefig(fname, dpi=300)
    pl.close('all')
 
class Store(object):
    """thread safe storage for optimization retry results."""
       
    def __init__(self, 
                 fun: Callable[[ArrayLike], float], # fitness function
                 bounds: Bounds, # bounds of the objective function arguments
                 check_interval: Optional[int] = 10, # sort evaluation memory after check_interval iterations
                 capacity: Optional[int] = 500, # capacity of the evaluation store
                 statistic_num: Optional[int] = 0,
                 plot_name: Optional[str] = None # requires statistic_num > 500
                ):    
        self.fun = fun
        self.lower, self.upper = _convertBounds(bounds)
        self.capacity = capacity
        self.check_interval = check_interval
        self.dim = len(self.lower)
        self.delta = []
        for k in range(self.dim):
            self.delta.append(self.upper[k] - self.lower[k])
        
        #shared between processes
        self.add_mutex = mp.Lock()    
        self.xs = mp.RawArray(ct.c_double, self.capacity * self.dim)
        self.ys = mp.RawArray(ct.c_double, self.capacity)  
        self.count_evals = mp.RawValue(ct.c_long, 0)   
        self.count_runs = mp.RawValue(ct.c_int, 0) 
        self.num_stored = mp.RawValue(ct.c_int, 0) 
        self.num_sorted = mp.RawValue(ct.c_int, 0)  
        self.count_stat_runs = mp.RawValue(ct.c_int, 0)  
        self.t0 = time.perf_counter()
        self.mean = mp.RawValue(ct.c_double, 0) 
        self.qmean = mp.RawValue(ct.c_double, 0) 
        self.best_y = mp.RawValue(ct.c_double, np.inf) 
        self.best_x = mp.RawArray(ct.c_double, self.dim)
        self.statistic_num = statistic_num
        self.plot_name = plot_name
        # statistics   
        self.statistic_num = statistic_num                         
        if statistic_num > 0:  # enable statistics                          
            self.time = mp.RawArray(ct.c_double, self.statistic_num)
            self.val = mp.RawArray(ct.c_double, self.statistic_num)
            self.si = mp.RawValue(ct.c_int, 0)
            self.sevals = mp.RawValue(ct.c_long, 0)
            self.bval = mp.RawValue(ct.c_double, np.inf)

    # register improvement - time and value
    def wrapper(self, x: ArrayLike):
        y = self.fun(x)
        self.sevals.value += 1
        if y < self.bval.value:
            self.bval.value = y
            si = self.si.value
            if si < self.statistic_num - 1:
                self.si.value = si + 1
            self.time[si] = dtime(self.t0)
            self.val[si] = y  
            logger.info(str(self.time[si]) + ' '  + 
                      str(self.sevals.value) + ' ' + 
                      str(int(self.sevals.value / self.time[si])) + ' ' + 
                      str(y) + ' ' + 
                      str(list(x)))
        return y
         
    def get_improvements(self):
        return np.array(list(zip(self.time[:self.si.value], self.val[:self.si.value])))
        
    # get num best values at evenly distributed times
    def get_statistics(self, num: int) -> List:
        ts = self.time[:self.si.value]
        ys = self.val[:self.si.value]
        mt = ts[-1]
        dt = 0.9999999 * mt / num
        conv = []
        ti = 0
        val = ys[0]
        for i in range(num):
            while ts[ti] < (i+1) * dt:
                ti += 1
                val = ys[ti]
            conv.append(val)
        return conv
    
    def eval_num(self, max_evals: int) -> int:
        return max_evals
                                             
    def replace(self, i: int, y: float, xs: ArrayLike):
        self.set_y(i, y)
        self.set_x(i, xs)
             
    def sort(self) -> int: # sort all entries to make room for new ones, determine best and worst
        """sorts all store entries, keep only the 90% best to make room for new ones."""
        ns = self.num_stored.value
        ys = np.asarray(self.ys[:ns])
        yi = ys.argsort()
        sortRuns = []
        for i in range(len(yi)):
            y = ys[yi[i]]
            xs = self.get_x(yi[i])
            sortRuns.append((y, xs))
        numStored = min(len(sortRuns),int(0.9*self.capacity)) # keep 90% best 
        for i in range(numStored):
            self.replace(i, sortRuns[i][0], sortRuns[i][1])
        self.num_sorted.value = numStored  
        self.num_stored.value = numStored  
        return numStored        
            
    def add_result(self, y: float, xs: ArrayLike, evals: int, limit=np.inf):
        """registers an optimization result at the store."""
        with self.add_mutex:
            self.incr_count_evals(evals)
            if y < limit:  
                self.count_stat_runs.value += 1
                if y < self.best_y.value:
                    self.best_y.value = y
                    self.best_x[:] = xs[:]
                    self.dump()
                if self.num_stored.value >= self.capacity-1:
                    self.sort()
                cnt = self.count_stat_runs.value
                diff = min(1E20, y - self.mean.value) # avoid overflow
                self.qmean.value += (cnt - 1)/ cnt * diff*diff ;
                self.mean.value += diff / cnt
                ns = self.num_stored.value
                self.num_stored.value = ns + 1
                self.replace(ns, y, xs)
            
    def get_x(self, pid: int):
        return self.xs[pid*self.dim:(pid+1)*self.dim]

    def get_x_best(self) -> np.ndarray:
        return np.array(self.best_x[:])
    
    def get_xs(self) -> np.ndarray:
        return np.array([self.get_x(i) for i in range(self.num_stored.value)])
    
    def get_y(self, pid: int) -> float:
        return self.ys[pid]

    def get_y_best(self) -> float:
        return self.best_y.value
    
    def get_ys(self) -> np.ndarray:
        return np.array(self.ys[:self.num_stored.value])
             
    def get_y_mean(self) -> float:
        return self.mean.value

    def get_y_standard_dev(self) -> float:
        cnt = self.count_stat_runs.value
        return 0 if cnt <= 0 else math.sqrt(self.qmean.value / cnt)

    def get_count_evals(self) -> int:
        return self.count_evals.value
 
    def get_count_runs(self) -> int:
        return self.count_runs.value
 
    def set_x(self, pid: int, xs: ArrayLike):
        self.xs[pid*self.dim:(pid+1)*self.dim] = xs[:]
       
    def set_y(self, pid: int, y: float):
        self.ys[pid] = y    
 
    def get_runs_compare_incr(self, limit: float):
        with self.add_mutex:
            if self.count_runs.value < limit:
                self.count_runs.value += 1
                return True
            else:
                return False 
       
    def incr_count_evals(self, evals: int):
        if self.count_runs.value % self.check_interval == self.check_interval-1:
            self.sort()
        self.count_evals.value += evals
            
    def dump(self):
        """logs the current status of the store if logger defined."""
        if not is_debug_active():
            return
        Ys = self.get_ys()
        vals = []
        for i in range(min(20, len(Ys))):
            vals.append(round(Ys[i],4))     
        dt = dtime(self.t0)                   
        message = '{0} {1} {2} {3} {4:.6f} {5:.2f} {6:.2f} {7!s} {8!s}'.format(
            dt, int(self.count_evals.value / dt), self.count_runs.value, self.count_evals.value, \
                self.best_y.value, self.get_y_mean(), self.get_y_standard_dev(), vals, self.best_x[:])
        logger.debug(message)

        
def _retry_loop(pid, rgs, store, optimize, num_retries, value_limit, stop_fitness = -np.inf):
    fun = store.wrapper if store.statistic_num > 0 else store.fun
        
    lower = store.lower
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        while store.get_runs_compare_incr(num_retries) and store.best_y.value > stop_fitness:      
            try:       
                rg = rgs[pid]
                sol, y, evals = optimize(fun, Bounds(store.lower, store.upper), None, 
                                         [rg.uniform(0.05, 0.1)]*len(lower), rg, store)
                store.add_result(y, sol, evals, value_limit)   
                if not store.plot_name is None: 
                    name = store.plot_name + "_retry_" + str(store.get_count_evals())
                    xs = np.array(store.get_xs())
                    ys = np.array(store.get_ys())
                    np.savez_compressed(name, xs=xs, ys=ys) 
                    plot(y, name, interp=False)    
            except Exception as ex:
                print(str(ex))
#        if pid == 0:
#            store.dump()

def _convertBounds(bounds):
    if bounds is None:
        raise ValueError('bounds need to be defined')
    if isinstance(bounds, Bounds):
        limits = np.array(new_bounds_to_old(bounds.lb,
                                                 bounds.ub,
                                                 len(bounds.lb)),
                               dtype=float).T
    else:
        limits = np.array(bounds, dtype='float').T
    if (np.size(limits, 0) != 2 or not
            np.all(np.isfinite(limits))):
        raise ValueError('bounds should be a sequence containing '
                         'real valued (min, max) pairs for each value'
                         ' in x')
    return limits[0], limits[1]
