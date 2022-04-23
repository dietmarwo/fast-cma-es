# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# parallel optimization retry of a multi-objective problem.

import numpy as np
import math, sys, time, warnings
import multiprocessing as mp
from multiprocessing import Process
from scipy.optimize import Bounds
from numpy.random import Generator, MT19937, SeedSequence
from fcmaes.optimizer import de_cma, logger, dtime
from fcmaes import retry, advretry

def minimize(fun,             
             bounds,
             weight_bounds,
             ncon = 0,
             value_exp = 2.0,
             value_limits = None,
             num_retries = 1024,
             logger = None,
             workers = mp.cpu_count(),
             popsize = 31, 
             max_evaluations = 50000, 
             capacity = None,
             optimizer = None,
             statistic_num = 0,
             plot_name = None
              ):   
    """Minimization of a multi objective function of one or more variables using parallel 
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
    weight_bounds : `Bounds`, optional
        Bounds on objective weights.
    ncon : int, optional
        number of constraints
    value_exp : float, optional
        exponent applied to the objective values for the weighted sum. 
    value_limits : sequence of floats, optional
        Upper limit for optimized objective values to be stored. 
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
    max_evaluations : int, optional
        Forced termination of all optimization runs after ``max_evaluations`` 
        function evaluations. Only used if optimizer is undefined, otherwise
        this setting is defined in the optimizer. 
    capacity : int, optional
        capacity of the evaluation store.
    optimizer : optimizer.Optimizer, optional
        optimizer to use. Default is a sequence of differential evolution and CMA-ES.
    plot_name : plot_name, optional
        if defined the pareto front is plotted during the optimization to monitor progress
     
    Returns
    -------
    xs, ys: list of argument vectors and corresponding value vectors of the optimization results. """

    if optimizer is None:
        optimizer = de_cma(max_evaluations, popsize)  
    if capacity is None: 
        capacity = num_retries
    store = retry.Store(fun, bounds, capacity = capacity, logger = logger, 
                        statistic_num = statistic_num, plot_name = plot_name)
    xs = np.array(mo_retry(fun, weight_bounds, ncon, value_exp, 
                           store, optimizer.minimize, num_retries, value_limits, workers))
    ys = np.array([fun(x) for x in xs])
    return xs, ys
    
def mo_retry(fun, weight_bounds, ncon, y_exp, store, optimize, num_retries, value_limits, 
          workers=mp.cpu_count()):
    sg = SeedSequence()
    rgs = [Generator(MT19937(s)) for s in sg.spawn(workers)]
    proc=[Process(target=_retry_loop,
            args=(pid, rgs, fun, weight_bounds, ncon, y_exp, 
                  store, optimize, num_retries, value_limits)) for pid in range(workers)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    store.sort()
    store.dump()
    return store.get_xs()

def _retry_loop(pid, rgs, fun, weight_bounds, ncon, y_exp, 
                store, optimize, num_retries, value_limits):
    
    if 'win' in sys.platform and not store.logger is None:
        store.logger = logger()       
    lower = store.lower
    wlb = np.array(weight_bounds.lb)
    wub = np.array(weight_bounds.ub)
    while store.get_runs_compare_incr(num_retries):      
        try:       
            rg = rgs[pid]
            w = rg.uniform(size=len(wub))          
            w /= _avg_exp(w, y_exp) # correct scaling
            w = wlb + w * (wub - wlb)
            wrapper = mo_wrapper(fun, w, ncon, y_exp)  
            x, y, evals = optimize(wrapper.eval, Bounds(store.lower, store.upper), None, 
                                     [rg.uniform(0.05, 0.1)]*len(lower), rg, store)
            objs = wrapper.mo_eval(x) # retrieve the objective values
            if value_limits is None or all([objs[i] < value_limits[i] for i in range(len(w))]):
                store.add_result(y, x, evals, math.inf)   
                if not store.plot_name is None:
                    name = store.plot_name + "_moretry_" + str(store.get_count_evals())
                    xs = np.array(store.get_xs())
                    ys = np.array([fun(x) for x in xs])
                    np.savez_compressed(name, xs=xs, ys=ys) 
                    plot(name, ncon, xs, ys)
        except Exception as ex:
            print(str(ex))
            
def pareto(xs, ys):
    """pareto front for argument vectors and corresponding function value vectors."""
    par = _pareto(ys)
    xp = xs[par]
    yp = ys[par]
    ya = np.argsort(yp.T[0])
    return xp[ya], yp[ya]
     
class mo_wrapper(object):
    """wrapper for multi objective functions applying the weighted sum approach."""
   
    def __init__(self, fun, weights, ncon, y_exp=2):
        self.fun = fun  
        self.weights = weights 
        self.ny = len(weights)
        self.nobj = self.ny - ncon
        self.ncon = ncon
        self.y_exp = y_exp

    def eval(self, x):
        y = self.fun(np.array(x))
        weighted = _avg_exp(self.weights*y, self.y_exp)
        if self.ncon > 0: # check constraint violations
            violations = np.array([i for i in range(self.nobj, self.ny) if y[i] > 0])
            if len(violations) > 0:
                weighted += sum(self.weights[violations])     
        return weighted
            
    def mo_eval(self, x):
        return self.fun(np.array(x))
        
def minimize_plot(name, optimizer, fun, bounds, weight_bounds, ncon = 0, 
                  value_limits = None, num_retries = 1024, 
             exp = 2.0, workers = mp.cpu_count(), 
             logger=logger(), statistic_num = 0, plot_name = None):
    time0 = time.perf_counter() # optimization start time
    name += '_' + optimizer.name
    logger.info('optimize ' + name) 
    xs, ys = minimize(fun, bounds, weight_bounds, ncon,
             value_exp = exp,
             value_limits = value_limits,
             num_retries = num_retries,              
             optimizer = optimizer,
             workers = workers,
             logger=logger, statistic_num = statistic_num, plot_name = plot_name)
    logger.info(name + ' time ' + str(dtime(time0))) 
    np.savez_compressed(name, xs=xs, ys=ys) 
    plot(name, ncon, xs, ys)
    
def plot(name, ncon, xs, ys, eps = 1E-2, all=True):   
    try:  
        if ncon > 0: # select feasible
            ycon = np.array([np.maximum(y[-ncon:], 0) for y in ys])  
            con = np.sum(ycon, axis=1)
            nobj = len(ys[0]) - ncon
            feasible = np.array([i for i in range(len(ys)) if con[i] < eps])
            if len(feasible) > 0:
                xs, ys = xs[feasible], np.array([ y[:nobj] for y in ys[feasible]])
            else:
                print("no feasible")
                return
        if all:
            retry.plot(ys, 'all_' + name + '.png', interp=False)
            xs, ys = pareto(xs, ys)
            for x, y in zip(xs, ys):
                print(str(list(y)) + ' ' + str([int(xi) for xi in x]))
        retry.plot(ys, 'front_' + name + '.png', interp=False)
    except Exception as ex:
        print(str(ex))

def adv_minimize_plot(name, optimizer, fun, bounds,
                   value_limit = math.inf, num_retries = 1024, logger=logger(), statistic_num = 0):
    time0 = time.perf_counter() # optimization start time
    name += '_smart_' + optimizer.name
    logger.info('smart optimize ' + name) 
    store = advretry.Store(lambda x:fun(x)[0], bounds, capacity=5000, logger=logger, 
                           num_retries=num_retries, statistic_num = statistic_num) 
    advretry.retry(store, optimizer.minimize, value_limit)
    xs = np.array(store.get_xs())
    ys = np.array([fun(x) for x in xs])
    retry.plot(ys, '_all_' + name + '.png', interp=False)
    np.savez_compressed(name , xs=xs, ys=ys)
    xs, front = pareto(xs, ys)
    logger.info(name+ ' time ' + str(dtime(time0))) 
    retry.plot(front, '_front_' + name + '.png')

def _avg_exp(y, y_exp):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weighted = sum([y[i]**y_exp for i in range(len(y))])**(1.0/y_exp)
    return weighted

def _pareto_values(ys):
    ys = ys[ys.sum(1).argsort()[::-1]]
    undominated = np.ones(ys.shape[0], dtype=bool)
    for i in range(ys.shape[0]):
        n = ys.shape[0]
        if i >= n:
            break
        undominated[i+1:n] = (ys[i+1:] >= ys[i]).any(1) 
        ys = ys[undominated[:n]]
    return ys

def _pareto(ys):
    pareto = np.arange(ys.shape[0])
    index = 0  # Next index to search for
    while index < len(ys):
        mask = np.any(ys < ys[index], axis=1)
        mask[index] = True
        pareto = pareto[mask]  # Remove dominated points
        ys = ys[mask]
        index = np.sum(mask[:index])+1
    return pareto
    
