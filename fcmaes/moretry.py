# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# parallel optimization retry of a multi-objective problem.

import numpy as np
import math, sys
import multiprocessing as mp
from multiprocessing import Process
from scipy.optimize import Bounds
from scipy import interpolate
from numpy.random import Generator, MT19937, SeedSequence
from fcmaes.optimizer import de_cma, logger
from fcmaes import retry

def minimize(fun,             
             bounds,
             weight_bounds,
             value_exp = 2.0,
             value_limits = None,
             num_retries = 1024,
             logger = None,
             workers = mp.cpu_count(),
             popsize = 31, 
             max_evaluations = 50000, 
             capacity = None,
             optimizer = None,
             statistic_num = 0
              ):   
    """Minimization of a multi objective function of one or more variables using parallel 
     optimization retry.
     
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
    weight_bounds : `Bounds`, optional
        Bounds on objective weights.
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
     
    Returns
    -------
    xs, ys: list of argument vectors and corresponding value vectors of the optimization results. """

    if optimizer is None:
        optimizer = de_cma(max_evaluations, popsize)  
    if capacity is None: 
        capacity = num_retries
    store = retry.Store(bounds, capacity = capacity, logger = logger, statistic_num = statistic_num)
    xs = np.array(mo_retry(fun, weight_bounds, value_exp, 
                           store, optimizer.minimize, num_retries, value_limits, workers))
    ys = np.array([fun(x) for x in xs])
    return xs, ys
    
def mo_retry(fun, weight_bounds, y_exp, store, optimize, num_retries, value_limits, 
          workers=mp.cpu_count()):
    sg = SeedSequence()
    rgs = [Generator(MT19937(s)) for s in sg.spawn(workers)]
    proc=[Process(target=_retry_loop,
            args=(pid, rgs, fun, weight_bounds, y_exp, 
                  store, optimize, num_retries, value_limits)) for pid in range(workers)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    store.sort()
    store.dump()
    return store.get_xs()

def _retry_loop(pid, rgs, fun, weight_bounds, y_exp, 
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
            wrapper = mo_wrapper(fun, w, y_exp)  
            x, y, evals = optimize(wrapper.eval, Bounds(store.lower, store.upper), None, 
                                     [rg.uniform(0.05, 0.1)]*len(lower), rg, store)
            objs = wrapper.mo_eval(x) # retrieve the objective values
            if value_limits is None or all([objs[i] < value_limits[i] for i in range(len(w))]):
                store.add_result(y, x, evals, math.inf)   
        except Exception as ex:
            continue

def pareto(xs, ys):
    """pareto front for argument vectors and corresponding function value vectors."""
    par = _pareto(ys)
    xp = xs[par]
    yp = ys[par]
    ya = np.argsort(yp.T[0])
    return xp[ya], yp[ya]
     
class mo_wrapper(object):
    """wrapper for multi objective functions applying the weighted sum approach."""
   
    def __init__(self, fun, weights, y_exp=2):
        self.fun = fun  
        self.nobj = len(weights)
        self.weights = weights 
        self.y_exp = y_exp
 
    def eval(self, x):
        y = self.fun(np.array(x))
        return _avg_exp(self.weights*y, self.y_exp)

    def mo_eval(self, x):
        return self.fun(np.array(x))

def plot(front, fname, interp=True):
    if len(front[0]) == 3:
        return plot3(front, fname)
    if len(front[0]) > 3:
        return plot3(front.T[:3].T, fname)        
    import matplotlib.pyplot as pl
    fig, ax = pl.subplots(1, 1)
    x = front[:, 0]; y = front[:, 1]
    if interp:
        xa = np.argsort(x)
        xs = x[xa]; ys = y[xa]
        x = []; y = []
        for i in range(len(xs)): # filter equal x values
            if i == 0 or xs[i] > xs[i-1] + 1E-5:
                x.append(xs[i]); y.append(ys[i])
        tck = interpolate.InterpolatedUnivariateSpline(x,y,k=1)
        x = np.linspace(min(x),max(x),1000)
        y = [tck(xi) for xi in x]
    ax.scatter(x, y, label=r"$\chi$", s=1)
    ax.grid()
    ax.set_xlabel(r'$f_1$')
    ax.set_ylabel(r'$f_2$')
    ax.legend()
    fig.savefig(fname, dpi=300)
    pl.close('all')

def plot3(front, fname):
    import matplotlib.pyplot as pl
    fig = pl.figure()
    ax = fig.add_subplot(projection='3d')
    x = front[:, 0]; y = front[:, 1]; z = front[:, 2]
    ax.scatter(x, y, z, label=r"$\chi$", s=1)
    ax.grid()
    ax.set_xlabel(r'$f_1$')
    ax.set_ylabel(r'$f_2$')
    ax.set_zlabel(r'$f_3$')
    ax.legend()
    pl.show()
    fig.savefig(fname, dpi=300)
    pl.close('all')

def _avg_exp(y, y_exp):
    return sum([y[i]**y_exp for i in range(len(y))])**(1.0/y_exp)

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
    
