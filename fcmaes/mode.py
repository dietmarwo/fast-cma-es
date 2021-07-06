# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

""" Numpy based implementation of multi objective
    Differential Evolution using the DE/pareto/1 strategy. 
    Derived and adapted for MO from its C++ counterpart 
    https://github.com/dietmarwo/fast-cma-es/blob/master/_fcmaescpp/deoptimizer.cpp
    
    Can switch to NSGA-II like population update via parameter 'nsga_update'.
    Then it works essentially like NSGA-II but instead of the tournament selection
    the whole population is sorted and the best individuals survive. To do this
    efficiently the crowd distance ordering is slightly inaccurate - each objective
    is ordered separately. 
    
    Compared to most Python NSGA-II implementations it supports parallel fitness function 
    evaluation. 
    
    Enables the comparison of DE and NSGA-II population update mechanism with everything else
    kept completely identical.
    
    
    Requires python 3.5 or higher. 
    
    Uses the following deviation from the standard DE algorithm:
    a) oscillating CR/F parameters. 
    
    You may keep parameters F and CR at their defaults since this implementation works well with the given settings for most problems,
    since the algorithm oscillates between different F and CR settings. 
    
    For expensive objective functions (e.g. machine learning parameter optimization) use the workers
    parameter to parallelize objective function evaluation. This causes delayed population update.
    It is usually preferrable if popsize > workers and workers = mp.cpu_count() to improve CPU utilization.  
"""

import numpy as np
import sys
import time
import ctypes as ct
from numpy.random import Generator, MT19937
from scipy.optimize import OptimizeResult
from fcmaes.evaluator import Evaluator
import multiprocessing as mp
from fcmaes.optimizer import dtime, logger
from fcmaes import retry, moretry

def minimize(mofun, 
             nobj,
             dim = None,
             bounds = None,
             ncon = 0, 
             popsize = 64, 
             max_evaluations = 100000, 
             workers = None,
             f = 0.5, 
             cr = 0.9, 
             nsga_update = False,
             pareto_update = False,
             rg = Generator(MT19937()),
             logger = None,
             plot_name = None):  
      
    """Minimization of a multi objjective function of one or more variables using
    Differential Evolution.
     
    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
            ``fun(x, *args) -> float``
        where ``x`` is an 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    nobj : int
        number of objectives
    dim : int
        dimension of the argument of the objective function
        either dim or bounds need to be defined
    bounds : sequence or `Bounds`, optional
        Bounds on variables. There are two ways to specify the bounds:
            1. Instance of the `scipy.Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.
    ncon : int, optional
        number of constraints, default is 0. 
        The objective function needs to return vectors of size nobj + ncon
    popsize : int, optional
        Population size.
    max_evaluations : int, optional
        Forced termination after ``max_evaluations`` function evaluations.
    workers : int or None, optional
        If not workers is None, function evaluation is performed in parallel for the whole population. 
        Useful for costly objective functions     
    f = float, optional
        The mutation constant. In the literature this is also known as differential weight, 
        being denoted by F. Should be in the range [0, 2].
    cr = float, optional
        The recombination constant. Should be in the range [0, 1]. 
        In the literature this is also known as the crossover probability.     
    nsga_update = boolean, optional
        Use of NSGA-II or DE population update. Default is False    
    pareto_update = boolean, optional
        Use the pareto front for population update if nsga_update = False. Default is False     
    rg = numpy.random.Generator, optional
        Random generator for creating random guesses.
    logger : logger, optional
        logger for log output for tell_one, If None, logging
        is switched off. Default is a logger which logs both to stdout and
        appends to a file ``optimizer.log``.
    trace_plots : trace_plots, optional
        flag indicating if the pareto front is plotted during the optimization to monitor progress
            
    Returns
    -------
    res : scipy.OptimizeResult
        The optimization result is represented as an ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, 
        ``fun`` the best function value, 
        ``nfev`` the number of function evaluations,
        ``nit`` the number of iterations,
        ``success`` a Boolean flag indicating if the optimizer exited successfully. """

    
    de = DE(dim, nobj, bounds, ncon, popsize, workers if not workers is None else 0, 
            f, cr, nsga_update, pareto_update, rg, logger, plot_name)
    try:
        if workers and workers > 1:
            x, y, evals, iterations, stop = de.do_optimize_delayed_update(mofun, max_evaluations, workers)
        else:      
            x, y, evals, iterations, stop = de.do_optimize(mofun, max_evaluations)
        return OptimizeResult(x=x, fun=y, nfev=evals, nit=iterations, status=stop, 
                              success=True)
    except Exception as ex:
        return OptimizeResult(x=None, fun=sys.float_info.max, nfev=0, nit=0, status=-1, success=False)  

class DE(object):
    
    def __init__(self, dim, nobj, bounds, ncon=0, popsize = 64, workers = 0,
                 F = 0.5, Cr = 0.9, nsga_update = False, pareto_update = False, 
                 rg = Generator(MT19937()), logger = None, plot_name = None):
        self.nobj = nobj
        self.ncon = ncon
        self.dim, self.lower, self.upper = _check_bounds(bounds, dim)
        if popsize is None:
            popsize = 31
        self.popsize = popsize
        self.workers = workers 
        self.rg = rg
        self.F0 = F
        self.Cr0 = Cr
        self.nsga_update = nsga_update
        self.pareto_update = pareto_update
        self.stop = 0
        self.iterations = 0
        self.evals = 0
        self.mutex = mp.Lock()
        self.p = 0
        self.best_p = None
        self.trace_plots = not plot_name is None
        self._init()
        if not logger is None:
            self.logger = logger
            self.n_evals = mp.RawValue(ct.c_long, 0)
            self.time_0 = time.perf_counter()
            self.name = plot_name if not plot_name is None else ''
            self.name += '_mode_' + str(popsize) + '_' + \
                ('nsga_update' if nsga_update else ('de_best_update' if pareto_update else 'de_all_update'))
        
    def ask(self):
        """ask for one new argument vector.
        
        Returns
        -------
        p : int population index 
        x : dim sized argument ."""
        
        p = self.p
        x = self._next_x(p)
        self.p = (self.p + 1) % self.popsize
        return p, x

    def tell(self, p, y, x):      
        """tell function value for a argument list retrieved by ask_one().
    
        Parameters
        ----------
        p : int population index 
        y : function value
        x : dim sized argument list
 
        Returns
        -------
        stop : int termination criteria, if != 0 loop should stop."""
        
        with self.mutex:  
            for dp in range(len(self.done)):
                if not self.done[dp]:
                    break
            self.nx[dp] = x
            self.ny[dp] = y      
            self.done[dp] = True
            if sum(self.done) >= self.popsize:
                done_p = np.arange(len(self.ny))
                done_p = done_p[self.done]
                p = self.popsize
                for dp in done_p:
                    self.x[p] = self.nx[dp]
                    self.y[p] = self.ny[dp]
                    self.done[dp] = False
                    if p >= len(self.y):
                        break
                    p += 1
                self.pop_update()
        
        if hasattr(self, 'logger'):
            self.n_evals.value += 1
            if self.n_evals.value % 100000 == 99999:           
                t = time.perf_counter() - self.time_0
                c = self.n_evals.value
                message = '"c/t={0:.2f} c={1:d} t={2:.2f} y={3!s} x={4!s}'.format(
                    c/t, c, t, str(list(y)), str(list(x)))
                self.logger.info(message)   
                if self.trace_plots:             
                    name = self.name + ' '  + str(self.n_evals.value)
                    np.savez_compressed(name, xs=self.x, ys=self.y) 
                    moretry.plot(name, self.ncon, self.x, self.y)

        return self.stop 

    def _init(self):
        self.x = np.empty((2*self.popsize, self.dim))
        self.y = np.empty((2*self.popsize, self.nobj + self.ncon))
        for i in range(self.popsize):
            self.x[i] = self._sample()
            self.y[i] = np.array([1E99]*(self.nobj + self.ncon))
        
        next_size = 2*(max(self.workers, self.popsize))
        self.done = np.zeros(next_size, dtype=bool)
        self.nx = np.empty((next_size, self.dim))
        self.ny = np.empty((next_size, self.nobj + self.ncon))
        self.vx = self.x.copy()
        self.vp = 0
                                    
    def do_optimize(self, fun, max_evals):
        self.fun = fun
        self.max_evals = max_evals    
        self.iterations = 0
        self.evals = 0
        while self.evals < self.max_evals:
            for p in range(self.popsize):
                x = self._next_x(p)
                self.y[self.popsize + p] = self.fun(x)
                self.x[self.popsize + p] = x
                self.evals += 1
            self.pop_update()
            if self.evals >= self.max_evals:
                break
        x, y = filter(self.x, self.y)
        return x, y, self.evals, self.iterations, self.stop

    def do_optimize_delayed_update(self, fun, max_evals, workers=mp.cpu_count()):
        self.fun = fun
        self.max_evals = max_evals    
        evaluator = Evaluator(self.fun)
        evaluator.start(workers)
        evals_x = {}
        self.iterations = 0
        self.evals = 0
        self.p = 0
        for _ in range(workers): # fill queue with initial population
            p, x = self.ask()
            evaluator.pipe[0].send((self.evals, x))
            evals_x[self.evals] = p, x # store x
            self.evals += 1
            
        while True: # read from pipe, tell de and create new x
            evals, y = evaluator.pipe[0].recv()            
            p, x = evals_x[evals] # retrieve evaluated x
            del evals_x[evals]
            self.tell(p, y, x) # tell evaluated x
            if self.stop != 0 or self.evals >= self.max_evals:
                break # shutdown worker if stop criteria met
            
            p, x = self.ask() # create new x          
            evaluator.pipe[0].send((self.evals, x))       
            evals_x[self.evals] = p, x  # store x
            self.evals += 1
            
        evaluator.stop()
        x, y = filter(self.x, self.y)
        return x, y, self.evals, self.iterations, self.stop

    def pop_update(self):
        domination = pareto(self.y, self.nobj, self.ncon)
        x = []
        y = []
        maxdom = int(max(domination))
        for dom in range(maxdom, -1, -1):
            domlevel = [p for p in range(len(domination)) if domination[p] == dom]
            if dom == maxdom: # store pareto front in self.best_p
                self.best_p =  domlevel
            if len(x) + len(domlevel) <= self.popsize:
                # whole level fits
                x = [*x, *self.x[domlevel]]
                y = [*y, *self.y[domlevel]]
            else: # sort for crowding
                nx = self.x[domlevel]
                ny = self.y[domlevel]    
                si = [0]
                if len(ny) > 1:                
                    cd = crowd_dist(ny)
                    si = np.flip(np.argsort(cd))
                for p in si:
                    if len(x) >= self.popsize:
                        break
                    x.append(nx[p])
                    y.append(ny[p])
                break # we have filled popsize members                                
        self.x[:self.popsize] = x[:self.popsize]
        self.y[:self.popsize] = y[:self.popsize]
        if self.nsga_update:
            self.vx = variation(self.x[:self.popsize], self.lower, self.upper, self.rg) 
       
    def _next_x(self, p):
        if self.nsga_update: # use NSGA-II update strategy.
            x = self.vx[self.vp]
            self.vp = (self.vp + 1) % self.popsize # only use the elite
            return x
        # use standard DE/pareto/1 strategy.
        if p == 0: # switch FR / CR every generation
            self.iterations += 1
            self.Cr = 0.5*self.Cr0 if self.iterations % 2 == 0 else self.Cr0
            self.F = 0.5*self.F0 if self.iterations % 2 == 0 else self.F0
        while True:
            if self.pareto_update and (not self.best_p is None) and len(self.best_p) > 3: 
                # sample from pareto front
                rb = self.rg.integers(0, len(self.best_p))
                rb = self.best_p[rb]
                r1, r2 = self.rg.integers(0, self.popsize, 2)
            else:
                # sample from whole population
                r1, r2, rb = self.rg.integers(0, self.popsize, 3)
                if r1 != p and r1 != rb and r1 != r2 and r2 != rb \
                        and r2 != p and rb != p:
                    break
        xp = self.x[p]
        xb = self.x[rb]
        x1 = self.x[r1]
        x2 = self.x[r2]
        x = self._feasible(xb + self.F * (x1 - x2))
        r = self.rg.integers(0, self.dim)
        tr = np.array(
            [i != r and self.rg.random() > self.Cr for i in range(self.dim)])    
        x[tr] = xp[tr]       
        return x
    
    def _sample(self):
        if self.upper is None:
            return self.rg.normal()
        else:
            return self.rg.uniform(self.lower, self.upper)
    
    def _feasible(self, x):
        if self.upper is None:
            return x
        else:
            return np.maximum(np.minimum(x, self.upper), self.lower)
                    
def _check_bounds(bounds, dim):
    if bounds is None and dim is None:
        raise ValueError('either dim or bounds need to be defined')
    if bounds is None:
        return dim, None, None
    else:
        return len(bounds.ub), np.asarray(bounds.lb), np.asarray(bounds.ub)
 
def filter(x, y):
    sorted = np.argsort([yi[0] for yi in y])
    x = x[sorted]
    y = y[sorted]
    y = np.array([yi for yi in y if yi[0] < 1E99])
    x = np.array(x[:len(y)])
    return x,y

def maxcon(cons):
    cons = np.minimum(cons, 1E99)
    maxc = max(cons)
    if maxc == 0:
        return 0
    else: # reduce number and value of constraint violations
        n = sum([c > 0 for c in cons])
        return n*maxc + sum(cons)

def pareto(ys, nobj, ncon):
    if ncon == 0:
        return pareto_levels(ys)
    else:
        yobj = np.array([y[:nobj] for y in ys])
        ycon = np.array([np.maximum(y[-ncon:], 0) for y in ys])    
        csum = [maxcon(cons) for cons in ycon]
        ci = np.argsort(csum)
        popn = len(ys)
        domination = np.zeros(popn)
        # first pareto front of feasible solutions
        cy = np.array([i for i in ci if csum[i] == 0])
        if len(cy) > 0:
            ypar = pareto_levels(yobj[cy])
            domination[cy] += ypar
        # then constraint violations   
        ci = np.array([i for i in ci if csum[i] > 0 and csum[i] < 1E99])  
        if len(ci) > 0:    
            maxcdom = len(ci)
            cdom = np.arange(maxcdom, 0, -1)
            domination[ci] += cdom
            if len(cy) > 0: # priorize feasible solutions
                domination[cy] += maxcdom + 1
    return domination   
 
def pareto_levels(ys):
    popn = len(ys)
    pareto = np.arange(popn)
    index = 0  # Next index to search for
    domination = np.zeros(popn)
    while index < len(ys):
        mask = np.any(ys < ys[index], axis=1)
        mask[index] = True
        pareto = pareto[mask]  # Remove dominated points
        domination[pareto] += 1
        ys = ys[mask]
        index = np.sum(mask[:index])+1
    return domination

def crowd_dist(y): # crowd distance for 1st objective
    n = len(y)
    y0 = np.array([yi[0] for yi in y])
    si = np.argsort(y0) # sort 1st objective
    y0_s = y0[si] # sorted
    d = y0_s[1:n] - y0_s[0:n-1] # neighbor distance
    if max(d) == 0:
        return np.zeros(n)
    dsum = np.zeros(n)
    dsum += np.array(list(d) + [0]) # distance to left
    dsum += np.array([0] + list(d)) # distance to right
    dsum[0] = 1E99 # keep borders
    dsum[-1] = 1E99
    ds = np.empty(n)
    ds[si] = dsum # inverse order
    return ds

# derived from https://github.com/ChengHust/NSGA-II/blob/master/GLOBAL.py
def variation(pop, lower, upper, rg, pro_c = 1, dis_c = 20, pro_m = 1, dis_m = 20):
    """Generate offspring individuals"""   
    pop = pop[:(len(pop) // 2) * 2][:]
    (n, d) = np.shape(pop)
    parent_1 = pop[:n // 2, :]
    parent_2 = pop[n // 2:, :]
    beta = np.zeros((n // 2, d))
    mu = rg.random((n // 2, d))
    beta[mu <= 0.5] = np.power(2 * mu[mu <= 0.5], 1 / (dis_c + 1))
    beta[mu > 0.5] = np.power(2 * mu[mu > 0.5], -1 / (dis_c + 1))
    beta = beta * ((-1)** rg.integers(2, size=(n // 2, d)))
    beta[rg.random((n // 2, d)) < 0.5] = 1
    beta[np.tile(rg.random((n // 2, 1)) > pro_c, (1, d))] = 1
    offspring = np.vstack(((parent_1 + parent_2) / 2 + beta * (parent_1 - parent_2) / 2,
                               (parent_1 + parent_2) / 2 - beta * (parent_1 - parent_2) / 2))
    site = rg.random((n, d)) < pro_m / d
    mu = rg.random((n, d))
    temp = site & (mu <= 0.5)
    lower, upper = np.tile(lower, (n, 1)), np.tile(upper, (n, 1))
    norm = (offspring[temp] - lower[temp]) / (upper[temp] - lower[temp])
    offspring[temp] += (upper[temp] - lower[temp]) * \
                           (np.power(2. * mu[temp] + (1. - 2. * mu[temp]) * np.power(1. - norm, dis_m + 1.),
                                     1. / (dis_m + 1)) - 1.)
    temp = site & (mu > 0.5)
    norm = (upper[temp] - offspring[temp]) / (upper[temp] - lower[temp])
    offspring[temp] += (upper[temp] - lower[temp]) * \
                           (1. - np.power(
                               2. * (1. - mu[temp]) + 2. * (mu[temp] - 0.5) * np.power(1. - norm, dis_m + 1.),
                               1. / (dis_m + 1.)))
    offspring = np.maximum(np.minimum(offspring, upper), lower)
    return offspring

def minimize_plot(name, fun, bounds, nobj, ncon = 0, popsize = 64, max_eval = 100000, nsga_update=False, 
                  pareto_update=False, workers = mp.cpu_count(), logger=logger(), plot_name = None):
    name += '_mode_' + str(popsize) + '_' + \
                ('nsga_update' if nsga_update else ('de_best_update' if pareto_update else 'de_all_update'))
    logger.info('optimize ' + name) 
    ret = minimize(fun, nobj, bounds = bounds, ncon = ncon, popsize = popsize, max_evaluations = max_eval,
                   nsga_update = nsga_update, pareto_update = pareto_update, workers=workers, 
                   logger=logger, plot_name = plot_name)
    ys = np.array(ret.fun)
    xs = np.array(ret.x)
    np.savez_compressed(name, xs=xs, ys=ys)
    moretry.plot(name, ncon, xs, ys)
