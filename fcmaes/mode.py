# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.
from __future__ import annotations

""" Numpy based implementation of multi objective
    Differential Evolution using either the DE/rand/1 strategy
    or a NSGA-II like population update (parameter 'nsga_update=True)'.
    Then it works similar to NSGA-II.
    
    Supports parallel fitness function evaluation. 
    
    Features enhanced multiple constraint ranking (https://www.jstage.jst.go.jp/article/tjpnsec/11/2/11_18/_article/-char/en/)
    improving its performance in handling constraints for engineering design optimization.

    Enables the comparison of DE and NSGA-II population update mechanism with everything else
    kept completely identical.
    
    Requires python 3.5 or higher. 
    
    Uses the following deviation from the standard DE algorithm:
    a) oscillating CR/F parameters. 
    
    You may keep parameters F and CR at their defaults since this implementation works well with the given settings for most problems,
    since the algorithm oscillates between different F and CR settings. 
    
    For expensive objective functions (e.g. machine learning parameter optimization) use the workers
    parameter to parallelize objective function evaluation. The workers parameter is limited by the 
    population size.
    
    The ints parameter is a boolean array indicating which parameters are discrete integer values. This 
    parameter was introduced after observing non optimal DE-results for the ESP2 benchmark problem: 
    https://github.com/AlgTUDelft/ExpensiveOptimBenchmark/blob/master/expensiveoptimbenchmark/problems/DockerCFDBenchmark.py
    If defined it causes a "special treatment" for discrete variables: They are rounded to the next integer value and
    there is an additional mutation to avoid getting stuck at local minima. This behavior is specified by the internal
    function _modifier which can be overwritten by providing the optional modifier argument. If modifier is defined,
    ints is ignored. 
    
    See https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/MODE.adoc for a detailed description.
"""

import numpy as np
import os, sys, time
import ctypes as ct
from numpy.random import Generator, MT19937
from scipy.optimize import Bounds

from fcmaes.evaluator import Evaluator, parallel_mo
from fcmaes import moretry
import multiprocessing as mp
from fcmaes.optimizer import dtime
from loguru import logger
from typing import Optional, Callable, Tuple
from numpy.typing import ArrayLike

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(mofun: Callable[[ArrayLike], ArrayLike], 
             nobj: int,
             ncon: int,
             bounds: Bounds,
             guess: Optional[np.ndarray] = None,
             popsize: Optional[int] = 64,
             max_evaluations: Optional[int] = 100000,
             workers: Optional[int] = 1,
             f: Optional[float] = 0.5,
             cr: Optional[float] = 0.9,
             pro_c: Optional[float] = 0.5,
             dis_c: Optional[float] = 15.0,
             pro_m: Optional[float] = 0.9,
             dis_m: Optional[float] = 20.0,
             nsga_update: Optional[bool] = True,
             pareto_update: Optional[int] = 0,
             ints: Optional[ArrayLike] = None,
             modifier: Callable = None,
             min_mutate: Optional[float] = 0.1,
             max_mutate: Optional[float] = 0.5,
             rg: Optional[Generator] = Generator(MT19937()),
             store: Optional[store] = None) -> Tuple[np.ndarray, np.ndarray]:  
      
    """Minimization of a multi objjective function of one or more variables using
    Differential Evolution.
     
    Parameters
    ----------
    mofun : callable
        The objective function to be minimized.
            ``mofun(x) -> ndarray(float)``
        where ``x`` is an 1-D array with shape (n,)
    nobj : int
        number of objectives
    ncon : int
        number of constraints, default is 0. 
        The objective function needs to return vectors of size nobj + ncon
    bounds : sequence or `Bounds`
        Bounds on variables. There are two ways to specify the bounds:
            1. Instance of the `scipy.Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.
    guess : ndarray, shape (popsize,dim) or Tuple
        Initial guess. 
    popsize : int, optional
        Population size.
    max_evaluations : int, optional
        Forced termination after ``max_evaluations`` function evaluations. 
    workers : int or None, optional
        workers > 1, function evaluation is performed in parallel for the whole population. 
        Useful for costly objective functions     
    f = float, optional
        The mutation constant. In the literature this is also known as differential weight, 
        being denoted by F. Should be in the range [0, 2].
    cr = float, optional
        The recombination constant. Should be in the range [0, 1]. 
        In the literature this is also known as the crossover probability.   
    pro_c, dis_c, pro_m, dis_m = float, optional
        NSGA population update parameters, usually leave at default.   
    nsga_update = boolean, optional
        Use of NSGA-II/SBX or DE population update. Default is True    
    pareto_update = float, optional
        Only applied if nsga_update = False. Favor better solutions for sample generation. Default 0 - 
        use all population members with the same probability.   
    ints = list or array, optional
        indicating which parameters are discrete integer values. If defined these parameters will be
        rounded to the next integer and some additional mutation of discrete parameters are performed.
    min_mutate = float, optional
        Determines the minimal mutation rate for discrete integer parameters.
    max_mutate = float, optional
        Determines the maximal mutation rate for discrete integer parameters. 
    modifier = callable, optional
        used to overwrite the default behaviour induced by ints. If defined, the ints parameter is
        ignored. Modifies all generated x vectors.
    rg = numpy.random.Generator, optional
        Random generator for creating random guesses.
    store : result store, optional
        if defined the optimization results are added to the result store. For multi threaded execution.
        use workers=1 if you call minimize from multiple threads
            
    Returns
    -------
    x, y: list of argument vectors and corresponding value vectors of the optimization results. """

    try:   
        mode = MODE(nobj, ncon, bounds, popsize,
            f, cr, pro_c, dis_c, pro_m, dis_m, nsga_update, pareto_update, rg, ints, min_mutate, max_mutate, modifier)
        mode.set_guess(guess, mofun, rg)
        if workers <= 1:
            x, y, = mode.minimize_ser(mofun, max_evaluations)
        else:
            x, y = mode.minimize_par(mofun, max_evaluations, workers)
        if not store is None:
            store.add_results(x, y)
        return x, y
    except Exception as ex:
        print(str(ex))  
        return None, None  

    
class store():
    
    """Result store. Used for multi threaded execution of minimize to collect optimization results.
     
    Parameters
    ----------
    dim : int
        dimension - number of variables
    nobj : int
        number of objectives
    capacity : int, optional
        capacity of the store collecting all solutions. If full, its content is replaced by its
        pareto front. 
    """
    
    def __init__(self, dim, nobj, capacity = mp.cpu_count()*512):    
        self.dim = dim
        self.nobj = nobj
        self.capacity = capacity
        self.add_mutex = mp.Lock()    
        self.xs = mp.RawArray(ct.c_double, self.capacity * self.dim)
        self.ys = mp.RawArray(ct.c_double, self.capacity * self.nobj)  
        self.num_stored = mp.RawValue(ct.c_int, 0) 
        self.num_added = mp.RawValue(ct.c_int, 0) 

    def add_result(self, x, y):
        with self.add_mutex:
            self.num_added.value += 1
            i = self.num_stored.value
            if i < self.capacity:                
                self.set_x(i, x) 
                self.set_y(i, y)
                i += 1
            self.num_stored.value = i

    def add_results(self, xs, ys):
        with self.add_mutex:
            self.num_added.value += 1
            i = self.num_stored.value
            for j in range(len(xs)):
                if i < self.capacity:                
                    self.set_x(i, xs[j]) 
                    self.set_y(i, ys[j])
                    i += 1
                else:
                    self.get_front(update=True)
                    i = self.num_stored.value
                    if i > 0.9*self.capacity: # give up
                        return
            self.num_stored.value = i
                      
    def get_front(self, update=False):
        stored = self.num_stored.value
        xs = np.array([self.get_x(i) for i in range(stored)])
        ys = np.array([self.get_y(i) for i in range(stored)])
        xf, yf = moretry.pareto(xs, ys)
        if update:
            for i in range(len(yf)):                   
                self.set_x(i, xf[i]) 
                self.set_y(i, yf[i])
            self.num_stored.value = len(yf)
        return xf, yf
       
    def get_xs(self):
        return np.array([self.get_x(i) for i in range(self.num_stored.value)])

    def get_ys(self):
        return np.array([self.get_y(i) for i in range(self.num_stored.value)])
    
    def get_x(self, i):
        return self.xs[i*self.dim:(i+1)*self.dim]
 
    def set_x(self, i, x):
        self.xs[i*self.dim:(i+1)*self.dim] = x[:]          
     
    def get_y(self, i):
        return self.ys[i*self.nobj:(i+1)*self.nobj]
 
    def set_y(self, i, y):
        self.ys[i*self.nobj:(i+1)*self.nobj] = y[:]          


class MODE(object):
    
    def __init__(self, 
                nobj: int,
                ncon: int, 
                bounds: Bounds,
                popsize: Optional[int] = 64, 
                F: Optional[float] = 0.5, 
                Cr: Optional[float] = 0.9, 
                pro_c: Optional[float] = 0.5,
                dis_c: Optional[float] = 15.0,
                pro_m: Optional[float] = 0.9,
                dis_m: Optional[float] = 20.0,
                nsga_update: Optional[bool] = True,
                pareto_update: Optional[int] = 0,
                rg: Optional[Generator] = Generator(MT19937()),
                ints: Optional[ArrayLike] = None,
                min_mutate: Optional[float] = 0.1,
                max_mutate: Optional[float] = 0.5,   
                modifier: Callable = None):
        self.nobj = nobj
        self.ncon = ncon
        self.dim, self.lower, self.upper = _check_bounds(bounds, None)
        if popsize is None:
            popsize = 64
        if popsize % 2 == 1 and nsga_update: # nsga update requires even popsize
            popsize += 1
        self.popsize = popsize
        self.rg = rg
        self.F0 = F
        self.Cr0 = Cr
        self.pro_c = pro_c
        self.dis_c = dis_c
        self.pro_m = pro_m
        self.dis_m = dis_m    
        self.nsga_update = nsga_update
        self.pareto_update = pareto_update
        self.stop = 0
        self.iterations = 0
        self.evals = 0
        self.mutex = mp.Lock()
        self.p = 0
        # nsga update doesn't support mixed integer
        self.ints = None if (ints is None or nsga_update) else np.array(ints)
        self.min_mutate = min_mutate
        self.max_mutate = max_mutate        
        # use default variable modifier for int variables if modifier is None
        if modifier is None and not ints is None:
            self.lower = self.lower.astype(float)
            self.upper = self.upper.astype(float)
            self.modifier = self._modifier
        else:
            self.modifier = modifier
        self._init()
               
    def set_guess(self, guess, mofun, rg = None):
        if not guess is None:
            if isinstance(guess, np.ndarray):
                ys = np.array([mofun(x) for x in guess])
            else:
                guess, ys = guess
            if rg is None:
                rg = Generator(MT19937())
            choice = rg.choice(len(ys), self.popsize, 
                                    replace = (len(ys) < self.popsize))
            self.tell(ys[choice], guess[choice])
                   
    def ask(self) -> np.ndarray:
        for p in range(self.popsize):
            self.x[p + self.popsize] = self._next_x(p)
        return self.x[self.popsize:]
                
    def tell(self, ys: np.ndarray, xs: Optional[np.ndarray] = None):
        if not xs is None:
            for p in range(self.popsize):
                self.x[p + self.popsize] = xs[p]
        for p in range(self.popsize):
            self.y[p + self.popsize] = ys[p]
        self.pop_update()
                         
    def _init(self):
        self.x = np.empty((2*self.popsize, self.dim))
        self.y = np.empty((2*self.popsize, self.nobj + self.ncon))
        for i in range(self.popsize):
            self.x[i] = self._sample()
            self.y[i] = np.array([1E99]*(self.nobj + self.ncon))
        self.vx = self.x.copy()
        self.vp = 0
        self.ycon = None
        self.eps = 0

    def minimize_ser(self, 
                     fun: Callable[[ArrayLike], ArrayLike], 
                     max_evaluations: Optional[int] = 100000) -> Tuple[np.ndarray, np.ndarray]:
        evals = 0
        while evals < max_evaluations:
            xs = self.ask()
            ys = np.array([fun(x) for x in xs])
            self.tell(ys)
            evals += self.popsize
        return xs, ys

        
    def minimize_par(self, 
                     fun: Callable[[ArrayLike], ArrayLike], 
                     max_evaluations: Optional[int] = 100000, 
                     workers: Optional[int] = mp.cpu_count()) -> Tuple[np.ndarray, np.ndarray]:
        fit = parallel_mo(fun, self.nobj + self.ncon, workers)
        evals = 0
        while evals < max_evaluations:
            xs = self.ask()
            ys = fit(xs)
            self.tell(ys)
            evals += self.popsize
        fit.stop()
        return xs, ys
                                    
    def pop_update(self):
        y0 = self.y
        x0 = self.x
        if self.nobj == 1:
            yi = np.flip(np.argsort(self.y[:,0]))
            y0 = self.y[yi]
            x0 = self.x[yi]    
        domination, self.ycon, self.eps = pareto_domination(y0, self.nobj, self.ncon, self.ycon, self.eps)
        x = []
        y = []
        maxdom = int(max(domination))
        for dom in range(maxdom, -1, -1):
            domlevel = [p for p in range(len(domination)) if domination[p] == dom]
            if len(domlevel) == 0:
                continue
            if len(x) + len(domlevel) <= self.popsize:
                # whole level fits
                x = [*x, *x0[domlevel]]
                y = [*y, *y0[domlevel]]
            else: # sort for crowding
                nx = x0[domlevel]
                ny = y0[domlevel]    
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
            self.vx = variation(self.x[:self.popsize], self.lower, self.upper, self.rg, 
                pro_c = self.pro_c, dis_c = self.dis_c, pro_m = self.pro_m, dis_m = self.dis_m) 
       
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
            if self.pareto_update > 0: # sample elite solutions
                r1, r2 = self.rg.integers(0, self.popsize, 2)
                rb = int(self.popsize * (self.rg.random() ** (1.0 + self.pareto_update)))
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
        if not self.modifier is None:
            x = self.modifier(x)   
        return x.clip(self.lower, self.upper)
    
    def _sample(self):
        if self.upper is None:
            return self.rg.normal()
        else:
            return self.rg.uniform(self.lower, self.upper)
    
    def _feasible(self, x):
        if self.upper is None:
            return x
        else:
            return np.clip(x, self.lower, self.upper)
        
    # default modifier for integer variables
    def _modifier(self, x):
        x_ints = x[self.ints]
        n_ints = len(self.ints)
        lb = self.lower[self.ints]
        ub = self.upper[self.ints]
        to_mutate = self.rg.uniform(self.min_mutate, self.max_mutate)
        # mututate some integer variables
        x_ints = np.array([x if self.rg.random() > to_mutate/n_ints else 
                           int(self.rg.uniform(lb[i], ub[i]))
                           for i, x in enumerate(x_ints)])
        return x   
    
    def _is_dominated(self, y, p):
        return np.all(np.fromiter((y[i] >= self.y[p, i] for i in range(len(y))), dtype=bool))

                    
def _check_bounds(bounds, dim):
    if bounds is None and dim is None:
        raise ValueError('either dim or bounds need to be defined')
    if bounds is None:
        return dim, None, None
    else:
        return len(bounds.ub), np.asarray(bounds.lb), np.asarray(bounds.ub)

def _filter(x, y):
    ym = np.amax(y,axis=1)
    sorted = np.argsort(ym)
    x = x[sorted]
    y = y[sorted]
    y = np.array([yi for yi in y if yi[0] < 1E99])
    x = np.array(x[:len(y)])
    return x,y

def objranks(objs):
    ci = objs.argsort(axis=0)
    rank = np.empty_like(ci)
    ar = np.arange(objs.shape[0])
    for i in range(objs.shape[1]): 
        rank[ci[:,i], i] = ar 
    rank = np.sum(rank, axis=1)
    return rank

def ranks(cons, feasible, eps):
    ci = cons.argsort(axis=0)
    rank = np.empty_like(ci)
    ar = np.arange(cons.shape[0])
    for i in range(cons.shape[1]): 
        rank[ci[:,i], i] = ar
    rank[feasible] = 0
    alpha = np.sum(np.greater(cons, eps), axis=1) / cons.shape[1] # violations
    alpha = np.tile(alpha, (cons.shape[1],1)).T
    rank = rank*alpha
    rank = np.sum(rank, axis=1)
    return rank

def get_valid(xs, ys, nobj):
    valid = (ys.T[nobj:].T <= 0).all(axis=1)
    return xs[valid], ys[valid]

def pareto_sort(x0, y0, nobj, ncon):
    domination, _, _ = pareto_domination(y0, nobj, ncon)
    x = []
    y = []
    maxdom = int(max(domination))
    for dom in range(maxdom, -1, -1):
        domlevel = [p for p in range(len(domination)) if domination[p] == dom]
        if len(domlevel) == 0:
            continue
        nx = x0[domlevel]
        ny = y0[domlevel]    
        si = [0]
        if len(ny) > 1:                
            cd = crowd_dist(ny)
            si = np.flip(np.argsort(cd))
        for p in si:
            x.append(nx[p])
            y.append(ny[p])                             
    return np.array(x), np.array(y)

def pareto_domination(ys, nobj, ncon, last_ycon = None, last_eps = 0):
    if ncon == 0:
        return pareto_levels(ys), None, 0
    else:
        eps = 0 # adjust tolerance to small constraint violations
        if not last_ycon is None and np.amax(last_ycon) < 1E90:
            eps = 0.5*(last_eps + 0.5*np.mean(last_ycon, axis=0))
            if np.amax(eps) < 1E-8: # ignore small eps
                eps = 0
        
        yobj = np.array([y[:nobj] for y in ys])
        ycon = np.array([np.maximum(y[-ncon:], 0) for y in ys])  
        popn = len(ys)              
        feasible = np.less_equal(ycon, eps).all(axis=1)
        
        csum = ranks(ycon, feasible, eps)
        if sum(feasible) > 0:
            csum += objranks(yobj)
        
        ci = np.argsort(csum)
        domination = np.zeros(popn)
        # first pareto front of feasible solutions
        cy = np.fromiter((i for i in ci if feasible[i]), dtype=int)
        if len(cy) > 0:
            ypar = pareto_levels(yobj[cy])
            domination[cy] = ypar        

        # then constraint violations   
        ci = np.fromiter((i for i in ci if not feasible[i]), dtype=int) 
        if len(ci) > 0:    
            cdom = np.arange(len(ci), 0, -1)
            domination[ci] += cdom
            if len(cy) > 0: # priorize feasible solutions
                domination[cy] += len(ci) + 1
                
        return domination, ycon, eps
 
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
    y0 = np.fromiter((yi[0] for yi in y), dtype=float)
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
    dis_c *= 0.5 + 0.5*rg.random() # vary spread factors randomly 
    dis_m *= 0.5 + 0.5*rg.random() 
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
    if pro_c < 1.0:
        beta[np.tile(rg.random((n // 2, 1)) > pro_c, (1, d))] = 1
    parent_mean = (parent_1 + parent_2) * 0.5
    parent_diff = (parent_1 - parent_2) * 0.5
    offspring = np.vstack((parent_mean + beta * parent_diff, parent_mean - beta * parent_diff))
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
    offspring = np.clip(offspring, lower, upper)
    return offspring

def feasible(xs, ys, ncon, eps = 1E-2):
    if ncon > 0: # select feasible
        ycon = np.array([np.maximum(y[-ncon:], 0) for y in ys])  
        con = np.sum(ycon, axis=1)
        nobj = len(ys[0]) - ncon
        feasible = np.fromiter((i for i in range(len(ys)) if con[i] < eps), dtype=int)
        if len(feasible) > 0:
            xs, ys = xs[feasible], np.array([y[:nobj] for y in ys[feasible]])
        else:
            print("no feasible")
    return xs, ys

def is_feasible(y, nobj, eps = 1E-2):
    ncon = len(y) - nobj
    if ncon == 0:
        return True
    else:
        c = np.sum(np.maximum(y[-ncon:], 0))
        return c < eps

class wrapper(object):
    """thread safe wrapper for objective function monitoring evaluation count and optimization result."""
   
    def __init__(self, 
                 fun: Callable[[ArrayLike], ArrayLike], 
                 nobj: int, 
                 store: Optional[store] = None, 
                 interval: Optional[int] = 100000, 
                 plot: Optional[bool] = False, 
                 name: Optional[str] = None):
        self.fun = fun
        self.nobj = nobj
        self.n_evals = mp.RawValue(ct.c_long, 0)
        self.time_0 = time.perf_counter()
        self.best_y = mp.RawArray(ct.c_double, nobj)  
        for i in range(nobj):
            self.best_y[i] = sys.float_info.max
        self.store = store
        self.interval = interval
        self.plot = plot
        self.name = name
        self.lock = mp.Lock()
    
    def __call__(self, x: ArrayLike) -> np.ndarray:
        try:
            y = self.fun(x)
            with self.lock:
                self.n_evals.value += 1
            if not self.store is None and is_feasible(y, self.nobj):
                self.store.add_result(x, y[:self.nobj])
            improve = False
            for i in range(self.nobj):
                if y[i] < self.best_y[i]:
                    improve = True 
                    self.best_y[i] = y[i] 
            improve = improve# and self.n_evals.value > 10000
            if self.n_evals.value % self.interval == 0 or improve:
                constr = np.maximum(y[self.nobj:], 0) 
                logger.info(
                    str(dtime(self.time_0)) + ' ' + 
                    str(self.n_evals.value) + ' ' + 
                    str(round(self.n_evals.value/(1E-9 + dtime(self.time_0)),0)) + ' ' + 
                    str(self.best_y[:]) + ' ' + str(list(constr)) + ' ' + str(list(x))) 
                if (not self.store is None) and (not self.name is None):
                    try:
                        xs, ys = self.store.get_front()
                        num = self.store.num_stored.value
                        name = self.name + '_' + str(num)
                        np.savez_compressed(name, xs=xs, ys=ys)
                        if self.plot:
                            moretry.plot(name, 0, xs, ys, all=False)
                    except Exception as ex:
                        print(str(ex))                                                
            return y
        except Exception as ex:
            print(str(ex))  
            return None  
 
def minimize_plot(name: str, 
                 fun: Callable[[ArrayLike], ArrayLike], 
                 nobj: int,
                 ncon: int, 
                 bounds: Bounds,
                 popsize: Optional[int] = 64, 
                 max_evaluations: Optional[int] = 100000, 
                 nsga_update: Optional[bool] = True,
                 pareto_update: Optional[int] = 0,
                 ints: Optional[ArrayLike] = None,
                 workers: Optional[int] = mp.cpu_count()) -> Tuple[np.ndarray, np.ndarray]:     
    name += '_mode_' + str(popsize) + '_' + \
                ('nsga_update' if nsga_update else ('de_update_' + str(pareto_update)))
    logger.info('optimize ' + name) 
    xs, ys = minimize(fun, nobj, ncon, bounds, popsize = popsize, max_evaluations = max_evaluations,
                   nsga_update = nsga_update, pareto_update = pareto_update, workers=workers, ints=ints)
    np.savez_compressed(name, xs=xs, ys=ys)
    moretry.plot(name, ncon, xs, ys)
    return xs, ys
