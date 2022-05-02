# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

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
from fcmaes.evaluator import Evaluator
from fcmaes import moretry
import multiprocessing as mp
from fcmaes.optimizer import logger, dtime

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(mofun, 
             nobj,
             ncon, 
             bounds,
             popsize = 64, 
             max_evaluations = 100000, 
             workers = None,
             f = 0.5, 
             cr = 0.9, 
             nsga_update = False,
             pareto_update = 0,
             ints = None,
             modifier = None,
             rg = Generator(MT19937()),
             store = None):  
      
    """Minimization of a multi objjective function of one or more variables using
    Differential Evolution.
     
    Parameters
    ----------
    mofun : callable
        The objective function to be minimized.
            ``mofun(x) -> list(float)``
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
    pareto_update = float, optional
        Only applied if nsga_update = False. Favor better solutions for sample generation. Default 0 - 
        use all population members with the same probability.   
    ints = list or array, optional
        indicating which parameters are discrete integer values. If defined these parameters will be
        rounded to the next integer and some additional mutation of discrete parameters are performed.
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

    mode = MODE(nobj, ncon, bounds, popsize, workers if not workers is None else 0, 
            f, cr, nsga_update, pareto_update, rg, ints, modifier)
    try:
        if workers and workers > 1:
            x, y, evals, iterations, stop = mode.do_optimize_delayed_update(mofun, max_evaluations, workers)
        else:      
            x, y, evals, iterations, stop = mode.do_optimize(mofun, max_evaluations)
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
        capacity of the result store.
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
    
    def __init__(self, nobj, ncon, bounds, popsize = 64, workers = 0,
                 F = 0.5, Cr = 0.9, nsga_update = False, pareto_update = False, 
                 rg = Generator(MT19937()), ints = None, modifier = None):
        self.nobj = nobj
        self.ncon = ncon
        self.dim, self.lower, self.upper = _check_bounds(bounds, None)
        if popsize is None:
            popsize = 64
        if popsize % 2 == 1 and nsga_update: # nsga update requires even popsize
            popsize += 1
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
        # nsga update doesn't support mixed integer
        self.ints = None if nsga_update else np.array(ints)
        # use default variable modifier for int variables if modifier is None
        if modifier is None and not self.ints is None:
            # adjust bounds because ints are rounded
            self.lower = self.lower.astype(float)
            self.upper = self.upper.astype(float)
            self.lower[self.ints] -= .499999999
            self.upper[self.ints] += .499999999
            self.modifier = self._modifier
        else:
            self.modifier = modifier
        self._init()
                
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
        if self._is_dominated(y, p):
            return self.stop
        
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
        x, y = _filter(self.x, self.y)
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
        x, y = _filter(self.x, self.y)
        return x, y, self.evals, self.iterations, self.stop

    def pop_update(self):
        y0 = self.y
        x0 = self.x
        if self.nobj == 1:
            yi = np.flip(np.argsort(self.y[:,0]))
            y0 = self.y[yi]
            x0 = self.x[yi]    
        domination = pareto(y0, self.nobj, self.ncon)
        x = []
        y = []
        maxdom = int(max(domination))
        for dom in range(maxdom, -1, -1):
            domlevel = [p for p in range(len(domination)) if domination[p] == dom]
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
        
    # default modifier for integer variables
    def _modifier(self, x):
        x_ints = x[self.ints]
        n_ints = len(self.ints)
        lb = self.lower[self.ints]
        ub = self.upper[self.ints]
        min_mutate = 0.5
        max_mutate = max(1.0, n_ints/20.0)
        to_mutate = self.rg.uniform(min_mutate, max_mutate)
        # mututate some integer variables
        x_ints = np.array([x if self.rg.random() > to_mutate/n_ints else 
                           self.rg.uniform(lb[i], ub[i])
                           for i, x in enumerate(x_ints)])
        # round to int values
        x[self.ints] = np.around(x_ints,0)
        return x   
    
    def _is_dominated(self, y, p):
        return np.all(np.array([y[i] >= self.y[p, i] for i in range(len(y))]))

                    
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

def ranks(cons):
    feasible = np.less_equal(cons, 0)
    ci = cons.argsort(axis=0)
    rank = np.empty_like(ci)
    ar = np.arange(cons.shape[0])
    for i in range(cons.shape[1]): 
        rank[ci[:,i], i] = ar
    rank[feasible] = 0
    alpha = np.sum(np.greater(cons, 0), axis=1) / cons.shape[1] # violations
    alpha = np.tile(alpha, (cons.shape[1],1)).T
    rank = rank*alpha
    rank = np.sum(rank, axis=1)
    return rank
     
def pareto(ys, nobj, ncon):
    if ncon == 0:
        return pareto_levels(ys)
    else:
        yobj = np.array([y[:nobj] for y in ys])
        ycon = np.array([np.maximum(y[-ncon:], 0) for y in ys])    
        csum = ranks(ycon)
        feasible = np.less_equal(csum, 0)
        if sum(feasible) > 0:
            csum += objranks(yobj)
        ci = np.argsort(csum)
        popn = len(ys)
        domination = np.zeros(popn)
        # first pareto front of feasible solutions
        cy = np.array([i for i in ci if feasible[i]])
        if len(cy) > 0:
            ypar = pareto_levels(yobj[cy])
            domination[cy] += ypar        
        # then constraint violations   
        ci = np.array([i for i in ci if not feasible[i]])  
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
    if pro_c < 1.0:
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


def feasible(xs, ys, ncon, eps = 1E-2):
    if ncon > 0: # select feasible
        ycon = np.array([np.maximum(y[-ncon:], 0) for y in ys])  
        con = np.sum(ycon, axis=1)
        nobj = len(ys[0]) - ncon
        feasible = np.array([i for i in range(len(ys)) if con[i] < eps])
        if len(feasible) > 0:
            xs, ys = xs[feasible], np.array([ y[:nobj] for y in ys[feasible]])
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
   
    def __init__(self, fun, nobj, store = None, interval = 100000, plot = False, 
                 name = None, logger=logger()):
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
        self.name = (fun.name if hasattr(fun, 'name') else 
                     (fun.__name__ if hasattr(fun, '__name__') else fun.__class__.__name__)) \
                        if name is None else name
        self.logger = logger
    
    def __call__(self, x):
        try:
            y = self.fun(x)
            if not self.store is None and is_feasible(y, self.nobj):
                self.store.add_result(x, y[:self.nobj])
            improve = False
            for i in range(self.nobj):
                if y[i] < self.best_y[i]:
                    improve = True 
                    self.best_y[i] = y[i] 
            improve = improve and self.n_evals.value > 100
            self.n_evals.value += 1
            if self.n_evals.value % self.interval == 0 or improve:
                constr = np.maximum(y[self.nobj:], 0) 
                self.logger.info(
                    str(dtime(self.time_0)) + ' ' + 
                    str(self.n_evals.value) + ' ' + 
                    str(round(self.n_evals.value/(1E-9 + dtime(self.time_0)),0)) + ' ' + 
                    str(self.best_y[:]) + ' ' + str(list(constr)) + ' ' + str(list(x))) 
                if not self.store is None:
                    try:
                        xs, ys = self.store.get_front(True)
                        num = self.store.num_stored.value
                        # self.logger.info(str(num) + ' ' + 
                        #               ', '.join(['(' + ', '.join([str(round(yi,3)) for yi in y]) + ')' for y in ys]))
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
        

def minimize_plot(name, fun, nobj, ncon, bounds, popsize = 64, max_evaluations = 100000, nsga_update=False, 
                  pareto_update=0, workers = mp.cpu_count()):
    name += '_mode_' + str(popsize) + '_' + \
                ('nsga_update' if nsga_update else ('de_update_' + str(pareto_update)))
    logger.info('optimize ' + name) 
    xs, ys = minimize(fun, nobj, ncon, bounds, popsize = popsize, max_evaluations = max_evaluations,
                   nsga_update = nsga_update, pareto_update = pareto_update, workers=workers)
    np.savez_compressed(name, xs=xs, ys=ys)
    moretry.plot(name, ncon, xs, ys)
