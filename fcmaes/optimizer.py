# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

import numpy as np
from numpy.random import MT19937, Generator
from scipy.optimize import Bounds, minimize, shgo, differential_evolution, dual_annealing, basinhopping
import sys
import time
import math
import logging

from fcmaes import cmaes, cmaescpp, decpp, dacpp

_logger = None

def logger(logfile = 'optimizer.log'):
    '''default logger used by the parallel retry. Logs both to stdout and into a file.'''
    global _logger
    if _logger is None:
        formatter = logging.Formatter('%(message)s')
        file_handler = logging.FileHandler(filename=logfile)
        file_handler.setLevel(logging.INFO)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        stdout_handler.setFormatter(formatter) 
        _logger = logging.getLogger('optimizer')
        _logger.addHandler(file_handler)
        _logger.addHandler(stdout_handler)
        _logger.setLevel(logging.INFO)
    return _logger

def eprint(*args, **kwargs):
    """print message to stderr."""
    print(*args, file=sys.stderr, **kwargs)

def scale(lower, upper):
    """scaling = 0.5 * difference of the bounds."""
    return 0.5 * (np.asarray(upper) - np.asarray(lower))

def typical(lower, upper):
    """typical value = mean of the bounds."""
    return 0.5 * (np.asarray(upper) + np.asarray(lower))

def fitting(guess, lower, upper):
    """fit a guess into the bounds."""
    return np.minimum(np.asarray(upper), np.maximum(np.asarray(guess), np.asarray(lower)))

def is_terminate(runid, iterations, val):
    """dummy is_terminate call back."""
    return False    

def random_x(lower, upper):
    """feasible random value uniformly distributed inside the bounds."""
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    return lower + np.multiply(upper - lower, np.random.rand(lower.size))
    
def dtime(t0):
    """time since t0."""
    return round(time.perf_counter() - t0, 2)

class single_objective:
    """Utility class to create a fcmaes problem from a pagmo problem."""
      
    def __init__(self, pagmo_prob):
        self.pagmo_prob = pagmo_prob
        self.name = pagmo_prob.get_name() 
        self.fun = self.fitness
        lb, ub = pagmo_prob.get_bounds()
        self.bounds = Bounds(lb, ub)
         
    def fitness(self,X):
        try:
            val = self.pagmo_prob.fitness(X)
        except Exception as ex:
            return sys.float_info.max
        return val[0]

def de_cma(max_evaluations = 50000, popsize=31, stop_fittness = math.inf, 
           de_max_evals = None, cma_max_evals = None):
    """Sequence differential evolution -> CMA-ES."""

    if de_max_evals is None:
        de_max_evals = int(0.5*max_evaluations)
    if cma_max_evals is None:
        cma_max_evals = int(0.5*max_evaluations)
    opt1 = De_cpp(max_evaluations = de_max_evals, stop_fittness = stop_fittness)
    opt2 = Cma_cpp(popsize=popsize, max_evaluations = cma_max_evals, 
                   stop_fittness = stop_fittness)
    return Sequence([opt1, opt2])

def da_cma(max_evaluations = 50000, da_max_evals = None, cma_max_evals = None,
           popsize=31, stop_fittness = math.inf):
    """Sequence differential evolution -> CMA-ES."""

    if da_max_evals is None:
        da_max_evals = int(0.5*max_evaluations)
    if cma_max_evals is None:
        cma_max_evals = int(0.5*max_evaluations)
    opt1 = Da_cpp(max_evaluations = da_max_evals, stop_fittness = stop_fittness)
    opt2 = Cma_cpp(popsize=popsize, max_evaluations = cma_max_evals, 
                   stop_fittness = stop_fittness)
    return Sequence([opt1, opt2])

class Optimizer(object):
    """Provides different optimization methods for use with parallel retry."""
    
    def __init__(self, max_evaluations=50000, name=''):
        self.max_evaluations = max_evaluations  
        self.name = name  

    def max_eval_num(self, store=None):
        return self.max_evaluations if store is None else \
                store.eval_num(self.max_evaluations)
                
    def get_count_runs(self, store=None):
        return 0 if store is None else \
                store.get_count_runs()
                        
class Sequence(Optimizer):
    """Sequence of optimizers."""
    
    def __init__(self, optimizers):
        Optimizer.__init__(self)
        self.optimizers = optimizers 
        self.max_evaluations = 0 
        for optimizer in self.optimizers:
            self.name += optimizer.name + ' '
            self.max_evaluations += optimizer.max_evaluations
                  
    def minimize(self, fun, bounds, guess=None, sdevs=None, rg=Generator(MT19937()), store=None):
        evals = 0
        for optimizer in self.optimizers:
            ret = optimizer.minimize(fun, bounds, guess, sdevs, rg, store)
            guess = ret[0]
            evals += ret[2]
        return ret[0], ret[1], evals

class Cma_python(Optimizer):
    """CMA_ES Python implementation."""
    
    def __init__(self, max_evaluations=50000,
                 popsize = 31, guess=None, stop_fittness = None):        
        Optimizer.__init__(self, max_evaluations, 'cma py')
        self.popsize = popsize
        self.stop_fittness = stop_fittness
        self.guess = guess

    def minimize(self, fun, bounds, guess=None, sdevs=0.3, rg=Generator(MT19937()), store=None):
        ret = cmaes.minimize(fun, bounds, 
                self.guess if guess is None else guess,
                input_sigma=sdevs, 
                max_evaluations = self.max_eval_num(store), 
                popsize=self.popsize, 
                stop_fittness = self.stop_fittness,
                rg=rg, runid=self.get_count_runs(store))     
        return ret.x, ret.fun, ret.nfev

class Cma_cpp(Optimizer):
    """CMA_ES C++ implementation."""
   
    def __init__(self, max_evaluations=50000,
                 popsize = 31, guess=None, stop_fittness = None):        
        Optimizer.__init__(self, max_evaluations, 'cma cpp')
        self.popsize = popsize
        self.stop_fittness = stop_fittness
        self.guess = guess

    def minimize(self, fun, bounds, guess=None, sdevs=0.3, rg=Generator(MT19937()), store=None):
        ret = cmaescpp.minimize(fun, bounds, 
                self.guess if guess is None else guess,
                input_sigma=sdevs, 
                max_evaluations = self.max_eval_num(store), 
                popsize=self.popsize, 
                stop_fittness = self.stop_fittness,
                rg=rg, runid = self.get_count_runs(store))     
        return ret.x, ret.fun, ret.nfev

class De_cpp(Optimizer):
    """Differential Evolution C++ implementation."""
    
    def __init__(self, max_evaluations=50000,
                 popsize = None, stop_fittness = None, 
                 keep = 200, f = 0.5, cr = 0.9):        
        Optimizer.__init__(self, max_evaluations, 'de cpp')
        self.popsize = popsize
        self.stop_fittness = stop_fittness
        self.keep = keep
        self.f = f
        self.cr = cr

    def minimize(self, fun, bounds, guess=None, sdevs=None, rg=Generator(MT19937()), store=None):
        ret = decpp.minimize(fun, len(bounds.lb), bounds, 
                popsize=self.popsize, 
                max_evaluations = self.max_eval_num(store), 
                stop_fittness = self.stop_fittness,
                keep = self.keep, f = self.f, cr = self.cr,
                rg=rg, runid = self.get_count_runs(store))
        return ret.x, ret.fun, ret.nfev
    
class Da_cpp(Optimizer):
    """Dual Annealing C++ implementation."""
    
    def __init__(self, max_evaluations=50000,
                 stop_fittness = None, use_local_search=True, guess = None):        
        Optimizer.__init__(self, max_evaluations, 'da cpp',)
        self.stop_fittness = stop_fittness
        self.use_local_search = use_local_search
        self.guess = guess
 
    def minimize(self, fun, bounds, guess=None, sdevs=None, rg=Generator(MT19937()), store=None):
        ret = dacpp.minimize(fun, bounds, 
                             self.guess if guess is None else guess,
                            max_evaluations = self.max_eval_num(store), 
                            use_local_search = self.use_local_search,
                            rg=rg, runid = self.get_count_runs(store))
        return ret.x, ret.fun, ret.nfev
        
class Dual_annealing(Optimizer):
    """scipy dual_annealing."""
 
    def __init__(self, max_evaluations=50000,
                 rg=Generator(MT19937()), use_local_search=True):        
        Optimizer.__init__(self, max_evaluations, 'scipy da')
        self.no_local_search = not use_local_search
 
    def minimize(self, fun, bounds, guess=None, sdevs=None, rg=Generator(MT19937()), store=None):
        ret = dual_annealing(fun, bounds=list(zip(bounds.lb, bounds.ub)),
            maxfun = self.max_eval_num(store), 
            no_local_search = self.no_local_search,
            x0=guess,
            seed = int(rg.uniform(0, 2**32 - 1)))
        return ret.x, ret.fun, ret.nfev

class Differential_evolution(Optimizer):
    """scipy differential_evolution."""
 
    def __init__(self, max_evaluations=50000, store=None,
                 popsize = 15):        
        Optimizer.__init__(self, max_evaluations, 'scipy de')
        self.popsize = popsize
 
    def minimize(self, fun, bounds, guess=None, sdevs=None, rg=Generator(MT19937()), store=None):
        popsize = self.popsize 
        maxiter = int(self.max_eval_num(store) / (popsize * len(bounds.lb)) - 1)
        ret = differential_evolution(fun, bounds=bounds, maxiter=maxiter,
                                      seed = int(rg.uniform(0, 2**32 - 1)))
        return ret.x, ret.fun, ret.nfev

class CheckBounds(object):
    
    def __init__(self, bounds):
        self.bounds = bounds
        
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        inb =  np.less_equal(x, self.bounds.ub).all() and \
            np.greater_equal(x, self.bounds.lb).all()
        return inb

class Basin_hopping(Optimizer):
    """scipy basin hopping."""
 
    def __init__(self, max_evaluations=50000, store=None):        
        Optimizer.__init__(self, max_evaluations, 'scipy basin hopping')
         
    def minimize(self, fun, bounds, guess=None, sdevs=None, rg=Generator(MT19937()), store=None):
        localevals = 200
        maxiter = int(self.max_eval_num(store) / localevals)         
        if guess is None:
            guess = rg.uniform(bounds.lb, bounds.ub)
        
        ret = basinhopping(fun, guess, niter=maxiter, 
                           minimizer_kwargs={"method": 'SLSQP', 
                                             "bounds":bounds},
                           accept_test=CheckBounds(bounds),
                           seed = int(rg.uniform(0, 2**32 - 1)))
        return ret.x, ret.fun, ret.nfev

class Minimize(Optimizer):
    """scipy minimize."""
 
    def __init__(self, max_evaluations=50000, store=None):        
        Optimizer.__init__(self, max_evaluations, 'scipy minimize')
 
    def minimize(self, fun, bounds, guess=None, sdevs=None, rg=Generator(MT19937()), store=None):
        if guess is None:
            guess = rg.uniform(bounds.lb, bounds.ub)
        ret = minimize(fun, x0=guess, bounds=bounds)
        return ret.x, ret.fun, ret.nfev
 
class Shgo(Optimizer):
    """scipy shgo."""

    def __init__(self, max_evaluations=50000, store=None):        
        Optimizer.__init__(self, max_evaluations, 'scipy shgo')
 
    def minimize(self, fun, bounds, guess=None, sdevs=None, rg=Generator(MT19937()), store=None):
        ret = shgo(fun, bounds=list(zip(bounds.lb, bounds.ub)), 
                   options={'maxfev': self.max_eval_num(store)})
        return ret.x, ret.fun, ret.nfev
