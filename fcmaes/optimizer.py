# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

import numpy as np
from scipy.optimize import minimize, shgo, differential_evolution, dual_annealing, Bounds
import sys
import time
import logging
import random

from fcmaes import cmaes
from fcmaes import cmaescpp 

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

def seed_random():    
    """makes sure the c++ random generator for this process is initialized properly"""
    if sys.platform.startswith('linux'):
        cmaescpp.seed_random() 

class Optimizer(object):
    """Provides different optimization methods for use with parallel retry."""
       
    def __init__(self, store, popsize = 31, stop_fittness = None):        
        self.popsize = popsize
        self.stop_fittness = stop_fittness
        # store provides the (changing) upper limit of the number of function evaluations
        self.store = store 
         
    def cma_python(self, fun, guess, bounds, sdevs, rg): 
        """CMA_ES Python implementation."""
        ret = cmaes.minimize(fun, bounds, guess,
                input_sigma=sdevs, max_evaluations=self.store.eval_num(), 
                popsize=self.popsize, stop_fittness = self.stop_fittness,
                rg=rg, runid=self.store.get_count_runs())     
        return ret.x, ret.fun, ret.nfev

    def cma_cpp(self, fun, guess, bounds, sdevs, rg):
        """CMA_ES C++ implementation."""
        ret = cmaescpp.minimize(fun, bounds, guess,
                input_sigma=sdevs, max_evaluations=self.store.eval_num(), 
                popsize=self.popsize, stop_fittness = self.stop_fittness,
                rg=rg, runid=self.store.get_count_runs())     
        return ret.x, ret.fun, ret.nfev
    
    def dual_annealing(self, fun, guess, bounds, sdevs, rg):
        """scipy dual_annealing."""
        ret = dual_annealing(fun, bounds=list(zip(bounds.lb, bounds.ub)), 
                             maxfun=self.store.eval_num(), 
                             seed=random.randint(0, 2**32 - 1))
        return ret.x, ret.fun, ret.nfev

    def differential_evolution(self, fun, guess, bounds, sdevs, rg):
        """scipy differential_evolution."""
        popsize = 15 # default value for differential_evolution
        maxiter = int(self.store.eval_num() / (popsize * len(bounds.lb)) - 1)
        ret = differential_evolution(fun, bounds=bounds, maxiter=maxiter,
                                      seed=random.randint(0, 2**32 - 1))
        return ret.x, ret.fun, ret.nfev
    
    def minimize(self, fun, guess, bounds, sdevs, rg):
        """scipy minimize."""
        ret = minimize(fun, bounds=bounds)
        return ret.x, ret.fun, ret.nfev
 
    def shgo(self, fun, guess, bounds, sdevs, rg):
        """scipy shgo."""
        ret = shgo(fun, bounds=list(zip(bounds.lb, bounds.ub)), 
                   options={'maxfev': self.store.eval_num()})
        return ret.x, ret.fun, ret.nfev
