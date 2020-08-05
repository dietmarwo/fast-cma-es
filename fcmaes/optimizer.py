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

from fcmaes import cmaes, cmaescpp, decpp, dacpp, hhcpp, gcldecpp, lcldecpp, ldecpp

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
            self.name += optimizer.name + ' -> '
            self.max_evaluations += optimizer.max_evaluations
        self.name = self.name[:-4]
                  
    def minimize(self, fun, bounds, guess=None, sdevs=None, rg=Generator(MT19937()), store=None):
        evals = 0
        for optimizer in self.optimizers:
            ret = optimizer.minimize(fun, bounds, guess, sdevs, rg, store)
            guess = ret[0]
            evals += ret[2]
        return ret[0], ret[1], evals

class Choice(Optimizer):
    """Random choice of optimizers."""
    
    def __init__(self, optimizers):
        Optimizer.__init__(self)
        self.optimizers = optimizers 
        self.max_evaluations = optimizers[0].max_evaluations 
        for optimizer in self.optimizers:
            self.name += optimizer.name + ' | '
        self.name = self.name[:-3]
                  
    def minimize(self, fun, bounds, guess=None, sdevs=None, rg=Generator(MT19937()), store=None):
        choice = rg.integers(0, len(self.optimizers))
        opt = self.optimizers[choice]
        return opt.minimize(fun, bounds, guess, sdevs, rg, store)

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

def de2_cma(max_evaluations = 50000, popsize=31, stop_fittness = math.inf, 
           de_max_evals = None, cma_max_evals = None):
    """Sequence differential evolution -> CMA-ES."""

    if de_max_evals is None:
        de_max_evals = int(0.5*max_evaluations)
    if cma_max_evals is None:
        cma_max_evals = int(0.5*max_evaluations)
    opt1 = Choice([GCLDE_cpp(de_max_evals), De_cpp(de_max_evals)])
    opt2 = Cma_cpp(cma_max_evals, popsize=popsize, stop_fittness = stop_fittness)
    return Sequence([opt1, opt2])

def de3_cma(max_evaluations = 50000, popsize=31, stop_fittness = math.inf, 
           de_max_evals = None, cma_max_evals = None):
    """Sequence differential evolution -> CMA-ES."""

    if de_max_evals is None:
        de_max_evals = int(0.5*max_evaluations)
    if cma_max_evals is None:
        cma_max_evals = int(0.5*max_evaluations)
    opt1 =  Choice([GCLDE_cpp(de_max_evals), Cma_cpp(de_max_evals), De_cpp(de_max_evals)])
    opt2 = Cma_cpp(cma_max_evals, popsize=popsize, stop_fittness = stop_fittness)
    return Sequence([opt1, opt2])

def gclde_cma(max_evaluations = 50000, popsize=31, stop_fittness = math.inf, 
           de_max_evals = None, cma_max_evals = None, workers = None):
    """Sequence G-CL-differential evolution -> CMA-ES."""

    if de_max_evals is None:
        de_max_evals = int(2.0*max_evaluations/3.0)
    if cma_max_evals is None:
        cma_max_evals = int(max_evaluations/3.0)
    opt1 = GCLDE_cpp(max_evaluations = de_max_evals, stop_fittness = stop_fittness, workers = workers)
    opt2 = Cma_cpp(popsize=popsize, max_evaluations = cma_max_evals, 
                   stop_fittness = stop_fittness, workers = workers)
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
                 popsize = 31, guess=None, stop_fittness = None, workers = None):        
        Optimizer.__init__(self, max_evaluations, 'cma cpp')
        self.popsize = popsize
        self.stop_fittness = stop_fittness
        self.guess = guess
        self.workers = workers

    def minimize(self, fun, bounds, guess=None, sdevs=0.3, rg=Generator(MT19937()), 
                 store=None, workers = None):
        ret = cmaescpp.minimize(fun, bounds, 
                self.guess if guess is None else guess,
                input_sigma=sdevs, 
                max_evaluations = self.max_eval_num(store), 
                popsize=self.popsize, 
                stop_fittness = self.stop_fittness,
                rg=rg, runid = self.get_count_runs(store), 
                workers = self.workers if workers is None else workers)     
        return ret.x, ret.fun, ret.nfev

class Cma_orig(Optimizer):
    """CMA_ES original implementation."""
   
    def __init__(self, max_evaluations=50000,
                 popsize = 31, guess=None, stop_fittness = None):        
        Optimizer.__init__(self, max_evaluations, 'cma orig')
        self.popsize = popsize
        self.stop_fittness = stop_fittness
        self.guess = guess

    def minimize(self, fun, bounds, guess=None, sdevs=0.3, rg=Generator(MT19937()), store=None):
        lower = bounds.lb
        upper = bounds.ub
        guess = self.guess if guess is None else guess
        if guess is None:
            guess = rg.uniform(lower, upper)
        max_evaluations = self.max_eval_num(store)   
        try:
            import cma
        except ImportError as e:
            raise ImportError("Please install CMA (pip install cma)") 
        try: 
            es = cma.CMAEvolutionStrategy(guess, 0.1,  {'bounds': [lower, upper], 
                                                             'typical_x': guess,
                                                             'scaling_of_variables': scale(lower, upper),
                                                             'popsize': self.popsize,
                                                             'CMA_stds': sdevs,
                                                             'verbose': -1,
                                                             'verb_disp': -1})
            evals = 0
            for i in range(max_evaluations):
                X, Y = es.ask_and_eval(fun)
                es.tell(X, Y)
                evals += self.popsize
                if es.stop():
                    break 
                if evals > max_evaluations:
                    break    
            return es.result.xbest, es.result.fbest, evals
        except Exception as ex:
            print(ex)
  
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

class LDe_cpp(Optimizer):
    """Local Differential Evolution C++ implementation."""
    
    def __init__(self, max_evaluations=50000,
                 popsize = None, stop_fittness = None, 
                 keep = 200, f = 0.5, cr = 0.9):        
        Optimizer.__init__(self, max_evaluations, 'lde cpp')
        self.popsize = popsize
        self.stop_fittness = stop_fittness
        self.keep = keep
        self.f = f
        self.cr = cr

    def minimize(self, fun, bounds, guess=None, sdevs=0.3, rg=Generator(MT19937()), store=None):
        ret = ldecpp.minimize(fun, bounds, guess, sdevs,
                popsize=self.popsize, 
                max_evaluations = self.max_eval_num(store), 
                stop_fittness = self.stop_fittness,
                keep = self.keep, f = self.f, cr = self.cr,
                rg=rg, runid = self.get_count_runs(store))
        return ret.x, ret.fun, ret.nfev

class GCLDE_cpp(Optimizer):
    """GCL-Differential Evolution C++ implementation."""
    
    def __init__(self, max_evaluations=50000,
                 popsize = None, stop_fittness = None, 
                 pbest = 0.7, f0 = 0.001, cr0 = 0.1, workers = None):        
        Optimizer.__init__(self, max_evaluations, 'gclde cpp')
        self.popsize = popsize
        self.stop_fittness = stop_fittness
        self.pbest = pbest
        self.f0 = f0
        self.cr0 = cr0
        self.workers = workers

    def minimize(self, fun, bounds, guess=None, sdevs=None, rg=Generator(MT19937()), 
                 store=None, workers = None):
        ret = gcldecpp.minimize(fun, bounds, 
                popsize=self.popsize, 
                max_evaluations = self.max_eval_num(store), 
                stop_fittness = self.stop_fittness,
                pbest = self.pbest, f0 = self.f0, cr0 = self.cr0,
                rg=rg, runid = self.get_count_runs(store),
                workers = self.workers if workers is None else workers)
        return ret.x, ret.fun, ret.nfev

class LCLDE_cpp(Optimizer):
    """LCL-Differential Evolution C++ implementation."""
    
    def __init__(self, max_evaluations=50000,
                 popsize = None, stop_fittness = None, 
                 pbest = 0.7, f0 = 0.001, cr0 = 0.1, workers = None):        
        Optimizer.__init__(self, max_evaluations, 'lclde cpp')
        self.popsize = popsize
        self.stop_fittness = stop_fittness
        self.pbest = pbest
        self.f0 = f0
        self.cr0 = cr0
        self.workers = workers

    def minimize(self, fun, bounds, guess=None, sdevs=0.3, rg=Generator(MT19937()), 
                 store=None, workers = None):
        ret = lcldecpp.minimize(fun, bounds, guess, sdevs,
                popsize=self.popsize, 
                max_evaluations = self.max_eval_num(store), 
                stop_fittness = self.stop_fittness,
                pbest = self.pbest, f0 = self.f0, cr0 = self.cr0,
                rg=rg, runid = self.get_count_runs(store),
                workers = self.workers if workers is None else workers)

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

class Hh_cpp(Optimizer):
    """Harris hawks C++ implementation."""
    
    def __init__(self, max_evaluations=50000,
                 popsize = 31, stop_fittness = None):        
        Optimizer.__init__(self, max_evaluations, 'hh cpp')
        self.popsize = popsize
        self.stop_fittness = stop_fittness

    def minimize(self, fun, bounds, guess=None, sdevs=None, rg=Generator(MT19937()), store=None):
        ret = hhcpp.minimize(fun, len(bounds.lb), bounds, 
                popsize=self.popsize, 
                max_evaluations = self.max_eval_num(store), 
                stop_fittness = self.stop_fittness,
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
            return self.pagmo_prob.fitness(X)[0]
        except Exception as ex:
            return sys.float_info.max

class NLopt(Optimizer):
    """NLopt_algo wrapper."""

    def __init__(self, algo, max_evaluations=50000, store=None):        
        Optimizer.__init__(self, max_evaluations, 'NLopt ' + algo.get_algorithm_name())
        self.algo = algo
 
    def minimize(self, fun, bounds, guess=None, sdevs=None, rg=Generator(MT19937()), store=None):
        self.fun = fun
        opt = self.algo
        opt.set_min_objective(self.nlfunc)
        opt.set_lower_bounds(bounds.lb)
        opt.set_upper_bounds(bounds.ub)
        opt.set_maxeval(self.max_eval_num(store))
        opt.set_initial_step(sdevs)
        if guess is None:
            guess = rg.uniform(bounds.lb, bounds.ub)
        x = opt.optimize(guess)
        y = opt.last_optimum_value()
        return x, y, opt.get_numevals()
    
    def nlfunc(self, x, _):
        try:
            return self.fun(x)
        except Exception as ex:
            return sys.float_info.max
    
