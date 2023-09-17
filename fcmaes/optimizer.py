# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

import numpy as np
from numpy.random import MT19937, Generator
from scipy.optimize import Bounds, minimize, shgo, differential_evolution, dual_annealing, basinhopping
import sys
import time
from loguru import logger
import ctypes as ct
import multiprocessing as mp 
from fcmaes.evaluator import serial, parallel
from fcmaes import crfmnes, crfmnescpp, pgpecpp, cmaes, de, cmaescpp, decpp, dacpp, gcldecpp, lcldecpp, ldecpp, csmacpp, bitecpp

from typing import Optional, Callable, Tuple, Union
from numpy.typing import ArrayLike

def eprint(*args, **kwargs):
    """print message to stderr."""
    print(*args, file=sys.stderr, **kwargs)

def scale(lower: ArrayLike, 
          upper: ArrayLike) -> np.ndarray:
    """scaling = 0.5 * difference of the bounds."""
    return 0.5 * (np.asarray(upper) - np.asarray(lower))

def typical(lower: ArrayLike, 
            upper: ArrayLike) -> np.ndarray:
    """typical value = mean of the bounds."""
    return 0.5 * (np.asarray(upper) + np.asarray(lower))

def fitting(guess: ArrayLike, 
            lower: ArrayLike, 
            upper: ArrayLike) -> np.ndarray:
    """fit a guess into the bounds."""
    return np.clip(np.asarray(guess), np.asarray(upper), np.asarray(lower))

def is_terminate(runid: int, 
                 iterations: int, 
                 val: float) -> bool:
    """dummy is_terminate call back."""
    return False    

def random_x(lower: ArrayLike, upper: ArrayLike) -> np.ndarray:
    """feasible random value uniformly distributed inside the bounds."""
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    return lower + np.multiply(upper - lower, np.random.rand(lower.size))
    
def dtime(t0: float) -> float:
    """time since t0."""
    return round(time.perf_counter() - t0, 2)

class wrapper(object):
    """Fitness function wrapper for use with parallel retry."""

    def __init__(self, 
                 fit: Callable[[ArrayLike], float]):
        self.fit = fit
        self.evals = mp.RawValue(ct.c_int, 0) 
        self.best_y = mp.RawValue(ct.c_double, np.inf) 
        self.t0 = time.perf_counter()

    def __call__(self, x: ArrayLike) -> float:
        try:
            self.evals.value += 1
            y = self.fit(x)
            y0 = y if np.isscalar(y) else sum(y)
            if y0 < self.best_y.value:
                self.best_y.value = y0
                logger.info(str(dtime(self.t0)) + ' '  + 
                              str(self.evals.value) + ' ' + 
                              str(round(self.evals.value/(1E-9 + dtime(self.t0)),0)) + ' ' + 
                              str(self.best_y.value) + ' ' + 
                              str(list(x)))
            return y
        except Exception as ex:
            print(str(ex))  
            return sys.float_info.max  
    
class Optimizer(object):
    """Provides different optimization methods for use with parallel retry."""
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000, 
                 name: Optional[str] = ''):
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
    
    def __init__(self, optimizers: ArrayLike):
        Optimizer.__init__(self)
        self.optimizers = optimizers 
        self.max_evaluations = 0 
        for optimizer in self.optimizers:
            self.name += optimizer.name + ' -> '
            self.max_evaluations += optimizer.max_evaluations
        self.name = self.name[:-4]

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Bounds, 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[Union[float, ArrayLike, Callable]] = None, 
                 rg: Optional[Generator] = Generator(MT19937()), 
                 store=None) -> Tuple[np.ndarray, float, int]:
        evals = 0
        y = np.inf
        for optimizer in self.optimizers:
            ret = optimizer.minimize(fun, bounds, guess, sdevs, rg, store)
            if ret[1] < y:
                y = ret[1]
                x = ret[0]
            guess = x
            evals += ret[2]
        return x, y, evals
                  
class Choice(Optimizer):
    """Random choice of optimizers."""
    
    def __init__(self, optimizers: ArrayLike):
        Optimizer.__init__(self)
        self.optimizers = optimizers 
        self.max_evaluations = optimizers[0].max_evaluations 
        for optimizer in self.optimizers:
            self.name += optimizer.name + ' | '
        self.name = self.name[:-3]
                  
    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Bounds, 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[Union[float, ArrayLike, Callable]] = None, 
                 rg: Optional[Generator] = Generator(MT19937()), 
                 store=None) -> Tuple[np.ndarray, float, int]:
        
        choice = rg.integers(0, len(self.optimizers))
        opt = self.optimizers[choice]
        return opt.minimize(fun, bounds, guess, sdevs, rg, store)

def de_cma(max_evaluations: Optional[int] = 50000, 
           popsize: Optional[int] = 31, 
           stop_fitness: Optional[float] = -np.inf, 
           de_max_evals: Optional[int] = None, 
           cma_max_evals: Optional[int] = None, 
           ints: Optional[ArrayLike] = None, 
           workers: Optional[int]  = None) -> Sequence:
    """Sequence differential evolution -> CMA-ES."""

    de_evals = np.random.uniform(0.1, 0.5)
    if de_max_evals is None:
        de_max_evals = int(de_evals*max_evaluations)
    if cma_max_evals is None:
        cma_max_evals = int((1.0-de_evals)*max_evaluations)
    opt1 = De_cpp(popsize=popsize, max_evaluations = de_max_evals, 
                  stop_fitness = stop_fitness, ints=ints, workers = workers)
    opt2 = Cma_cpp(popsize=popsize, max_evaluations = cma_max_evals, 
                   stop_fitness = stop_fitness, workers = workers)
    return Sequence([opt1, opt2])

def de_cma_py(max_evaluations: Optional[int] = 50000, 
           popsize: Optional[int] = 31, 
           stop_fitness: Optional[float] = -np.inf, 
           de_max_evals: Optional[int] = None, 
           cma_max_evals: Optional[int] = None, 
           ints: Optional[ArrayLike] = None, 
           workers: Optional[int]  = None) -> Sequence:
    """Sequence differential evolution -> CMA-ES in python."""

    de_evals = np.random.uniform(0.1, 0.5)
    if de_max_evals is None:
        de_max_evals = int(de_evals*max_evaluations)
    if cma_max_evals is None:
        cma_max_evals = int((1.0-de_evals)*max_evaluations)
    opt1 = De_python(popsize=popsize, max_evaluations = de_max_evals, 
                     stop_fitness = stop_fitness, ints=ints, workers = workers)
    opt2 = Cma_python(popsize=popsize, max_evaluations = cma_max_evals, 
                   stop_fitness = stop_fitness, workers = workers)
    return Sequence([opt1, opt2])

def da_cma(max_evaluations: Optional[int] = 50000, 
           popsize: Optional[int] = 31, 
           da_max_evals: Optional[int] = None, 
           cma_max_evals: Optional[int] = None, 
           stop_fitness: Optional[float] = -np.inf) -> Sequence:
    """Sequence dual annealing -> CMA-ES."""

    da_evals = np.random.uniform(0.1, 0.5)
    if da_max_evals is None:
        da_max_evals = int(da_evals*max_evaluations)
    if cma_max_evals is None:
        cma_max_evals = int((1.0-da_evals)*max_evaluations)
    opt1 = Da_cpp(max_evaluations = da_max_evals, stop_fitness = stop_fitness)
    opt2 = Cma_cpp(popsize=popsize, max_evaluations = cma_max_evals, 
                   stop_fitness = stop_fitness)
    return Sequence([opt1, opt2])

def de_crfmnes(max_evaluations: Optional[int] = 50000, 
               popsize: Optional[int] = 32, 
               stop_fitness: Optional[float] = -np.inf, 
               de_max_evals: Optional[int] = None, 
               crfm_max_evals: Optional[int] = None, 
               ints: Optional[ArrayLike] = None, 
               workers: Optional[int]  = None) -> Sequence:
    """Sequence differential evolution -> CRFMNES."""

    de_evals = np.random.uniform(0.1, 0.5)
    if de_max_evals is None:
        de_max_evals = int(de_evals*max_evaluations)
    if crfm_max_evals is None:
        crfm_max_evals = int((1.0-de_evals)*max_evaluations)
    opt1 = De_cpp(popsize=popsize, max_evaluations = de_max_evals, 
                  stop_fitness = stop_fitness, ints=ints, workers = workers)
    opt2 = Crfmnes_cpp(popsize=popsize, max_evaluations = crfm_max_evals, 
                   stop_fitness = stop_fitness, workers = workers)
    return Sequence([opt1, opt2])

def crfmnes_bite(max_evaluations: Optional[int] = 50000, 
                popsize: Optional[int] = 32, 
                stop_fitness: Optional[float] = -np.inf, 
                crfm_max_evals: Optional[int] = None, 
                bite_max_evals: Optional[int] = None, 
                M: Optional[int] = 1) -> Sequence:
    """Sequence CRFMNES -> Bite."""

    crfmnes_evals = np.random.uniform(0.1, 0.5)
    if crfm_max_evals is None:
        crfm_max_evals = int(crfmnes_evals*max_evaluations)
    if bite_max_evals is None:
        bite_max_evals = int((1.0-crfmnes_evals)*max_evaluations)
    opt1 = Crfmnes_cpp(popsize=popsize, max_evaluations = crfm_max_evals, 
                  stop_fitness = stop_fitness)
    opt2 = Bite_cpp(popsize=popsize, max_evaluations = bite_max_evals, 
                   stop_fitness = stop_fitness, M=M)
    return Sequence([opt1, opt2])

def cma_bite(max_evaluations: Optional[int] = 50000, 
            popsize: Optional[int] = 32, 
            stop_fitness: Optional[float] = -np.inf, 
            cma_max_evals: Optional[int] = None, 
            bite_max_evals: Optional[int] = None, 
            M: Optional[int] = 1) -> Sequence:
    """Sequence CMA-ES -> Bite."""

    cma_evals = np.random.uniform(0.1, 0.5)
    if cma_max_evals is None:
        cma_max_evals = int(cma_evals*max_evaluations)
    if bite_max_evals is None:
        bite_max_evals = int((1.0-cma_evals)*max_evaluations)
    opt1 = Cma_cpp(popsize=popsize, max_evaluations = cma_max_evals, 
                  stop_fitness = stop_fitness, stop_hist = 0)
    opt2 = Bite_cpp(popsize=popsize, max_evaluations = bite_max_evals, 
                   stop_fitness = stop_fitness, M=M)
    return Sequence([opt1, opt2])

class Crfmnes(Optimizer):
    """CRFMNES Python implementation."""
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = 32, 
                 guess: Optional[ArrayLike] = None, 
                 stop_fitness: Optional[float] = -np.inf,
                 sdevs: Optional[float] = None, 
                 workers: Optional[int] = None):        

        Optimizer.__init__(self, max_evaluations, 'crfmnes')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.guess = guess
        self.sdevs = sdevs
        self.workers = workers

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = 0.3, 
                 rg: Optional[Generator] = Generator(MT19937()), 
                 store = None) -> Tuple[np.ndarray, float, int]:

        ret = crfmnes.minimize(fun, bounds, 
                self.guess if not self.guess is None else guess,
                input_sigma = self.sdevs if not self.sdevs is None else sdevs,
                max_evaluations = self.max_eval_num(store), 
                popsize=self.popsize, 
                stop_fitness = self.stop_fitness,
                rg=rg, runid=self.get_count_runs(store),
                workers = self.workers)     
        return ret.x, ret.fun, ret.nfev

class Crfmnes_cpp(Optimizer):
    """CRFMNES C++ implementation."""
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = 32, 
                 guess: Optional[ArrayLike] = None, 
                 stop_fitness: Optional[float] = -np.inf,
                 sdevs: Optional[float] = None, 
                 workers: Optional[int] = None):        
       
        Optimizer.__init__(self, max_evaluations, 'crfmnes cpp')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.guess = guess
        self.sdevs = sdevs
        self.workers = workers

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = 0.3, 
                 rg: Optional[Generator] = Generator(MT19937()), 
                 store = None) -> Tuple[np.ndarray, float, int]:

        ret = crfmnescpp.minimize(fun, bounds, 
                self.guess if not self.guess is None else guess,
                input_sigma = self.sdevs if not self.sdevs is None else sdevs,
                max_evaluations = self.max_eval_num(store), 
                popsize=self.popsize, 
                stop_fitness = self.stop_fitness,
                rg=rg, runid=self.get_count_runs(store), 
                workers = self.workers)     
        return ret.x, ret.fun, ret.nfev

class Pgpe_cpp(Optimizer):
    """PGPE C++ implementation."""
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 500000,
                 popsize: Optional[int] = 640, 
                 guess: Optional[ArrayLike] = None, 
                 stop_fitness: Optional[float] = -np.inf,
                 sdevs: Optional[float] = None, 
                 workers: Optional[int] = None):
               
        Optimizer.__init__(self, max_evaluations, 'pgpe cpp')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.guess = guess
        self.sdevs = sdevs
        self.workers = workers

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = 0.1, 
                 rg: Optional[Generator] = Generator(MT19937()), 
                 store = None) -> Tuple[np.ndarray, float, int]:

        ret = pgpecpp.minimize(fun, bounds, 
                self.guess if not self.guess is None else guess,
                input_sigma = self.sdevs if not self.sdevs is None else sdevs,
                max_evaluations = self.max_eval_num(store), 
                popsize=self.popsize, 
                stop_fitness = self.stop_fitness,
                rg=rg, runid=self.get_count_runs(store), 
                workers = self.workers)     
        return ret.x, ret.fun, ret.nfev

class Cma_python(Optimizer):
    """CMA_ES Python implementation."""
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = 31, 
                 guess: Optional[ArrayLike] = None, 
                 stop_fitness: Optional[float] = -np.inf,
                 sdevs: Optional[float] = None, 
                 workers: Optional[int] = None,        
                 update_gap: Optional[int] = None, 
                 normalize: Optional[bool] = True):  
           
        Optimizer.__init__(self, max_evaluations, 'cma py')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.update_gap = update_gap
        self.guess = guess
        self.sdevs = sdevs
        self.normalize = normalize
        self.workers = workers

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[Union[float, ArrayLike, Callable]] = 0.1, 
                 rg: Optional[Generator] = Generator(MT19937()), 
                 store = None) -> Tuple[np.ndarray, float, int]:

        ret = cmaes.minimize(fun, bounds, 
                self.guess if not self.guess is None else guess,
                input_sigma= self.sdevs if not self.sdevs is None else sdevs,
                max_evaluations = self.max_eval_num(store), 
                popsize=self.popsize, 
                stop_fitness = self.stop_fitness,
                rg=rg, runid=self.get_count_runs(store),
                normalize = self.normalize,
                update_gap = self.update_gap,
                workers = self.workers)     
        return ret.x, ret.fun, ret.nfev

class Cma_cpp(Optimizer):
    """CMA_ES C++ implementation."""
   
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = 31, 
                 guess: Optional[ArrayLike] = None, 
                 stop_fitness: Optional[float] = -np.inf,
                 sdevs: Optional[float] = None, 
                 workers: Optional[int] = None,        
                 update_gap: Optional[int] = None, 
                 normalize: Optional[bool] = True,                 
                 delayed_update: Optional[bool] = True,   
                 stop_hist: Optional[int] = None): 
          
        Optimizer.__init__(self, max_evaluations, 'cma cpp')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.stop_hist = stop_hist
        self.guess = guess
        self.sdevs = sdevs
        self.update_gap = update_gap
        self.delayed_update = delayed_update
        self.normalize = normalize
        self.workers = workers

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[Union[float, ArrayLike, Callable]] = 0.1, 
                 rg=Generator(MT19937()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        
        ret = cmaescpp.minimize(fun, bounds,
                self.guess if not self.guess is None else guess,
                input_sigma = self.sdevs if not self.sdevs is None else sdevs,
                max_evaluations =self.max_eval_num(store),
                popsize = self.popsize,
                stop_fitness = self.stop_fitness,
                stop_hist = self.stop_hist,
                rg = rg, runid = self.get_count_runs(store),
		        update_gap = self.update_gap,
                normalize = self.normalize,
                delayed_update = self.delayed_update,
                workers = self.workers)   
        return ret.x, ret.fun, ret.nfev

class Cma_orig(Optimizer):
    """CMA_ES original implementation."""
   
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = 31, 
                 guess: Optional[ArrayLike] = None, 
                 stop_fitness: Optional[float] = -np.inf,
                 sdevs: Optional[float] = None): 
     
        Optimizer.__init__(self, max_evaluations, 'cma orig')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.guess = guess
        self.sdevs = sdevs

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[Union[float, ArrayLike]] = 0.3, 
                 rg: Optional[Generator] = Generator(MT19937()), 
                 store = None) -> Tuple[np.ndarray, float, int]:

        lower = bounds.lb
        upper = bounds.ub
        guess = self.guess if not self.guess is None else guess
        if guess is None:
            guess = rg.uniform(lower, upper)
        max_evaluations = self.max_eval_num(store)   
        input_sigma= self.sdevs if not self.sdevs is None else sdevs
        try:
            import cma
        except ImportError as e:
            raise ImportError("Please install CMA (pip install cma)") 
        try: 
            es = cma.CMAEvolutionStrategy(guess, 0.1,  {'bounds': [lower, upper], 
                                                             'typical_x': guess,
                                                             'scaling_of_variables': scale(lower, upper),
                                                             'popsize': self.popsize,
                                                             'CMA_stds': input_sigma,
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

class Cma_lw(Optimizer):
    """CMA lightweight Python implementation. See https://github.com/CyberAgentAILab/cmaes """
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = 31, 
                 guess: Optional[ArrayLike] = None, 
                 stop_fitness: Optional[float] = -np.inf,
                 sdevs: Optional[Union[float, ArrayLike]] = None, 
                 workers: Optional[int] = None):
      
        Optimizer.__init__(self, max_evaluations, 'cma_lw')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.guess = guess
        self.sdevs = sdevs
        self.workers = workers

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[Union[float, ArrayLike]] = 0.3, 
                 rg: Optional[Generator] = Generator(MT19937()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        
        try:
            import cmaes
        except ImportError as e:
            raise ImportError("Please install cmaes (pip install cmaes)") 

        if guess is None:
            guess = self.guess
        if guess is None:    
            guess = rg.uniform(bounds.lb, bounds.ub)
        bds = np.array([t for t in zip(bounds.lb, bounds.ub)])
        seed = int(rg.uniform(0, 2**32 - 1))
        optimizer = cmaes.CMA(mean=guess, sigma=np.mean(sdevs), bounds=bds, seed=seed, population_size=self.popsize)
        best_y = np.inf
        evals = 0
        fun = serial(fun) if (self.workers is None or self.workers <= 1) else parallel(fun, self.workers)  
        while evals < self.max_evaluations and not optimizer.should_stop():
            xs = [optimizer.ask() for _ in range(optimizer.population_size)]
            ys = fun(xs)
            solutions = []
            for i in range(optimizer.population_size):
                x = xs[i]
                y = ys[i]
                solutions.append((x, y))
                if y < best_y:
                    best_y = y
                    best_x = x
            optimizer.tell(solutions)
            evals += optimizer.population_size           
        if isinstance(fun, parallel):
            fun.stop()
        return best_x, best_y, evals

class Cma_awm(Optimizer):
    """CMA awm Python implementation. See https://github.com/CyberAgentAILab/cmaes """
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = 31, 
                 guess: Optional[ArrayLike] = None, 
                 stop_fitness: Optional[float] = -np.inf,
                 sdevs: Optional[Union[float, ArrayLike]] = None, 
                 continuous_space = None, 
                 discrete_space = None, 
                 workers: Optional[int] = None):
               
        Optimizer.__init__(self, max_evaluations, 'cma_awm')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.guess = guess
        self.sdevs = sdevs
        self.workers = workers
        self.continuous_space = continuous_space
        self.discrete_space = discrete_space

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[Union[float, ArrayLike]] = 0.3, 
                 rg: Optional[Generator] = Generator(MT19937()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        try:
            import cmaes
        except ImportError as e:
            raise ImportError("Please install cmaes (pip install cmaes)") 
              
        if guess is None:
            guess = self.guess
        if guess is None:    
            guess = rg.uniform(bounds.lb, bounds.ub)
        seed = int(rg.uniform(0, 2**32 - 1))
        optimizer = cmaes.CMAwM(mean=guess, sigma=np.mean(sdevs),       
                         continuous_space=self.continuous_space,  
                         discrete_space=self.discrete_space, 
                         seed=seed, population_size=self.popsize)
        best_y = 1E99
        evals = 0
        fun = serial(fun) if (self.workers is None or self.workers <= 1) else parallel(fun, self.workers)  
        while evals < self.max_evaluations and not optimizer.should_stop():
            asks = [optimizer.ask() for _ in range(optimizer.population_size)]
            ys = fun([x[0] for x in asks])
            solutions = []
            for i in range(optimizer.population_size):
                x = asks[i][1]
                y = ys[i]
                solutions.append((x, y))
                if y < best_y:
                    best_y = y
                    best_x = x
            optimizer.tell(solutions)
            evals += optimizer.population_size           
        if isinstance(fun, parallel):
            fun.stop()
        return best_x, best_y, evals

class Cma_sep(Optimizer):
    """CMA sep Python implementation. See https://github.com/CyberAgentAILab/cmaes """
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = 31, 
                 guess: Optional[ArrayLike] = None, 
                 stop_fitness: Optional[float] = -np.inf,
                 sdevs: Optional[Union[float, ArrayLike]] = None, 
                 workers: Optional[int] = None):
      
        Optimizer.__init__(self, max_evaluations, 'cma_sep')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.guess = guess
        self.sdevs = sdevs
        self.workers = workers

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[Union[float, ArrayLike]] = 0.3, 
                 rg: Optional[Generator] = Generator(MT19937()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        try:
            import cmaes
        except ImportError as e:
            raise ImportError("Please install cmaes (pip install cmaes)") 

        if guess is None:
            guess = self.guess
        if guess is None:    
            guess = rg.uniform(bounds.lb, bounds.ub)
        bds = np.array([t for t in zip(bounds.lb, bounds.ub)])
        seed = int(rg.uniform(0, 2**32 - 1))
        optimizer = cmaes.SepCMA(mean=guess, sigma=np.mean(sdevs), bounds=bds, seed=seed, population_size=self.popsize)
        best_y = np.inf
        evals = 0
        fun = serial(fun) if (self.workers is None or self.workers <= 1) else parallel(fun, self.workers)  
        while evals < self.max_evaluations and not optimizer.should_stop():
            xs = [optimizer.ask() for _ in range(optimizer.population_size)]
            ys = fun(xs)
            solutions = []
            for i in range(optimizer.population_size):
                x = xs[i]
                y = ys[i]
                solutions.append((x, y))
                if y < best_y:
                    best_y = y
                    best_x = x
            optimizer.tell(solutions)
            evals += optimizer.population_size          
        if isinstance(fun, parallel):
            fun.stop()
        return best_x, best_y, evals
      
class De_cpp(Optimizer):
    """Differential Evolution C++ implementation."""
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = None, 
                 stop_fitness: Optional[float] = -np.inf,
                 keep: Optional[int] = 200, 
                 f: Optional[float] = 0.5, 
                 cr: Optional[float] = 0.9, 
                 ints: Optional[ArrayLike] = None, 
                 workers: Optional[int] = None):
      
        Optimizer.__init__(self, max_evaluations, 'de cpp')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.keep = keep
        self.f = f
        self.cr = cr
        self.ints = ints
        self.workers = workers

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = None,  # ignored
                 rg: Optional[Generator] = Generator(MT19937()), 
                 store = None) -> Tuple[np.ndarray, float, int]:

        ret = decpp.minimize(fun, None, bounds, 
                popsize=self.popsize, 
                max_evaluations = self.max_eval_num(store), 
                stop_fitness = self.stop_fitness,
                keep = self.keep, f = self.f, cr = self.cr, ints=self.ints,
                rg=rg, runid = self.get_count_runs(store), 
                workers = self.workers)
        return ret.x, ret.fun, ret.nfev

class De_python(Optimizer):
    """Differential Evolution Python implementation."""
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = None,
                 stop_fitness: Optional[float] = -np.inf,
                 keep: Optional[int] = 200, 
                 f: Optional[float] = 0.5, 
                 cr: Optional[float] = 0.9, 
                 ints: Optional[ArrayLike] = None, 
                 workers: Optional[int] = None):
             
        Optimizer.__init__(self, max_evaluations, 'de py')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.keep = keep
        self.f = f
        self.cr = cr
        self.ints = ints
        self.workers = workers

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = None, # ignored
                 rg: Optional[Generator] = Generator(MT19937()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        
        ret = de.minimize(fun, None, 
                bounds, self.popsize, self.max_eval_num(store),
                stop_fitness = self.stop_fitness,
                keep = self.keep, f = self.f, cr = self.cr, ints=self.ints,
                rg=rg, workers = self.workers)
        return ret.x, ret.fun, ret.nfev

class Cma_ask_tell(Optimizer):
    """CMA ask tell implementation."""
    
    def __init__(self, max_evaluations=50000,
                 popsize = 31, guess=None, stop_fitness = -np.inf, sdevs = None):        
        Optimizer.__init__(self, max_evaluations, 'cma at')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.guess = guess
        self.sdevs = sdevs

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = None, # ignored
                 rg: Optional[Generator] = Generator(MT19937()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        
        es = cmaes.Cmaes(bounds,
                popsize = self.popsize, 
                input_sigma = self.sdevs if not self.sdevs is None else sdevs, 
                rg = rg)       
        iters = self.max_eval_num(store) // self.popsize
        evals = 0
        for j in range(iters):
            xs = es.ask()
            ys = [fun(x) for x in xs]
            evals += len(xs)
            stop = es.tell(ys)
            if stop != 0:
                break 
        return es.best_x, es.best_value, evals

class De_ask_tell(Optimizer):
    """Differential Evolution ask tell implementation."""
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = None,
                 stop_fitness: Optional[float] = -np.inf,
                 keep: Optional[int] = 200, 
                 f: Optional[float] = 0.5, 
                 cr: Optional[float] = 0.9):        
        Optimizer.__init__(self, max_evaluations, 'de at')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.keep = keep
        self.f = f
        self.cr = cr

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = None, # ignored
                 rg: Optional[Generator] = Generator(MT19937()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        
        dim = len(bounds.lb)
        popsize = 31 if self.popsize is None else self.popsize
        es = de.DE(dim, bounds, popsize = popsize, rg = rg, keep = self.keep, F = self.f, Cr = self.cr)  
        es.fun = fun  #remove
        max_evals = self.max_eval_num(store)
        while es.evals < max_evals:
            xs = es.ask()
            ys = [fun(x) for x in xs]
            stop = es.tell(ys, xs)
            if stop != 0:
                break 
        return es.best_x, es.best_value, es.evals

class random_search(Optimizer):
    """Random search."""
   
    def __init__(self, max_evaluations=50000):        
        Optimizer.__init__(self, max_evaluations, 'random')
 
    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = None, # ignored
                 rg: Optional[Generator] = Generator(MT19937()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        
        dim, x_min, y_min = len(bounds.lb), None, None
        max_chunk_size = 1 + 4e4 / dim
        evals = self.max_eval_num(store)
        budget = evals
        while budget > 0:
            chunk = int(max([1, min([budget, max_chunk_size])]))
            X = rg.uniform(bounds.lb, bounds.ub, size = [chunk, dim])
            F = [fun(x) for x in X]
            index = np.argmin(F) if len(F) else None
            if index is not None and (y_min is None or F[index] < y_min):
                x_min, y_min = X[index], F[index]
            budget -= chunk
        return x_min, y_min, evals

class LDe_cpp(Optimizer):
    """Local Differential Evolution C++ implementation."""
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = None,
                 stop_fitness: Optional[float] = -np.inf,
                 keep: Optional[int] = 200, 
                 f: Optional[float] = 0.5, 
                 cr: Optional[float] = 0.9, 
                 guess: Optional[ArrayLike] = None, 
                 sdev: Optional[Union[float, ArrayLike, Callable]] = None, 
                 ints: Optional[ArrayLike] = None):
               
        Optimizer.__init__(self, max_evaluations, 'lde cpp')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.keep = keep
        self.f = f
        self.cr = cr
        self.guess = guess
        self.sdevs = sdev
        self.ints = ints
        
    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = None, # ignored
                 rg: Optional[Generator] = Generator(MT19937()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        
        ret = ldecpp.minimize(fun, bounds, 
                self.guess if not self.guess is None else guess, 
                self.sdevs if not self.sdevs is None else sdevs,
                popsize=self.popsize, 
                max_evaluations = self.max_eval_num(store), 
                stop_fitness = self.stop_fitness,
                keep = self.keep, f = self.f, cr = self.cr, ints = self.ints,
                rg=rg, runid = self.get_count_runs(store))
        return ret.x, ret.fun, ret.nfev

class GCLDE_cpp(Optimizer):
    """GCL-Differential Evolution C++ implementation."""
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = None,
                 stop_fitness: Optional[float] = -np.inf,
                 pbest: Optional[float] = 0.7, 
                 f: Optional[float] = 0.5, 
                 cr: Optional[float] = 0.9, 
                 workers: Optional[int] = None):
                
        Optimizer.__init__(self, max_evaluations, 'gclde cpp')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.pbest = pbest
        self.f = f
        self.cr = cr
        self.workers = workers

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = None, # ignored
                 rg: Optional[Generator] = Generator(MT19937()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        
        ret = gcldecpp.minimize(fun, None, bounds, 
                popsize=self.popsize, 
                max_evaluations = self.max_eval_num(store), 
                stop_fitness = self.stop_fitness,
                pbest = self.pbest, f0 = self.f, cr0 = self.cr,
                rg=rg, runid = self.get_count_runs(store),
                workers = self.workers)
        return ret.x, ret.fun, ret.nfev

class LCLDE_cpp(Optimizer):
    """LCL-Differential Evolution C++ implementation."""
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = None,
                 stop_fitness: Optional[float] = -np.inf,
                 pbest: Optional[float] = 0.7, 
                 f: Optional[float] = 0.5, 
                 cr: Optional[float] = 0.9, 
                 guess: Optional[ArrayLike] = None, 
                 sdev: Optional[Union[float, ArrayLike, Callable]] = None, 
                 workers: Optional[int] = None):                 
       
        Optimizer.__init__(self, max_evaluations, 'lclde cpp')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.pbest = pbest
        self.f = f
        self.cr = cr
        self.workers = workers
        self.guess = guess
        self.sdevs = sdev

    def minimize(self, 
                fun: Callable[[ArrayLike], float], 
                bounds: Optional[Bounds], 
                guess: Optional[ArrayLike] = None, 
                sdevs: Optional[float] = 0.3, # ignored
                rg=Generator(MT19937()), 
                store = None) -> Tuple[np.ndarray, float, int]:                 
        
        ret = lcldecpp.minimize(fun, bounds, 
                self.guess if not self.guess is None else guess, 
                self.sdevs if not self.sdevs is None else sdevs,
                popsize=self.popsize, 
                max_evaluations = self.max_eval_num(store), 
                stop_fitness = self.stop_fitness,
                pbest = self.pbest, f0 = self.f, cr0 = self.cr,
                rg=rg, runid = self.get_count_runs(store),
                workers = self.workers)

        return ret.x, ret.fun, ret.nfev
    
class Da_cpp(Optimizer):
    """Dual Annealing C++ implementation."""
    
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 stop_fitness: Optional[float] = -np.inf,
                 use_local_search: Optional[bool] = True,
                 guess: Optional[ArrayLike] = None):                 
                
        Optimizer.__init__(self, max_evaluations, 'da cpp',)
        self.stop_fitness = stop_fitness
        self.use_local_search = use_local_search
        self.guess = guess
 
    def minimize(self, 
                fun: Callable[[ArrayLike], float], 
                bounds: Optional[Bounds], 
                guess: Optional[ArrayLike] = None, 
                sdevs: Optional[float] = None, # ignored
                rg=Generator(MT19937()), 
                store = None) -> Tuple[np.ndarray, float, int]:

        ret = dacpp.minimize(fun, bounds, 
                             self.guess if guess is None else guess,
                            max_evaluations = self.max_eval_num(store), 
                            use_local_search = self.use_local_search,
                            rg=rg, runid = self.get_count_runs(store))
        return ret.x, ret.fun, ret.nfev

class Csma_cpp(Optimizer):
    """SCMA C++ implementation."""
   
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = None,
                 guess: Optional[ArrayLike] = None, 
                 stop_fitness: Optional[float] = -np.inf,
                 sdevs: Optional[float] = None):
                
        Optimizer.__init__(self, max_evaluations, 'scma cpp')
        self.popsize = popsize
        self.stop_fitness = stop_fitness
        self.guess = guess
        self.sdevs = sdevs

    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = 0.16, 
                 rg=Generator(MT19937()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        
        ret = csmacpp.minimize(fun, bounds, 
                self.guess if guess is None else guess,
                self.sdevs if not self.sdevs is None else sdevs,
                max_evaluations = self.max_eval_num(store), 
                stop_fitness = self.stop_fitness,
                rg=rg, runid = self.get_count_runs(store))     
        return ret.x, ret.fun, ret.nfev

class Bite_cpp(Optimizer):
    """Bite C++ implementation."""
   
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 guess: Optional[ArrayLike] = None, 
                 stop_fitness: Optional[float] = -np.inf,                
                 M: Optional[int] = None,
                 popsize: Optional[int] = None,
                 stall_criterion: Optional[int] = None):
                
        Optimizer.__init__(self, max_evaluations, 'bite cpp')
        self.guess = guess
        self.stop_fitness = stop_fitness
        self.M = 1 if M is None else M 
        self.popsize = 0 if popsize is None else popsize 
        self.stall_criterion = 0 if stall_criterion is None else stall_criterion 

    def minimize(self,
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = None, # ignored
                 rg=Generator(MT19937()), 
                 store = None) -> Tuple[np.ndarray, float, int]:

        ret = bitecpp.minimize(fun, bounds, 
                self.guess if guess is None else guess,
                max_evaluations = self.max_eval_num(store), 
                stop_fitness = self.stop_fitness, M = self.M, popsize = self.popsize, 
                stall_criterion = self.stall_criterion,
                rg=rg, runid = self.get_count_runs(store))     
        return ret.x, ret.fun, ret.nfev
        
class Dual_annealing(Optimizer):
    """scipy dual_annealing."""
 
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 use_local_search: Optional[bool] = True):
                
        Optimizer.__init__(self, max_evaluations, 'scipy da')
        self.no_local_search = not use_local_search
 
    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = None, # ignored
                 rg=Generator(MT19937()), 
                 store = None) -> Tuple[np.ndarray, float, int]:                 
        ret = dual_annealing(fun, bounds=list(zip(bounds.lb, bounds.ub)),
            maxfun = self.max_eval_num(store), 
            no_local_search = self.no_local_search,
            x0=guess,
            seed = int(rg.uniform(0, 2**32 - 1)))
        return ret.x, ret.fun, ret.nfev

class Differential_evolution(Optimizer):
    """scipy differential_evolution."""
 
    def __init__(self, 
                 max_evaluations: Optional[int] = 50000,
                 popsize: Optional[int] = 31):
                
        Optimizer.__init__(self, max_evaluations, 'scipy de')
        self.popsize = popsize
 
    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = None, # ignored
                 rg=Generator(MT19937()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        
        popsize = self.popsize 
        maxiter = int(self.max_eval_num(store) / (popsize * len(bounds.lb)) - 1)
        ret = differential_evolution(fun, bounds=bounds, maxiter=maxiter,
                                      seed = int(rg.uniform(0, 2**32 - 1)))
        return ret.x, ret.fun, ret.nfev

class CheckBounds(object):
    
    def __init__(self, bounds: Bounds):
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
         
    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = None, # ignored
                 rg=Generator(MT19937()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
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
 
    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = None, # ignored
                 rg=Generator(MT19937()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        
        if guess is None:
            guess = rg.uniform(bounds.lb, bounds.ub)
        ret = minimize(fun, x0=guess, bounds=bounds)
        return ret.x, ret.fun, ret.nfev
 
class Shgo(Optimizer):
    """scipy shgo."""

    def __init__(self, max_evaluations=50000, store=None):        
        Optimizer.__init__(self, max_evaluations, 'scipy shgo')
 
    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = None, # ignored
                 rg=Generator(MT19937()), 
                 store = None) -> Tuple[np.ndarray, float, int]:
        
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
 
    def minimize(self, 
                 fun: Callable[[ArrayLike], float], 
                 bounds: Optional[Bounds], 
                 guess: Optional[ArrayLike] = None, 
                 sdevs: Optional[float] = None, # ignored
                 rg=Generator(MT19937()), 
                 store = None) -> Tuple[np.ndarray, float, int]:

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
    
