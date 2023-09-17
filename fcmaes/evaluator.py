# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.
from __future__ import annotations

""" Parallel objective function evaluator.
    Uses pipes to avoid re-spawning new processes for each eval_parallel call. 
    the objective function is distributed once to all processes and
    reused for all eval_parallel calls. Evaluator(fun) needs to be stopped after the
    whole optimization is finished to avoid a resource leak.
"""

from multiprocessing import Process, Pipe
import multiprocessing as mp
import ctypes as ct
import numpy as np
import sys, math, os  
from loguru import logger
from typing import Optional, Callable, Tuple
from numpy.typing import ArrayLike

pipe_limit = 64 # higher values can cause issues

def is_debug_active():
    try: # nasty but currently there is no other way
        for handler in logger._core.handlers.values():
            if handler['level'].no <= logger.level("DEBUG").no:
                return True
    except Exception as ex:   
        pass
    return False

def eval_parallel(xs: ArrayLike, 
                  evaluator: Evaluator):
    popsize = len(xs)
    ys = np.empty(popsize)
    i0 = 0
    i1 = min(popsize, pipe_limit)
    while True:
        _eval_parallel_segment(xs, ys, i0, i1, evaluator)
        if i1 >= popsize:
            break;
        i0 += pipe_limit
        i1 = min(popsize, i1 + pipe_limit)
    return ys
        
def eval_parallel_mo(xs: ArrayLike, 
                     evaluator: Evaluator, 
                     nobj: int):
    popsize = len(xs)
    ys = np.empty((popsize,nobj))
    i0 = 0
    i1 = min(popsize, pipe_limit)
    while True:
        _eval_parallel_segment(xs, ys, i0, i1, evaluator)
        if i1 >= popsize:
            break;
        i0 += pipe_limit
        i1 = min(popsize, i1 + pipe_limit)
    return ys
        
class Evaluator(object):
       
    def __init__(self, 
                 fun: Callable[[ArrayLike], float], # objective function
                ):   
        self.fun = fun 
        self.pipe = Pipe()
        self.read_mutex = mp.Lock() 
        self.write_mutex = mp.Lock() 
            
    def start(self, workers: Optional[int] = mp.cpu_count()):
        self.workers = workers
        self.proc=[Process(target=_evaluate, args=(self.fun, 
                self.pipe, self.read_mutex, self.write_mutex)) for _ in range(workers)]
        [p.start() for p in self.proc]
        
    def stop(self): # shutdown all workers 
        for _ in range(self.workers):
            self.pipe[0].send(None)
        [p.join() for p in self.proc]    
        for p in self.pipe:
            p.close()

def _eval_parallel_segment(xs, ys, i0, i1, evaluator):
    for i in range(i0, i1):
        evaluator.pipe[0].send((i, xs[i]))
    for _ in range(i0, i1):        
        i, y = evaluator.pipe[0].recv()
        ys[i] = y
    return ys

def _evaluate(fun, pipe, read_mutex, write_mutex): # worker
    while True:
        with read_mutex:
            msg = pipe[1].recv() # Read from the input pipe
        if msg is None: 
            break # shutdown worker
        try:
            i, x = msg
            y = fun(x)
        except Exception as ex:
            y =  sys.float_info.max
        with write_mutex:            
            pipe[1].send((i, y)) # Send result

def _check_bounds(bounds, guess, rg):
    if bounds is None and guess is None:
        raise ValueError('either guess or bounds need to be defined')
    if bounds is None:
        return None, None, np.asarray(guess)
    if guess is None:
        guess = rg.uniform(bounds.lb, bounds.ub)
    return np.asarray(bounds.lb), np.asarray(bounds.ub), np.asarray(guess)

def _get_bounds(dim, bounds, guess, rg):
    if bounds is None:
        if guess is None:
            guess = np.asarray(np.zeros(dim))
        return None, None, guess
    if guess is None:
        guess = rg.uniform(bounds.lb, bounds.ub)
    return np.asarray(bounds.lb), np.asarray(bounds.ub), np.asarray(guess)

class _fitness(object):
    """wrapper around the objective function, scales relative to boundaries."""
     
    def __init__(self, fun, lower, upper, normalize = None):
        self.fun = fun
        self.evaluation_counter = 0
        self.lower = lower
        self.normalize = False
        if not (lower is None or normalize is None):
            self.normalize = normalize
        if not lower is None:
            self.upper = upper
            self.scale = 0.5 * (upper - lower)
            self.typx = 0.5 * (upper + lower)

    def values(self, Xs): #enables parallel evaluation
        values = self.fun(Xs)
        self.evaluation_counter += len(Xs)
        return np.array(values)
    
    def closestFeasible(self, X):
        if self.lower is None:
            return X    
        else:
            if self.normalize:
                return np.clip(X, -1.0, 1.0)
            else:
                return np.clip(X, self.lower, self.upper)

    def encode(self, X):
        if self.normalize:
            return (X - self.typx) / self.scale
        else:
            return X
   
    def decode(self, X):
        if self.normalize:
            return (X * self.scale) + self.typx
        else:
            return X
         
def serial(fun):
    """Convert an objective function for serial execution for cmaes.minimize.
    
    Parameters
    ----------
    fun : objective function mapping a list of float arguments to a float value

    Returns
    -------
    out : function
        A function mapping a list of lists of float arguments to a list of float values
        by applying the input function in a loop."""
  
    return lambda xs : [_tryfun(fun, x) for x in xs]
        
def _func_serial(fun, num, pid, xs, ys):
    for i in range(pid, len(xs), num):
        ys[i] = _tryfun(fun, xs[i])

def _tryfun(fun, x):
    try:
        fit = fun(x)
        return fit if math.isfinite(fit) else sys.float_info.max
    except Exception:
        return sys.float_info.max
    
class parallel(object):
    """Convert an objective function for parallel execution for cmaes.minimize.
    
    Parameters
    ----------
    fun : objective function mapping a list of float arguments to a float value.
   
    represents a function mapping a list of lists of float arguments to a list of float values
    by applying the input function using parallel processes. stop needs to be called to avoid
    a resource leak"""
        
    def __init__(self, 
                 fun: Callable[[ArrayLike], float], 
                 workers: Optional[int] = mp.cpu_count()):
        self.evaluator = Evaluator(fun)
        self.evaluator.start(workers)
    
    def __call__(self, xs: ArrayLike) -> np.ndarray:
        return eval_parallel(xs, self.evaluator)

    def stop(self):
        self.evaluator.stop()

class parallel_mo(object):
        
    def __init__(self, 
                 fun: Callable[[ArrayLike], ArrayLike], 
                 nobj: int, 
                 workers: Optional[int] = mp.cpu_count()):
        self.nobj = nobj
        self.evaluator = Evaluator(fun)
        self.evaluator.start(workers)
    
    def __call__(self, xs: ArrayLike) -> np.ndarray:
        return eval_parallel_mo(xs, self.evaluator, self.nobj)

    def stop(self):
        self.evaluator.stop()

class callback(object):
    
    def __init__(self, fun: Callable[[ArrayLike], float]):
        self.fun = fun
    
    def __call__(self, n: int, x: ArrayLike) -> float:
        try:
            fit = self.fun(np.fromiter((x[i] for i in range(n)), dtype=float))
            return fit if math.isfinite(fit) else sys.float_info.max
        except Exception as ex:
            return sys.float_info.max
        
class callback_so(object):
    
    def __init__(self, 
                 fun: Callable[[ArrayLike], float], 
                 dim: int, 
                 is_terminate: Optional[Callable[[ArrayLike, float], bool]] = None):
        self.fun = fun
        self.dim = dim
        self.nobj = 1
        self.is_terminate = is_terminate
    
    def __call__(self, dim, x, y):
        try:
            arrTypeX = ct.c_double*(self.dim)
            xaddr = ct.addressof(x.contents)
            xbuf = np.frombuffer(arrTypeX.from_address(xaddr))
            arrTypeY = ct.c_double*(self.nobj)
            yaddr = ct.addressof(y.contents)   
            ybuf = np.frombuffer(arrTypeY.from_address(yaddr))  
            fit = self.fun(xbuf)
            ybuf[0] = fit if math.isfinite(fit) else sys.float_info.max
            return False if self.is_terminate is None else self.is_terminate(xbuf, ybuf) 
        except Exception as ex:
            print (ex)
            return False

class callback_mo(object):
    
    def __init__(self, 
                 fun: Callable[[ArrayLike], ArrayLike], 
                 dim: int, 
                 nobj: int, 
                 is_terminate: Optional[bool] = None):
        self.fun = fun
        self.dim = dim
        self.nobj = nobj
        self.is_terminate = is_terminate
    
    def __call__(self, dim: int, x, y):
        try:
            arrTypeX = ct.c_double*(dim)
            xaddr = ct.addressof(x.contents)
            xbuf = np.frombuffer(arrTypeX.from_address(xaddr))
            arrTypeY = ct.c_double*(self.nobj)
            yaddr = ct.addressof(y.contents)   
            ybuf = np.frombuffer(arrTypeY.from_address(yaddr))  
            ybuf[:] = self.fun(xbuf)[:]
            return False if self.is_terminate is None else self.is_terminate(xbuf, ybuf) 
        except Exception as ex:
            print (ex)
            return False

class callback_par(object):
    
    def __init__(self, 
                 fun: Callable[[ArrayLike], float], 
                 parfun: Callable[[ArrayLike], ArrayLike]):
        self.fun = fun
        self.parfun = parfun
    
    def __call__(self, popsize, n, xs_, ys_):
        try:
            arrType = ct.c_double*(popsize*n)
            addr = ct.addressof(xs_.contents)
            xall = np.frombuffer(arrType.from_address(addr))
            
            if self.parfun is None:
                for p in range(popsize):
                    ys_[p] = self.fun(xall[p*n : (p+1)*n])
            else:    
                xs = []
                for p in range(popsize):
                    x = xall[p*n : (p+1)*n]
                    xs.append(x)
                ys = self.parfun(xs)
                for p in range(popsize):
                    ys_[p] = ys[p]
        except Exception as ex:
            print (ex)

basepath = os.path.dirname(os.path.abspath(__file__))

try: 
    if sys.platform.startswith('linux'):
        libcmalib = ct.cdll.LoadLibrary(basepath + '/lib/libacmalib.so')  
    elif 'mac' in sys.platform or 'darwin' in sys.platform:
        libcmalib = ct.cdll.LoadLibrary(basepath + '/lib/libacmalib.dylib')  
    else:
        os.environ['PATH'] = (basepath + '/lib') + os.pathsep + os.environ['PATH']
        libcmalib = ct.cdll.LoadLibrary(basepath + '/lib/libacmalib.dll')
except Exception as ex:
    libcmalib = None
    
mo_call_back_type = ct.CFUNCTYPE(ct.c_bool, ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double))
  
call_back_type = ct.CFUNCTYPE(ct.c_double, ct.c_int, ct.POINTER(ct.c_double))  

call_back_par = ct.CFUNCTYPE(None, ct.c_int, ct.c_int, \
                                  ct.POINTER(ct.c_double), ct.POINTER(ct.c_double))  

