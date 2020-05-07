# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

import sys
import math
import numpy as np
import ctypes as ct
import multiprocessing as mp
from scipy.optimize import Bounds

class Wrapper(object):
    """thread safe wrapper for objective function monitoring evaluation count and optimization result."""
   
    def __init__(self, fun, dim):
        self.func = fun   
        self.statMutex = mp.Lock()    
        self.best_x = mp.RawArray(ct.c_double, dim)
        self.best_y = mp.RawValue(ct.c_double, sys.float_info.max) 
        self.count = mp.RawValue(ct.c_int, 0) 
    
    def eval(self, x):
        y = self.func(x)
        with self.statMutex:
            if y < self.best_y.value:
                self.best_y.value = y
                self.best_x[:] = x[:]
            self.count.value += 1
        return y
    
    def get_best_x(self):
        return self.best_x[:]

    def get_best_y(self):
        return self.best_y.value
    
    def get_count(self):
        return self.count.value
    
class Rosen(object):
    
    def __init__(self, dim):    
        _testfun.__init__(self, 'rosen', _rosen, [-5]*dim, [5]*dim)
  
class Rastrigin(object):
    
    def __init__(self, dim):    
        _testfun.__init__(self, 'rastrigin', _rastrigin, [-5.12]*dim, [5.12]*dim)

class Eggholder(object):
    
    def __init__(self):    
        _testfun.__init__(self, 'eggholder', _eggholder, [-512]*2, [512]*2)

class RastriginMean(object):
    
    def __init__(self, dim, n):    
        fun = lambda x: _rastrigin_mean(x, n)
        _testfun.__init__(self, 'rastrigin_mean', fun, [-5.12]*dim, [5.12]*dim)

class _testfun(object):
    def __init__(self, name, fun, lower, upper):    
        self.name = name 
        self.fun = fun
        self.bounds = Bounds(lower, upper)
        self.wrapper = Wrapper(self.fun, len(lower))   
    
def _rosen(xs, alpha=1e2):
    """Rosenbrock test objective function."""
    xs = [xs] if np.isscalar(xs[0]) else xs 
    xs = np.asarray(xs)
    f = [sum(alpha * (x[:-1]**2 - x[1:])**2 + (1. - x[:-1])**2) for x in xs]
    return f if len(f) > 1 else f[0]  # 1-element-list into scalar

def _rastrigin(x):
    """Rastrigin test objective function."""
    dim = len(x)
    x = np.asarray(x)
    return 10.0*dim + sum(x*x - 10.0*np.cos(2.0*math.pi*x))

def _modify(x, delta):
    dim = len(x)
    modified = np.asarray(x) + delta * np.random.randn(dim)
    return modified.tolist()

def _rastrigin_mean(x, n):
    """mean value of n Rastrigin function calls."""
    delta = 0.001
    sumy = 0
    for i in range(n):
        sumy += _rastrigin(_modify(x, delta))
    return sumy / n

def _eggholder(x):
    """Eggholder test objective function."""
    return (-(x[1] + 47.0)
                        * np.sin(np.sqrt(abs(x[0]/2.0 + (x[1] + 47.0))))
                        - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47.0))))
                        )