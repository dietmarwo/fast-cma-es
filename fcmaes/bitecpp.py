# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

""" Implements a stochastic non-linear
    bound-constrained derivative-free optimization method.
    Description is available at https://github.com/avaneev/biteopt
"""

import sys
import os
import math
import ctypes as ct
import numpy as np
from numpy.random import MT19937, Generator
from scipy.optimize import OptimizeResult
from fcmaes.cmaes import _check_bounds
from fcmaes.cmaescpp import callback, libcmalib, freemem
from fcmaes.dacpp import call_back_type

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun, 
             bounds=None, 
             x0=None, 
             popsize = 0, 
             max_evaluations = 100000, 
             stop_fitness = None, 
             M = 1,
             rg = Generator(MT19937()),
             runid=0):   
    """Minimization of a scalar function of one or more variables using a 
    C++ SCMA implementation called via ctypes.
     
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
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where 'n' is the number of independent variables.  
    popsize = int, optional
        CMA-ES population size.
    max_evaluations : int, optional
        Forced termination after ``max_evaluations`` function evaluations.
    stop_fitness : float, optional 
         Limit for fitness value. If reached minimize terminates.
    M : int, optional 
        Depth to use, 1 for plain CBiteOpt algorithm, >1 for CBiteOptDeep. Expected range is [1; 36].
    rg = numpy.random.Generator, optional
        Random generator for creating random guesses.
    runid : int, optional
        id used to identify the run for debugging / logging. 
           
    Returns
    -------
    res : scipy.OptimizeResult
        The optimization result is represented as an ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, 
        ``fun`` the best function value, 
        ``nfev`` the number of function evaluations,
        ``nit`` the number of CMA-ES iterations, 
        ``status`` the stopping critera and
        ``success`` a Boolean flag indicating if the optimizer exited successfully. """
    
    lower, upper, guess = _check_bounds(bounds, x0, rg)      
    n = guess.size   
    if lower is None:
        lower = [0]*n
        upper = [0]*n
    if stop_fitness is None:
        stop_fitness = -math.inf   
    array_type = ct.c_double * n 
    c_callback = call_back_type(callback(fun))
    try:
        res = optimizeBite_C(runid, c_callback, n, int(rg.uniform(0, 2**32 - 1)), 
                           array_type(*guess), array_type(*lower), array_type(*upper), 
                           max_evaluations, stop_fitness, popsize, M)

        x = np.array(np.fromiter(res, dtype=np.float64, count=n))
        val = res[n]
        evals = int(res[n+1])
        iterations = int(res[n+2])
        stop = int(res[n+3])
        freemem(res)
        return OptimizeResult(x=x, fun=val, nfev=evals, nit=iterations, status=stop, success=True)
    except Exception as ex:
        return OptimizeResult(x=None, fun=sys.float_info.max, nfev=0, nit=0, status=-1, success=False)

class callback(object):
    
    def __init__(self, fun):
        self.fun = fun
    
    def __call__(self, n, x):
        try:
            fit = self.fun([x[i] for i in range(n)])
            return fit if math.isfinite(fit) else sys.float_info.max
        except Exception:
            return sys.float_info.max

optimizeBite_C = libcmalib.optimizeBite_C
optimizeBite_C.argtypes = [ct.c_long, call_back_type, ct.c_int, ct.c_int, \
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
            ct.c_int, ct.c_double, ct.c_int, ct.c_int]

optimizeBite_C.restype = ct.POINTER(ct.c_double)         


