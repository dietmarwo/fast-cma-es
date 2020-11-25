# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

""" Implements a Sigma Adaptation Evolution Strategy 
    similar to CMA-ES, but mainly focuses on sigma adaptation.
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
from fcmaes.cmaescpp import libcmalib, freemem
from fcmaes.dacpp import call_back_type

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun, 
             bounds=None, 
             x0=None, 
             input_sigma = 0.166, 
             popsize = 0, 
             max_evaluations = 100000, 
             stop_fittness = None, 
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
    input_sigma : ndarray, shape (n,) or scalar
        Initial step size for each dimension.
    popsize = int, optional
        CMA-ES population size.
    max_evaluations : int, optional
        Forced termination after ``max_evaluations`` function evaluations.
    stop_fittness : float, optional 
         Limit for fitness value. If reached minimize terminates.
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
    if np.ndim(input_sigma) == 0:
        input_sigma = [input_sigma] * n
    if stop_fittness is None:
        stop_fittness = -math.inf   
    array_type = ct.c_double * n 
    c_callback = call_back_type(callback(fun))
    try:
        res = optimizeCsma_C(runid, c_callback, n, int(rg.uniform(0, 2**32 - 1)), 
                           array_type(*guess), array_type(*lower), array_type(*upper), 
                           array_type(*input_sigma), max_evaluations, stop_fittness, popsize)

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
  
optimizeCsma_C = libcmalib.optimizeCsma_C
optimizeCsma_C.argtypes = [ct.c_long, call_back_type, ct.c_int, ct.c_int, \
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
            ct.POINTER(ct.c_double), ct.c_int, ct.c_double, ct.c_int]

optimizeCsma_C.restype = ct.POINTER(ct.c_double)         
