# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

""" Eigen based implementation of the Harris hawks algorithm.
    Derived from derived from https://github.com/7ossam81/EvoloPy/blob/master/optimizers/HHO.py.
    See DOI: https://doi.org/10.1016/j.future.2019.02.028
"""

import sys
import os
import math
import ctypes as ct
import numpy as np
from numpy.random import MT19937, Generator
from scipy.optimize import OptimizeResult
from fcmaes.cmaescpp import callback

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun, 
             dim,
             bounds = None, 
             popsize = None, 
             max_evaluations = 100000, 
             stop_fittness = None, 
             rg = Generator(MT19937()),
             runid=0):   

    """Minimization of a scalar function of one or more variables using a 
    C++ Harris hawks implementation called via ctypes.
     
    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
            ``fun(x, *args) -> float``
        where ``x`` is an 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    dim : int
        dimension of the argument of the objective function
    bounds : sequence or `Bounds`, optional
        Bounds on variables. There are two ways to specify the bounds:
            1. Instance of the `scipy.Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.
    max_evaluations : int, optional
        Forced termination after ``max_evaluations`` function evaluations.
    stop_fittness : float, optional 
         Limit for fitness value. If reached minimize terminates.
    rg = numpy.random.Generator, optional
        Random generator for creating random guesses.
    runid : int, optional
        id used to identify the optimization run. 
            
    Returns
    -------
    res : scipy.OptimizeResult
        The optimization result is represented as an ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, 
        ``fun`` the best function value, 
        ``nfev`` the number of function evaluations,
        ``nit`` the number of iterations,
        ``success`` a Boolean flag indicating if the optimizer exited successfully. """
                
    lower = np.asarray(bounds.lb)
    upper = np.asarray(bounds.ub)
    n = dim  
    if popsize is None:
        popsize = 31
    if lower is None:
        lower = [0]*n
        upper = [0]*n
    if stop_fittness is None:
        stop_fittness = math.inf   
    array_type = ct.c_double * n   
    c_callback = call_back_type(callback(fun))
    seed = int(rg.uniform(0, 2**32 - 1))
    try:
        res = optimizeHH_C(runid, c_callback, n, seed,
                           array_type(*lower), array_type(*upper), 
                           max_evaluations, stop_fittness,  
                           popsize)
        x = np.array(np.fromiter(res, dtype=np.float64, count=n))
        val = res[n]
        evals = int(res[n+1])
        iterations = int(res[n+2])
        stop = int(res[n+3])
        freemem(res)
        return OptimizeResult(x=x, fun=val, nfev=evals, nit=iterations, status=stop, success=True)
    except Exception as ex:
        return OptimizeResult(x=None, fun=sys.float_info.max, nfev=0, nit=0, status=-1, success=False)  
      
basepath = os.path.dirname(os.path.abspath(__file__))

if sys.platform.startswith('linux'):
    libcmalib = ct.cdll.LoadLibrary(basepath + '/lib/libacmalib.so')  
elif 'mac' in sys.platform:
    libgtoplib = ct.cdll.LoadLibrary(basepath + '/lib/libacmalib.dylib')  
else:
    libcmalib = ct.cdll.LoadLibrary(basepath + '/lib/libacmalib.dll')  

call_back_type = ct.CFUNCTYPE(ct.c_double, ct.c_int, ct.POINTER(ct.c_double))  
optimizeHH_C = libcmalib.optimizeHH_C
optimizeHH_C.argtypes = [ct.c_long, call_back_type, ct.c_int, ct.c_int, \
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
            ct.c_int, ct.c_double, ct.c_int]

optimizeHH_C.restype = ct.POINTER(ct.c_double)         
freemem = libcmalib.free_mem
freemem.argtypes = [ct.POINTER(ct.c_double)]
  
