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
from scipy.optimize import OptimizeResult, Bounds
from fcmaes.evaluator import _check_bounds, mo_call_back_type, callback_so, libcmalib

from typing import Optional, Callable
from numpy.typing import ArrayLike

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun: Callable[[ArrayLike], float],
             bounds: Optional[Bounds] = None,
             x0: Optional[ArrayLike] = None,
             max_evaluations: Optional[int] = 100000,
             stop_fitness: Optional[float] = -np.inf,
             M: Optional[int] = 1,
             popsize: Optional[int] = 0,
             stall_criterion: Optional[int]  = 0,
             rg: Optional[Generator]  = Generator(MT19937()),
             runid: Optional[int] = 0) -> OptimizeResult:
    """Minimization of a scalar function of one or more variables using a 
    C++ SCMA implementation called via ctypes.
     
    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
            ``fun(x) -> float``
        where ``x`` is an 1-D array with shape (dim,)
    bounds : sequence or `Bounds`, optional
        Bounds on variables. There are two ways to specify the bounds:
            1. Instance of the `scipy.Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.
    x0 : ndarray, shape (dim,)
        Initial guess. Array of real elements of size (dim,),
        where 'dim' is the number of independent variables.  
    max_evaluations : int, optional
        Forced termination after ``max_evaluations`` function evaluations.
    stop_fitness : float, optional 
         Limit for fitness value. If reached minimize terminates.
    M : int, optional 
        Depth to use, 1 for plain CBiteOpt algorithm, >1 for CBiteOptDeep. Expected range is [1; 36].
    popsize = int, optional
        initial population size.
    stall_criterion : int, optional 
        Terminate if stall_criterion*128*evaluations stalled, Not used if <= 0
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
    dim = guess.size   
    if lower is None:
        lower = [0]*dim
        upper = [0]*dim 
    array_type = ct.c_double * dim 
    c_callback = mo_call_back_type(callback_so(fun, dim))
    res = np.empty(dim+4)
    res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
    try:
        optimizeBite_C(runid, c_callback, dim, int(rg.uniform(0, 2**32 - 1)), 
                           array_type(*guess), array_type(*lower), array_type(*upper), 
                           max_evaluations, stop_fitness, M, popsize, stall_criterion, res_p)
        x = res[:dim]
        val = res[dim]
        evals = int(res[dim+1])
        iterations = int(res[dim+2])
        stop = int(res[dim+3])
        return OptimizeResult(x=x, fun=val, nfev=evals, nit=iterations, status=stop, success=True)
    except Exception as ex:
        return OptimizeResult(x=None, fun=sys.float_info.max, nfev=0, nit=0, status=-1, success=False)

if not libcmalib is None: 
    
    optimizeBite_C = libcmalib.optimizeBite_C
    optimizeBite_C.argtypes = [ct.c_long, mo_call_back_type, ct.c_int, ct.c_int, \
                ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
                ct.c_int, ct.c_double, ct.c_int, ct.c_int, ct.c_int, ct.POINTER(ct.c_double)]
       


