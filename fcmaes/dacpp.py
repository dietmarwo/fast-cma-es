# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

""" Eigen based implementation of dual annealing.
    Derived from https://github.com/scipy/scipy/blob/master/scipy/optimize/_dual_annealing.py.
    Local search is fixed to LBFGS-B
"""

import sys
import os
import ctypes as ct
import numpy as np
from numpy.random import MT19937, Generator
from scipy.optimize import OptimizeResult, Bounds
from fcmaes.evaluator import _check_bounds, call_back_type, callback, libcmalib

from typing import Optional, Callable, Union
from numpy.typing import ArrayLike

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun: Callable[[ArrayLike], float], 
             bounds: Optional[Bounds] = None, 
             x0: Optional[ArrayLike] = None,
             max_evaluations: Optional[int] = 100000, 
             use_local_search: Optional[bool] = True,
             rg: Optional[Generator] = Generator(MT19937()),
             runid: Optional[int] = 0) -> OptimizeResult:   

    """Minimization of a scalar function of one or more variables using a 
    C++ Dual Annealing implementation called via ctypes.
     
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
    use_local_search : bool, optional
        If true local search is performed.
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
        ``nit`` the number of iterations,
        ``success`` a Boolean flag indicating if the optimizer exited successfully. """
                
    lower, upper, guess = _check_bounds(bounds, x0, rg)   
    dim = guess.size   
    if lower is None:
        lower = [0]*dim
        upper = [0]*dim
    array_type = ct.c_double * dim   
    c_callback = call_back_type(callback(fun))
    seed = int(rg.uniform(0, 2**32 - 1))
    res = np.empty(dim+4)
    res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
    try:
        optimizeDA_C(runid, c_callback, dim, seed,
                    array_type(*guess), array_type(*lower), array_type(*upper), 
                    max_evaluations, use_local_search, res_p)
        x = res[:dim]
        val = res[dim]
        evals = int(res[dim+1])
        iterations = int(res[dim+2])
        stop = int(res[dim+3])
        return OptimizeResult(x=x, fun=val, nfev=evals, nit=iterations, status=stop, success=True)
    except Exception as ex:
        return OptimizeResult(x=None, fun=sys.float_info.max, nfev=0, nit=0, status=-1, success=False)

if not libcmalib is None: 
          
    optimizeDA_C = libcmalib.optimizeDA_C
    optimizeDA_C.argtypes = [ct.c_long, call_back_type, ct.c_int, ct.c_int, \
                ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
                ct.c_int, ct.c_bool, ct.POINTER(ct.c_double)]

