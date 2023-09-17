# Copyright (c) Mingcheng Zuo, Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

"""Eigen based implementation of differential evolution (GCL-DE) derived from
    "A case learning-based differential evolution algorithm for global optimization of interplanetary trajectory design,
    Mingcheng Zuo, Guangming Dai, Lei Peng, Maocai Wang, Zhengquan Liu", https://doi.org/10.1016/j.asoc.2020.106451
"""

import sys
import os
import math
import ctypes as ct
import numpy as np
from numpy.random import MT19937, Generator
from scipy.optimize import OptimizeResult, Bounds
from fcmaes.evaluator import callback_par, call_back_par, parallel, libcmalib
from fcmaes.de import _check_bounds

from typing import Optional, Callable, Union
from numpy.typing import ArrayLike

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun: Callable[[ArrayLike], float],
             dim: Optional[int] = None,
             bounds: Optional[Bounds] = None,
             popsize: Optional[int] = None,
             max_evaluations: Optional[int] = 100000,
             stop_fitness: Optional[float] = -np.inf,
             pbest: Optional[float] = 0.7,
             f0: Optional[float] = 0.0,
             cr0: Optional[float] = 0.0,
             rg: Optional[Generator] = Generator(MT19937()),
             runid: Optional[int] = 0,
             workers: Optional[int] = None) -> OptimizeResult:
     
    """Minimization of a scalar function of one or more variables using a 
    C++ GCL Differential Evolution implementation called via ctypes.
     
    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
            ``fun(x) -> float``
        where ``x`` is an 1-D array with shape (dim,)
    dim : int
        dimension of the argument of the objective function
    bounds : sequence or `Bounds`
        Bounds on variables. There are two ways to specify the bounds:
            1. Instance of the `scipy.Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`.
    popsize : int, optional
        Population size.
    max_evaluations : int, optional
        Forced termination after ``max_evaluations`` function evaluations.
    stop_fitness : float, optional 
         Limit for fitness value. If reached minimize terminates.
    pbest = float, optional
        use low value 0 < pbest <= 1 to narrow search.
    f0 = float, optional
        The initial mutation constant. In the literature this is also known as differential weight, 
        being denoted by F. Should be in the range [0, 2].
    cr0 = float, optional
        The initial recombination constant. Should be in the range [0, 1]. 
        In the literature this is also known as the crossover probability.     
    rg = numpy.random.Generator, optional
        Random generator for creating random guesses.
    runid : int, optional
        id used to identify the run for debugging / logging. 
    workers : int or None, optional
        If not workers is None, function evaluation is performed in parallel for the whole population. 
        Useful for costly objective functions but is deactivated for parallel retry.      
 
           
    Returns
    -------
    res : scipy.OptimizeResult
        The optimization result is represented as an ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, 
        ``fun`` the best function value, 
        ``nfev`` the number of function evaluations,
        ``nit`` the number of iterations,
        ``success`` a Boolean flag indicating if the optimizer exited successfully. """
                
    dim, lower, upper = _check_bounds(bounds, dim)
    if popsize is None:
        popsize = int(dim*8.5+150)
    if lower is None:
        lower = [0]*dim
        upper = [0]*dim  
    parfun = None if (workers is None or workers <= 1) else parallel(fun, workers)
    array_type = ct.c_double * dim   
    c_callback_par = call_back_par(callback_par(fun, parfun))
    seed = int(rg.uniform(0, 2**32 - 1))
    res = np.empty(dim+4)
    res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
    try:
        optimizeGCLDE_C(runid, c_callback_par, dim, seed,
                           array_type(*lower), array_type(*upper), 
                           max_evaluations, pbest, stop_fitness,  
                           popsize, f0, cr0, res_p)
        x = res[:dim]
        val = res[dim]
        evals = int(res[dim+1])
        iterations = int(res[dim+2])
        stop = int(res[dim+3])
        res = OptimizeResult(x=x, fun=val, nfev=evals, nit=iterations, status=stop, success=True)
    except Exception as ex:
        res = OptimizeResult(x=None, fun=sys.float_info.max, nfev=0, nit=0, status=-1, success=False)
    if not parfun is None:
        parfun.stop()
    return res

if not libcmalib is None: 
          
    optimizeGCLDE_C = libcmalib.optimizeGCLDE_C
    optimizeGCLDE_C.argtypes = [ct.c_long, call_back_par, ct.c_int, ct.c_int, \
                ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
                ct.c_int, ct.c_double, ct.c_double, ct.c_int, \
                ct.c_double, ct.c_double, ct.POINTER(ct.c_double)]
         

