# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

"""Eigen based implementation of differential evolution using onl the DE/best/1 strategy.
    Uses two deviations from the standard DE algorithm:
    a) temporal locality introduced in 
        https://www.researchgate.net/publication/309179699_Differential_evolution_for_protein_folding_optimization_based_on_a_three-dimensional_AB_off-lattice_model
    b) reinitialization of individuals based on their age.
    To be used to further optimize a given solution. Initial population is created using a normal distribition 
    with mean=x0 and sdev = input_sigma (normalized over the bounds and can be defined separately for each variable)
    """

import sys
import os
import math
import ctypes as ct
import numpy as np
from numpy.random import MT19937, Generator
from scipy.optimize import OptimizeResult, Bounds
from fcmaes.evaluator import _check_bounds, mo_call_back_type, callback_so, libcmalib

from typing import Optional, Callable, Union
from numpy.typing import ArrayLike

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun: Callable[[ArrayLike], float], 
             bounds: Optional[Bounds] = None, 
             x0: Optional[ArrayLike] = None,
             input_sigma = 0.3, 
             popsize = None, 
             max_evaluations = 100000, 
             stop_fitness = -np.inf, 
             keep = 200,
             f = 0.5,
             cr = 0.9,
             ints = None,
             min_mutate = 0.1,
             max_mutate = 0.5, 
             rg = Generator(MT19937()),
             runid=0) -> OptimizeResult:  
     
    """Minimization of a scalar function of one or more variables using a 
    C++ Differential Evolution implementation called via ctypes.
     
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
    input_sigma : ndarray, shape (dim,) or scalar
        Initial step size for each dimension.
    popsize : int, optional
        Population size.
    max_evaluations : int, optional
        Forced termination after ``max_evaluations`` function evaluations.
    stop_fitness : float, optional 
         Limit for fitness value. If reached minimize terminates.
    keep = float, optional
        changes the reinitialization probability of individuals based on their age. Higher value
        means lower probablity of reinitialization.
    f = float, optional
        The mutation constant. In the literature this is also known as differential weight, 
        being denoted by F. Should be in the range [0, 2].
    cr = float, optional
        The recombination constant. Should be in the range [0, 1]. 
        In the literature this is also known as the crossover probability.   
    ints = list or array of bool, optional
        indicating which parameters are discrete integer values. If defined these parameters will be
        rounded to the next integer and some additional mutation of discrete parameters are performed.    
    min_mutate = float, optional
        Determines the minimal mutation rate for discrete integer parameters.
    max_mutate = float, optional
        Determines the maximal mutation rate for discrete integer parameters. 
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
    if popsize is None:
        popsize = 31
    if lower is None:
        lower = [0]*dim
        upper = [0]*dim
    if ints is None:
        ints = [False]*dim
    if callable(input_sigma):
        input_sigma=input_sigma()
    if np.ndim(input_sigma) == 0:
        input_sigma = [input_sigma] * dim  
    array_type = ct.c_double * dim   
    bool_array_type = ct.c_bool * dim 
    c_callback = mo_call_back_type(callback_so(fun, dim))
    seed = int(rg.uniform(0, 2**32 - 1))
    res = np.empty(dim+4)
    res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
    try:
        optimizeLDE_C(runid, c_callback, dim, 
                           array_type(*guess), array_type(*input_sigma), seed,
                           array_type(*lower), array_type(*upper), 
                           max_evaluations, keep, stop_fitness,  
                           popsize, f, cr, min_mutate, max_mutate, bool_array_type(*ints), res_p)
        x = res[:dim]
        val = res[dim]
        evals = int(res[dim+1])
        iterations = int(res[dim+2])
        stop = int(res[dim+3])
        return OptimizeResult(x=x, fun=val, nfev=evals, nit=iterations, status=stop, success=True)
    except Exception as ex:
        return OptimizeResult(x=None, fun=sys.float_info.max, nfev=0, nit=0, status=-1, success=False)  

if not libcmalib is None: 
          
    optimizeLDE_C = libcmalib.optimizeLDE_C
    optimizeLDE_C.argtypes = [ct.c_long, mo_call_back_type, ct.c_int, 
                ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_int, \
                ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
                ct.c_int, ct.c_double, ct.c_double, ct.c_int, \
                ct.c_double, ct.c_double, ct.c_double, ct.c_double, 
                ct.POINTER(ct.c_bool), ct.POINTER(ct.c_double)]

