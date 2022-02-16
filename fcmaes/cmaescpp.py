# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

""" Eigen based implementation of active CMA-ES.
    Derived from http://cma.gforge.inria.fr/cmaes.m which follows
    https://www.researchgate.net/publication/227050324_The_CMA_Evolution_Strategy_A_Comparing_Review
"""

import sys
import os
import math
import ctypes as ct
import numpy as np
from numpy.random import MT19937, Generator
from scipy.optimize import OptimizeResult
from fcmaes.cmaes import _check_bounds
from fcmaes.decpp import mo_call_back_type, callback, libcmalib

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun, 
             bounds=None, 
             x0=None, 
             input_sigma = 0.3, 
             popsize = 31, 
             max_evaluations = 100000, 
             accuracy = 1.0, 
             stop_fitness = None, 
             rg = Generator(MT19937()),
             runid=0,
             workers = 1, 
             normalize = True,
             update_gap = None):   
    """Minimization of a scalar function of one or more variables using a 
    C++ CMA-ES implementation called via ctypes.
     
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
    popsize = int, optional
        CMA-ES population size.
    max_evaluations : int, optional
        Forced termination after ``max_evaluations`` function evaluations.
    accuracy : float, optional
        values > 1.0 reduce the accuracy.
    stop_fitness : float, optional 
         Limit for fitness value. If reached minimize terminates.
    rg = numpy.random.Generator, optional
        Random generator for creating random guesses.
    runid : int, optional
        id used by the is_terminate callback to identify the CMA-ES run. 
    workers : int or None, optional
        If not workers is None, function evaluation is performed in parallel for the whole population. 
        Useful for costly objective functions but is deactivated for parallel retry.      
    normalize : boolean, optional
        pheno -> if true geno transformation maps arguments to interval [-1,1] 
    update_gap : int, optional
        number of iterations without distribution update
           
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
    if workers is None:
        workers = 0
    mu = int(popsize/2)
    if callable(input_sigma):
        input_sigma=input_sigma()
    if np.ndim(input_sigma) == 0:
        input_sigma = [input_sigma] * dim
    if stop_fitness is None:
        stop_fitness = math.inf    
    array_type = ct.c_double * dim 
    c_callback = mo_call_back_type(callback(fun, dim))
    res = np.empty(dim+4)
    res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
    try:
        optimizeACMA_C(runid, c_callback, dim, array_type(*guess), array_type(*lower), array_type(*upper), 
                array_type(*input_sigma), max_evaluations, stop_fitness, mu, 
                popsize, accuracy, int(rg.uniform(0, 2**32 - 1)), normalize, -1 if update_gap is None else update_gap, 
                workers, res_p)
        x = res[:dim]
        val = res[dim]
        evals = int(res[dim+1])
        iterations = int(res[dim+2])
        stop = int(res[dim+3])
        return OptimizeResult(x=x, fun=val, nfev=evals, nit=iterations, status=stop, success=True)
    except Exception as ex:
        return OptimizeResult(x=None, fun=sys.float_info.max, nfev=0, nit=0, status=-1, success=False)

optimizeACMA_C = libcmalib.optimizeACMA_C
optimizeACMA_C.argtypes = [ct.c_long, mo_call_back_type, ct.c_int, \
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
            ct.POINTER(ct.c_double), ct.c_int, ct.c_double, ct.c_int, ct.c_int, \
            ct.c_double, ct.c_long, ct.c_bool, ct.c_int, 
            ct.c_int, ct.POINTER(ct.c_double)]


