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
             popsize = 32, 
             max_evaluations = 100000, 
             stop_fitness = -math.inf, 
             rg = Generator(MT19937()),
             runid=0,
             normalize = False,
             use_constraint_violation = True,
             penalty_coef = 1E5
             ):
       
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
    input_sigma : float, optional
        Initial step size.
    popsize = int, optional
        CMA-ES population size.
    max_evaluations : int, optional
        Forced termination after ``max_evaluations`` function evaluations.
    stop_fitness : float, optional 
         Limit for fitness value. If reached minimize terminates.
    rg = numpy.random.Generator, optional
        Random generator for creating random guesses.
    runid : int, optional
        id used by the is_terminate callback to identify the CMA-ES run.     
    normalize : boolean, optional
        if true pheno -> geno transformation maps arguments to interval [-1,1] 
           
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
    if popsize is None:
        popsize = 32      
    if popsize % 2 == 1: # requires even popsize
        popsize += 1
    if lower is None:
        lower = [0]*dim
        upper = [0]*dim
    if callable(input_sigma):
        input_sigma=input_sigma()
    if np.ndim(input_sigma) > 0:
        input_sigma = np.mean(input_sigma)
    if stop_fitness is None:
        stop_fitness = -math.inf    
    array_type = ct.c_double * dim 
    c_callback = mo_call_back_type(callback(fun, dim))
    res = np.empty(dim+4)
    res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
    try:
        optimizeCRFMNES_C(runid, c_callback, dim, array_type(*guess), 
                       array_type(*lower), array_type(*upper), 
                input_sigma, max_evaluations, stop_fitness,
                popsize, int(rg.uniform(0, 2**32 - 1)), penalty_coef, 
                use_constraint_violation, normalize, res_p)
        x = res[:dim]
        val = res[dim]
        evals = int(res[dim+1])
        iterations = int(res[dim+2])
        stop = int(res[dim+3])
        return OptimizeResult(x=x, fun=val, nfev=evals, nit=iterations, status=stop, success=True)
    except Exception as ex:
        return OptimizeResult(x=None, fun=sys.float_info.max, nfev=0, nit=0, status=-1, success=False)

optimizeCRFMNES_C = libcmalib.optimizeCRFMNES_C
optimizeCRFMNES_C.argtypes = [ct.c_long, mo_call_back_type, ct.c_int, \
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
            ct.c_double, ct.c_int, ct.c_double, ct.c_int, 
            ct.c_long, ct.c_double, 
            ct.c_bool, ct.c_bool, ct.POINTER(ct.c_double)]


