# Copyright (c) Mingcheng Zuo, Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

"""Eigen based implementation of differential evolution (LCL-DE) derived from
    "A case learning-based differential evolution algorithm for global optimization of interplanetary trajectory design,
    Mingcheng Zuo, Guangming Dai, Lei Peng, Maocai Wang, Zhengquan Liu", https://doi.org/10.1016/j.asoc.2020.106451
    To be used to further optimize a given solution. Initial population is created using a normal distribition 
    with mean=x0 and sdev = input_sigma (normalized over the bounds and can be defined separately for each variable)
"""

import sys
import os
import math
import ctypes as ct
import numpy as np
from numpy.random import MT19937, Generator
from scipy.optimize import OptimizeResult
from fcmaes.cmaescpp import callback_par, call_back_par, freemem, libcmalib
from fcmaes.cmaes import parallel, _check_bounds

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun, 
             bounds=None, 
             x0=None, 
             input_sigma = 0.3, 
             popsize = None, 
             max_evaluations = 100000, 
             stop_fitness = None, 
             pbest = 0.7,
             f0 = 0.0,
             cr0 = 0.0,
             rg = Generator(MT19937()),
             runid=0,
             workers = None):  
     
    """Minimization of a scalar function of one or more variables using a 
    C++ LCL Differential Evolution implementation called via ctypes.
     
    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
            ``fun(x, *args) -> float``
        where ``x`` is an 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    bounds : sequence or `Bounds`
        Bounds on variables. There are two ways to specify the bounds:
            1. Instance of the `scipy.Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`.
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where 'n' is the number of independent variables.  
    input_sigma : ndarray, shape (n,) or scalar
        Initial step size for each dimension.
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

    lower, upper, guess = _check_bounds(bounds, x0, rg)      
    n = guess.size  
    if popsize is None:
        popsize = int(n*8.5+150)
    if lower is None:
        lower = [0]*n
        upper = [0]*n
    if np.ndim(input_sigma) == 0:
        input_sigma = [input_sigma] * n
    if stop_fitness is None:
        stop_fitness = math.inf   
    parfun = None if workers is None else parallel(fun, workers)
    array_type = ct.c_double * n   
    c_callback_par = call_back_par(callback_par(fun, parfun))
    seed = int(rg.uniform(0, 2**32 - 1))
    try:
        res = optimizeLCLDE_C(runid, c_callback_par, n, 
                           array_type(*guess), array_type(*input_sigma), seed,
                           array_type(*lower), array_type(*upper), 
                           max_evaluations, pbest, stop_fitness,  
                           popsize, f0, cr0)
        x = np.array(np.fromiter(res, dtype=np.float64, count=n))
        val = res[n]
        evals = int(res[n+1])
        iterations = int(res[n+2])
        stop = int(res[n+3])
        freemem(res)
        if not parfun is None:
            parfun.stop() # stop all parallel evaluation processes
        return OptimizeResult(x=x, fun=val, nfev=evals, nit=iterations, status=stop, success=True)
    except Exception as ex:
        if not workers is None:
            fun.stop() # stop all parallel evaluation processes
        return OptimizeResult(x=None, fun=sys.float_info.max, nfev=0, nit=0, status=-1, success=False)  
      
optimizeLCLDE_C = libcmalib.optimizeLCLDE_C
optimizeLCLDE_C.argtypes = [ct.c_long, call_back_par, ct.c_int, 
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_int, \
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
            ct.c_int, ct.c_double, ct.c_double, ct.c_int, \
            ct.c_double, ct.c_double]

optimizeLCLDE_C.restype = ct.POINTER(ct.c_double)         
