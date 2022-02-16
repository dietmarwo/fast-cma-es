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
from scipy.optimize import OptimizeResult
from fcmaes.decpp import libcmalib
from fcmaes.cmaes import _check_bounds

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun, 
             bounds=None, 
             x0=None, 
             input_sigma = 0.3, 
             popsize = None, 
             max_evaluations = 100000, 
             stop_fitness = None, 
             keep = 200,
             f = 0.5,
             cr = 0.9,
             rg = Generator(MT19937()),
             runid=0):  
     
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
    if callable(input_sigma):
        input_sigma=input_sigma()
    if np.ndim(input_sigma) == 0:
        input_sigma = [input_sigma] * dim
    if stop_fitness is None:
        stop_fitness = math.inf   
    array_type = ct.c_double * dim   
    c_callback = call_back_type(callback(fun))
    seed = int(rg.uniform(0, 2**32 - 1))
    res = np.empty(dim+4)
    res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
    try:
        optimizeLDE_C(runid, c_callback, dim, 
                           array_type(*guess), array_type(*input_sigma), seed,
                           array_type(*lower), array_type(*upper), 
                           max_evaluations, keep, stop_fitness,  
                           popsize, f, cr, res_p)
        x = res[:dim]
        val = res[dim]
        evals = int(res[dim+1])
        iterations = int(res[dim+2])
        stop = int(res[dim+3])
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
        except Exception as ex:
            return sys.float_info.max

class callback_par(object):
    
    def __init__(self, fun, parfun):
        self.fun = fun
        self.parfun = parfun
    
    def __call__(self, popsize, n, xs_, ys_):
        try:
            arrType = ct.c_double*(popsize*n)
            addr = ct.addressof(xs_.contents)
            xall = np.frombuffer(arrType.from_address(addr))
            
            if self.parfun is None:
                for p in range(popsize):
                    ys_[p] = self.fun(xall[p*n : (p+1)*n])
            else:    
                xs = []
                for p in range(popsize):
                    x = xall[p*n : (p+1)*n]
                    xs.append(x)
                ys = self.parfun(xs)
                for p in range(popsize):
                    ys_[p] = ys[p]
        except Exception as ex:
            print (ex)

call_back_type = ct.CFUNCTYPE(ct.c_double, ct.c_int, ct.POINTER(ct.c_double))  
call_back_par = ct.CFUNCTYPE(None, ct.c_int, ct.c_int, \
                                  ct.POINTER(ct.c_double), ct.POINTER(ct.c_double))  
      
optimizeLDE_C = libcmalib.optimizeLDE_C
optimizeLDE_C.argtypes = [ct.c_long, call_back_type, ct.c_int, 
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_int, \
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
            ct.c_int, ct.c_double, ct.c_double, ct.c_int, \
            ct.c_double, ct.c_double, ct.POINTER(ct.c_double)]
     
