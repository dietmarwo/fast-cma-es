# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

""" Eigen based implementation of Fast Moving Natural Evolution Strategy 
    for High-Dimensional Problems (CR-FM-NES), see https://arxiv.org/abs/2201.11422 .
    Derived from https://github.com/nomuramasahir0/crfmnes .
"""

import sys
import os
import math
import ctypes as ct
import numpy as np
from numpy.random import MT19937, Generator
from scipy.optimize import OptimizeResult, Bounds
from fcmaes.evaluator import _check_bounds, _get_bounds, callback_par, parallel, call_back_par, libcmalib

from typing import Optional, Callable, Union
from numpy.typing import ArrayLike

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun: Callable[[ArrayLike], float], 
             bounds: Optional[Bounds] = None, 
             x0: Optional[ArrayLike] = None,
             input_sigma = 0.3, 
             popsize = 32, 
             max_evaluations = 100000, 
             workers = None,
             stop_fitness = -np.inf, 
             rg = Generator(MT19937()),
             runid=0,
             normalize = False,
             use_constraint_violation = True,
             penalty_coef = 1E5
             ) -> OptimizeResult:
       
    """Minimization of a scalar function of one or more variables using a 
    C++ CR-FM-NES implementation called via ctypes.
     
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
    workers : int or None, optional
        If workers > 1, function evaluation is performed in parallel for the whole population. 
        Useful for costly objective functions but is deactivated for parallel retry.  
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
    array_type = ct.c_double * dim   
    parfun = None if (workers is None or workers <= 1) else parallel(fun, workers)  
    c_callback_par = call_back_par(callback_par(fun, parfun))
    res = np.empty(dim+4)
    res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
    try:
        optimizeCRFMNES_C(runid, c_callback_par, dim, array_type(*guess), 
                       array_type(*lower), array_type(*upper), 
                input_sigma, max_evaluations, stop_fitness,
                popsize, int(rg.uniform(0, 2**32 - 1)), penalty_coef, 
                use_constraint_violation, normalize, res_p)
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

class CRFMNES_C:

    def __init__(self,
                dim: int, 
                bounds: Optional[Bounds] = None, 
                x0: Optional[ArrayLike] = None,
                input_sigma: Optional[Union[float, ArrayLike, Callable]] = 0.3, 
                popsize: Optional[int] = 32,   
                rg: Optional[Generator] = Generator(MT19937()),
                runid: Optional[int] = 0,
                normalize: Optional[bool] = False,
                use_constraint_violation: Optional[bool] = True,
                penalty_coef: Optional[float] = 1E5
                ):
       
        """Minimization of a scalar function of one or more variables using a 
        C++ CR-FM-NES implementation called via ctypes.
         
        Parameters
        ----------
        dim : int
            dimension of the argument of the objective function
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
        rg = numpy.random.Generator, optional
            Random generator for creating random guesses.
        runid : int, optional
            id used by the is_terminate callback to identify the CMA-ES run.     
        normalize : boolean, optional
            if true pheno -> geno transformation maps arguments to interval [-1,1]"""

        lower, upper, guess = _get_bounds(dim, bounds, x0, rg)      
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
        array_type = ct.c_double * dim   
        try:
            self.ptr = initCRFMNES_C(runid, dim, array_type(*guess), 
                           array_type(*lower), array_type(*upper), 
                    input_sigma, popsize, int(rg.uniform(0, 2**32 - 1)), penalty_coef, 
                    use_constraint_violation, normalize)
            self.popsize = popsize
            self.dim = dim            
        except Exception as ex:
            print (ex)
            pass
    
    def __del__(self):
        destroyCRFMNES_C(self.ptr)
            
    def ask(self) -> np.ndarray:
        try:
            lamb = self.popsize
            n = self.dim
            res = np.empty(lamb*n)
            res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
            askCRFMNES_C(self.ptr, res_p)
            xs = np.empty((lamb, n))
            for p in range(lamb):
                xs[p,:] = res[p*n : (p+1)*n]
            return xs
        except Exception as ex:
            print (ex)
            return None

    def tell(self, ys: np.ndarray):
        try:
            array_type_ys = ct.c_double * len(ys)
            return tellCRFMNES_C(self.ptr, array_type_ys(*ys))
        except Exception as ex:
            print (ex)
            return -1        

    def population(self) -> np.ndarray:
        try:
            lamb = self.popsize
            n = self.dim
            res = np.empty(lamb*n)
            res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
            populationCRFMNES_C(self.ptr, res_p)
            xs = np.array(lamb, n)
            for p in range(lamb):
                xs[p] = res[p*n : (p+1)*n]
                return xs
        except Exception as ex:
            print (ex)
            return None
        
    def result(self) -> OptimizeResult:
        res = np.empty(self.dim+4)
        res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
        try:
            resultCRFMNES_C(self.ptr, res_p)
            x = res[:self.dim]
            val = res[self.dim]
            evals = int(res[self.dim+1])
            iterations = int(res[self.dim+2])
            stop = int(res[self.dim+3])
            res = OptimizeResult(x=x, fun=val, nfev=evals, nit=iterations, status=stop, success=True)
        except Exception as ex:
            res = OptimizeResult(x=None, fun=sys.float_info.max, nfev=0, nit=0, status=-1, success=False)
        return res

if not libcmalib is None: 

    optimizeCRFMNES_C = libcmalib.optimizeCRFMNES_C
    optimizeCRFMNES_C.argtypes = [ct.c_long, call_back_par, ct.c_int, \
                ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
                ct.c_double, ct.c_int, ct.c_double, ct.c_int, 
                ct.c_long, ct.c_double, 
                ct.c_bool, ct.c_bool, ct.POINTER(ct.c_double)]
          
    initCRFMNES_C = libcmalib.initCRFMNES_C
    initCRFMNES_C.argtypes = [ct.c_long, ct.c_int, \
                ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
                ct.c_double, ct.c_int,
                ct.c_long, ct.c_double, 
                ct.c_bool, ct.c_bool]
    
    initCRFMNES_C.restype = ct.c_void_p   
    
    destroyCRFMNES_C = libcmalib.destroyCRFMNES_C
    destroyCRFMNES_C.argtypes = [ct.c_void_p]
    
    askCRFMNES_C = libcmalib.askCRFMNES_C
    askCRFMNES_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
    
    tellCRFMNES_C = libcmalib.tellCRFMNES_C
    tellCRFMNES_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
    tellCRFMNES_C.restype = ct.c_int
    
    populationCRFMNES_C = libcmalib.populationCRFMNES_C
    populationCRFMNES_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
    
    resultCRFMNES_C = libcmalib.resultCRFMNES_C
    resultCRFMNES_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
