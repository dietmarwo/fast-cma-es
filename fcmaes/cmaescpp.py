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
from scipy.optimize import OptimizeResult, Bounds
from fcmaes.evaluator import _check_bounds, _get_bounds, mo_call_back_type, callback_so, callback_par, call_back_par, parallel, libcmalib

from typing import Optional, Callable, Union
from numpy.typing import ArrayLike

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun: Callable[[ArrayLike], float], 
             bounds: Optional[Bounds] = None, 
             x0: Optional[ArrayLike] = None,
             input_sigma: Optional[Union[float, ArrayLike, Callable]] = 0.3,
             popsize: Optional[int] = 31,
             max_evaluations: Optional[int]  = 100000,
             accuracy: Optional[float] = 1.0, 
             stop_fitness: Optional[float] = -np.inf, 
             stop_hist: Optional[float] = None,
             rg: Optional[Generator] = Generator(MT19937()),
             runid: Optional[int] = 0,
             workers: Optional[int] = 1, 
             normalize: Optional[bool] = True,
             delayed_update: Optional[bool] = True,
             update_gap: Optional[int] = None
             ) -> OptimizeResult:
   
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
    stop_hist : float, optional 
         Set to 0 if you want to prevent premature termination because 
         there is no progress
    rg = numpy.random.Generator, optional
        Random generator for creating random guesses.
    runid : int, optional
        id used by the is_terminate callback to identify the CMA-ES run. 
    workers : int or None, optional
        If not workers is None, function evaluation is performed in parallel for the whole population. 
        Useful for costly objective functions but is deactivated for parallel retry.      
    normalize : boolean, optional
        if true pheno -> geno transformation maps arguments to interval [-1,1]
    delayed_update : boolean, optional
        if true uses delayed update / C++ parallelism, i false uses Python multithreading
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
    if stop_hist is None:
        stop_hist = -1;
    array_type = ct.c_double * dim 
    c_callback = mo_call_back_type(callback_so(fun, dim))
    parfun = None if delayed_update == True or workers is None or workers <= 1 else parallel(fun, workers)
    c_callback_par = call_back_par(callback_par(fun, parfun))
    res = np.empty(dim+4)
    res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
    try:
        optimizeACMA_C(runid, c_callback, c_callback_par, 
                dim, array_type(*guess), array_type(*lower), array_type(*upper), 
                array_type(*input_sigma), max_evaluations, stop_fitness, stop_hist, mu, 
                popsize, accuracy, int(rg.uniform(0, 2**32 - 1)), 
                normalize, delayed_update, -1 if update_gap is None else update_gap,
                workers, res_p)
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

class ACMA_C:

    def __init__(self,
        dim, 
        bounds: Optional[Bounds] = None, 
        x0: Optional[ArrayLike] = None,
        input_sigma: Optional[Union[float, ArrayLike, Callable]] = 0.3, 
        popsize: Optional[int] = 31,  
        max_evaluations: Optional[int] = 100000, 
        accuracy: Optional[float] = 1.0, 
        stop_fitness: Optional[float] = -np.inf, 
        stop_hist: Optional[float] = None,
        rg: Optional[Generator] = Generator(MT19937()),
        runid: Optional[int] = 0,
        normalize: Optional[bool] = True,
        delayed_update: Optional[bool] = True,
        update_gap: Optional[int] = None
     ):
       
        """Parameters
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
        stop_hist : float, optional 
             Set to 0 if you want to prevent premature termination because 
             there is no progress
        rg = numpy.random.Generator, optional
            Random generator for creating random guesses.
        runid : int, optional
            id used by the is_terminate callback to identify the CMA-ES run.     
        normalize : boolean, optional
            if true pheno -> geno transformation maps arguments to interval [-1,1]
        delayed_update : boolean, optional
            if true uses delayed update / C++ parallelism, i false uses Python multithreading
        update_gap : int, optional
            number of iterations without distribution update"""
             
        lower, upper, guess = _get_bounds(dim, bounds, x0, rg)     
        if lower is None:
            lower = [0]*dim
            upper = [0]*dim
        mu = int(popsize/2)
        if callable(input_sigma):
            input_sigma=input_sigma()
        if np.ndim(input_sigma) == 0:
            input_sigma = [input_sigma] * dim
        if stop_hist is None:
            stop_hist = -1;
        array_type = ct.c_double * dim 
        try:
            self.ptr = initACMA_C(runid,
                dim, array_type(*guess), array_type(*lower), array_type(*upper), 
                array_type(*input_sigma), max_evaluations, stop_fitness, stop_hist, mu, 
                popsize, accuracy, int(rg.uniform(0, 2**32 - 1)), 
                normalize, delayed_update, -1 if update_gap is None else update_gap)
            self.popsize = popsize
            self.dim = dim            
        except Exception as ex:
            print (ex)
            pass
    
    def __del__(self):
        destroyACMA_C(self.ptr)
            
    def ask(self) -> np.array:
        try:
            popsize = self.popsize
            n = self.dim
            res = np.empty(popsize*n)
            res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
            askACMA_C(self.ptr, res_p)
            xs = np.empty((popsize, n))
            for p in range(popsize):
                xs[p,:] = res[p*n : (p+1)*n]
            return xs
        except Exception as ex:
            print (ex)
            return None

    def tell(self, 
             ys: np.ndarray, 
             xs: Optional[np.ndarray] = None) -> int:  
        if not xs is None:
            return self.tell_x_(ys, xs)
        try:
            array_type_ys = ct.c_double * len(ys)
            return tellACMA_C(self.ptr, array_type_ys(*ys))
        except Exception as ex:
            print (ex)
            return -1    

    def tell_x_(self, ys: np.ndarray, xs: np.ndarray):
        try:
            flat_xs = xs.flatten()
            array_type_xs = ct.c_double * len(flat_xs)
            array_type_ys = ct.c_double * len(ys)
            return tellXACMA_C(self.ptr, array_type_ys(*ys), array_type_xs(*flat_xs))
        except Exception as ex:
            print (ex)
            return -1 
        
    def population(self) -> np.array:
        try:
            popsize = self.popsize
            n = self.dim
            res = np.empty(popsize*n)
            res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
            populationACMA_C(self.ptr, res_p)
            xs = np.array(popsize, n)
            for p in range(popsize):
                xs[p] = res[p*n : (p+1)*n]
                return xs
        except Exception as ex:
            print (ex)
            return None

    def result(self) -> OptimizeResult:
        res = np.empty(self.dim+4)
        res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
        try:
            resultACMA_C(self.ptr, res_p)
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

    optimizeACMA_C = libcmalib.optimizeACMA_C
    optimizeACMA_C.argtypes = [ct.c_long, mo_call_back_type, call_back_par, ct.c_int, \
                ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
                ct.POINTER(ct.c_double), ct.c_int, ct.c_double, ct.c_double, ct.c_int, ct.c_int, \
                ct.c_double, ct.c_long, ct.c_bool, ct.c_bool, ct.c_int, 
                ct.c_int, ct.POINTER(ct.c_double)]
    
    initACMA_C = libcmalib.initACMA_C
    initACMA_C.argtypes = [ct.c_long, ct.c_int, \
                ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
                ct.POINTER(ct.c_double), ct.c_int, ct.c_double, ct.c_double, ct.c_int, 
                ct.c_int, ct.c_double, ct.c_long, ct.c_bool, ct.c_bool, ct.c_int]
                    
    initACMA_C.restype = ct.c_void_p   
    
    destroyACMA_C = libcmalib.destroyACMA_C
    destroyACMA_C.argtypes = [ct.c_void_p]
    
    askACMA_C = libcmalib.askACMA_C
    askACMA_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
    
    tellACMA_C = libcmalib.tellACMA_C
    tellACMA_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
    tellACMA_C.restype = ct.c_int
    
    tellXACMA_C = libcmalib.tellXACMA_C
    tellXACMA_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
    tellXACMA_C.restype = ct.c_int
    
    populationACMA_C = libcmalib.populationACMA_C
    populationACMA_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
    
    resultACMA_C = libcmalib.resultACMA_C
    resultACMA_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
