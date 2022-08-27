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
import multiprocessing as mp
from fcmaes.cmaes import _check_bounds
from fcmaes.decpp import libcmalib
from fcmaes.evaluator import Evaluator, eval_parallel

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun, 
             bounds=None, 
             x0=None, 
             input_sigma = 0.3, 
             popsize = 32, 
             max_evaluations = 100000, 
             workers = None,
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
    if stop_fitness is None:
        stop_fitness = -math.inf    
    parfun = None if (workers is None or workers <= 1) else parallel(fun, workers)
    array_type = ct.c_double * dim   
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
    
class parallel(object):
    """Convert an objective function for parallel execution for cmaes.minimize.
    
    Parameters
    ----------
    fun : objective function mapping a list of float arguments to a float value.
   
    represents a function mapping a list of lists of float arguments to a list of float values
    by applying the input function using parallel processes. stop needs to be called to avoid
    a resource leak"""
        
    def __init__(self, fun, workers = mp.cpu_count()):
        self.evaluator = Evaluator(fun)
        self.evaluator.start(workers)
    
    def __call__(self, xs):
        return eval_parallel(xs, self.evaluator)

    def stop(self):
        self.evaluator.stop()
        
class callback(object):
    
    def __init__(self, fun):
        self.fun = fun
    
    def __call__(self, n, x):
        try:
            fit = self.fun(np.array([x[i] for i in range(n)]))
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
optimizeCRFMNES_C = libcmalib.optimizeCRFMNES_C
optimizeCRFMNES_C.argtypes = [ct.c_long, call_back_par, ct.c_int, \
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
            ct.c_double, ct.c_int, ct.c_double, ct.c_int, 
            ct.c_long, ct.c_double, 
            ct.c_bool, ct.c_bool, ct.POINTER(ct.c_double)]


