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
from fcmaes.cmaes import _check_bounds, serial, parallel

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun, 
             bounds=None, 
             x0=None, 
             input_sigma = 0.3, 
             popsize = 31, 
             max_evaluations = 100000, 
             max_iterations = 100000,  
             accuracy = 1.0, 
             stop_fittness = None, 
             is_terminate = None, 
             rg = Generator(MT19937()),
             runid=0,
             workers = None):   
    """Minimization of a scalar function of one or more variables using a 
    C++ CMA-ES implementation called via ctypes.
     
    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
            ``fun(x, *args) -> float``
        where ``x`` is an 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    bounds : sequence or `Bounds`, optional
        Bounds on variables. There are two ways to specify the bounds:
            1. Instance of the `scipy.Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where 'n' is the number of independent variables.  
    input_sigma : ndarray, shape (n,) or scalar
        Initial step size for each dimension.
    popsize = int, optional
        CMA-ES population size.
    max_evaluations : int, optional
        Forced termination after ``max_evaluations`` function evaluations.
    max_iterations : int, optional
        Forced termination after ``max_iterations`` iterations.
    accuracy : float, optional
        values > 1.0 reduce the accuracy.
    stop_fittness : float, optional 
         Limit for fitness value. If reached minimize terminates.
    is_terminate : callable, optional
        Callback to be used if the caller of minimize wants to 
        decide when to terminate. 
    rg = numpy.random.Generator, optional
        Random generator for creating random guesses.
    runid : int, optional
        id used by the is_terminate callback to identify the CMA-ES run. 
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
        ``nit`` the number of CMA-ES iterations, 
        ``status`` the stopping critera and
        ``success`` a Boolean flag indicating if the optimizer exited successfully. """
    
    lower, upper, guess = _check_bounds(bounds, x0, rg)      
    n = guess.size   
    if lower is None:
        lower = [0]*n
        upper = [0]*n
    mu = int(popsize/2)
    if np.ndim(input_sigma) == 0:
        input_sigma = [input_sigma] * n
    if stop_fittness is None:
        stop_fittness = math.inf   
    if is_terminate is None:    
        is_terminate=_is_terminate_false
        use_terminate = False 
    else:
        use_terminate = True 
    parfun = None if workers is None else parallel(fun, workers)
    array_type = ct.c_double * n 
    c_callback_par = call_back_par(callback_par(fun, parfun))
    c_is_terminate = is_terminate_type(is_terminate)
    try:
        res = optimizeACMA_C(runid, c_callback_par, n, array_type(*guess), array_type(*lower), array_type(*upper), 
                           array_type(*input_sigma), max_iterations, max_evaluations, stop_fittness, mu, 
                           popsize, accuracy, use_terminate, c_is_terminate, int(rg.uniform(0, 2**32 - 1)))

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

class callback(object):
    
    def __init__(self, fun):
        self.fun = fun
    
    def __call__(self, n, x):
        try:
            fit = self.fun([x[i] for i in range(n)])
            return fit if math.isfinite(fit) else sys.float_info.max
        except Exception:
            return sys.float_info.max

#https://stackoverflow.com/questions/7543675/how-to-convert-pointer-to-c-array-to-python-array

class callback_par(object):
    
    def __init__(self, fun, parfun):
        self.fun = fun
        self.parfun = parfun
    
    def __call__(self, popsize, n, xs_, ys_):
        try:
            #xall = np.array(np.fromiter(xs_, dtype = np.float64, count = popsize*n))
            
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

def _is_terminate_false(runid, iterations, val):
    return False 
  
basepath = os.path.dirname(os.path.abspath(__file__))

if sys.platform.startswith('linux'):
    libcmalib = ct.cdll.LoadLibrary(basepath + '/lib/libacmalib.so')  
elif 'mac' in sys.platform:
    libgtoplib = ct.cdll.LoadLibrary(basepath + '/lib/libacmalib.dylib')  
else:
    libcmalib = ct.cdll.LoadLibrary(basepath + '/lib/libacmalib.dll')  

call_back_par = ct.CFUNCTYPE(None, ct.c_int, ct.c_int, \
                                  ct.POINTER(ct.c_double), ct.POINTER(ct.c_double))  
is_terminate_type = ct.CFUNCTYPE(ct.c_bool, ct.c_long, ct.c_int, ct.c_double)    

optimizeACMA_C = libcmalib.optimizeACMA_C
optimizeACMA_C.argtypes = [ct.c_long, call_back_par, ct.c_int, \
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
            ct.POINTER(ct.c_double), ct.c_int, ct.c_int, ct.c_double, ct.c_int, ct.c_int, \
            ct.c_double, ct.c_bool, is_terminate_type, ct.c_long]

optimizeACMA_C.restype = ct.POINTER(ct.c_double)         

freemem = libcmalib.free_mem
freemem.argtypes = [ct.POINTER(ct.c_double)]
