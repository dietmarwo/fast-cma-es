# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

"""Eigen based implementation of differential evolution using the DE/best/1 strategy.
    Uses three deviations from the standard DE algorithm:
    a) temporal locality introduced in 
        https://www.researchgate.net/publication/309179699_Differential_evolution_for_protein_folding_optimization_based_on_a_three-dimensional_AB_off-lattice_model
    b) reinitialization of individuals based on their age.
    c) oscillating CR/F parameters.
    
    The ints parameter is a boolean array indicating which parameters are discrete integer values. This 
    parameter was introduced after observing non optimal results for the ESP2 benchmark problem: 
    https://github.com/AlgTUDelft/ExpensiveOptimBenchmark/blob/master/expensiveoptimbenchmark/problems/DockerCFDBenchmark.py
    If defined it causes a "special treatment" for discrete variables: They are rounded to the next integer value and
    there is an additional mutation to avoid getting stuck to local minima."""
    
import sys
import os
import ctypes as ct
import numpy as np
from numpy.random import MT19937, Generator
from scipy.optimize import OptimizeResult, Bounds
from fcmaes.evaluator import mo_call_back_type, callback_so, libcmalib
from fcmaes.de import _check_bounds

from typing import Optional, Callable, Tuple, Union
from numpy.typing import ArrayLike

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun: Callable[[ArrayLike], float],
             dim: Optional[int] = None,
             bounds: Optional[Bounds] = None,
             popsize: Optional[int] = 31,
             max_evaluations: Optional[int] = 100000,
             stop_fitness: Optional[float] = -np.inf,
             keep: Optional[int] = 200,
             f: Optional[float] = 0.5,
             cr: Optional[float] = 0.9,
             rg: Optional[Generator] = Generator(MT19937()),
             ints: Optional[ArrayLike] = None,
             min_mutate: Optional[float] = 0.1,
             max_mutate: Optional[float] = 0.5,
             workers: Optional[int] = 1,
             is_terminate: Optional[Callable[[ArrayLike, float], bool]] = None,
             x0: Optional[ArrayLike] = None,
             input_sigma: Optional[Union[float, ArrayLike, Callable]] = 0.3,
             min_sigma: Optional[float] = 0,
             runid: Optional[int] = 0) -> OptimizeResult: 
     
    """Minimization of a scalar function of one or more variables using a 
    C++ Differential Evolution implementation called via ctypes.
     
    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
            ``fun(x) -> float``
        where ``x`` is an 1-D array with shape (dim,)
    dim : int
        dimension of the argument of the objective function
    bounds : sequence or `Bounds`, optional
        Bounds on variables. There are two ways to specify the bounds:
            1. Instance of the `scipy.Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.
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
    ints = list or array of bool, optional
        indicating which parameters are discrete integer values. If defined these parameters will be
        rounded to the next integer and some additional mutation of discrete parameters are performed.       
    min_mutate = float, optional
        Determines the minimal mutation rate for discrete integer parameters.
    max_mutate = float, optional
        Determines the maximal mutation rate for discrete integer parameters. 
    workers : int or None, optional
        If not workers is None, function evaluation is performed in parallel for the whole population. 
        Useful for costly objective functions but is deactivated for parallel retry.      
    is_terminate : callable, optional
        Callback to be used if the caller of minimize wants to decide when to terminate.
    x0 : ndarray, shape (dim,)
        Initial guess. Array of real elements of size (dim,),
        where 'dim' is the number of independent variables.  
    input_sigma : ndarray, shape (dim,) or scalar
        Initial sigma for each dimension.
    min_sigma = float, optional
        minimal sigma limit. If 0, uniform random distribution is used (requires bounds).
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
    
    dim, lower, upper = _check_bounds(bounds, dim)
    if popsize is None:
        popsize = 31
    if x0 is None:
        x0 = np.zeros(dim) if lower is None else rg.uniform(bounds.lb, bounds.ub)
    if lower is None:
        lower = [0]*dim
        upper = [0]*dim
        if min_sigma == 0:
            min_sigma = 0.1 # no uniform random generation possible without bounds
    if callable(input_sigma):
        input_sigma=input_sigma()
    if np.ndim(input_sigma) == 0:
        input_sigma = [input_sigma] * dim
    if ints is None:
        ints = [False]*dim
    if workers is None:
        workers = 0 
    array_type = ct.c_double * dim   
    bool_array_type = ct.c_bool * dim 
    c_callback = mo_call_back_type(callback_so(fun, dim, is_terminate))
    seed = int(rg.uniform(0, 2**32 - 1))
    res = np.empty(dim+4)
    res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
    try:
        optimizeDE_C(runid, c_callback, dim, seed,
                           array_type(*lower), array_type(*upper), 
                           array_type(*x0), array_type(*input_sigma), min_sigma,
                           bool_array_type(*ints), max_evaluations, keep, stop_fitness,  
                           popsize, f, cr, min_mutate, max_mutate, workers, res_p)
        x = res[:dim]
        val = res[dim]
        evals = int(res[dim+1])
        iterations = int(res[dim+2])
        stop = int(res[dim+3])
        return OptimizeResult(x=x, fun=val, nfev=evals, nit=iterations, status=stop, success=True)
    except Exception as ex:
        return OptimizeResult(x=None, fun=sys.float_info.max, nfev=0, nit=0, status=-1, success=False)  
      
class DE_C:

    def __init__(self,                
                 dim: Optional[int] = None,
                 bounds: Optional[Bounds] = None,
                 popsize: Optional[int] = 31,
                 keep: Optional[int] = 200,
                 f: Optional[float] = 0.5,
                 cr: Optional[float] = 0.9,
                 rg: Optional[Generator] = Generator(MT19937()),
                 ints: Optional[ArrayLike] = None,
                 min_mutate: Optional[float] = 0.1,
                 max_mutate: Optional[float] = 0.5,
                 x0: Optional[ArrayLike] = None,
                 input_sigma: Optional[Union[float, ArrayLike, Callable]] = 0.3,
                 min_sigma: Optional[float] = 0,
        ):      
        dim, lower, upper = _check_bounds(bounds, dim)     
        if popsize is None:
            popsize = 31
        if lower is None:
            lower = [0]*dim
            upper = [0]*dim
            if min_sigma == 0:
                min_sigma = 0.1 # no uniform random generation possible without bounds
        if x0 is None:
            x0 = rg.uniform(bounds.lb, bounds.ub)
        if callable(input_sigma):
            input_sigma=input_sigma()
        if np.ndim(input_sigma) == 0:
            input_sigma = [input_sigma] * dim
        if ints is None:
            ints = [False]*dim
        array_type = ct.c_double * dim   
        bool_array_type = ct.c_bool * dim 
        seed = int(rg.uniform(0, 2**32 - 1))
        try:
            self.ptr = initDE_C(0, dim, seed,
                           array_type(*lower), array_type(*upper), 
                           array_type(*x0), array_type(*input_sigma), min_sigma,
                           bool_array_type(*ints),
                           keep, popsize, f, cr, min_mutate, max_mutate)
            self.popsize = popsize
            self.dim = dim            
        except Exception as ex:
            print (ex)
            pass
 
    def __del__(self):
        destroyDE_C(self.ptr)
            
    def ask(self) -> np.array:
        try:
            popsize = self.popsize
            n = self.dim
            res = np.empty(popsize*n)
            res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
            askDE_C(self.ptr, res_p)
            xs = np.empty((popsize, n))
            for p in range(popsize):
                xs[p,:] = res[p*n : (p+1)*n]
            return xs
        except Exception as ex:
            print (ex)
            return None

    def tell(self, ys: np.ndarray):
        try:
            array_type_ys = ct.c_double * len(ys)
            return tellDE_C(self.ptr, array_type_ys(*ys))
        except Exception as ex:
            print (ex)
            return -1        

    def population(self) -> np.array:
        try:
            popsize = self.popsize
            n = self.dim
            res = np.empty(popsize*n)
            res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
            populationDE_C(self.ptr, res_p)
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
            resultDE_C(self.ptr, res_p)
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
    
    optimizeDE_C = libcmalib.optimizeDE_C
    optimizeDE_C.argtypes = [ct.c_long, mo_call_back_type, ct.c_int, ct.c_int, \
                ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
                ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_double, \
                ct.POINTER(ct.c_bool), \
                ct.c_int, ct.c_double, ct.c_double, ct.c_int, \
                ct.c_double, ct.c_double, ct.c_double, ct.c_double, 
                ct.c_int, ct.POINTER(ct.c_double)]
        
    initDE_C = libcmalib.initDE_C
    initDE_C.argtypes = [ct.c_long, ct.c_int, ct.c_int, \
                ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
                ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_double, \
                ct.POINTER(ct.c_bool), \
                ct.c_double, ct.c_int, \
                ct.c_double, ct.c_double, ct.c_double, ct.c_double]
    
    initDE_C.restype = ct.c_void_p   
    
    destroyDE_C = libcmalib.destroyDE_C
    destroyDE_C.argtypes = [ct.c_void_p]
    
    askDE_C = libcmalib.askDE_C
    askDE_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
    
    tellDE_C = libcmalib.tellDE_C
    tellDE_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
    tellDE_C.restype = ct.c_int
    
    populationDE_C = libcmalib.populationDE_C
    populationDE_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
    
    resultDE_C = libcmalib.resultDE_C
    resultDE_C.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]

