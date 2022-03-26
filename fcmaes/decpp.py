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
import math
import ctypes as ct
import numpy as np
from numpy.random import MT19937, Generator
from scipy.optimize import OptimizeResult
from fcmaes import de

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun, 
             dim = None,
             bounds = None, 
             popsize = None, 
             max_evaluations = 100000, 
             stop_fitness = None, 
             keep = 200,
             f = 0.5,
             cr = 0.9,
             rg = Generator(MT19937()),
             ints = None,
             workers = 1,
             is_terminate = None,
             runid=0):  
     
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
    workers : int or None, optional
        If not workers is None, function evaluation is performed in parallel for the whole population. 
        Useful for costly objective functions but is deactivated for parallel retry.      
    is_terminate : callable, optional
        Callback to be used if the caller of minimize wants to decide when to terminate.
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
    
    dim, lower, upper = de._check_bounds(bounds, dim)
    if popsize is None:
        popsize = 31
    if lower is None:
        lower = [0]*dim
        upper = [0]*dim
    if ints is None:
        ints = [False]*dim
    if workers is None:
        workers = 0
    if stop_fitness is None:
        stop_fitness = math.inf   
    array_type = ct.c_double * dim   
    bool_array_type = ct.c_bool * dim 
    c_callback = mo_call_back_type(callback(fun, dim, is_terminate))
    seed = int(rg.uniform(0, 2**32 - 1))
    res = np.empty(dim+4)
    res_p = res.ctypes.data_as(ct.POINTER(ct.c_double))
    try:
        optimizeDE_C(runid, c_callback, dim, seed,
                           array_type(*lower), array_type(*upper), bool_array_type(*ints),
                           max_evaluations, keep, stop_fitness,  
                           popsize, f, cr, workers, res_p)
        x = res[:dim]
        val = res[dim]
        evals = int(res[dim+1])
        iterations = int(res[dim+2])
        stop = int(res[dim+3])
        return OptimizeResult(x=x, fun=val, nfev=evals, nit=iterations, status=stop, success=True)
    except Exception as ex:
        return OptimizeResult(x=None, fun=sys.float_info.max, nfev=0, nit=0, status=-1, success=False)  

class callback(object):
    
    def __init__(self, fun, dim, is_terminate = None):
        self.fun = fun
        self.dim = dim
        self.nobj = 1
        self.is_terminate = is_terminate
    
    def __call__(self, dim, x, y):
        try:
            arrTypeX = ct.c_double*(self.dim)
            xaddr = ct.addressof(x.contents)
            xbuf = np.frombuffer(arrTypeX.from_address(xaddr))
            arrTypeY = ct.c_double*(self.nobj)
            yaddr = ct.addressof(y.contents)   
            ybuf = np.frombuffer(arrTypeY.from_address(yaddr))  
            fit = self.fun(xbuf)
            ybuf[0] = fit if math.isfinite(fit) else sys.float_info.max
            return False if self.is_terminate is None else self.is_terminate(xbuf, ybuf) 
        except Exception as ex:
            print (ex)
            return False


class callback_mo(object):
    
    def __init__(self, fun, dim, nobj, is_terminate = None):
        self.fun = fun
        self.dim = dim
        self.nobj = nobj
        self.is_terminate = is_terminate
    
    def __call__(self, dim, x, y):
        try:
            arrTypeX = ct.c_double*(dim)
            xaddr = ct.addressof(x.contents)
            xbuf = np.frombuffer(arrTypeX.from_address(xaddr))
            arrTypeY = ct.c_double*(self.nobj)
            yaddr = ct.addressof(y.contents)   
            ybuf = np.frombuffer(arrTypeY.from_address(yaddr))  
            ybuf[:] = self.fun(xbuf)[:]
            return False if self.is_terminate is None else self.is_terminate(xbuf, ybuf) 
        except Exception as ex:
            print (ex)
            return False


mo_call_back_type = ct.CFUNCTYPE(ct.c_bool, ct.c_int, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double))  

basepath = os.path.dirname(os.path.abspath(__file__))

if sys.platform.startswith('linux'):
    libcmalib = ct.cdll.LoadLibrary(basepath + '/lib/libacmalib.so')  
elif 'mac' in sys.platform or 'darwin' in sys.platform:
    libcmalib = ct.cdll.LoadLibrary(basepath + '/lib/libacmalib.dylib')  
else:
    os.environ['PATH'] = (basepath + '/lib') + os.pathsep + os.environ['PATH']
    libcmalib = ct.cdll.LoadLibrary(basepath + '/lib/libacmalib.dll')  
      
optimizeDE_C = libcmalib.optimizeDE_C
optimizeDE_C.argtypes = [ct.c_long, mo_call_back_type, ct.c_int, ct.c_int, \
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_bool), \
            ct.c_int, ct.c_double, ct.c_double, ct.c_int, \
            ct.c_double, ct.c_double, ct.c_int, ct.POINTER(ct.c_double)]
    
