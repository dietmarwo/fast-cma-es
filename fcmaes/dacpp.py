# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

import sys
import os
import ctypes as ct
import numpy as np
from numpy.random import MT19937, Generator
from scipy.optimize import OptimizeResult
from fcmaes.cmaes import _check_bounds
from fcmaes.cmaescpp import _c_func

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun,
             bounds=None, 
             x0=None, 
             max_evaluations = 100000, 
             use_local_search = True,
             rg = Generator(MT19937()),
             runid=0):   
                
    lower, upper, guess = _check_bounds(bounds, x0, rg)   
    n = guess.size   
    if lower is None:
        lower = [0]*n
        upper = [0]*n
    array_type = ct.c_double * n   
    c_callback = call_back_type(_c_func(fun))
    seed = int(rg.uniform(0, 2**32 - 1))
    try:
        res = optimizeDA_C(runid, c_callback, n, seed,
                           array_type(*guess), array_type(*lower), array_type(*upper), 
                           max_evaluations, use_local_search)
        x = np.array(np.fromiter(res, dtype=np.float64, count=n))
        val = res[n]
        evals = int(res[n+1])
        iterations = int(res[n+2])
        stop = int(res[n+3])
        freemem(res)
        return OptimizeResult(x=x, fun=val, nfev=evals, nit=iterations, status=stop, success=True)
    except Exception as ex:
        return OptimizeResult(x=None, fun=sys.float_info.max, nfev=0, nit=0, status=-1, success=False)
  
basepath = os.path.dirname(os.path.abspath(__file__))

if sys.platform.startswith('linux'):
    libcmalib = ct.cdll.LoadLibrary(basepath + '/lib/libacmalib.so')  
elif sys.platform.contains('mac'):
    libgtoplib = ct.cdll.LoadLibrary(basepath + '/lib/libacmalib.dylib')  
else:
    libcmalib = ct.cdll.LoadLibrary(basepath + '/lib/libacmalib.dll')
    
call_back_type = ct.CFUNCTYPE(ct.c_double, ct.c_int, ct.POINTER(ct.c_double))  
optimizeDA_C = libcmalib.optimizeDA_C
optimizeDA_C.argtypes = [ct.c_long, call_back_type, ct.c_int, ct.c_int, \
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
            ct.c_int, ct.c_bool]

optimizeDA_C.restype = ct.POINTER(ct.c_double)         
freemem = libcmalib.free_mem
freemem.argtypes = [ct.POINTER(ct.c_double)]
 

