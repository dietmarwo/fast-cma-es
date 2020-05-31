# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

import sys
import os
import math
import ctypes as ct
import numpy as np
from numpy.random import MT19937, Generator
from scipy.optimize import OptimizeResult
from fcmaes.cmaescpp import _c_func

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

def minimize(fun, 
             dim,
             bounds = None, 
             popsize = None, 
             max_evaluations = 100000, 
             stop_fittness = None, 
             keep = 200.0,
             f = 0.5,
             cr = 0.9,
             rg = Generator(MT19937()),
             runid=0):   
                
    lower = np.asarray(bounds.lb)
    upper = np.asarray(bounds.ub)
    n = dim  
    if popsize is None:
        popsize = n*15
    if lower is None:
        lower = [0]*n
        upper = [0]*n
    if stop_fittness is None:
        stop_fittness = math.inf   
    array_type = ct.c_double * n   
    c_callback = call_back_type(_c_func(fun))
    seed = int(rg.uniform(0, 2**32 - 1))
    try:
        res = optimizeDE_C(runid, c_callback, n, seed,
                           array_type(*lower), array_type(*upper), 
                           max_evaluations, keep, stop_fittness,  
                           popsize, f, cr)
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
optimizeDE_C = libcmalib.optimizeDE_C
optimizeDE_C.argtypes = [ct.c_long, call_back_type, ct.c_int, ct.c_int, \
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), \
            ct.c_int, ct.c_double, ct.c_double, ct.c_int, \
            ct.c_double, ct.c_double]

optimizeDE_C.restype = ct.POINTER(ct.c_double)         
freemem = libcmalib.free_mem
freemem.argtypes = [ct.POINTER(ct.c_double)]
  

