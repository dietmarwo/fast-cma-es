# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

import sys
import math
import os
import numpy as np
import ctypes as ct
from scipy.optimize import Bounds

basepath = os.path.dirname(os.path.abspath(__file__))
if sys.platform.startswith('linux'):
    libcmalib = ct.cdll.LoadLibrary(basepath + '/../fcmaes/lib/libacmalib.so')  
elif 'mac' in sys.platform:
    libcmalib = ct.cdll.LoadLibrary(basepath + '/../fcmaes/lib/libacmalib.dylib')  
else:
    os.environ['PATH'] = (basepath + '/lib') + os.pathsep + os.environ['PATH']
    libcmalib = ct.cdll.LoadLibrary(basepath + '/../fcmaes/lib/libacmalib.dll')  

objectives_re_C = libcmalib.objectives_re_C
objectives_re_C.argtypes = [ct.c_char_p, ct.POINTER(ct.c_double)]

objectives_re_C.restype = ct.POINTER(ct.c_double)         

bounds_re_C = libcmalib.bounds_re_C
bounds_re_C.argtypes = [ct.c_char_p]

bounds_re_C.restype = ct.POINTER(ct.c_double)         

freemem = libcmalib.free_mem
freemem.argtypes = [ct.POINTER(ct.c_double)]

def objectives_re(name, x, numVars, numObjs):
    array_type = ct.c_double * numVars   
    try:
        res = objectives_re_C(ct.create_string_buffer(name.encode('utf-8')), array_type(*x))
        y = np.array(np.fromiter(res, dtype=np.float64, count=numObjs))
        freemem(res)
        return y
    except Exception as ex:
        return None 

def bounds_re(name):
    try:
        res = bounds_re_C(ct.create_string_buffer(name.encode('utf-8')))
        numVars = int(res[0])
        numObjs = int(res[1])
        numConstr = int(res[2])
        lower = np.array(res[3:3+numVars])
        upper = np.array(res[3+numVars:3+2*numVars])
        freemem(res)
        return numVars, numObjs, numConstr, lower, upper
    except Exception as ex:
        return None 
  
class re_problem(object):
    
    def __init__(self, name, weight_bounds = Bounds([0, 0], [1, 1])):
        self.name = name
        if bounds_re(name) is None:
            raise Exception("re function is not implemented")
        self.numVars, self.numObjs, self.numConstr, lower, upper = bounds_re(name)
        self.bounds = Bounds(lower, upper)
        self.weight_bounds = weight_bounds # weighting of objectives
 
    def fun(self, x):
        y = objectives_re(self.name, x, self.numVars, self.numObjs)
        return y

from fcmaes.optimizer import de_cma, Bite_cpp
from fcmaes import moretry
from moexamples import mo_retry

def main():
    rep = re_problem('RE21', weight_bounds = Bounds([0, 10], [0.001, 100]) )
    #rep = re_problem('RE31', weight_bounds = Bounds([0.1, 0.0001, 0.1], [1, 0.001, 1]) )
    #rep = re_problem('RE24', weight_bounds = Bounds([0.1, 0.1], [1, 1]) )
    #rep = re_problem('RE42', weight_bounds = Bounds([0.2, 0.2, 0.2, 1000], [1, 1, 1, 1000]) )
    mo_retry(rep, de_cma(10000), '_decma_front', num_retries = 3200, exp = 2.0)

if __name__ == '__main__':
    main()

