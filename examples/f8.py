# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# This example is taken from https://mintoc.de/index.php/F-8_aircraft

# The F-8 aircraft control problem is based on a very simple aircraft model. 
# The control problem was introduced by Kaya and Noakes and aims at controlling 
# an aircraft in a time-optimal way from an initial state to a terminal state.

# Uses compiled differential equations based on the Ascent library 
# https://github.com/AnyarInc/Ascent see 
# https://github.com/dietmarwo/fast-cma-es/blob/master/_fcmaescpp/ascent.cpp

# Compare with the F8 results 
# given in http://www.midaco-solver.com/data/pub/The_Oracle_Penalty_Method.pdf

from scipy.integrate import ode
from scipy.optimize import Bounds
import numpy as np
import multiprocessing as mp
import ctypes as ct
import math
import warnings
import time 
from fcmaes.optimizer import logger, Sequence, Cma_cpp, GCLDE_cpp, de_cma, dtime
from fcmaes import advretry, retry, gcldecpp, csmacpp
from fcmaes.cmaescpp import libcmalib, freemem

ksi = 0.05236
    
def f8(t, y, w):
    return [
            -0.877*y[0] + y[2] - 0.088*y[0]*y[2] + 0.47*y[0]**2 - 0.019*y[1]**2 - y[0]**2*y[2] + 3.846*y[0]**3 
            +0.215*ksi - 0.28*y[0]**2*ksi + 0.47*y[0]*ksi**2 - 0.63*ksi**3 
            -(0.215*ksi - 0.28*y[0]**2*ksi - 0.63*ksi**3)*2*w,
            y[2],
            -4.208*y[0] - 0.396*y[2] - 0.47*y[0]**2 - 3.564*y[0]**3
            +20.967*ksi - 6.265*y[0]**2*ksi + 46*y[0]*ksi**2 -61.4*ksi**3
            -(20.967*ksi - 6.265*y[0]**2*ksi - 61.4*ksi**3)*2*w
            ]
 
times = 0.1+np.arange(0,1,0.00001)
rtol = 1e-8
atol = 1e-8
# shared with all parallel processes
best_f = mp.RawValue(ct.c_double, math.inf) 
f_evals = mp.RawValue(ct.c_int, 0) 

def obj_f(X):
    try:
        t = 0.
        y = [0.4655, 0., 0.]
        I = ode(f8)
        I.set_integrator("dopri5", nsteps=10**6, rtol=rtol, atol=atol)
        I.set_initial_value(y, t)
        n = len(X)     
        for i in range(n):
            if X[i] == 0:
                continue
            #  bang-bang type switches starting with w(t) = 1.
            w = (i + 1) % 2
            t += X[i]
            I.set_f_params(w)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y = I.integrate(t)
        val0 = np.sum(X)
        penalty = np.sum(np.abs(y))
        # estimated fixed weight for penalty 
        val = 0.1*val0 + penalty
        global f_evals
        f_evals.value += 1
        global best_f
        if best_f.value > val:
            best_f.value = val
            # monitor value and constraint violation
            print("val = {0:.8f} penalty = {1:.8f} f(xmin) = {2:.5f} nfev = {3}"
                  .format(val0, penalty, val, f_evals.value))
        return val
    except Exception:
        return 1E10 # fail


def obj_f_c(X):
    try:
        y = np.asarray([0.4655, 0., 0.])
        array_type = ct.c_double * y.size 
        n = len(X)     
        for i in range(n):
            if X[i] == 0:
                continue
            #  bang-bang type switches starting with w(t) = 1.
            w = (i + 1) % 2
            ry = integrateF8_C(array_type(*y), w, X[i], 0.1)
            y = np.array(np.fromiter(ry, dtype=np.float64, count=y.size))
            freemem(ry)

        val0 = np.sum(X)
        penalty = np.sum(np.abs(y))
        # estimated fixed weight for penalty 
        val = 0.1*val0 + penalty
        global f_evals
        f_evals.value += 1
        global best_f
        if best_f.value > val:
            best_f.value = val
            # monitor value and constraint violation
            print("val = {0:.8f} penalty = {1:.8f} f(xmin) = {2:.5f} nfev = {3}"
                  .format(val0, penalty, val, f_evals.value))
        return val
    except Exception as e:
        return 1E10 # fail

    
def test_default_cordinated_retry(dim = 6):
    # coordinated retry with default optimizer
    return advretry.minimize(obj_f_c, bounds(dim), logger=logger()) 

def test_gclde_cordinated_retry(dim = 6):
    # coordinated retry with GCLDE->CMA sequence optimizer
    return advretry.minimize(obj_f_c, bounds(dim), logger=logger(), 
                             optimizer=Sequence([GCLDE_cpp(750), Cma_cpp(750, popsize=13)])) 

def test_cma_cordinated_retry(dim = 6):
    # coordinated retry with CMA-ES optimizer with reduced popsize
    # faster for small dimension, use default for dim > 12
    return advretry.minimize(obj_f_c, bounds(dim), logger=logger(), optimizer=Cma_cpp(2000, popsize=13))
        
def test_de_cma_parallel_retry(dim = 6):
    # parallel optimization retry using a DE -> CMA-ES sequence
    t0 = time.perf_counter();
    evals = 0
    for i in range(1000):
        ret = retry.minimize(obj_f_c, bounds(dim), logger=logger(), optimizer=de_cma(50000), 
                             value_limit = 10)

        evals += ret.nfev
        print("{0}: time = {1:.1f} fun = {2:.3f} nfev = {3}"
              .format(i+1, dtime(t0), ret.fun, evals)) 
    return ret

def test_cma_parallel_eval(dim = 6):
    # parallel function evaluation using CMA-ES
    t0 = time.perf_counter();
    evals = 0
    for i in range(1000):
        ret = csmacpp.minimize(obj_f_c, bounds(dim), popsize=32, max_evaluations = 50000, workers = mp.cpu_count())
        evals += ret.nfev
        print("{0}: time = {1:.1f} fun = {2:.3f} nfev = {3}"
              .format(i+1, dtime(t0), ret.fun, evals)) 
    return ret

def test_gclde_parallel_eval(dim = 6):
    # parallel function evaluation using GCL_DE
    t0 = time.perf_counter();
    evals = 0
    for i in range(100000):
        ret = gcldecpp.minimize(obj_f_c, bounds(dim), popsize=256, max_evaluations = 200000, 
                                workers = mp.cpu_count())
        evals += ret.nfev
        print("{0}: time = {1:.1f} fun = {2:.3f} nfev = {3}"
              .format(i+1, dtime(t0), ret.fun, evals)) 
    return ret
    
def bounds(n):
    lb = [0]*n
    ub = [2]*n
    return Bounds(lb,ub)

integrateF8_C = libcmalib.integrateF8_C
integrateF8_C.argtypes = [ct.POINTER(ct.c_double), ct.c_double, ct.c_double, ct.c_double]
integrateF8_C.restype = ct.POINTER(ct.c_double)   

if __name__ == '__main__':
    
    dim = 6
    #ret = test_default_cordinated_retry(dim)
    #ret = test_gclde_cordinated_retry(dim)
    #ret = test_cma_cordinated_retry(dim)
    ret = test_de_cma_parallel_retry(dim)
    #ret = test_cma_parallel_eval(dim)
    #ret = test_gclde_parallel_eval(dim)

