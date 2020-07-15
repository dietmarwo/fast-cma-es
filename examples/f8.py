# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# This example is taken from https://mintoc.de/index.php/F-8_aircraft

# The F-8 aircraft control problem is based on a very simple aircraft model. 
# The control problem was introduced by Kaya and Noakes and aims at controlling 
# an aircraft in a time-optimal way from an initial state to a terminal state.

# please do: 'pip install jitcode', https://github.com/neurophysik/jitcode

# For your own differential equations optimization problem check jitcode argument 
# 'helpers' for common subexpressions which could speed up things further. See 
# for example https://github.com/neurophysik/jitcode/blob/master/examples/SW_of_Roesslers.py

from jitcode import jitcode, y
from scipy.integrate import ode
from symengine import Symbol
from scipy.optimize import Bounds
import numpy as np
import multiprocessing as mp
import ctypes as ct
import math
import warnings
import time 
from fcmaes.optimizer import logger, Sequence, Cma_cpp, GCLDE_cpp, dtime
from fcmaes import advretry, cmaes, cmaescpp, gcldecpp

# use only the compiled function, not the integrator-wrapper because of parallelism
def get_compiled_function(f, control_pars):
    dummy = jitcode(f, verbose=False, control_pars = control_pars)
    dummy.compile_C()
    return dummy.f

ksi = 0.05236
w = Symbol("w")
    
f8_equations = [
    -0.877*y(0) + y(2) - 0.088*y(0)*y(2) + 0.47*y(0)**2 - 0.019*y(1)**2 - y(0)**2*y(2) + 3.846*y(0)**3 
    +0.215*ksi - 0.28*y(0)**2*ksi + 0.47*y(0)*ksi**2 - 0.63*ksi**3 
    -(0.215*ksi - 0.28*y(0)**2*ksi - 0.63*ksi**3)*2*w,
    y(2),
    -4.208*y(0) - 0.396*y(2) - 0.47*y(0)**2 - 3.564*y(0)**3
    +20.967*ksi - 6.265*y(0)**2*ksi + 46*y(0)*ksi**2 -61.4*ksi**3
    -(20.967*ksi - 6.265*y(0)**2*ksi - 61.4*ksi**3)*2*w
    ]
 
times = 0.1+np.arange(0,1,0.00001)
rtol = 1e-8
atol = 1e-8
# speed up evaluation of the objective function by compiling the differential equations
compiled_f8 = get_compiled_function(f8_equations, [w])
# shared with all parallel processes
best_f = mp.RawValue(ct.c_double, math.inf) 

def obj_f(X):
    try:
        t = 0.
        y = [0.4655, 0., 0.]
        I = ode(compiled_f8)
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
        global best_f
        if best_f.value > val:
            best_f.value = val
            # monitor value and constraint violation
            print("val = {0:.8f} penalty = {1:.8f} f(xmin) = {2:.5f}"
                  .format(val0, penalty, val))
        return val
    except Exception:
        return 1E10 # fail
    
def test_cordinated_retry(dim = 6):
    # coordinated retry with default optimizer
    return advretry.minimize(obj_f, bounds(dim), logger=logger()) 

def test_cordinated_retry_GCL(dim = 6):
    # coordinated retry with GCLDE->CMA sequence optimizer
    return advretry.minimize(obj_f, bounds(dim), logger=logger(), 
                             optimizer=Sequence([GCLDE_cpp(750), Cma_cpp(750, popsize=13)])) 

def test_cordinated_retry_cma(dim = 6):
    # coordinated retry with CMA-ES optimizer with reduced popsize
    # faster for small dimension, use default for dim > 12
    return advretry.minimize(obj_f, bounds(dim), logger=logger(), optimizer=Cma_cpp(2000, popsize=13))
        
def test_cma_parallel(dim = 6):
    # parallel function evaluation using CMA-ES
    t0 = time.perf_counter();
    for i in range(100000):
        ret = cmaes.minimize(obj_f, bounds(dim), popsize=32, max_evaluations = 4000000, workers = mp.cpu_count())
        print("{0}: time = {1:.1f} fun = {2:.3f}"
              .format(i+1, dtime(t0), ret.fun)) 
    return ret

def test_cmacpp_parallel(dim = 6):
    # parallel function evaluation using CMA-ES cpp
    t0 = time.perf_counter();
    for i in range(100000):
        ret = cmaescpp.minimize(obj_f, bounds(dim), popsize=32, max_evaluations = 4000000, workers = mp.cpu_count())
        print("{0}: time = {1:.1f} fun = {2:.3f}"
              .format(i+1, dtime(t0), ret.fun)) 
    return ret

def test_gcldecpp_parallel(dim = 6):
    # parallel function evaluation using GCL_DE
    t0 = time.perf_counter();
    for i in range(100000):
        ret = gcldecpp.minimize(obj_f, bounds(dim), popsize=256, max_evaluations = 500000, 
                                workers = mp.cpu_count())
        print("{0}: time = {1:.1f} fun = {2:.3f}"
              .format(i+1, dtime(t0), ret.fun)) 
    return ret
    
def bounds(n):
    lb = [0]*n
    ub = [2]*n
    return Bounds(lb,ub)

if __name__ == '__main__':
    
    dim = 6
    #ret = test_cordinated_retry(dim)
    #ret = test_cordinated_retry_GCL(dim)
    #ret = test_cordinated_retry_cma(dim)
    #ret = test_cmacpp_parallel(dim)
    ret = test_gcldecpp_parallel(dim)

