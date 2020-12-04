# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# This problem consists of minimizing the weight of a tension/compression spring subject to constraints on 
# shear stress, surge frequency and minimum deflection
# The design variables are:

# - x1: the mean coil diameter
# - x2: the wire diameter
# - x3: the number of active coils

# which are subject to four inequality constraints.
# fcmaes retry used with a penalty for constraint violations 'weight_penalty'
# works as well as scipy minimize.

# This example is taken from https://www.sciencedirect.com/science/article/abs/pii/S0096300306015098

import math
import time
import numpy as np
from scipy.optimize import Bounds, minimize
from fcmaes.optimizer import dtime, random_x, logger
from fcmaes import retry, advretry

bounds = Bounds([0.01, 0.01, 0.01], [20, 20, 20])

def _feasible(x):
    x = np.array(x)
    return np.maximum(np.minimum(x, bounds.ub), bounds.lb)

def constraint_ineq(x):
    return [ x[1]**3 * x[2] / (71785 * x[0]**4) - 1,
             1 - (4*x[1]**2 - x[0]*x[1]) / (12566*(x[1]*x[0]**3 - x[0]**4)) - 1/(5108*x[0]**2),
             140.45*x[0]/(x[1]**2 * x[2]) - 1,
             1 - (x[1] + x[0]) / 1.5]

def penalty(x):
    return - np.sum(np.minimum(constraint_ineq(x), 0))

def weight(x):   
    return (x[2] + 2) * x[1]*x[0]**2

def weight_penalty(x): 
    try:
        val = weight(x) 
        return val + 100000*penalty(x)
    except ZeroDivisionError:
        return 1E99
        
def print_result(ret, best, t0, i):
    x = _feasible(ret.x) # make sure result is _feasible
    w = weight(x)
    val = weight_penalty(x) # add penalty for ineq constraint violation
    if val < best:
        pen = penalty(x) # check ineq constraint
        best = min(val, best)
        print("{0}: time = {1:.1f} best = {2:.8f} f(xmin) = {3:.8f} ineq = {4:.5f}"
              .format(i+1, dtime(t0), best, w, pen))
    return best

def test_minimize_SLSQP(fun, num):
    ineq_cons = {'type': 'ineq', 'fun' : constraint_ineq}

    best = math.inf
    t0 = time.perf_counter();
    for i in range(num):
        guess = random_x(bounds.lb, bounds.ub)
        ret = minimize(fun, x0 = guess, bounds = bounds,
                       method='SLSQP', constraints=[ineq_cons])
        if ret.success:
            best = print_result(ret, best, t0, i)

if __name__ == '__main__':
    
    # try as alternative 
    # test_minimize_SLSQP(weight, 10000)
    
    t0 = time.perf_counter();
    ret = advretry.minimize(weight_penalty, bounds, logger = logger(), num_retries = 320)
    #ret = retry.minimize(weight_penalty, bounds, logger = logger(), num_retries=32)
    print_result(ret, 10000, t0, 0)
