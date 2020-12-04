# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# A cylindrical vessel is capped at both ends by hemispherical heads. The objective is to
# minimize the total weight, including the weight of material, forming and welding. There are four design variables:

# - x1: thickness of the shell
# - x2: thickness of the head
# - x3: inner radius
# - x4 length of cylindrical section of the vessel, not including the head

# x1 and x2 are integer multiples of 0.0625 inch, witch are
# the available thickness of rolled steel plates, 
# and x3 and x4 are continuous.
# x1 and x2 are made discrete by calling '_feasible(x)' in the objective function 'feasable_cost_penalty()
# This works well with derivative free optimizers, but never do this with scipy.minimize. 
# Try 'test_minimize_SLSQP' to see why. But this can be fixed by using 'cost_int_penalty' as
# objective function which adds a penalty for the "multiples of 0.0625" violation. 
# With fcmaes retry usually you can just "discretize" arguments. 

# This example is taken from https://www.sciencedirect.com/science/article/abs/pii/S0096300306015098

import math
import time
import numpy as np
from scipy.optimize import Bounds, minimize
from fcmaes import retry, advretry
from fcmaes.optimizer import dtime, random_x, logger

fac = 0.0625
bounds = Bounds([1.1, 0.6, 0, 0], [240, 240, 240, 240]) # six inequalities
#bounds = Bounds([0, 0, 0, 0], [240, 240, 240, 240]) # four inequalities

def discrete(x):
    feasible_x = x - x % fac
    if feasible_x < bounds.lb[0]:
        feasible_x += fac
    return feasible_x

def _feasible(x):
    x = np.array(x)
    x[0] = discrete(x[0])
    x[1] = discrete(x[1])
    return np.maximum(np.minimum(x, bounds.ub), bounds.lb)

def constraint_ineq(x):
    return [x[0] - 0.0193*x[2],
            x[1] - 0.00954*x[2],
            math.pi*x[2]**2 * x[3] + (4/3)*math.pi*x[2]**3 - 1296000]

def weight(x):   
    return 0.6224*x[0]*x[2]*x[3] + 1.7781*x[1]*x[2]**2 \
                + 3.1661 * x[0]**2 * x[3] + 19.84*x[0]**2 * x[2]

def penalty(x):
    return -np.sum(np.minimum(constraint_ineq(x), 0))

def feasable_cost(x): 
    x = _feasible(x)   
    return weight(x)

def feasable_cost_penalty(x): 
    return feasable_cost(x) + 100000*penalty(x)

def penalty_int(x):
    feasible_x = x - x % fac
    if x - feasible_x < 0.5 * fac:
        return x - feasible_x
    else:
        return feasible_x + fac - x

def penalty_int_all(x):
    return penalty_int(x[0]) + penalty_int(x[1])

def cost_int_penalty(x): 
    return feasable_cost(x) + 100000*penalty_int_all(x)

def print_result(ret, best, t0, i):
    val = feasable_cost_penalty(ret.x) 
    x = _feasible(ret.x) # make sure result is _feasible
    if val < best:
        best = val
        print("{0}: time = {1:.1f} best = {2:.8f} f(xmin) = {3:.5f} ineq = {4:.8f} x = {5:s}"
              .format(i+1, dtime(t0), best, weight(x), penalty(x), str(x)))
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
    # test_minimize_SLSQP(feasable_cost, 10000)
    # works much better
    # test_minimize_SLSQP(cost_int_penalty, 10000)
    
    t0 = time.perf_counter();
    ret = advretry.minimize(feasable_cost_penalty, bounds, logger = logger(), num_retries=320)
    #ret = retry.minimize(feasable_cost_penalty, bounds, logger = logger(), num_retries=32)
    print_result(ret, 10000, t0, 0)
