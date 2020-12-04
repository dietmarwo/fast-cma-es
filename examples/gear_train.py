# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# A compound gear train is to be designed to achieve a specific gear ratio between the
# driver and driven shafts. The objective of the gear train design is to find the number
# of teeth in each of the four gears so as to minimize 
# - the error between the obtained gear ratio and a required gear rafio of 1/6.931 and 
# - the maximum size of any of the four gears. 
# Since the number of teeth must be integers, all four variables are strictly integers.
# This example is taken from https://link.springer.com/chapter/10.1007/3-540-45356-3_84 

# This example illustrates that for relatively simple tasks no dedicated mixed integer
# solver is required. 500 solutions with different upper bounds are computed in about 80 sec
# on an AMD 3950x 16 core processor. The continuous input vector is made discrete by
# calling '_feasible(x)' in the objective function 'feasable_ratio()'. The second objective
# is modeled as boxed boundary. By incrementally increasing the bounds on the teeth number
# the whole pareto front is listed. 

from fcmaes import retry, advretry
import math
import time
import numpy as np
from scipy.optimize import Bounds
from fcmaes.optimizer import dtime

fac = 1.0
lowerbound = [12, 12, 12, 12]

def ratio(x):   
    return (1 / 6.931 - x[0]*x[1] / (x[2]*x[3])) ** 2

# map floats to discrete integer variables
def discrete(x):
    feasible_x = x - x % fac
    if feasible_x < 12:
        feasible_x += fac
    if feasible_x > 60:
        feasible_x = 60
    return feasible_x
 
def _feasible(x):
    x = np.array(x)
    return [discrete(v) for v in x]
 
def feasable_ratio(x): 
    x = _feasible(x)   
    return ratio(x) 

def print_result(ret, best, t0, i):
    x = _feasible(ret.x) # make sure result is _feasible / discrete
    val = ratio(x)
    if val < best:
        best = min(val, best)
    print("{0}: time = {1:.1f} best = {2:.3E} f(xmin) = {3:.3E} x = {4:s}"
              .format(i, dtime(t0), best, val, str(x)))
    return best
 
def test_optimizer(fun, n):
    best = math.inf
    t0 = time.perf_counter()
    for i in range(n):
        max_x = i + 12
        bounds = Bounds(lowerbound, [max_x+0.99]*4)
        ret = retry.minimize(fun, bounds, max_evaluations=10000, num_retries=100)
        best = print_result(ret, best, t0, max_x)

def test_optimizer_adv(fun, n):
    best = math.inf
    t0 = time.perf_counter()
    for i in range(n):
        max_x = i + 12
        bounds = Bounds(lowerbound, [max_x+0.99]*4)
        ret = advretry.minimize(fun, bounds, min_evaluations=5000, num_retries=100)
        best = print_result(ret, best, t0, max_x)
 
if __name__ == '__main__':
    
    test_optimizer_adv(feasable_ratio, 500)
    #test_optimizer(feasable_ratio, 500)
