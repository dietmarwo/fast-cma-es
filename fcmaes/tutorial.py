# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

import math
import time
from fcmaes import astro, advretry, retry, cmaes, cmaescpp
from fcmaes.optimizer import dtime, random_x, Cma_python
from scipy.optimize import minimize, differential_evolution, dual_annealing

def test_advretry(problem, value_limit, num):
    best = math.inf
    t0 = time.perf_counter();    
    for i in range(num):
        ret = advretry.minimize(problem.fun, bounds = problem.bounds, num_retries = 4000, 
                   value_limit = value_limit)
        best = min(ret.fun, best)
        print("{0}: time = {1:.1f} best = {2:.1f} f(xmin) = {3:.1f}"
              .format(i+1, dtime(t0), best, ret.fun))

def test_advretry_cma_python(problem, value_limit, num):
    best = math.inf
    t0 = time.perf_counter();    
    for i in range(num):
        ret = advretry.minimize(problem.fun, bounds = problem.bounds, num_retries = 4000, 
                   optimizer = Cma_python(2000), value_limit = value_limit)
        best = min(ret.fun, best)
        print("{0}: time = {1:.1f} best = {2:.1f} f(xmin) = {3:.1f}"
              .format(i+1, dtime(t0), best, ret.fun))

def test_retry(problem, num, log=None):
    best = math.inf
    t0 = time.perf_counter();    
    for i in range(num):
        ret = retry.minimize(problem.fun, bounds = problem.bounds, 
                num_retries = 2000, max_evaluations = 100000, logger=log)
        best = min(ret.fun, best)
        print("{0}: time = {1:.1f} best = {2:.1f} f(xmin) = {3:.1f}"
              .format(i+1, dtime(t0), best, ret.fun))

def test_retry_cma_python(problem, num):
    best = math.inf
    t0 = time.perf_counter();    
    for i in range(num):
        ret = retry.minimize(problem.fun, bounds = problem.bounds, 
                num_retries = 2000, optimizer = Cma_python(100000))
        best = min(ret.fun, best)
        print("{0}: time = {1:.1f} best = {2:.1f} f(xmin) = {3:.1f}"
              .format(i+1, dtime(t0), best, ret.fun))

def test_cma_python(problem, num):
    best = math.inf
    t0 = time.perf_counter();
    for i in range(num):
        ret = cmaes.minimize(problem.fun, max_evaluations = 100000, bounds = problem.bounds)
        if best > ret.fun or i % 100 == 99:
            print("{0}: time = {1:.1f} best = {2:.1f} f(xmin) = {3:.1f}"
              .format(i+1, dtime(t0), best, ret.fun))
        best = min(ret.fun, best)

def test_cma_cpp(problem, num):
    best = math.inf
    t0 = time.perf_counter();
    for i in range(num):
        ret = cmaescpp.minimize(problem.fun, max_evaluations = 100000, bounds = problem.bounds)
        if best > ret.fun or i % 100 == 99:
            print("{0}: time = {1:.1f} best = {2:.1f} f(xmin) = {3:.1f}"
              .format(i+1, dtime(t0), best, ret.fun))
        best = min(ret.fun, best)
        
def test_ask_tell(problem, num):    
    best = math.inf
    t0 = time.perf_counter();
    for i in range(num):
        es = cmaes.Cmaes(bounds = problem.bounds)
        iters = 3000
        for _ in range(iters):
            xs = es.ask()
            ys = [problem.fun(x) for x in xs]
            stop = es.tell(ys)
            if stop != 0:
                break 
        best = min(es.best_value, best)
        print("{0}: time = {1:.1f} best = {2:.1f} f(xmin) = {3:.1f}"
              .format(i+1, dtime(t0), best, es.best_value))

def test_cma_parallel(problem, num):
    best = math.inf
    t0 = time.perf_counter();
    for i in range(num):
        ret = cmaes.minimize(problem.fun, is_parallel = True, bounds = problem.bounds)
        if best > ret.fun or i % 100 == 99:
            print("{0}: time = {1:.1f} best = {2:.1f} f(xmin) = {3:.1f}"
              .format(i+1, dtime(t0), best, ret.fun))
        best = min(ret.fun, best)

def test_differential_evolution(problem, num):
    best = math.inf
    t0 = time.perf_counter();
    for i in range(num):
        ret = differential_evolution(problem.fun, bounds = problem.bounds)
        if best > ret.fun or i % 100 == 99:
            print("{0}: time = {1:.1f} best = {2:.1f} f(xmin) = {3:.1f}"
              .format(i+1, dtime(t0), best, ret.fun))
        best = min(ret.fun, best)

def test_dual_annealing(problem, num):
    best = math.inf
    lb = problem.bounds.lb
    ub = problem.bounds.ub
    t0 = time.perf_counter();
    for i in range(num):
        ret = dual_annealing(problem.fun, bounds = list(zip(lb, ub)))
        if best > ret.fun or i % 100 == 99:
            print("{0}: time = {1:.1f} best = {2:.1f} f(xmin) = {3:.1f}"
              .format(i+1, dtime(t0), best, ret.fun))
        best = min(ret.fun, best)

def test_scipy_minimize(problem, num):
    best = math.inf
    t0 = time.perf_counter();
    for i in range(num):
        guess = random_x(problem.bounds.lb, problem.bounds.ub)
        ret = minimize(problem.fun, x0 = guess, bounds = problem.bounds)
        if best > ret.fun or i % 20000 == 19999:
            print("{0}: time = {1:.1f} best = {2:.1f} f(xmin) = {3:.1f}"
              .format(i+1, dtime(t0), best, ret.fun))
        best = min(ret.fun, best)

if __name__ == '__main__':

    problem = astro.Gtoc1()
#     problem = astro.Cassini1()
#     problem = astro.Cassini2()
#     problem = astro.Rosetta()
#     problem = astro.Messenger()
#     problem = astro.MessFull()

    test_advretry(problem, 20.0, 10)
#     test_advretry_cma_python(problem, 20.0, 10)
#     test_retry(problem, 10, logger())
#     test_retry_cma_python(problem, 10) 
#     test_cma_python(problem, 1000)
#     test_cma_cpp(problem, 1000)
#     test_ask_tell(problem, 10000)
#     test_cma_parallel(problem, 1000)
#     test_differential_evolution(problem, 1000)
#     test_dual_annealing(problem, 1000)
#     test_scipy_minimize(problem, 200000)

