'''
Created on Feb 14, 2020

@author: dwolz
'''

import time
import math
from scipy.optimize import minimize, differential_evolution, dual_annealing, shgo
from fcmaes.optimizer import dtime, random_x, typical, scale

from fcmaes import astro

def test_scipy_minimize(problem, num):
    best = math.inf
    t0 = time.perf_counter();
    for i in range(num):
        guess = random_x(problem.bounds.lb, problem.bounds.ub)
        ret = minimize(problem.fun, x0 = guess, bounds = problem.bounds)
        best = min(ret.fun, best)
        print("{0}: time = {1:.1f} best = {2:.1f} f(xmin) = {3:.1f}"
              .format(i+1, dtime(t0), best, ret.fun))

def test_shgo(problem, num):
    best = math.inf
    lb = problem.bounds.lb
    ub = problem.bounds.ub
    t0 = time.perf_counter();
    for i in range(num):
        ret = shgo(problem.fun, bounds = list(zip(lb, ub)), 
                   n = 300, sampling_method = 'sobol')
        best = min(ret.fun, best)
        print("{0}: time = {1:.1f} best = {2:.1f} f(xmin) = {3:.1f}"
              .format(i+1, dtime(t0), best, ret.fun))

def test_dual_annealing(problem, num):
    best = math.inf
    lb = problem.bounds.lb
    ub = problem.bounds.ub
    t0 = time.perf_counter();
    for i in range(num):
        ret = dual_annealing(problem.fun, bounds = list(zip(lb, ub)))
        best = min(ret.fun, best)
        print("{0}: time = {1:.1f} best = {2:.1f} f(xmin) = {3:.1f}"
              .format(i+1, dtime(t0), best, ret.fun))

def test_differential_evolution(problem, num):
    best = math.inf
    t0 = time.perf_counter();
    for i in range(num):
        ret = differential_evolution(problem.fun, bounds = problem.bounds)
        best = min(ret.fun, best)
        print("{0}: time = {1:.1f} best = {2:.1f} f(xmin) = {3:.1f}"
              .format(i+1, dtime(t0), best, ret.fun))

def test_cma_original(problem, num):
    import cma
    best = math.inf
    lb = problem.bounds.lb
    ub = problem.bounds.ub
    t0 = time.perf_counter();
    for i in range(num):
        guess = random_x(problem.bounds.lb, problem.bounds.ub)
        es = cma.CMAEvolutionStrategy(guess, 1.0,  
                                      {'bounds': [lb, ub], 'popsize': 32, 
                                        'typical_x': typical(lb, ub),
                                        'scaling_of_variables': scale(lb, ub),
                                        'verbose': -1, 'verb_disp': -1})
        for j in range(100000):
            X, Y = es.ask_and_eval(problem.fun)
            es.tell(X, Y)
            if es.stop():
                break 
        best = min(es.result.fbest, best)
        print("{0}: time = {1:.1f} best = {2:.1f} f(xmin) = {3:.1f}"
              .format(i+1, dtime(t0), best, es.result.fbest))

from fcmaes import cmaes

def test_cma_python(problem, num):
    best = math.inf
    t0 = time.perf_counter();
    for i in range(num):
        ret = cmaes.minimize(problem.fun, bounds = problem.bounds)
        best = min(ret.fun, best)
        print("{0}: time = {1:.1f} best = {2:.1f} f(xmin) = {3:.1f}"
              .format(i+1, dtime(t0), best, ret.fun))
        
from fcmaes import cmaescpp

def test_cma_cpp(problem, num):
    best = math.inf
    t0 = time.perf_counter();
    for i in range(num):
        ret = cmaescpp.minimize(problem.fun, bounds = problem.bounds)
        best = min(ret.fun, best)
        print("{0}: time = {1:.1f} best = {2:.1f} f(xmin) = {3:.1f}"
              .format(i+1, dtime(t0), best, ret.fun))

from fcmaes import retry

def test_retry_python(problem, num):
    ret = retry.minimize(problem.fun, bounds = problem.bounds, num_retries = num, 
                   max_evaluations = 50000)

def test_retry_cpp(problem, num):
    ret = retry.minimize(problem.fun, bounds = problem.bounds, num_retries = num, 
                   max_evaluations = 50000, useCpp = True)

from fcmaes import advretry

def test_advretry_python(problem, value_limit, num):
    best = math.inf
    t0 = time.perf_counter();    
    for i in range(num):
        ret = advretry.minimize(problem.fun, bounds = problem.bounds, num_retries = 4000, 
                   value_limit = value_limit, logger = None)
        best = min(ret.fun, best)
        print("{0}: time = {1:.1f} best = {2:.1f} f(xmin) = {3:.1f}"
              .format(i+1, dtime(t0), best, ret.fun))

def test_advretry_cpp(problem, value_limit, num):
    best = math.inf
    t0 = time.perf_counter();    
    for i in range(num):
        ret = advretry.minimize(problem.fun, bounds = problem.bounds, num_retries = 4000, 
                   value_limit = value_limit, logger = None, useCpp = True)
        best = min(ret.fun, best)
        print("{0}: time = {1:.1f} best = {2:.1f} f(xmin) = {3:.1f}"
              .format(i+1, dtime(t0), best, ret.fun))

if __name__ == '__main__':
    problem = astro.Gtoc1()
#     problem = astro.Cassini1()
#     problem = astro.Messenger()
#     problem = astro.MessFull()
 
#     test_scipy_minimize(problem, 1000)
#     test_shgo(problem, 2)
#     test_dual_annealing(problem, 100)
#     test_differential_evolution(problem, 100)
#     test_cma_original(problem, 100)
    test_cma_python(problem, 100)
#     test_cma_cpp(problem, 100)
#     test_retry_python(problem, 5000)
#     test_retry_cpp(problem, 5000)
#     test_advretry_python(problem, -1000000, 10)
#     test_advretry_cpp(problem, -1000000, 10)

