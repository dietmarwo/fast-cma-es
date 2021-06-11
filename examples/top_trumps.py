# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# The gbea TopTrump benchmark is a carefully designed real world benchmark. 
# Both its single objective and multi-objective fitness functions reflect the requirements of 
# a real world TopTrump card game designer. Its simulation based tests are efficiently implemented, 
# so that it is possible to compare optimization algorithms investing limited CPU time, specially 
# if parallelization is applied. To do so the socket based interface is replaced by a much simpler 
# ctypes based interface. 

# https://github.com/ttusar/coco-gbea
# https://doi.org/10.5281/zenodo.2594848
# https://github.com/ttusar/coco-gbea/blob/main/code-experiments/rw-problems/GBEA.md
# https://www.researchgate.net/publication/334220017_Single-_and_multi-objective_game-benchmark

import sys
import os
import numpy as np
import ctypes as ct
from scipy.optimize import Bounds

basepath = os.path.dirname(os.path.abspath(__file__))
if sys.platform.startswith('linux'):
    librw = ct.cdll.LoadLibrary(basepath + '/../fcmaes/lib/librw_top_trumps.so')  
else:
    os.environ['PATH'] = (basepath + '/lib') + os.pathsep + os.environ['PATH']
    librw = ct.cdll.LoadLibrary(basepath + '/../fcmaes/lib/librw_top_trumps.dll')  

# configurable number of simulations
evaluate_rw_top_trumps = librw.evaluate_rw_top_trumps
evaluate_rw_top_trumps.argtypes = [ct.c_int, ct.c_char_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
                                   ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]       

rw_top_trumps_bounds = librw.rw_top_trumps_bounds
rw_top_trumps_bounds.argtypes = [ct.c_int, ct.c_int, 
                                 ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]   

def objectives_rw(name, numObjs, function, instance, rep, x):
    try:
        x = [round(xi, 0) for xi in x]
        x = np.array(x)
        y = np.empty(numObjs)
        x_p = x.ctypes.data_as(ct.POINTER(ct.c_double))
        y_p = y.ctypes.data_as(ct.POINTER(ct.c_double))  

        evaluate_rw_top_trumps(rep, ct.create_string_buffer(name.encode('utf-8')), 
                               numObjs, function, instance, len(x), x_p, y_p)
        
        return y
    except Exception as ex:
        return None 

def bounds_rw(dim, instance):
    try:
        lb = np.empty(4)   
        ub = np.empty(4) 
        lb_p = lb.ctypes.data_as(ct.POINTER(ct.c_double))
        ub_p = ub.ctypes.data_as(ct.POINTER(ct.c_double))  
        rw_top_trumps_bounds(instance, 4, lb_p, ub_p)
        lb[:] -= (0.5 - 1e-6)
        ub[:] += (0.5 - 1e-6)
        lower = np.empty(dim)
        upper = np.empty(dim)
        for i in range(dim):
            lower[i] = lb[i % 4]
            upper[i] = ub[i % 4]
        return lower, upper
    except Exception as ex:
        return None 

class tt_problem(object):

    def __init__(self, suite, name, dim, numObjs, function, instance, rep = 2000, weight_bounds = Bounds([0, 0], [1, 1])):
        self.suite = suite
        self.name = name
        self.dim = dim
        self.numObjs = numObjs
        self.function = function
        self.instance = instance
        self.rep = rep
        lb, ub = bounds_rw(dim, instance)
        self.weight_bounds = weight_bounds
        self.bounds = Bounds(lb, ub)

    def fun(self, x):
        return objectives_rw(self.suite, self.numObjs, self.function, self.instance, self.rep, x)

from fcmaes.optimizer import de_cma, Bite_cpp, Cma_cpp, LDe_cpp, dtime,  De_cpp, random_search, wrapper, logger
from fcmaes import moretry, advretry, retry, mode

def mo_minimize_plot(problem, opt, name, exp = 3.0, num_retries = 256):
    moretry.minimize_plot(name, opt, wrapper(problem.fun), problem.bounds, problem.weight_bounds, 
                          num_retries = num_retries, exp = exp)

def minimize_plot(problem, opt, name, num_retries = 256):
    retry.minimize_plot(name, opt, problem.fun, problem.bounds,
                          num_retries = num_retries)

from fcmaes.optimizer import wrapper

def main():
    
    suite = 'rw-top-trumps'
    function = 5
    instance = 5
    dim = 128
    nobj = 1
    rep = 2000
    name = suite + '_f' + str(function) + 'i' + str(instance) + 'd' + str(dim)
    
    problem = tt_problem(suite, name, dim, nobj, function, instance, rep)

    minimize_plot(problem, random_search(10000), name + '_10k64', num_retries = 64)
    minimize_plot(problem, Cma_cpp(10000), name + '_10k64', num_retries = 64)
    minimize_plot(problem, De_cpp(10000), name + '_10k64', num_retries = 64)
    minimize_plot(problem, Bite_cpp(10000, M=16), name + '_10k64', num_retries = 64)
 
    suite = 'rw-top-trumps-biobj'
    function = 3
    instance = 5
    dim = 128
    nobj = 2
    rep = 2000
    name = suite + '_f' + str(function) + 'i' + str(instance) + 'd' + str(dim)
    problem = tt_problem(suite, name, dim, nobj, function, instance, rep)    
    
    mo_minimize_plot(problem, random_search(4000), name + '_4k512', num_retries = 512)
    mo_minimize_plot(problem, Cma_cpp(4000), name + '_4k512', num_retries = 512)
    mo_minimize_plot(problem, De_cpp(4000), name + '_4k512', num_retries = 512)
    mo_minimize_plot(problem, Bite_cpp(4000, M=16), name + '_4k512', num_retries = 512)
    mode.minimize_plot(name, problem.fun, problem.bounds, 2, popsize = 200, nsga_update=True, max_eval = 1000000)
    mode.minimize_plot(name, problem.fun, problem.bounds, 2, popsize = 200, nsga_update=False, max_eval = 1000000)
    
if __name__ == '__main__':
    main()

