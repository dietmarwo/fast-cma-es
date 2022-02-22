# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# This example is inspired by 
# https://github.com/AlgTUDelft/ExpensiveOptimBenchmark/blob/master/expensiveoptimbenchmark/problems/TSP.py[TSP.py]
# but using numba to speed up the objective function. 

# It implements a "noisy Travelers Salesman Problem". Goal is to minimize the "robust" tour length, the maximal 
# length for "iter_num" iterations using random noise: noise_factor*rnd(0,1) for each transfer. 

# See https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/TSP.adoc for a detailed description.

import numpy as np
import tsplib95
import networkx
from numba import njit, numba
from scipy.optimize import Bounds
import time
import math
import ctypes as ct
import multiprocessing as mp 
from fcmaes.optimizer import logger, Bite_cpp, Cma_cpp, De_cpp, De_python, dtime
from fcmaes import retry, modecpp, de  

# do 'pip install tsplib95'

@njit(fastmath=True) 
def evaluate_tsp(x, W, d, noise_factor, iter_num):
    robust_total_route_length = 0
    order = np.argsort(x) + 1   
    for _ in range(iter_num):
        total_route_length = 0
        total_route_length += W[0, order[0]] + np.random.random() * noise_factor            
        total_route_length += W[order[d-1], 0] + np.random.random() * noise_factor    
        for i in range(d-1):
            total_route_length += W[order[i], order[i+1]] + np.random.random() * noise_factor
        robust_total_route_length = max(total_route_length, robust_total_route_length)
    return robust_total_route_length

class TSP:

    def __init__(self, name, W, noise_factor, iter_num):
        self.name = name
        self.d = W.shape[0] - 1
        self.W = W
        self.noise_factor = noise_factor
        self.iter_num = iter_num
        self.evals = mp.RawValue(ct.c_long, 0)  # writable across python processes
        self.best_y = mp.RawValue(ct.c_double, math.inf) 
        self.t0 = time.perf_counter()

    def __call__(self, x):  
        y = evaluate_tsp(x, self.W, self.d, self.noise_factor, self.iter_num)
        self.evals.value += 1
        if y < self.best_y.value:
            self.best_y.value = y            
            logger().info("evals = {0}: time = {1:.1f} y = {2:.5f} x= {3:s}"
                          .format(self.evals.value, dtime(self.t0), y, 
                                  '[' + ", ".join([f"{xi:.16f}" for xi in x]) + ']'
                    ))
        return y
    
    def mofun(self, xs):
        y = self(xs)
        return np.array([y])
   
    def bounds(self):
        return Bounds(np.zeros(self.d), np.array([1]*self.d))    
 
    def __str__(self):
        return f"TSP(name={self.name},evals={self.evals},iter={self.iter_num})"

    def optimize(self):
        self.bestY = 1E99
        self.bestX = []  
        return retry.minimize(self, self.bounds(), optimizer=Bite_cpp(50000,stall_iterations=3), num_retries=32)   
        #return retry.minimize(self, self.bounds(), optimizer=De_cpp(50000), num_retries=32)    
        #return retry.minimize(self, self.bounds(), optimizer=Cma_cpp(50000), num_retries=320)  
        #return retry.minimize(self, self.bounds(), optimizer=De_python(50000), num_retries=32)   

   
    def optimize_mo(self, nsga_update = True):
        self.bestY = 1E99
        self.bestX = []
        return modecpp.retry(self.mofun, 1, 0, self.bounds(), num_retries=320, popsize = 48, 
                  max_evaluations = 1000000, nsga_update = nsga_update, logger = logger())


def load_tsplib(path, noise_factor=1, iter_num=100):
    instance = tsplib95.load(path)
    W = networkx.to_numpy_matrix(instance.get_graph())
    return TSP(instance.name, W, noise_factor, iter_num)

if __name__ == '__main__':
    
    # see http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/STSP.html for solutions   
   
    tsp = load_tsplib('data/tsp/br17.tsp').optimize()

    # see https://www.math.uwaterloo.ca/tsp/history/tspinfo/gr666_info.html 
    # optimal solution without noise for gr666 is 294358.
    #tsp = load_tsplib('data/tsp/gr666.tsp').optimize()
    
    #tsp = load_tsplib('data/tsp/berlin52.tsp').optimize_mo()


