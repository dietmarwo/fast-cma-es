# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.
# See https://github.com/netotz/alpha-neighbor-p-center-problem/blob/main/README.md
# See https://www.researchgate.net/publication/257196448_Optimal_algorithms_for_the_a-neighbor_p-center_problem
#
# This implements the original variant of the problem using continuous optimization and
# a variant (commented out) where instead of the alpha-best we use the sum of the best alpha distances.
# Users and facilities are taken from TSP/JSON files. 
# See https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/Service.adoc for a detailed description. 

import json, sys
import numpy as np
from numba import njit
import numba
from fcmaes.optimizer import Bite_cpp, wrapper, logger
from fcmaes import retry
from scipy.optimize import Bounds  

import sys 
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")
logger.add("log_{time}.txt", format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")

@njit(fastmath=True) 
def next_free_(used, p):
    while used[p]:
        p = (p + 1) % used.size
    used[p] = True
    return p

@njit(fastmath=True) 
def selection_(s, n):
    disjoined_s = np.zeros(s.size, dtype=numba.int32)
    used = np.zeros(n, dtype=numba.boolean)
    for i in range(s.size):
        disjoined_s[i] = next_free_(used, s[i])
    return disjoined_s

@njit(fastmath=True)
def calc_distance_(users, facilities):
    distance = np.zeros((len(users), len(facilities)), dtype=numba.int32)
    for i in range(len(users)):
        for j in range(len(facilities)): # round to next int
            dx = users[i,0] - facilities[j,0]
            dy = users[i,1] - facilities[j,1]
            distance[i,j] = int(round(np.sqrt(dx*dx + dy*dy), 0))
    return distance      

@njit(fastmath=True) 
def fitness_(selection, distance, alpha):
    selected = distance[:,selection] 
    partitioned = np.partition(selected, alpha)    
    return max([max(d[:alpha]) for d in partitioned])
    # variant where instead of the alpha-best we use the sum of the best alpha distances
    # return max([np.sum(d[:alpha]) for d in partitioned])        

class ANPCP():
    
    def __init__(self, p, alpha):
        self.p = p
        self.alpha = alpha
        self.dim = self.p
             
    def init_json(self, json_file):   
        with open(json_file) as json_file:
            anpcp = json.load(json_file)      
            self.facilities = anpcp['facilities']
            self.users = anpcp['users']
            self.distance = np.array(anpcp['distances'], dtype = int)
            self.fnum = len(self.facilities)
            self.unum = len(self.users)
            self.bounds = Bounds([0]*self.dim, [self.fnum-1E-9]*self.dim)  
   
    def init_tsp(self, tsp_file):   
        with open(tsp_file) as csvfile:
            lines = csvfile.readlines()
            users = []
            facilities = [] 
            for line in lines:
                row = line.split()
                if len(row) != 4 or not row[0][0].isdigit():
                    continue
                coords = np.array([float(row[1]), float(row[2])])
                if row[3] == '1':
                    facilities.append(coords)
                else:
                    users.append(coords)
            self.users = np.array(users)
            self.facilities = np.array(facilities)        
            self.unum = len(self.users)
            self.fnum = len(self.facilities)        
            self.distance = calc_distance_(self.users, self.facilities)
            self.bounds = Bounds([0]*self.dim, [self.fnum-1E-9]*self.dim)  
            
    def random_x(self, seed = 123):
        rng = np.random.default_rng(seed)
        return rng.integers(0, self.fnum, self.dim)

    def get_selection(self, x):
        return selection_(x.astype(int), self.fnum)

    def fitness(self, x):
        selection = selection_(x.astype(int), self.fnum)
        return fitness_(selection, self.distance, self.alpha)
    
def optimize(anpcp, opt, num_retries = 32):
    ret = retry.minimize(wrapper(anpcp.fitness), 
                               anpcp.bounds, num_retries = num_retries, 
                               optimizer=opt)
    print("selection = ", anpcp.get_selection(ret.x))
    print("value = ", ret.fun)

if __name__ == '__main__':
        
    # anpcp = ANPCP(6, 2) # p = 6, alpha = 2   
    # anpcp.init_tsp('data/att48_36_12_4.anpcp.tsp')
    # anpcp = ANPCP(12, 2) # p = 12, alpha = 2
    # anpcp.init_tsp('data/rat783_588_195_4.anpcp.tsp')   
    anpcp = ANPCP(12, 2) # p = 12, alpha = 2
    anpcp.init_tsp('data/rl1323_993_330_4.anpcp.tsp')
    popsize = 7 + 12*anpcp.dim
    max_evaluations = 300000
    opt = Bite_cpp(max_evaluations, popsize=popsize, M=8)
    optimize(anpcp, opt, num_retries = 32)
    
