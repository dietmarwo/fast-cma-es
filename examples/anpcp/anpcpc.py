# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.
# 
# See https://github.com/netotz/alpha-neighbor-p-center-problem/blob/main/README.md
# See https://www.researchgate.net/publication/257196448_Optimal_algorithms_for_the_a-neighbor_p-center_problem
#
# This implements a continuous variation of the problem: 
# We don't offer a set of facilities to choose from. Only p, the number of chosen facilities is given and we
# search for optimal coordinates. Users are taken from the TSP/JSON files as before and the boundaries
# of the coordinates are determined using the facilities defined there. 
# Additionally a variant is implemented (commented out) where instead of the alpha-best we 
# use the sum of the best alpha distances.
# Note that we internally use the square of the distance to speed up the computation of the distance matrix.  
# See https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/Service.adoc for a detailed description. 

import json, sys
import numpy as np
from numba import njit
from fcmaes.optimizer import Bite_cpp, Cma_cpp, wrapper, logger
from fcmaes import retry
from scipy.optimize import Bounds  

import sys 
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")
logger.add("log_{time}.txt", format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")

@njit(fastmath=True)
def calc_distance_(users, facilities_x, facilities_y):
    distance = np.zeros((len(users), len(facilities_x)))
    for i in range(len(users)):
        for j in range(len(facilities_x)): # round to next int
            dx = users[i,0] - facilities_x[j]
            dy = users[i,1] - facilities_y[j]
            # we use the square of the distance because it is faster to compute
            distance[i,j] = dx*dx + dy*dy 
    return distance      

@njit(fastmath=True) 
def fitness_(facilities_x, facilities_y, users, alpha):
    distance = calc_distance_(users, facilities_x, facilities_y) 
    partitioned = np.partition(distance, alpha)    
    return max([max(d[:alpha]) for d in partitioned])
    # variant where instead of the alpha-best we use the sum of the best alpha distances
    # return max([np.sum(d[:alpha]) for d in partitioned])
        
class ANPCPC():
    
    def __init__(self, p, alpha):
        self.p = p
        self.alpha = alpha
        self.dim = self.p * 2
             
    def init_json(self, json_file):   
        with open(json_file) as json_file:
            anpcp = json.load(json_file)      
            self.users = anpcp['users']
            self.facilities = np.array(anpcp['facilities'])
            self.unum = len(self.users)
            self.bounds = Bounds([0]*self.dim, [np.max(self.facilities)]*self.dim)  
   
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
            self.fnum = len(facilities)        
            self.bounds = Bounds([0]*self.dim, [np.max(self.facilities)]*self.dim)  
            
    def random_x(self, seed = 123):
        rng = np.random.default_rng(seed)
        return rng.random(self.dim) * np.max(self.facilities)

    def get_facilities(self, x):
        facilities_x = x[:self.p]
        facilities_y = x[self.p:]
        return np.array([[facilities_x[i], facilities_y[i]] \
                                    for i in range(len(facilities_x))])

    def fitness(self, x):
        facilities_x = x[:self.p]
        facilities_y = x[self.p:]
        return fitness_(facilities_x, facilities_y, self.users, self.alpha)
    
def optimize(anpcpc, opt, num_retries = 32):
    ret = retry.minimize(wrapper(anpcpc.fitness), 
                               anpcpc.bounds, num_retries = num_retries, 
                               optimizer=opt)
    print("facility locations = ", anpcpc.get_facilities(ret.x))
    print("value = ", np.sqrt(ret.fun))
    
if __name__ == '__main__':
    # anpcpc = ANPCPC(6, 2) # p = 6, alpha = 2   
    # anpcpc.init_tsp('data/att48_36_12_4.anpcp.tsp')    
    # anpcpc = ANPCPC(12, 2) # p = 12, alpha = 2
    # anpcpc.init_tsp('data/rat783_588_195_4.anpcp.tsp')   
    anpcpc = ANPCPC(12, 2) # p = 12, alpha = 2
    anpcpc.init_tsp('data/rl1323_993_330_4.anpcp.tsp')
    
    max_evaluations = 100000
    opt = Bite_cpp(max_evaluations, M=6)
    # opt = Cma_cpp(max_evaluations)
    optimize(anpcpc, opt, num_retries = 32)
    
