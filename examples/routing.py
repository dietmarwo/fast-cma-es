# see https://github.com/krishna-praveen/Capacitated-Vehicle-Routing-Problem
# See https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/Routing.adoc for a detailed description.

# Tested using https://docs.conda.io/en/main/miniconda.html on Linux Mint 21.2

import numpy as np
from numba import njit
from fcmaes.optimizer import Bite_cpp, De_cpp, Crfmnes_cpp, Crfmnes, Cma_cpp, de_cma, wrapper
from fcmaes import mode, modecpp, moretry, retry
from scipy.optimize import Bounds

import sys 
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")
logger.add("log_{time}.txt", format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")

@njit(fastmath=True)
def fitness_(seq, distance, demands, capacity):
    n = len(seq)
    seq += 1
    sum_demand = 0
    sum_dist = 0
    last = 0
    vehicles = 1
    for i in range(n+1):
        customer = seq[i] if i < n else 0
        demand = demands[customer]
        if sum_demand + demand > capacity:
            # end vehicle tour, return to base
            sum_dist += distance[last, 0]
            sum_demand = 0
            vehicles += 1
            last = 0
        # go to customer
        sum_demand += demand
        sum_dist += distance[last, customer]
        last = customer
    return np.array([-float(vehicles), sum_dist])

def parse(filename):
    with open(filename) as csvfile:
        lines = csvfile.readlines()
        demands = []
        coords = [] 
        for line in lines:
            row = line.split()
            if len(row) < 5 or not row[0][0].isdigit():
                continue
            demands.append(float(row[3]))
            coords.append(np.array([float(row[1]), float(row[2])])) 
    n = len(demands)
    distances = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            distances[i,j] = np.linalg.norm(coords[i] - coords[j])
    return np.array(demands), distances
        
class Routing():
    def __init__(self, filename, capacity):
        self.capacity = capacity
        self.demands, self.distance = parse(filename)
        self.dim = len(self.demands) - 1
        self.bounds = Bounds([0]*self.dim, [1]*self.dim)
        self.bounds_capacity = Bounds([40] + [0]*(self.dim), [500] + [1]*self.dim)

    def fitness(self, x):   
        return fitness_(np.argsort(x), self.distance, self.demands, self.capacity)

    def fitness_capacity(self, x):   
        y = fitness_(np.argsort(x[1:]), self.distance, self.demands, x[0])
        return np.array([x[0], y[1]])

    def fitness_so(self, x):   
        return fitness_(np.argsort(x), self.distance, self.demands, self.capacity)[1]
    
    def dump(self, seq, y=0, capa=None):
        if capa is None:
            capa = self.capacity
        n = len(seq)
        seq += 1
        sum_demand = 0
        sum_dist = 0
        last = 0
        vehicles = 1
        tour = [0]
        print ("tour ", str(list(seq)))
        print ("y ", y)
        for i in range(n+1):
            customer = seq[i] if i < n else 0
            demand = self.demands[customer]
            if sum_demand + demand > capa:
                sum_dist += self.distance[last, 0]
                print ("vehicle ", vehicles, "tour", tour + [0], "demands", sum_demand, "distance", sum_dist)
                sum_demand = 0
                vehicles += 1
                tour = [0]
                last = 0
            sum_demand += demand
            sum_dist += self.distance[last, customer]
            tour.append(customer)
            last = customer
        print ("vehicle ", vehicles, "tour", tour, "demands", sum_demand, "distance", sum_dist)
        return np.array([float(vehicles), sum_dist])
  
def optimize(filename, capacity, popsize, max_evaluations):
    routing = Routing(filename, capacity)    
    x, y = modecpp.retry(mode.wrapper(routing.fitness, 2, interval = 10000000), 2, 0,
                 routing.bounds, popsize = popsize, 
                 max_evaluations = max_evaluations, 
                 nsga_update=True, num_retries = 320)
    pname = "routing." + str(popsize) + "." + str(max_evaluations)
    np.savez_compressed(pname, xs=x, ys=y)
    moretry.plot(pname, 0, x, y, all=False, interp=True)#, plot3d=True)
    for i in range(len(y)):
        routing.dump(np.argsort(x[i]), y[i])

def optimize_capacity(filename, popsize, max_evaluations, num_retries = 640):
    routing = Routing(filename, 0)    
    x, y = modecpp.retry(mode.wrapper(routing.fitness_capacity, 2, interval = 10000000), 2, 0,
                 routing.bounds_capacity, popsize = popsize, 
                 max_evaluations = max_evaluations, 
                 nsga_update=True, num_retries = num_retries)
    pname = "routing." + str(popsize) + "." + str(max_evaluations)
    np.savez_compressed(pname, xs=x, ys=y)
    moretry.plot(pname, 0, x, y, all=False, interp=True)#, plot3d=True)
    routing.dump(np.argsort(x[-1][1:]), y[-1], y[-1][0])
   
def optimize_so(filename, capacity, opt, num_retries = 320):
    routing = Routing(filename, capacity)
    name = "routing." + str(opt.max_evaluations)    
    ret = retry.minimize_plot(name, opt, wrapper(routing.fitness_so), 
                               routing.bounds, num_retries = num_retries)
    routing.dump(np.argsort(ret.x), ret.fun)
   
def dump(filename, capacity, seq):
    routing = Routing(filename, capacity)     
    routing.dump(np.array(seq)-1) 

if __name__ == '__main__':
    filename = "data/Input_Data.txt"
    capacity = 70
    popsize = 128
    max_evaluations = 100000
    
    optimize_capacity(filename, 2*popsize, 2*max_evaluations)
    #optimize(filename, capacity, popsize, max_evaluations)
    #optimize_so(filename, capacity, Bite_cpp(max_evaluations))
    #optimize_so(filename, capacity, Crfmnes_cpp(max_evaluations,popsize=popsize))
    #optimize_so(filename, capacity, de_cma(max_evaluations,popsize=popsize))
    #optimize_so(filename, capacity, De_cpp(max_evaluations,popsize=popsize))
    #dump(filename, capacity, [1, 2, 4, 3, 5, 16, 14, 12, 10, 11, 13, 17, 18, 19, 15, 7, 6, 9, 8, 20, 24, 25, 23, 22, 21])

    