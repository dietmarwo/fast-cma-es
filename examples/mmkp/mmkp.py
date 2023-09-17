# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.
# 
# Solving Multiple-Choice Multidimensional Knapsack Problem MMKP
#
# See https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/Oneforall.adoc for a detailed description.

import numpy as np
import os
from numba import njit
import numba
from datetime import datetime
from fcmaes.optimizer import crfmnes_bite, wrapper
from fcmaes import retry
from scipy.optimize import Bounds

import sys 
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}")
logger.add("log_{time}.txt")

@njit(fastmath=True)
def fitness_(x, n, l, avail, values, resources):
    vsum = 0
    rsum = np.zeros(l, dtype=numba.int32)
    for i in range(n):
        vsum += values[i][x[i]]
        rsum += resources[i][x[i]]   
    rsum = np.maximum(rsum - avail, np.zeros(l, dtype=numba.int32))
    pen = np.sum(rsum)
    return vsum, pen

def parse(filename):
    n = 0
    with open(filename) as csvfile:
        lines = csvfile.readlines()
        avail = None
        values = []
        resources = [] 
        for line in lines:
            row = line.split()               
            if len(row) < 1:
                continue
            if n == 0:
                n = int(row[0])
                l = int(row[1])
                m = int(row[2])
            else:
                if avail is None:
                    avail = [int(r) for r in row]
                else:
                    if row[0] == 'Solutions':
                        break
                    if len(row) > 1:
                        values.append(float(row[0]))
                        resources.append([int(r) for r in row[1:]])
        bsol = lines[-1].split()
        bval = float(bsol[-1] )
        bsol = np.array([int(r) for r in bsol[:-1]], dtype=int)
    return n, l, m, bval, bsol, \
        np.array(avail, dtype=int), \
        np.reshape(np.array(values), (n,l)),\
        np.reshape(np.array(resources, dtype=int), (n,l,m))
       
class MMKP():
    def __init__(self, problem):
        self.problem = problem
        filename = 'problems/' + problem
        self.n, self.l, self.m, self.best_val, self.best_sol,\
                self.avail, self.values, self.resources = parse(filename)
        self.dim = self.n
        self.bounds = Bounds([0]*self.dim, [self.l-1E-12]*self.dim)

    def deviation(self, vsum):
        return 100 * ((self.best_val - vsum) / self.best_val)

    def fitness_pen(self, x):   
        return fitness_(x.astype(int), self.n, self.l, self.avail, \
                self.values, self.resources)

    def fitness(self, x):   
        vsum, penalty = self.fitness_pen(x)
        if penalty > 0:
            penalty += 100    
        return self.deviation(vsum) + penalty    
    
    def dump(self, x):
        x = x.astype(int)
        vsum, _ = self.fitness_pen(x)
        lines = []
        lines.append('Instance Name : ' + self.problem + '\n')
        lines.append('Date : ' + str(datetime.today().date()) + '\n')
        lines.append('Score = ' + str(vsum) + 
            ' Deviation = ' + str(round(self.deviation(vsum),2)) + " %\n")
        lines.append('Solution\n')
        lines.append(' '.join([str(xi) for xi in x]))
        filename = 'solutions_co/' + self.problem + '.txt'
        with open(filename, 'w') as f:
            f.writelines(lines)
        logger.info(''.join(lines))
       
def optimize(mmkp, opt, num_retries = 32):
    ret = retry.minimize(wrapper(mmkp.fitness), 
                               mmkp.bounds, num_retries = num_retries, 
                               stop_fitness = 1E-12, optimizer=opt)
    x = ret.x.astype(int)
    mmkp.dump(ret.x)

def opt_dir(dir):
    files = os.listdir(dir)
    files.sort()
    for file in files:
        problem = file.split('.')[0]
        mmkp = MMKP(problem)
        dim = mmkp.dim
        popsize = 500#100 + dim
        max_evaluations = 10000000#popsize*20000
        opt = crfmnes_bite(max_evaluations, popsize=popsize, M=4, stop_fitness = 1E-12)
        optimize(mmkp, opt, num_retries = 1024)

if __name__ == '__main__':
    opt_dir('problems')
    
    # x = np.array([4,8,9,5,8,8,9,7,8,4,4,9,4,9,1,8,8,6,4,8], dtype=int)
    # mmkp = MMKP('I04')
    # print(mmkp.fitness_pen(x))