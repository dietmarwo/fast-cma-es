# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# See https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/Media.adoc for a detailed description.

# Tested using https://docs.conda.io/en/main/miniconda.html on Linux Mint 21.2

import numpy as np
from numba import njit
import numba
from fcmaes.optimizer import De_cpp, wrapper
from fcmaes import retry, moretry, modecpp, mode
from scipy.optimize import Bounds    
import math
from multiprocessing import Pool
import multiprocessing as mp
import ctypes as ct

import sys 
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}")
logger.add("log_{time}.txt")

@njit(fastmath=True)
def num_true(nodes):
    n = 0
    for b in nodes:
        if b:
            n += 1
    return n

@njit(fastmath=True)
def sum_weights(nodes, weights):
    sum = 0
    for i in range(len(nodes)):
        if nodes[i]:
            sum += weights[i]
    return sum

def add_vertice(v, vmap):
    if v in vmap:
        return vmap[v]
    else:
        i = len(vmap)
        vmap[v] = i
        return i

def circles_arr(circs):
    max_circ = max([len(c) for c in circs])
    acircs = np.empty((len(circs), max_circ), dtype=int) 
    for c in range(len(circs)):
        circ = circs[c]
        for i in range(len(circ)):
            acircs[c, i] = circ[i]
    return acircs
   
class fb_graph():
    
    def __init__(self, vmap, source, target, circs):
        self.nodes = np.array(list(vmap.values()), dtype=int)
        self.source = source
        self.target = target    
        self.nnodes = len(vmap)
        self.ncircles = len(circs)
        self.circs = circs
        self.acircles = circles_arr(circs)
        self.circle_size = np.array([len(circ) for circ in circs])
        rng = np.random.default_rng(601) # use seeded random weights
        self.weights = rng.random(self.nnodes)

def parse_fb(edge_file, circle_file):
    lines = open(edge_file, 'r').readlines()
    source = []
    target = []
    vmap = {}
    for line in lines:
        vs = line.split()
        source.append(add_vertice(vs[0], vmap))
        target.append(add_vertice(vs[1], vmap))
    lines = open(circle_file, 'r').readlines()  
    circs = []
    for line in lines:
        vs = line.split()[1:]
        if len(vs) > 1: #drop circles of 1
            for v in vs: # add circ nodes to vmap
                add_vertice(v, vmap)
            circs.append([vmap[v] for v in vs])
    return fb_graph(vmap, np.array(source, dtype=int), np.array(target, dtype=int),  
                 circs)

@njit(fastmath=True)
def fb_covered(source, target, circles, circle_size, nds):
    num = 0
    nedges = len(source)
    for i in range(nedges): # check all edges
        if nds[source[i]] or nds[target[i]]:
            num += 1   
    for i in range(len(circle_size)):
        circle = circles[i] # check all circle pairs
        size = circle_size[i]
        for j in range(size-1):
            for k in range(j+1, size): # add if both are covered
                if nds[circle[j]] and nds[circle[k]]:
                    num += 1.0/math.sqrt(size)
    return num 

@njit(fastmath=True)
def nodes(x):  
    nds = np.empty(len(x), dtype=numba.boolean)
    for i in range(len(x)):
        nds[i] = True if x[i] > 0.5 else False 
    return nds

class problem_fb():
    
    def __init__(self, g):
        self.dim = g.nnodes
        self.bounds = Bounds([0]*self.dim, [1.99999]*self.dim)  
        self.g = g   
        self.best_y = mp.RawValue(ct.c_double, np.inf) 
        self.max_cost, self.max_cov = self.cost(np.array([1]*self.dim)) 
        
    def cost(self, x):
        nds = nodes(x.astype(int))
        cov = fb_covered(self.g.source, self.g.target, self.g.acircles, 
                         self.g.circle_size, nds)
        cost = sum_weights(nds, self.g.weights)
        return cost, cov        
    
    def fitness(self, x):
        cost, cov = self.cost(x)
        cost /= self.max_cost # to be minimized
        cov /= -self.max_cov # to be maximized
        return [cost, cov]

    def fitness_so(self, x):
        cost, cov = self.cost(x)
        cost /= self.max_cost # to be minimized
        cov /= -self.max_cov # to be maximized
        cost = max(0.3, cost) # target 30% cost
        y = 2*cost + cov
        if y < self.best_y.value:
            self.best_y.value = y
            nds = nodes(x.astype(int))
            print("n,cov", cost, cov, num_true(nds), len(nds))
        return y
    
def opt_mo(g): 
    prob = problem_fb(g)     
    pname = "fb1912_mo500k.256.32.de"    
    x, y = modecpp.retry(mode.wrapper(prob.fitness, 2), 
                         2, 0, prob.bounds, popsize = 256, 
                     max_evaluations = 500000, ints = [True]*prob.dim,
                     nsga_update=False, num_retries = 32,
                     workers=32)
    np.savez_compressed(pname, xs=x, ys=y)
    moretry.plot(pname, 0, x, y, all=False)
   
def opt_so(g): 
    prob = problem_fb(g)  
    pname = "fb1912_so3000k.512.32.de" 
    res = retry.minimize_plot(pname, De_cpp(3000000, 
                              popsize = 512, ints = [True]*prob.dim), 
                              wrapper(prob.fitness_so), 
                              prob.bounds, 
                              num_retries=32)   
    print (nodes(res.x.astype(int)))


if __name__ == '__main__':
    g = parse_fb("1912.edges", "1912.circles")

    #opt_so(g)
    opt_mo(g)
    