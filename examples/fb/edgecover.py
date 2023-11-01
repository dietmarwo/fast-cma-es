# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# See https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/Media.adoc for a detailed description.

# Tested using https://docs.conda.io/en/main/miniconda.html on Linux Mint 21.2

import networkx as nx
import numpy as np
from numba import njit
import numba
from fcmaes.optimizer import De_cpp, wrapper
from scipy.optimize import Bounds    
import random
import time, math
from multiprocessing import Pool
import multiprocessing as mp
import ctypes as ct

import sys 
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")
logger.add("log_{time}.txt", format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")

def plot_graph(g):
    import matplotlib.pyplot as plt
    pos = nx.circular_layout(g)
    nx.draw(g, pos, with_labels=True)
    labels = nx.get_edge_attributes(g, 'weight')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)
    plt.show()

'''apply jgrapht

    inspired by https://github.com/danielslz/minimum-vertex-cover/blob/main/utils.py
'''

def nx_to_jgraph(g):
    import jgrapht
    jg = jgrapht.create_graph(directed=False, weighted=False,
                             allowing_self_loops=False, allowing_multiple_edges=False)
    jg.add_vertices_from(list(g.nodes))
    jg.add_edges_from(list(g.edges))
    return jg

def solve_jg(g):
    import jgrapht
    jg = nx_to_jgraph(g)
    start_time = time.time()
    mvc = jgrapht.algorithms.vertexcover.greedy(jg)
    #mvc = jgrapht.algorithms.vertexcover.edgebased(jg)
    #mvc = jgrapht.algorithms.vertexcover.clarkson(jg)
    #mvc = jgrapht.algorithms.vertexcover.baryehuda_even(jg)
    mvc_size = int(mvc[0])
    print ("jgraph mvc size", mvc_size , ' of nodes: ', len(list(g.nodes())), 
           ' time = ', round(time.time()-start_time, 3), ' sec')

'''greedy algorithm

    inspired by https://github.com/sliao7/CSE6140-Final-Project-Minimum-Vertex-Cover/blob/main/code/SA.py
'''
    
def initial_solution(g):
    solution = list(g.nodes())
    # sort nodes for degree, low degree has better chance not to uncover an edge
    for _, node in \
            sorted(list(zip(list(dict(g.degree(solution)).values()), solution))):
        remove = True
        for neigbor in g.neighbors(node): # all neighbors covered?
            if neigbor not in solution:
                remove = False # bad luck, would uncover an edge
        if remove:    
            solution.remove(node)                   
    return solution

def remove_node(g, solution, mean, start_time, max_time):
    solution = solution.copy()
    uncovered = []
    while len(uncovered) == 0:
        to_delete = random.choice(solution)
        for neighbor in g.neighbors(to_delete):
            if neighbor not in solution:
                uncovered.append(neighbor)
                uncovered.append(to_delete)
        solution.remove(to_delete)  
    i = 0
    max_i = mean * 10
    while len(uncovered) > 0 and i < max_i and \
            time.time() - start_time < max_time:
        i += 1
        # delete node from solution
        next_solution = solution.copy()
        next_uncovered = uncovered.copy()
        to_delete = random.choice(solution)
        solution.remove(to_delete) 
        for neighbor in g.neighbors(to_delete):
            if neighbor not in solution:
                uncovered.append(neighbor)
                uncovered.append(to_delete)            
        # add node to solution
        to_add = random.choice(uncovered)
        solution.append(to_add)
        for neighbor in g.neighbors(to_add):
            if neighbor not in solution:
                uncovered.remove(neighbor)
                uncovered.remove(to_add)      
        # update solution if uncovered shrink        
        if len(next_uncovered) < len(uncovered) or \
            (len(next_uncovered) == len(uncovered) and \
                i > mean and random.random() < 1.0/mean):  
            solution = next_solution.copy()
            uncovered = next_uncovered.copy()
    return solution, uncovered, i

def solve_greedy(g, seed, max_time):
    print("seed", seed)
    random.seed(seed)
    start_time = time.time()
    solution = initial_solution(g)
    iters = []
    mean = 10000
    while time.time() - start_time < max_time:
        next_solution, uncovered, i = remove_node(g, solution, mean, start_time, max_time)
        iters.append(i)
        mean = np.mean(iters)
        if len(uncovered) == 0:  # all covered ?
            solution = next_solution
            print(round(time.time()-start_time,3), len(solution), i, int(mean))   

    print(round(time.time()-start_time,3), len(solution))
    print('Solution: ({}) {}'.format(
        len(solution), solution))
    return solution

def run_solve(g, max_time):
    return solve_greedy(g, random.randint(0, 100000000), max_time)
        
def solve_multiprocessing(g, num, max_time): 
    with Pool(processes=num) as pool:
        solutions = pool.starmap(run_solve, [[g, max_time] for _ in range(num)])
    return solutions

def add_vertice(v, vmap):
    if v in vmap:
        return vmap[v]
    else:
        i = len(vmap)
        vmap[v] = i
        return i
    
def nx_graph(filename):
    g = nx.Graph()
    vmap = {}
    lines = open(filename, 'r').readlines()
    for line in lines:
        vs = line.split()
        source = add_vertice(vs[0], vmap)
        target = add_vertice(vs[1], vmap)
        g.add_edge(source, target)            
    return g

''' optimization approach using numba'''

@njit(fastmath=True)
def num_true(nodes):
    n = 0
    for b in nodes:
        if b:
            n += 1
    return n

@njit(fastmath=True)
def numba_covered(source, target, nodes):
    num = 0
    nedges = len(source)
    for i in range(nedges):
        if nodes[source[i]] or nodes[target[i]]:
            num += 1
    return nedges - num 

def not_covered(g, nodes):
    return numba_covered(g.source, g.target, nodes)

@njit(fastmath=True)
def nodes(x):  
    nds = np.empty(len(x), dtype=numba.boolean)
    for i in range(len(x)):
        nds[i] = True if x[i] > 0.5 else False 
    return nds

class graph():
    
    def __init__(self, g):
        self.nodes = np.array(g.nodes(), dtype=int)
        self.source = np.array([n for n, _ in g.edges()], dtype=int)
        self.target = np.array([n for _, n in g.edges()], dtype=int)    

class problem():
    
    def __init__(self, g):
        self.dim = len(g.nodes())
        self.bounds = Bounds([0]*self.dim, [1.999999999999]*self.dim)  
        self.g = graph(g)   
        self.best_n = mp.RawValue(ct.c_double, np.inf) 
    
    def fitness(self, x):
        nds = nodes(x.astype(int))
        ncov = not_covered(self.g, nds)
        n = num_true(nds)        
        return n + 2*ncov
          
def solve_opt(g): 
    prob = problem(g)  
    res = retry.minimize(wrapper(prob.fitness), 
                     prob.bounds, 
                     optimizer=De_cpp(500000, ints = [True]*prob.dim), 
                     num_retries=32)
    nds = nodes(res.x.astype(int))
    ncov = not_covered(prob.g, nds)
    n = num_true(nds)  
    print ("nodes = ", n, " of population = ", len(nds),
           " % = ", int(100*n/len(nds)), " edges not covered = ", ncov)

if __name__ == '__main__':

    g = nx_graph("1912.edges")
    
    #solve_jg(g)
    #solve_greedy(g, 47, 1)
    
    solve_multiprocessing(g, 16, 10)

    #solve_opt(g)
    
    #plot_graph(g)