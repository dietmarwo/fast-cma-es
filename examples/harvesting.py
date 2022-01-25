# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.
# 
# This code implements a variant of the multi objective flexible job shop problem
# and is derived from https://github.com/dietmarwo/fast-cma-es/blob/master/examples/jobshop.py 

# It supports both a multiobjective and the single objective variant utilizing Numba and the optimizers
# https://github.com/dietmarwo/fast-cma-es/blob/master/_fcmaescpp/modeoptimizer.cpp (MO) and
# https://github.com/dietmarwo/fast-cma-es/blob/master/_fcmaescpp/include/biteopt.h (SO).

# The FJSP variant implemented here is related to "asteroid harvesting":

# N movable identical factories are deployed on N asteroids to perform operations associated to m jobs.
# As in FJSP the operations need to be executed in the order specified by the job. 
# The equivalent to a machine in FJSP is a factory deployment to a specific asteroid. Its resources
# determine its capability to execute job operations. Therefore - using this simplified
# model - asteroid harvesting can be viewed as a FJSP with two constraints:

# a) Moving factories is expensive, therefore a factory can only be deployed once on an asteroid, 
# it is active for a single consecutive time window. 
# b) The upper limit of active machines (factory deployments) is determined by N, the number
# of factories. 

import math
import pandas as pd
import numpy as np
import sys, math, time
from pathlib import Path
from fcmaes import retry, advretry, mode, modecpp, moretry
from fcmaes.optimizer import logger, Bite_cpp, Cma_cpp, De_cpp, de_cma, dtime, Differential_evolution
from scipy.optimize import Bounds
import ctypes as ct
import multiprocessing as mp 
from multiprocessing import Process
from numba import njit, numba
from numpy.random import Generator, MT19937, SeedSequence
from jobshop import gantt, read_fjs, retry_modecpp, reorder, job_indices, filter_tasks, get_selection    
import jobshop

@njit(fastmath=True) 
def get_order(x, n_operations):
    return np.argsort(x[n_operations:2*n_operations])

@njit(fastmath=True)
def filtered_tasks(x, task_data, n_operations, n_machines, job_indices, job_ids):
    selection = get_selection(x, n_operations, n_machines)
    operations = filter_tasks(task_data, selection, n_operations)
    order = get_order(x, n_operations)
    tasks = reorder(operations, order, n_operations, job_ids[order], job_indices)
    return tasks 

class fitness: 

    def __init__(self, task_data, bounds, n_jobs, n_operations, n_machines, max_active, name):
        self.evals = mp.RawValue(ct.c_long, 0)  # writable across python processes
        self.best_y = mp.RawValue(ct.c_double, math.inf) # writable across python processes
        self.t0 = time.perf_counter()
        self.task_data = task_data   
        self.n_jobs = n_jobs
        self.n_operations = n_operations
        self.n_machines = n_machines 
        self.max_active = max_active 
        self.bounds = bounds
        self.name = name
        self.nobj = 3
        self.ncon = 1
        self.weights = np.array([1.0, 0.02, 0.001, 1000])  
        self.job_indices, self.job_ids = job_indices(task_data)

    def chart(self, x):
        max_time = x[-1]
        start = x[-self.n_machines-1:-1]*max_time
        duration = x[-2*self.n_machines-1:-self.n_machines-1]*max_time
        tasks = filtered_tasks(x, self.task_data, self.n_operations, self.n_machines,
                               self.job_indices, self.job_ids)
        chart(tasks, self.n_jobs, self.n_machines, self.max_active, start, duration)
        
    def fun(self, x):     
        tasks = filtered_tasks(x, self.task_data, self.n_operations, self.n_machines,
                               self.job_indices, self.job_ids)
        max_time = x[-1]
        start = x[-self.n_machines-1:-1]*max_time
        duration = x[-2*self.n_machines-1:-self.n_machines-1]*max_time
        machine_time, job_time, machine_work_time, min_work, fails = \
            exec_tasks(tasks, self.n_jobs, self.n_machines, self.max_active, start, duration)
        if fails is None: # timing error
            return np.array([0, 0, 0, 10000])
        span = np.amax(machine_time)
        work = np.sum(machine_work_time)
        wmax = np.amax(machine_work_time)
        ys = np.array([span, work, wmax, fails])            
        y = sum(self.weights*ys) # weighted sum  
        self.evals.value += 1
        if y < self.best_y.value:
            self.best_y.value = y  
            logger().info("evals = {0}: time = {1:.1f} t = {2:.0f} f = {3:.0f} y = {4:.2f} s = {5:.2f} w = {6:.0f} m = {7:.0f} m= {8:s} j= {9:s} w= {10:s}"
                .format(self.evals.value, dtime(self.t0), max_time, fails, y, span, work, wmax,
                        str([int(si) for si in machine_time]),
                        str([int(oi) for oi in job_time]),
                        str([int(oi) for oi in machine_work_time]),
                        ))
        return ys   

    def __call__(self, x):   
        ys = self.fun(x)
        return sum(self.weights*ys) # weighted sum  

def retry_modecpp(fit, retry_num = 32, popsize = 128, max_eval = 500000, workers=mp.cpu_count()):
    xf, yf = jobshop.retry_modecpp(fit, retry_num, popsize, max_eval, workers)
    xs = []; ys = []
    for i in range(len(yf)):
        if yf[i][-1] == 0: # filter valid solutions
            xs.append(xf[i])
            ys.append(yf[i][:fit.nobj])
    return np.array(xs), np.array(ys)
     
def optall():
    for i in range(1,16):
        optimize(i)
        
def optimize(bi): 
    name = "BrandimarteMk" + str(bi)
    tasks, n_jobs, n_machines, n_operations, sum_time = read_fjs("data/1_Brandimarte/" + name + ".fjs")
     
    dim = 2*n_operations + 2*n_machines + 1
    lower_bound = np.zeros(dim)
    lower_bound[:] = 0.0000001 
    upper_bound = np.zeros(dim)
    upper_bound[:] = 0.9999999
    upper_bound[-1] = sum_time
    bounds = Bounds(lower_bound, upper_bound)
    
    max_active = 4
    multi_objective = True 
    
    fit = fitness(tasks, bounds, n_jobs, n_operations, n_machines, max_active, name)
    
    if multi_objective:
        xs, front = retry_modecpp(fit, retry_num=32, popsize = 48, max_eval = 960000, workers=16)
        logger().info(name + " retry_modecpp(mo_problem, retry_num=32, popsize = 48, max_eval = 960000, workers=16)" )
        logger().info(str([tuple(y) for y in front]))
        x = xs[0]
    else:    
        store = retry.Store(fit, bounds, logger=logger()) 
        logger().info(name + " Bite_cpp(960000,M=1).minimize, num_retries=256)" )
        retry.retry(store, Bite_cpp(960000,M=1).minimize, num_retries=256)    
        x = store.get_xs()[0]
    
    fit.chart(x)     
    
@njit(fastmath=True)     
def adjust_timing(start, duration, max_active):
    inf = 10000000
    start = start.copy()
    i_sort = np.argsort(start)
    for k in range(max_active):
        start[i_sort[k]] = 0  # first max_active available from start
    stop = start + duration 
    j_sort = np.argsort(stop)
    for k in range(max_active):
        stop[j_sort[-1 - k]] = inf  # last max_active available forever
    n = len(start)
    moved = True
    while moved:
        active = 0
        i = 0
        j = 0
        moved = False
        while i < n and j < n:
            ci = i_sort[i]  # current start
            cj = j_sort[j]  # current stop
            if start[ci] >= stop[cj]:
                active -= 1
                j += 1
            else:
                if active < max_active:
                    active += 1               
                    i += 1
                else:
                    if (ci != cj): # move start after next machine stops
                        for i in range(n):
                            if start[i] == stop[ci]: # other machine already waiting for stop[cj]
                                return False, start, stop # otherwise can lead to an infinite loop
                        start[ci] = stop[cj]
                        stop[ci] = start[ci] + duration[ci]      
                        moved = True
                        break;
                    else:                            
                        j += 1
        if moved:
            i_sort = np.argsort(start)  # sort again
            j_sort = np.argsort(stop) 
    return True, start, stop
    
@njit(fastmath=True) 
def exec_tasks(tasks, n_jobs, n_machines, max_active, start, duration):
    success, start, stop = adjust_timing(start, duration, max_active)
    if not success:
        return None, None, None, None, None
    machine_time = start # we initialize with the machine startup times
    machine_work_time = np.zeros(n_machines)
    job_time = np.zeros(n_jobs)
    fails = 0
    for task in tasks:
        job = int(task[0])
        machine = int(task[2])
        time = task[3]
        # previous task needs to be finished and machine needs to be available
        end_time = max(machine_time[machine], job_time[job]) + time
        if end_time > stop[machine]: # machine already shut down
            fails += 1 # failure to execute task at all
            continue
        machine_time[machine] = end_time
        job_time[job] = end_time  
        machine_work_time[machine] += time
    return machine_time, job_time, machine_work_time, np.amin(machine_work_time), fails

def scheduling(tasks, n_jobs, n_machines, max_active, start, duration):
    success, start, stop = adjust_timing(start, duration, max_active)
    if not success:
        print("timing error")
    machine_time = start # we initialize with the machine startup times
    job_time = np.zeros(n_jobs)
    fails = 0
    solution = {'machine': [], 'start': [], 'end': [], 'job': [], 'task':[]}
    for task in tasks:
        job = int(task[0])
        machine = int(task[2])
        time = task[3]
        start = max(machine_time[machine], job_time[job])
        end = start + time
        if end > stop[machine]: # machine already shut down
            fails += 1 # failure to execute task at all
            continue
        machine_time[machine] = end
        job_time[job] = end  
        solution['machine'].append(int(machine))
        solution['start'].append(int(start))
        solution['end'].append(int(end))
        solution['job'].append(int(job))
        solution['task'].append(int(task[1]))
    print('fails = ', fails)
    return solution
    
def chart(tasks, n_jobs, n_machines, max_active, start, duration):
    solution = scheduling(tasks, n_jobs, n_machines, max_active, start, duration)
    print(solution)
    gantt(solution)

def main():
   optimize(1)
   #optall()
    
if __name__ == '__main__':
    main()
