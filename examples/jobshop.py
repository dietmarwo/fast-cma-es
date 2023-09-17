# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.
# 
# This code implements the multi objective flexible job shop problem. It supports both the 
# multiobjective and the single objective variant utilizing Numba and the optimizers
# https://github.com/dietmarwo/fast-cma-es/blob/master/_fcmaescpp/modeoptimizer.cpp (MO) and
# https://github.com/dietmarwo/fast-cma-es/blob/master/_fcmaescpp/include/biteopt.h (SO).

# FJSP is to determine the most appropriate machine for each operation (called machine selection)
# and the sequence of the operations on machines (called operation sequencing). The optimization
# objective of the FJSP is to minimize some indicators, e.g. makespan, maximum tardiness and total flow
# time
# 
# See https://www.honda-ri.de/pubs/pdf/3949.pdf for an alternative implementation of the same problem.
# See https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/JobShop.adoc for a detailed description.

# Tested using https://docs.conda.io/en/main/miniconda.html on Linux Mint 21.2

import math
import pandas as pd
import numpy as np
import sys, math, time
from pathlib import Path
from fcmaes import retry, advretry, mode, modecpp, moretry
from fcmaes.optimizer import Bite_cpp, Cma_cpp, De_cpp, de_cma, dtime, Differential_evolution
from scipy.optimize import Bounds
import ctypes as ct
import multiprocessing as mp 
from numba import njit, numba
import seaborn as sns
import matplotlib.pyplot as plt

import sys 
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}")
logger.add("log_{time}.txt")

def read_fjs(filename):
    inf = 1000000
    with open(filename) as f:
        lines = f.readlines()
    first_line = lines[0].split()
    n_jobs = int(first_line[0])
    n_machines = int(first_line[1])
    nb_operations = [int(lines[j + 1].split()[0]) for j in range(n_jobs)]
    n_operations = np.sum(nb_operations)
    nb_tasks = sum(nb_operations[j] for j in range(n_jobs))
    processing_time = [[inf for m in range(n_machines)] for t in range(nb_tasks)]
    # For each job, for each operation, the corresponding task id
    operation_task = [[0 for o in range(nb_operations[j])] for j in range(n_jobs)]
    sum_time = 0
    id = 0
    for j in range(n_jobs):
        line = lines[j + 1].split()
        tmp = 0
        for o in range(nb_operations[j]):
            n_machines_operation = int(line[tmp + o + 1])
            for i in range(n_machines_operation):
                machine = int(line[tmp + o + 2 * i + 2]) - 1
                time = int(line[tmp + o + 2 * i + 3])
                processing_time[id][machine] = time
                sum_time += time
            operation_task[j][o] = id
            id += 1
            tmp += 2 * n_machines_operation
    tasks = []
    for job in range(n_jobs):
        jtasks = operation_task[job]
        for task in jtasks:
            times = processing_time[task]
            for machine in range(n_machines):
                time = times[machine]
                if time < inf:
                    tasks.append((job, task, machine, time))
    
    return np.array(tasks), n_jobs, n_machines, n_operations, sum_time

def gantt(data):
    df = pd.DataFrame.from_dict(data)
    df['duration']=df.end-df.start    
    p_start=df.start.min()
    p_end=df.end.max()
    p_duration=(p_end-p_start+1)
    df['rel_start']=df.start.apply(lambda x: (x-p_start))
    x_ticks=[i for i in range(int(p_duration+1))]
    x_labels=[(p_start+i) for i in x_ticks]   
    plt.figure(figsize=(8,4))
    cols = sns.color_palette()
    machines = list(df.machine) 
    mi = np.argsort(machines) 
    y = 0
    last = machines[mi[0]]
    for i in mi:
        if machines[i] != last:
            last = machines[i]
            y += 1
        plt.barh(y='M' + str(machines[i]), left=df.rel_start[i], 
                 width=df.duration[i],  color=cols[df.job[i] % len(cols)])
        plt.text(x=df.rel_start[i], y=y, s = str(df.task[i]))
    plt.gca().invert_yaxis()
    plt.xticks(ticks=x_ticks, labels=x_labels)
    plt.title('Gantt Chart')
    plt.xticks(rotation=60)
    plt.grid(axis='x', alpha=0.5)
    plt.savefig('gantt.png')
    plt.show()

def scheduling(tasks, n_jobs, n_machines):
    machine_time = np.zeros(n_machines)
    job_time = np.zeros(n_jobs)
    solution = {'machine': [], 'start': [], 'end': [], 'job': [], 'task':[]}
    for task in tasks:
        job = int(task[0])
        machine = int(task[2])
        time = task[3]
        # previous task needs to be finished and machine needs to be available
        start = max(machine_time[machine], job_time[job])
        end = start + time
        machine_time[machine] = end
        job_time[job] = end  
        solution['machine'].append(int(machine))
        solution['start'].append(int(start))
        solution['end'].append(int(end))
        solution['job'].append(int(job))
        solution['task'].append(int(task[1]))
    return solution

def chart(tasks, n_jobs, n_machines):
    solution = scheduling(tasks, n_jobs, n_machines)
    logger.info(solution)
    gantt(solution)
       
@njit(fastmath=True)        
def job_indices(tasks):
    indices = []
    ids = []
    n = tasks.shape[0]
    j = 0
    last = tasks[0]
    indices.append(j)
    for i in range(n+1):
        if i == n or tasks[i][0] != last[0] or tasks[i][1] != last[1]: # new operation
            ids.append(last[0])
            if (i < n and tasks[i][0] != last[0]):
                indices.append(j+1)
            j += 1
        if i < n:    
            last = tasks[i]
    return np.array(indices), np.array(ids)

@njit(fastmath=True) 
def reorder(tasks, order, n_operations, job_ids, job_indices):
    ids = job_ids[order]
    ordered = np.empty((n_operations,4))
    op_index = np.zeros(n_operations, dtype=numba.int32)
    for i in range(n_operations):
        machine = ids[i]
        index = job_indices[machine] + op_index[machine]
        op_index[machine] += 1
        ordered[i] = tasks[index]
    return ordered
        
@njit(fastmath=True) 
def exec_tasks(tasks, n_jobs, n_machines):
    machine_time = np.zeros(n_machines)
    machine_work_time = np.zeros(n_machines)
    job_time = np.zeros(n_jobs)
    for task in tasks:
        job = int(task[0])
        machine = int(task[2])
        time = task[3]
        # previous task needs to be finished and machine needs to be available
        end_time = max(machine_time[machine], job_time[job]) + time
        machine_time[machine] = end_time
        job_time[job] = end_time  
        machine_work_time[machine] += time
        #print("exec job {0} task {1} machine {2} start {3} end {4}".format(job, int(task[1]), machine, end_time-time, end_time))
    return machine_time, job_time, machine_work_time

@njit(fastmath=True) 
def filter_tasks(x, tasks, n_operations, n_machines):
    n = tasks.shape[0]
    operations = np.empty((n_operations,4))
    j = 0
    last = tasks[0]
    last_i = 0
    for i in range(n+1):
        if i == n or tasks[i][0] != last[0] or tasks[i][1] != last[1]: # new operation
            m = i - last_i
            sel_i = int(x[j]*10*n_machines) % m
            selected = tasks[last_i + sel_i]
            operations[j,:] = selected
            last_i = i
            j += 1
        if i < n:    
            last = tasks[i]
    return operations

@njit(fastmath=True)
def filtered_tasks(x, task_data, n_operations, n_machines, job_indices, job_ids):
    operations = filter_tasks(x, task_data, n_operations, n_machines)
    order = np.argsort(x[n_operations:])
    tasks = reorder(operations, order, n_operations, job_ids, job_indices)
    return tasks

class fitness: 

    def __init__(self, task_data, bounds, n_jobs, n_operations, n_machines, name):
        self.evals = mp.RawValue(ct.c_long, 0)  # writable across python processes
        self.best_y = mp.RawValue(ct.c_double, np.inf) # writable across python processes
        self.t0 = time.perf_counter()
        self.task_data = task_data   
        self.n_jobs = n_jobs
        self.n_operations = n_operations
        self.n_machines = n_machines 
        self.bounds = bounds
        self.name = name
        self.nobj = 3
        self.ncon = 0
        self.weights = np.array([1.0, 0.02, 0.001]) # only used for single objective optimization
        self.job_indices, self.job_ids = job_indices(task_data)
    
    def chart(self, x):
        tasks = filtered_tasks(x, self.task_data, self.n_operations, self.n_machines,
                                self.job_indices, self.job_ids)
        chart(tasks, self.n_jobs, self.n_machines)
        
    def fun(self, x): # multi objective function         
        tasks = filtered_tasks(x, self.task_data, self.n_operations, self.n_machines,
                               self.job_indices, self.job_ids)
        machine_time, job_time, machine_work_time = exec_tasks(tasks, self.n_jobs, self.n_machines)
        span = np.amax(machine_time)
        work = np.sum(machine_work_time)
        wmax = np.amax(machine_work_time)
        ys = np.array([span, work, wmax])      
        y = sum(self.weights*ys) # weighted sum  
        self.evals.value += 1
        if y < self.best_y.value:
            self.best_y.value = y  
            logger.info("evals = {0}: time = {1:.1f} y = {2:.2f} s = {3:.0f} w = {4:.0f} m = {5:.0f} m= {6:s} j= {7:s} w= {8:s}"
                .format(self.evals.value, dtime(self.t0), y, span, work, wmax,
                        str([int(si) for si in machine_time]),
                        str([int(oi) for oi in job_time]),
                        str([int(oi) for oi in machine_work_time]),
                        ))
        return ys      

    def __call__(self, x): # single objective function        
        ys = self.fun(x)
        return sum(self.weights*ys) # weighted sum  
 
def optall(multi_objective = True):
    for i in range(1,16):
        optimize(i, multi_objective)

def optimize(bi, multi_objective = True): 
    name = "BrandimarteMk" + str(bi)
    tasks, n_jobs, n_machines, n_operations, _ = read_fjs("data/1_Brandimarte/" + name + ".fjs")
    
    dim = 2*n_operations
    lower_bound = np.zeros(dim)
    lower_bound[:] = 0.0000001 
    upper_bound = np.zeros(dim)
    upper_bound[:] = 0.9999999
    bounds = Bounds(lower_bound, upper_bound)
          
    fit = fitness(tasks, bounds, n_jobs, n_operations, n_machines, name)
    if multi_objective:
        xs, front = modecpp.retry(fit.fun, fit.nobj, fit.ncon, fit.bounds, num_retries=32, popsize = 48, 
                  max_evaluations = 960000, nsga_update = True, workers=16)
        logger.info(name + " modecpp.retry(num_retries=32, popsize = 48, max_evals = 960000, nsga_update = True, workers=16" )
        logger.info(str([tuple(y) for y in front]))
    else:    
        store = retry.Store(fit, bounds) 
        logger.info(name + " Bite_cpp(960000,M=1).minimize, num_retries=256)" )
        retry.retry(store, Bite_cpp(960000,M=1).minimize, num_retries=256)    
    
    return fit, xs
    
def main():
    #optall(multi_objective = True)
    fit, xs = optimize(1, multi_objective = True)
    fit.chart(xs[0]) 
    
if __name__ == '__main__':
    main()