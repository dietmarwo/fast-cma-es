# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# Solving Solomons Benchmark for the 
# Vehicles Benchmark Problem with Time Windows.
# See http://web.cba.neu.edu/~msolomon/problems.htm
# See https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/Oneforall.adoc for a detailed description.

import numpy as np
import os
from numba import njit
from datetime import datetime
from fcmaes.optimizer import crfmnes_bite, wrapper
from fcmaes import retry
from scipy.optimize import Bounds

def parse_problem(filename):
    with open(filename) as csvfile:
        lines = csvfile.readlines()
        demand = []
        coord = [] 
        ready = [] 
        due = [] 
        service = []
        for line in lines:
            row = line.split()
            if len(row) == 2 and row[0][0].isdigit():
                number = int(row[0])
                capacity = int(row[1])                
            if len(row) < 5 or not row[0][0].isdigit():
                continue
            coord.append(np.array([float(row[1]), float(row[2])])) 
            demand.append(float(row[3]))
            ready.append(float(row[4]))
            due.append(float(row[5]))
            service.append(float(row[6]))
    n = len(demand)
    dtimes = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            dtimes[i,j] = np.linalg.norm(coord[i] - coord[j])
    return number, capacity, dtimes, np.array(demand), np.array(ready),\
                np.array(due), np.array(service)

@njit(fastmath=True)
def fitness_(seq, capacity, dtime, demands, readys, dues, services):
    n = len(seq)
    seq += 1
    sum_demand = 0
    sum_dtime = 0
    time = 0
    last = 0
    vehicles = 1
    for i in range(0, n+1):
        customer = seq[i] if i < n else 0
        demand = demands[customer]
        ready = readys[customer]
        due = dues[customer]
        service = services[customer]
        if sum_demand + demand > capacity or \
                time + dtime[last, customer] > due: 
            # end vehicle tour, return to base
            dt = dtime[last, 0]
            sum_dtime += dt
            time = 0
            sum_demand = 0
            vehicles += 1
            last = 0
        # go to customer
        dt = dtime[last, customer]
        time += dt 
        if time < ready:
            time = ready
        time += service       
        sum_demand += demand
        sum_dtime += dt
        last = customer
    return np.array([float(vehicles), sum_dtime])
    
class VRTPW():
    def __init__(self, problem):
        self.problem = problem
        filename = 'problems/' + problem + '.txt'
        self.vnumber, self.capacity, self.dtime, self.demand, self.ready,\
            self.due, self.service = parse_problem(filename)
        self.dim = len(self.demand) - 1
        self.bounds = Bounds([0]*self.dim, [1]*self.dim)
        
    def fitness(self, x):
        fit = fitness_(np.argsort(x), self.capacity, self.dtime, self.demand, \
                    self.ready, self.due, self.service)   
        return 10*fit[0] + fit[1] 
    
    def dump(self, seq, y, problem, opt_name=''):
        lines = []
        lines.append('Instance Name : ' + self.problem + '\n')
        lines.append('Date : ' + str(datetime.today().date()) + '\n')
        lines.append('Solution\n')
        n = len(seq)
        seq += 1
        sum_dtime = 0
        last = 0
        vehicles = 1
        sum_demand = 0
        time = 0
        tour = []
        for i in range(0, n+1):
            customer = seq[i] if i < n else 0
            demand = self.demand[customer]
            ready = self.ready[customer]
            due = self.due[customer]
            service = self.service[customer]
            if sum_demand + demand > self.capacity or \
                        time + self.dtime[last, customer] > due: 
                dt = self. dtime[last, 0]
                sum_dtime += dt
                time = 0
                lines.append('Route ' + str(vehicles) + ' : ' + ' '.join(map(str, tour)) + '\n')
                sum_demand = 0
                vehicles += 1
                tour = []
                last = 0
            dt = self.dtime[last, customer]
            time += dt 
            if time < ready:
                time = ready
            time += service
            sum_demand += demand
            sum_dtime += dt
            if customer != 0:
                tour.append(customer)
            last = customer
        print ("vehicles ", vehicles-1, "demands", sum_demand, "dtime", sum_dtime)
        filename = 'solutions/' + problem + '.txt'
        with open(filename, 'w') as f:
            f.writelines(lines)
        print(''.join(lines))
    
def optimize(vrtpw, opt, num_retries = 64):
    ret = retry.minimize(wrapper(vrtpw.fitness), 
                        vrtpw.bounds, num_retries = num_retries, optimizer=opt)
    vrtpw.dump(np.argsort(ret.x), ret.fun, vrtpw.problem, opt.name.replace(' ','_') + '_')

def opt_dir(dir):
    files = os.listdir(dir)
    files.sort()
    for file in files:
        problem = file.split('.')[0]
        vrtpw = VRTPW(problem) 
        popsize = 500
        max_evaluations = popsize*20000
        opt = crfmnes_bite(max_evaluations, popsize=popsize, M=6)
        optimize(vrtpw, opt, num_retries = 64)
    
if __name__ == '__main__':
    opt_dir('problems')

