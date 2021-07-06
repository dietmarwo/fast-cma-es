# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# multi objective optimization experiments

import numpy as np
import random
import math
import time
import glob
from scipy.optimize import Bounds
from fcmaes.optimizer import de_cma, Bite_cpp, random_search, dtime, logger
from fcmaes import moretry, retry, mode
from deap import base
from deap import creator
from deap import tools
import array
import deap.benchmarks as db

from fcmaes.astro import Cassini1, Cassini2, Tandem

class cassini1_mo: 

    def __init__(self):
        self.base = Cassini1()
        self.bounds = self.base.bounds
        self.weight_bounds = Bounds([1, 0.01], [100, 1]) # weighting of objectives
        self.name = self.base.name
 
    def fun(self, x):
        dv = self.base.fun(np.array(x)) # delta velocity, original objective (km/s)
        mission_time = sum(x[1:]) # mission time (days)
        y = np.empty(2)
        y[0] = dv       
        y[1] = mission_time
        return y

class cassini2_mo: 

    def __init__(self):
        self.base = Cassini2()
        self.bounds = self.base.bounds
        self.weight_bounds = Bounds([1, 0.01], [100, 1]) # weighting of objectives
        self.name = self.base.name
 
    def fun(self, x):
        dv = self.base.fun(np.array(x)) # delta velocity, original objective (km/s)
        mission_time = sum(x[4:9]) # mission time (days)
        y = np.empty(2)
        y[0] = dv       
        y[1] = mission_time
        return y

class tandem_mo: 

    def __init__(self, constrained=False):
        self.base = Tandem(5, constrained=constrained)
        self.bounds = self.base.bounds
        self.weight_bounds = Bounds([1, 0], [1, 0]) # ignore 2nd objective
        self.name = self.base.name
 
    def fun(self, x):
        final_mass = self.base.fun(np.array(x)) # original objective (-kg)
        mission_time = sum(x[4:8]) # mission time (days)
        y = np.empty(2)
        y[0] = final_mass       
        y[1] = mission_time
        return y

class zdt1: 

    def __init__(self, dim):
        self.fun = db.zdt1
        self.bounds = Bounds([0]*dim, [1]*dim)
        self.weight_bounds = Bounds([0.01, 0.01], [1, 1]) 
        self.name = 'zdt1(' + str(dim) + ')'

class schaffer: 

    def __init__(self, dim):
        self.fun = db.schaffer_mo
        self.bounds = Bounds([-1000]*dim, [1000]*dim)
        self.weight_bounds = Bounds([0.01, 0.01], [1, 1]) 
        self.name = 'schaffer(' + str(dim) + ')'

class poloni: 

    def __init__(self, dim):
        self.fun = db.poloni
        self.bounds = Bounds([-math.pi]*dim, [math.pi]*dim)
        self.weight_bounds = Bounds([0.01, 0.01], [1, 1]) 
        self.name = 'poloni(' + str(dim) + ')'

class fonseca: 

    def __init__(self, dim):
        self.fun = db.fonseca
        self.bounds = Bounds([-4]*dim, [4]*dim) 
        self.weight_bounds = Bounds([0.01, 0.01], [1, 1]) 
        self.name = 'fonseca(' + str(dim) + ')'
 

def uniform(bounds):
    return [random.uniform(b[0], b[1]) for b in bounds]

FirstCall = True
# adapted copy from https://github.com/ppgaluzio/MOBOpt/blob/master/mobopt/_NSGA2.py
def nsgaII(NObj, objective, pbounds, seed=None, NGEN=20000, MU=400, CXPB=0.9):
    random.seed(seed)

    global FirstCall
    if FirstCall:
        creator.create('FitnessMin', base.Fitness, weights=(-1.0,)*NObj)
        creator.create('Individual', array.array, typecode='d',
                       fitness=creator.FitnessMin)
        FirstCall = False
    toolbox = base.Toolbox()

    NDIM = len(pbounds)

    toolbox.register('attr_float', uniform, pbounds)

    toolbox.register('individual',
                     tools.initIterate,
                     creator.Individual,
                     toolbox.attr_float)

    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    toolbox.register('evaluate', objective)

    toolbox.register('mate',
                     tools.cxSimulatedBinaryBounded,
                     low=pbounds[:, 0].tolist(),
                     up=pbounds[:, 1].tolist(),
                     eta=20.0)

    toolbox.register('mutate',
                     tools.mutPolynomialBounded,
                     low=pbounds[:, 0].tolist(),
                     up=pbounds[:, 1].tolist(),
                     eta=20.0,
                     indpb=1.0/NDIM)

    toolbox.register('select', tools.selNSGA2)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('min', np.min, axis=0)
    stats.register('max', np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = 'gen', 'evals', 'std', 'min', 'avg', 'max'

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)

    for gen in range(1, NGEN):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        # print(logbook.stream)

    front = np.array([ind.fitness.values for ind in pop])
    return pop, logbook, front
  
def nsgaII_test(problem, fname, NGEN=2000, MU=100, value_limits = None):
    time0 = time.perf_counter() # optimization start time
    name = problem.name 
    logger().info('optimize ' + name + ' nsgaII') 
    pbounds = np.array(list(zip(problem.bounds.lb, problem.bounds.ub)))
    pop, logbook, front = nsgaII(2, problem.fun, pbounds, NGEN=NGEN, MU=MU) 
    logger().info(name + ' nsgaII time ' + str(dtime(time0)))    
    name = 'nsgaII_' + str(NGEN) + '_' + str(MU) + name + '_' + fname
    np.savez_compressed(name, xs=pop, ys=front)
    if not value_limits is None:
        front = np.array(
            [y for y in front if all([y[i] < value_limits[i] for i in range(len(y))])])
    retry.plot(front, name)
       
def plot_all(folder, fname):
    files = glob.glob(folder + '/*.npz', recursive=True)
    xs = []
    ys = []
    for file in files:
        with np.load(file) as data:
            xs += list(data['xs'])
            ys += list(data['ys'])
    xs = np.array(xs); ys = np.array(ys)         
    xs, front = moretry.pareto(xs, ys)
    retry.plot(ys, fname + '_all.png', interp=False)
    retry.plot(front, fname + '_front.png')

def de_minimize_plot(problem, name, popsize = 64, max_eval = 100000, nobj = 2):
    mode.minimize_plot(problem.name + '_' + name, problem.fun, problem.bounds, nobj, popsize = popsize, 
                       max_eval = max_eval, nsga_update=False, plot_name = "nsga_off")

def nsga_minimize_plot(problem, name, popsize = 64, max_eval = 100000, nobj = 2):
    mode.minimize_plot(problem.name + '_' + name, problem.fun, problem.bounds, nobj, popsize = popsize, 
                       max_eval = max_eval, nsga_update=True, plot_name = "nsga_on")

def minimize_plot(problem, opt, name, exp = 2.0, num_retries = 1024, value_limits=None):
    moretry.minimize_plot(problem.name + '_' + name, opt, 
                          problem.fun, problem.bounds, problem.weight_bounds, 
                          num_retries = num_retries, exp = exp, value_limits = value_limits)

def adv_minimize_plot(problem, opt, name, value_limit = math.inf, num_retries = 10240):
    moretry.adv_minimize_plot(problem.name + '_' + name, opt, 
                              problem.fun, problem.bounds, value_limit = value_limit,
                              num_retries = num_retries)

if __name__ == '__main__':
    
    de_minimize_plot(zdt1(20), '100k64')
    de_minimize_plot(schaffer(20), '100k64')
    de_minimize_plot(poloni(20), '100k64')
    de_minimize_plot(fonseca(20), '100k64')
    
    nsga_minimize_plot(zdt1(20), '100k64')
    nsga_minimize_plot(schaffer(20), '100k64')
    nsga_minimize_plot(poloni(20), '100k64')
    nsga_minimize_plot(fonseca(20), '100k64')
        
    minimize_plot(zdt1(20), Bite_cpp(M=16), '50k1k')
    minimize_plot(schaffer(20), Bite_cpp(M=16), '50k1k')
    minimize_plot(poloni(20), Bite_cpp(M=16), '50k1k', exp=1.0)
    minimize_plot(fonseca(20), Bite_cpp(M=16), '50k1k', exp=3.0)
     
    minimize_plot(zdt1(20), de_cma(), '50k1k')
    minimize_plot(schaffer(20), de_cma(), '50k1k')
    minimize_plot(poloni(20), de_cma(), '50k1k', exp=1.0)
    minimize_plot(fonseca(20), de_cma(), '50k1k', exp=3.0)
         
    minimize_plot(zdt1(20), random_search(), '50k1k')
    minimize_plot(schaffer(20), random_search(), '50k1k')
    minimize_plot(poloni(20), random_search(), '50k1k', exp=1.0)
    minimize_plot(fonseca(20), random_search(), '50k1k', exp=3.0)
     
    minimize_plot(cassini1_mo(), de_cma(50000), '50k4k', num_retries=4096, value_limits=[40, 7000])    
    minimize_plot(cassini1_mo(), Bite_cpp(50000, M=16), '50k4k', num_retries=4096, value_limits=[40, 7000])
    minimize_plot(cassini1_mo(), random_search(50000), '50k4k', num_retries=4096, value_limits=[40, 7000])
     
    minimize_plot(cassini2_mo(), de_cma(50000), '50k4k', num_retries=4096, value_limits=[40, 7000])   
    minimize_plot(cassini2_mo(), Bite_cpp(50000, M=16), '50k4k', num_retries=4096, value_limits=[40, 7000])
    minimize_plot(cassini2_mo(), random_search(50000), '50k4k', num_retries=4096, value_limits=[40, 7000])
     
    minimize_plot(tandem_mo(), de_cma(100000), '100k10k', num_retries=10240, exp=1.0)
    minimize_plot(tandem_mo(), Bite_cpp(100000, M=16), '100k10k', num_retries=10240, exp=1.0)
    minimize_plot(tandem_mo(), random_search(100000), '100k10k', num_retries=10240, exp=1.0)
    
    nsgaII_test(zdt1(20), '_front.png')
    nsgaII_test(schaffer(20), '_front.png')
    nsgaII_test(poloni(20),  '_front.png')
    nsgaII_test(fonseca(20), '_front.png')

    adv_minimize_plot(tandem_mo(), de_cma(1500), '_smart', value_limit = 0, num_retries = 60000)

      
