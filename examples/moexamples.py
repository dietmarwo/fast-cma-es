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
from numpy.random import Generator, MT19937
from fcmaes.optimizer import de_cma, Bite_cpp, Cma_cpp, dtime, logger
from fcmaes import moretry, advretry
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
        self.weight_bounds = Bounds([10, 0.01], [1000, 1]) # weighting of objectives
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
        self.weight_bounds = Bounds([10, 0.01], [1000, 1]) # weighting of objectives
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

def monte_carlo(fun, bounds, n=500000, value_limits=None):
    rg = Generator(MT19937())
    ys = []
    for _ in range(n):
        x = rg.uniform(bounds.lb, bounds.ub)
        y = fun(x)
        if value_limits is None or all([y[i] < value_limits[i] for i in range(len(y))]):
            ys.append(y)
    ys = np.array(ys)
    pareto = moretry._pareto(ys)
    return ys[pareto]
    
def mo_retry(problem, opt, fname, value_limits = None, num_retries = 1024, exp = 2.0):
    time0 = time.perf_counter() # optimization start time
    name = problem.name 
    logger().info('optimize ' + name + ' ' + opt.name) 
    xs, ys = moretry.minimize(problem.fun,
             problem.bounds, problem.weight_bounds, 
             value_exp = exp,
             value_limits = value_limits,
             num_retries = num_retries,              
             optimizer = opt,
             logger=logger())
    np.savez_compressed(name + '_' + fname, xs=xs, ys=ys)
    xs, front = moretry.pareto(xs, ys)
    logger().info(name + ' ' + opt.name + ' time ' + str(dtime(time0))) 
    moretry.plot(front, name + '_' + fname + '.png')

def mo_adv_retry(problem, opt, fname, value_limit = math.inf, num_retries = 1024):
    time0 = time.perf_counter() # optimization start time
    name = problem.name 
    logger().info('smart optimize ' + name + ' ' + opt.name) 
    store = advretry.Store(problem.bounds, capacity=5000, logger=logger(), 
                           num_retries=num_retries) 
    ret = advretry.retry(problem.base.fun, store, opt.minimize, num_retries, value_limit)
    xs = np.array(store.get_xs())
    ys = np.array([problem.fun(x) for x in xs])
    np.savez_compressed(name + '_' + fname, xs=xs, ys=ys)
    xs, front = moretry.pareto(xs, ys)
    logger().info(name + ' ' + opt.name + ' time ' + str(dtime(time0))) 
    moretry.plot(front, name + '_' + fname + '.png')

def monte(problem, fname, n=500000, value_limits = None):
    time0 = time.perf_counter() # optimization start time
    name = problem.name 
    logger().info('monte carlo ' + name) 
    front = monte_carlo(problem.fun, problem.bounds, n=n, value_limits=value_limits)
    logger().info(name + ' monte carlo time ' + str(dtime(time0))) 
    moretry.plot(front, 'monte_' +  name + '_' +  fname)
  
def nsgaII_test(problem, fname, NGEN=2000, MU=100, value_limits = None):
    time0 = time.perf_counter() # optimization start time
    name = problem.name 
    logger().info('optimize ' + name + ' nsgaII') 
    pbounds = np.array(list(zip(problem.bounds.lb, problem.bounds.ub)))
    pop, logbook, front = nsgaII(2, problem.fun, pbounds, NGEN=NGEN, MU=MU) 
    logger().info(name + ' nsgaII time ' + str(dtime(time0)))    
    if not value_limits is None:
        front = np.array(
            [y for y in front if all([y[i] < value_limits[i] for i in range(len(y))])])
    moretry.plot(front, 'nsgaII_' + name + '_' + fname)
   
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
    moretry.plot(ys, fname + '_all.png', interp=False)
    moretry.plot(front, fname + '_front.png')

if __name__ == '__main__':
    
    plot_all("tandem/*", "tandem")
    import sys
    sys.exit()
    
    mo_retry(cassini1_mo(), Bite_cpp(M=16), '_bite_front', value_limits=[40, 2300])
    mo_retry(cassini2_mo(), Bite_cpp(M=16), '_bite_front', value_limits=[40, 2000])
    
    mo_retry(cassini1_mo(), de_cma(), '_decma_front', value_limits=[40, 2300])
    mo_retry(cassini2_mo(), de_cma(), '_decma_front', value_limits=[40, 2000])

    mo_retry(zdt1(20), Bite_cpp(M=16), '_bite_front')
    mo_retry(schaffer(20), Bite_cpp(M=16), '_bite_front')
    mo_retry(poloni(20), Bite_cpp(M=16), '_bite_front', exp=1.0)
    mo_retry(fonseca(20), Bite_cpp(M=16), '_bite_front', exp=3.0)

    mo_retry(zdt1(20), de_cma(), '_decma_front')
    mo_retry(schaffer(20), de_cma(), '_decma_front')
    mo_retry(poloni(20), de_cma(), '_decma_front', exp=1.0)
    mo_retry(fonseca(20), de_cma(), '_decma_front', exp=3.0)
    
    mo_retry(zdt1(20), Cma_cpp(), '_cma_front')
    mo_retry(schaffer(20), Cma_cpp(), '_cma_front')
    mo_retry(poloni(20), Cma_cpp(), '_cma_front', exp=1.0)
    mo_retry(fonseca(20), Cma_cpp(), '_cma_front', exp=3.0)
    
    monte(zdt1(20), '_front.png')
    monte(schaffer(20), '_front.png')
    monte(poloni(20),  '_front.png')
    monte(fonseca(20), '_front.png')

    nsgaII_test(cassini1_mo(), '_front.png', value_limits = [50, 2300])
    nsgaII_test(zdt1(20), '_front.png')
    nsgaII_test(schaffer(20), '_front.png')
    nsgaII_test(poloni(20),  '_front.png')
    nsgaII_test(fonseca(20), '_front.png')
    
    mo_retry(tandem_mo(), de_cma(100000), '_de_cma_front', num_retries=4096, exp=1.0)
    # takes some time
    # mo_adv_retry(tandem_mo(), de_cma(1500), '_smart_front', value_limit = 0, num_retries = 40960)
     