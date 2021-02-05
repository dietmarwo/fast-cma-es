# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# Test for fcmaes coordinated retry applied to https://www.esa.int/gsp/ACT/projects/gtop/
# using https://github.com/esa/pygmo2 / pagmo2 optimization algorithms.
# 
# Please install pygmo before executing this test:
# pip install pygmo 

from fcmaes.astro import Messenger, Cassini2, Rosetta, Gtoc1, Cassini1, Sagas, Tandem, MessFull
from fcmaes.optimizer import logger, Optimizer, Sequence
from fcmaes.advretry import minimize
import pygmo as pg
import numpy as np
import math
from numpy.random import MT19937, Generator

class pygmo_udp(object):
    """Wraps a fcmaes fitness function as pygmo udp."""
    def __init__(self, fun, bounds): 
        self.fun = fun
        self.bounds = bounds
              
    def fitness(self, x):
        return [self.fun(x)]

    def get_bounds(self):
        return (self.bounds.lb, self.bounds.ub)

def de_cma_pyg(max_evaluations = 50000, popsize=31, stop_fittness = -math.inf, 
           de_max_evals = None, cma_max_evals = None):
    """Sequence de1220 -> cmaes pagmo."""

    deEvals = np.random.uniform(0.1, 0.3)
    if de_max_evals is None:
        de_max_evals = int(deEvals*max_evaluations)
    if cma_max_evals is None:
        cma_max_evals = int((1.0-deEvals)*max_evaluations)
    opt1 = De_pyg(popsize=popsize, max_evaluations = de_max_evals, stop_fittness = stop_fittness)
    opt2 = Cma_pyg(popsize=popsize, max_evaluations = cma_max_evals, 
                   stop_fittness = stop_fittness)
    return Sequence([opt1, opt2])

class Cma_pyg(Optimizer):
    """CMA_ES pagmo implementation."""
   
    def __init__(self, max_evaluations=50000, popsize = 31, guess=None):        
        Optimizer.__init__(self, max_evaluations, 'cma pagmo')
        self.popsize = popsize
        self.guess = guess

    def minimize(self, fun, bounds, guess=None, sdevs=0.3, rg=Generator(MT19937()), store=None):
        gen = int(self.max_eval_num(store) / self.popsize + 1)       
        algo = pg.algorithm(pg.cmaes(gen=gen, force_bounds = True, 
                    sigma0 = np.mean(sdevs), seed = int(rg.uniform(0, 2**32 - 1))))
        udp = pygmo_udp(fun, bounds)    
        prob = pg.problem(udp) 
        pop = pg.population(prob, self.popsize)
        if not guess is None: 
            scale = np.multiply(0.5 * (bounds.ub - bounds.lb), sdevs)
            for i in range(self.popsize):
                xi = np.random.normal(guess, scale)
                xi = np.maximum(np.minimum(xi, bounds.ub), bounds.lb)
                pop.set_x(i, xi)
        pop = algo.evolve(pop)
        return pop.champion_x, pop.champion_f, prob.get_fevals()

class De_pyg(Optimizer):
    """Differential Evolution pagmo implementation."""
    
    def __init__(self, max_evaluations=50000, popsize = None):        
        Optimizer.__init__(self, max_evaluations, 'de1220 pagmo')
        self.popsize = popsize

    def minimize(self, fun, bounds, guess=None, sdevs=None, rg=Generator(MT19937()), store=None):
        gen = int(self.max_eval_num(store) / self.popsize + 1)
        algo = pg.algorithm(pg.de1220(gen=gen, seed = int(rg.uniform(0, 2**32 - 1))))
        udp = pygmo_udp(fun, bounds)    
        prob = pg.problem(udp) 
        pop = pg.population(prob, self.popsize)
        pop = algo.evolve(pop)
        return pop.champion_x, pop.champion_f, prob.get_fevals()

def _test_optimizer(opt, problem, num_retries = 10000, num = 1, value_limit = 100.0, 
                    stop_val = -1E99, log = logger()):
    log.info("Testing coordinated retry " + opt.name +  ' ' + problem.name )
    for _ in range(num):
        ret = minimize(problem.fun, problem.bounds, value_limit, num_retries, log, 
                       optimizer=opt, stop_fitness = stop_val)

def main():
    numRuns = 10
    min_evals = 1500
    algo = de_cma_pyg(min_evals)
    fac = 1.005
    _test_optimizer(algo, Gtoc1(), num_retries = 10000, num = numRuns, 
                    value_limit = -300000.0, stop_val = -1581950/fac)
    _test_optimizer(algo, Cassini1(), num_retries = 4000, num = numRuns, 
                    value_limit = 20.0, stop_val = 4.9307*fac)
    _test_optimizer(algo, Cassini2(), num_retries = 6000, num = numRuns, 
                    value_limit = 20.0, stop_val = 8.383*fac)
    _test_optimizer(algo, Messenger(), num_retries = 8000, num = numRuns, 
                    value_limit = 20.0, stop_val = 8.63*fac)
    _test_optimizer(algo, Rosetta(), num_retries = 4000, num = numRuns, 
                    value_limit = 20.0, stop_val = 1.3433*fac)
    _test_optimizer(algo, Sagas(), num_retries = 4000, num = numRuns, 
                    value_limit = 100.0, stop_val = 18.187*fac)
    _test_optimizer(algo, Tandem(5), num_retries = 20000, num = numRuns, 
                    value_limit = -300.0, stop_val = -1500.6/fac)
    _test_optimizer(algo, MessFull(), num_retries = 50000, num = numRuns, 
                    value_limit = 12.0, stop_val = 1.959*fac)
 
if __name__ == '__main__':
    main()
    