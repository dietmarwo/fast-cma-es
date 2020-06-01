# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

import math
import os
from numpy.random import Generator, MT19937, SeedSequence
from scipy.optimize import OptimizeResult, Bounds
import multiprocessing as mp
from multiprocessing import Process
from fcmaes.retry import Store

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

def minimize(prob, 
             algo,
             value_limit = math.inf,
             num_retries = 100*mp.cpu_count(),
             logger = None,
             workers = mp.cpu_count(),
             popsize = 1, 
             ):   
    """Minimization of a scalar function of one or more variables using parallel retry.
       Similar to fcmaes.retry but works with pygmo / pagmo problems + algorithms.
       For problems with equality/inequality contraints or multiple objectives fcmaes.retry cannot 
       be used since a fcmaes objective function is expected to return a single value. 
       pygmo / pagmo support both contraints and multiple objectives. Alternatively
       you can use https://esa.github.io/pygmo2/archipelago.html but it is a bit tricky
       to configure it to use multiprocessing. As default it uses multithreading which means
       it scales less with the number of available processor cores. 
       
    Parameters
    ----------
    prob : pygmo/pagmo problem, https://esa.github.io/pagmo2/docs/cpp/problem.html
        The objective function to be minimized.
    algo : pygmo/pagmo algorithm, https://esa.github.io/pagmo2/docs/cpp/algorithm.html
        The optimizer
    value_limit : float, optional
        Upper limit for optimized function values to be stored. 
    num_retries : int, optional
        Number of optimization retries.    
    logger : logger, optional
        logger for log output of the retry mechanism. If None, logging
        is switched off. Default is a logger which logs both to stdout and
        appends to a file ``optimizer.log``.
    workers : int, optional
        number of parallel processes used. Default is mp.cpu_count()
    popsize = int, optional
        population size 
     
    Returns
    -------
    res : scipy.OptimizeResult
        The optimization result is represented as an ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, 
        ``fun`` the best function value, ``nfev`` the number of function evaluations,
        ``success`` a Boolean flag indicating if the optimizer exited successfully. """

    lb, ub = prob.get_bounds()
    bounds = Bounds(lb, ub)
    store = Store(bounds, logger = logger)
    return retry(store, prob, algo, num_retries, value_limit, popsize, workers)
                 
def retry(store, prob, algo, num_retries, value_limit = math.inf, popsize=1, workers=mp.cpu_count()):
    try:
        import pygmo as pg
    except ImportError as e:
        raise ImportError("Please install PYGMO (pip install pygmo) to use PAGMO optimizers") from e
    sg = SeedSequence()
    rgs = [Generator(MT19937(s)) for s in sg.spawn(workers)]
    proc=[Process(target=_retry_loop,
            args=(pid, rgs, store, prob, algo, num_retries, value_limit, popsize, pg)) for pid in range(workers)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    store.sort()
    store.dump()
    return OptimizeResult(x=store.get_x_best(), fun=store.get_y_best(), 
                          nfev=store.get_count_evals(), success=True)
        
def _retry_loop(pid, rgs, store, prob, algo, num_retries, value_limit, popsize, pg):
    
    while store.get_runs_compare_incr(num_retries):      
        try:            
            seed = int(rgs[pid].uniform(0, 2**32 - 1))
            pop = pg.population(prob, popsize, seed=seed)
            pop = algo.evolve(pop)
        except Exception:
            pass  # ignore "Maximum number of iteration reached"      
        sol = pop.champion_x
        y = pop.champion_f
        evals = pop.problem.get_fevals()
         
        feasible = prob.feasibility_x(pop.champion_x)
        if feasible:
            store.add_result(y[0], sol, evals, value_limit)
            store.dump()

