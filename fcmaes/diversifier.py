# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.
from __future__ import annotations

""" Numpy based implementation of an diversifying wrapper / parallel retry mechanism. 

Uses the archive from CVT MAP-Elites (https://arxiv.org/abs/1610.05729)
and generalizes ideas from CMA-ME (https://arxiv.org/pdf/1912.02400.pdf)
to other wrapped algorithms. 

Both the parallel retry and the archive based modification of the fitness 
function enhance the diversification of the optimization result.
The resulting archive may be stored and can be used to continue the
optimization later.   

Requires a QD-fitness function returning both an fitness value and a
behavior vector used to determine the corresponding archive niche using
Voronoi tesselation. 

Returns an archive of niche-elites containing also for each niche statistics 
about the associated solutions.     
"""

import numpy as np
from numpy.random import Generator, MT19937, SeedSequence
from multiprocessing import Process
from scipy.optimize import Bounds
from fcmaes.optimizer import logger, dtime, de_cma, Optimizer
import multiprocessing as mp
import ctypes as ct
from time import perf_counter
from fcmaes.mapelites import Archive, update_archive
from fcmaes import advretry
import threadpoolctl

import logging
from typing import Optional, Callable, Tuple, Dict
from numpy.typing import ArrayLike

def minimize(qd_fitness: Callable[[ArrayLike], Tuple[float, np.ndarray]], 
            bounds: Bounds,
            desc_bounds: Bounds,
            niche_num: Optional[int] = 4000,
            samples_per_niche: Optional[int] = 20,
            retries: Optional[int] = None,
            workers: Optional[int] = mp.cpu_count(),
            archive: Optional[Archive] = None,
            opt_params: Optional[Dict] = {},
            logger: Optional[logging.Logger] = logger()) -> Archive:
    
    """Wraps an fcmaes optmizer/solver by hijacking its tell function.
    Works as CVT Map-Elites in maintaining an archive of diverse elites. 
    But this archive is not used to derive solution vectors, but to reevaluate them. 
    For each fitness result it determines its niche. The "told" fitness is
    determined relative to its local elite. If it is better the evaluated solution
    becomes the new niche-elite.  
    This way the wrapped solver is "tricked" to follow a QD-goal: Finding empty niches
    and improving all niches. This works not only for CMA-ES, but also for other 
    solvers: DE, CR-FM-NES and PGPE. Both their Python and C++ versions are supported. 
     
    Parameters
    ----------
    solver : evolutionary algorithm, needs to support ask/tell 
    qd_fitness : callable
        The objective function to be minimized. Returns a fitness value and a behavior vector. 
            ``qd_fitness(x) -> float, array``
        where ``x`` is an 1-D array with shape (n,)
    bounds : `Bounds`
        Bounds on variables. Instance of the `scipy.Bounds` class.
    desc_bounds : `Bounds`
        Bounds on behavior descriptors. Instance of the `scipy.Bounds` class.        
    niche_num : int, optional
        Number of niches.
    samples_per_niche : int, optional
        Number of samples used for niche computation.  
    retries : int, optional
        Number of optimization runs.
    workers : int, optional
        Number of spawned parallel worker processes.
    archive : Archive, optional
        If defined MAP-elites is continued for this archive.
    opt_params : dictionary, optional (or a list/tuple/array of these)
        Parameters selecting and configuring the wrapped solver.
        'solver' - supported are 'CMA','CMA_CPP','CRMFNES','CRMFNES_CPP','DE','DE_CPP','PGPE'
                    default is 'CMA_CPP'
        'popsize' - population size, default = 32
        'sigma' -  initial distribution sigma, default = rg.uniform(0.03, 0.3)**2)
        'mean' - initial distribution mean, default=rg.uniform(bounds.lb, bounds.ub)) 
        'max_evals' - maximal number of evaluations per run, default = 50000
        'stall_criterion' - how many iterations without progress allowed, default = 50 iterations 
        If a list/tuple/array of parameters are given, the corresponding solvers are called in a 
        sequence.      
    logger : logger, optional
        logger for log output of the retry mechanism. If None, logging
        is switched off. Default is a logger which logs both to stdout and
        appends to a file ``optimizer.log``.
        
    Returns
    -------
    archive : Archive
        Resulting archive of niches. Can be stored for later continuation of MAP-elites."""

    if retries is None:
        retries = workers
    dim = len(bounds.lb)
    if archive is None: 
        archive = Archive(dim, desc_bounds, niche_num)
        archive.init_niches(samples_per_niche)       
    t0 = perf_counter()   
    qd_fitness.archive = archive # attach archive for logging
    count = mp.RawValue(ct.c_long, 0)      
    minimize_parallel_(archive, qd_fitness, bounds, workers, 
                       opt_params, count, retries)
    if not logger is None:
        ys = np.sort(archive.get_ys())[:min(100, archive.capacity)] # best fitness values
        logger.info(f'best {min(ys):.3f} worst {max(ys):.3f} ' + 
                 f'mean {np.mean(ys):.3f} stdev {np.std(ys):.3f} time {dtime(t0)} s')
    return archive

def apply_advretry(fitness: Callable[[ArrayLike], float], 
                   descriptors: Callable[[ArrayLike], np.ndarray], 
                   bounds: Bounds, 
                   archive: Archive, 
                   optimizer: Optional[Optimizer] = None, 
                   num_retries: Optional[int] = 1000, 
                   workers: Optional[int] = mp.cpu_count(),
                   max_eval_fac: Optional[float] = 5.0,
                   logger: Optional[logging.Logger] = logger()):
        
    """Unifies the QD world with traditional optimization. It converts
    a QD-archive into a multiprocessing store used by the fcmaes smart
    boundary management meta algorithm (advretry). Then advretry is applied
    to find the global optimum. Finally the updated store is feed back into
    the QD-archive. For this we need a descriptor generating function 
    'descriptors' which may require reevaluation of the new solutions.  
    
     
    Parameters
    ----------
    solver : evolutionary algorithm, needs to support ask/tell 
    fitness : callable
        The objective function to be minimized. Returns a fitness value. 
            ``fitness(x) -> float``
    descriptors : callable
        Generates the descriptors for a solution. Returns a behavior vector. 
            ``descriptors(x) -> array``
        where ``x`` is an 1-D array with shape (n,)
    bounds : `Bounds`
        Bounds on variables. Instance of the `scipy.Bounds` class.
    archive : Archive
        Improves the solutions if this archive.
    optimizer : optimizer.Optimizer, optional
        Optimizer to use. Default is a sequence of differential evolution and CMA-ES.
    num_retries : int, optional
        Number of optimization runs.
    workers : int, optional
        Number of spawned parallel worker processes.
    max_eval_fac : int, optional
        Final limit of the number of function evaluations = max_eval_fac*min_evaluations  
    logger : logger, optional
        logger for log output of the retry mechanism. If None, logging
        is switched off. Default is a logger which logs both to stdout and
        appends to a file ``optimizer.log``."""

    if optimizer is None:
        optimizer = de_cma(1500)
    # generate advretry store
    store = advretry.Store(fitness, bounds, num_retries=num_retries, 
                           max_eval_fac=max_eval_fac, logger=logger) 
    # select only occupied entries
    ys = archive.get_ys()    
    valid = (ys < np.inf)
    ys = ys[valid]
    xs = archive.get_xs()[valid]
    t0 = perf_counter() 
    # transfer to advretry store
    for i in range(len(ys)):
        store.add_result(ys[i], xs[i], 0)
    # perform parallel retry
    advretry.retry(store, optimizer.minimize, workers=workers)
    # transfer back to archive
    ys = store.get_ys()    
    xs = store.get_xs()
    descs = [descriptors(x) for x in xs] # may involve reevaluating fitness
    niches = archive.index_of_niches(descs)
    for i in range(len(ys)):
        archive.set(niches[i], (ys[i], descs[i]), xs[i])
    archive.argsort()
    if not logger is None:
        ys = np.sort(archive.get_ys())[:min(100, archive.capacity)] # best fitness values
        logger.info(f'best {min(ys):.3f} worst {max(ys):.3f} ' + 
                 f'mean {np.mean(ys):.3f} stdev {np.std(ys):.3f} time {dtime(t0)} s')    

def minimize_parallel_(archive, fitness, bounds, workers, 
                         opt_params, count, retries):
    sg = SeedSequence()
    rgs = [Generator(MT19937(s)) for s in sg.spawn(workers)]
    stopProcess = mp.RawValue(ct.c_bool, False)
    proc=[Process(target=run_minimize_,
            args=(archive, fitness, bounds, rgs[p],
                  opt_params, count, retries, p, stopProcess)) for p in range(workers)]
    [p.start() for p in proc]
    [p.join() for p in proc]
                    
def run_minimize_(archive, fitness, bounds, rg, opt_params, count, retries, p, stopProcess):  
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        if isinstance(opt_params, (list, tuple, np.ndarray)):
            for params in opt_params: # call MAP-Elites
                if 'elites' == params.get('solver') and p % opt_params[0].get('use', 2) == 0:  
                    #print('elites ' + str(p) + ' started') 
                    run_map_elites_(archive, fitness, bounds, rg, stopProcess, params)
                    #print('elites '+ str(p) + ' finished')
                    return
        while count.value < retries: # call solvers in loop
            count.value += 1      
            best_x = None
            if isinstance(opt_params, (list, tuple, np.ndarray)):
                for params in opt_params: # call in sequence
                    if 'elites' == params.get('solver'):
                        continue
        #           print(params.get('solver') + str(p) + ' started')
                    if best_x is None:
                        best_x = minimize_(archive, fitness, bounds, rg, stopProcess, p, params)
                    else:
                        best_x = minimize_(archive, fitness, bounds, rg, stopProcess, p, params, x0 = best_x)
        #           print(params.get('solver') + str(p) + ' finished')
            else:        
                    minimize_(archive, fitness, bounds, rg, opt_params) 
        stopProcess.value = True # stop all processes      

from fcmaes.mapelites import variation_,  iso_dd_
                
def run_map_elites_(archive, fitness, bounds, rg, stopProcess, opt_params = {}):    
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        popsize = opt_params.get('popsize', 32)  
        use_sbx = opt_params.get('use_sbx', True)     
        dis_c = opt_params.get('dis_c', 20)   
        dis_m = opt_params.get('dis_m', 20)  
        iso_sigma = opt_params.get('iso_sigma', 0.1)
        line_sigma = opt_params.get('line_sigma', 0.2)
        select_n = archive.capacity
        while not stopProcess.value:                
            if use_sbx:
                pop = archive.random_xs(select_n, popsize, rg)
                xs = variation_(pop, bounds.lb, bounds.ub, rg, dis_c, dis_m)
            else:
                x1 = archive.random_xs(select_n, popsize, rg)
                x2 = archive.random_xs(select_n, popsize, rg)
                xs = iso_dd_(x1, x2, bounds.lb, bounds.ub, rg, iso_sigma, line_sigma)    
            yds = [fitness(x) for x in xs]
            descs = np.array([yd[1] for yd in yds])
            niches = archive.index_of_niches(descs)
            for i in range(len(yds)):
                archive.set(niches[i], yds[i], xs[i])
            archive.argsort()   
            select_n = archive.get_occupied()   

def minimize_(archive, fitness, bounds, rg, stopProcess, p, opt_params, x0 = None):  
    es = get_solver_(bounds, opt_params, rg, p, x0) 
    max_evals = opt_params.get('max_evals', 50000)
    stall_criterion = opt_params.get('stall_criterion', 50)
    old_ys = None
    last_improve = 0
    max_iters = int(max_evals/es.popsize)
    best_x = None
    best_y = np.inf
    for iter in range(max_iters):
        xs = es.ask()
        ys, real_ys = update_archive(archive, xs, fitness)
        # update best real fitness
        yi = np.argmin(real_ys)
        ybest = real_ys[yi] 
        if ybest < best_y:
            best_y = ybest
            best_x = xs[yi]
        if not old_ys is None:
            if (np.sort(ys) < old_ys).any():
                last_improve = iter          
        if last_improve + stall_criterion < iter:
            break
        stop = es.tell(ys)
        if stop != 0 or stopProcess.value:
            #print('stop = ', stop)
            break 
        old_ys = np.sort(ys)
    return best_x # real best solution

from fcmaes import cmaes, cmaescpp, crfmnescpp, pgpecpp, decpp, crfmnes, de

def get_solver_(bounds, opt_params, rg, p, x0 = None):
    dim = len(bounds.lb)
    popsize = opt_params.get('popsize', 31) 
    #popsize -= int(p // 2)  
    sigma = opt_params.get('sigma',rg.uniform(0.03, 0.3)**2)
    mean = opt_params.get('mean', rg.uniform(bounds.lb, bounds.ub)) \
                if x0 is None else x0
    name = opt_params.get('solver', 'CMA_CPP')
    if name == 'CMA':
        return cmaes.Cmaes(bounds, x0 = mean,
                          popsize = popsize, input_sigma = sigma, rg = rg)
    elif name == 'CMA_CPP':
        return cmaescpp.ACMA_C(dim, bounds, x0 = mean, #stop_hist = 0,
                          popsize = popsize, input_sigma = sigma, rg = rg)
    elif name == 'CRMFNES':
        return crfmnes.CRFMNES(dim, bounds, x0 = mean,
                          popsize = popsize, input_sigma = sigma, rg = rg)
    elif name == 'CRMFNES_CPP':
        return crfmnescpp.CRFMNES_C(dim, bounds, x0 = mean,
                          popsize = popsize, input_sigma = sigma, rg = rg)
    elif name == 'DE':
        return de.DE(dim, bounds, popsize = popsize, rg = rg)
    elif name == 'DE_CPP':
        return decpp.DE_C(dim, bounds, popsize = popsize, rg = rg)
    elif name == 'PGPE':
        return pgpecpp.PGPE_C(dim, bounds, x0 = mean,
                          popsize = popsize, input_sigma = sigma, rg = rg)
    else:
        print ("invalid solver")
        return None
            
