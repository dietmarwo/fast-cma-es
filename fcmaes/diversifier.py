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
from fcmaes.optimizer import dtime, de_cma, Optimizer
import multiprocessing as mp
import ctypes as ct
from time import perf_counter
from fcmaes.mapelites import Archive, update_archive, rng
from fcmaes import advretry
from fcmaes.evaluator import is_debug_active
from loguru import logger
import threadpoolctl

from typing import Optional, Callable, Tuple, Dict
from numpy.typing import ArrayLike

def minimize(qd_fitness: Callable[[ArrayLike], Tuple[float, np.ndarray]], 
            bounds: Bounds,
            qd_bounds: Bounds,
            niche_num: Optional[int] = 10000,
            samples_per_niche: Optional[int] = 20,
            max_evals: Optional[int] = None,
            workers: Optional[int] = mp.cpu_count(),
            archive: Optional[Archive] = None,
            opt_params: Optional[Dict] = {},
            use_stats: Optional[bool] = False,            
            ) -> Archive:
    
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
    qd_bounds : `Bounds`
        Bounds on behavior descriptors. Instance of the `scipy.Bounds` class.        
    niche_num : int, optional
        Number of niches.
    samples_per_niche : int, optional
        Number of samples used for niche computation. 
        If samples_per_niche > 0 cvt-clustering is used, else grid-clustering is used. 
    max_evals : int, optional
        Number of fitness evaluations.
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
    use_stats : bool, optional 
        If True, archive accumulates statistics of the solutions
                
    Returns
    -------
    archive : Archive
        Resulting archive of niches. Can be stored for later continuation of MAP-elites."""

    if max_evals is None:
        max_evals = workers*50000
    dim = len(bounds.lb)
    if archive is None: 
        archive = Archive(dim, qd_bounds, niche_num, use_stats)
        archive.init_niches(samples_per_niche)
        # initialize archive with random values
        archive.set_xs(rng.uniform(bounds.lb, bounds.ub, (niche_num, dim)))         
    t0 = perf_counter()   
    qd_fitness.archive = archive # attach archive for logging     
    minimize_parallel_(archive, qd_fitness, bounds, workers, opt_params, max_evals)
    if is_debug_active():
        ys = np.sort(archive.get_ys())[:min(100, archive.capacity)] # best fitness values
        logger.debug(f'best {min(ys):.3f} worst {max(ys):.3f} ' + 
                 f'mean {np.mean(ys):.3f} stdev {np.std(ys):.3f} time {dtime(t0)} s')
    return archive

def apply_advretry(fitness: Callable[[ArrayLike], float], 
                   qd_fitness: Callable[[ArrayLike], Tuple[float, np.ndarray]], 
                   bounds: Bounds, 
                   archive: Archive, 
                   optimizer: Optional[Optimizer] = None, 
                   num_retries: Optional[int] = 1000, 
                   workers: Optional[int] = mp.cpu_count(),
                   max_eval_fac: Optional[float] = 5.0,
                   xs: Optional[np.ndarray] = None,
                   ys: Optional[np.ndarray] = None,
                   x_conv: Callable[[ArrayLike], ArrayLike] = None):
        
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
    qf_fun : callable
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
    xs : ndarray, optional
        Used to initialize advretry. If undefined the archive content is used. 
        If xs is defined, ys must be too
    ys : ndarray, optional
        Used to initialize advretry. If undefined the archive content is used.  
    x_conv : callable, optional
        If defined converts the x in xs to solutions suitable for the given archive.
        If undefined it is assumed that the x in xs are valid archive solutons.   
    """

    if optimizer is None:
        optimizer = de_cma(1500)
    # generate advretry store
    store = advretry.Store(fitness, bounds, num_retries=num_retries, 
                           max_eval_fac=max_eval_fac)  
                         
    # select only occupied entries
    if xs is None:
        ys = archive.get_ys()    
        valid = (ys < np.inf)
        ys = ys[valid]
        xs = archive.get_xs()[valid]
    t0 = perf_counter() 
    # transfer to advretry store
    for i in range(len(ys)):
        store.add_result(ys[i], xs[i], 1)
    # perform parallel retry
    advretry.retry(store, optimizer.minimize, workers=workers)
    # transfer back to archive 
    xs = store.get_xs()
    if not x_conv is None:
        xs = [x_conv(x) for x in xs]
    yds = [qd_fitness(x) for x in xs]
    descs = np.array([yd[1] for yd in yds])
    ys = np.array([yd[0] for yd in yds])
    niches = archive.index_of_niches(descs)
    for i in range(len(ys)):
        archive.set(niches[i], (ys[i], descs[i]), xs[i])
    archive.argsort()
    if is_debug_active():
        ys = np.sort(archive.get_ys())[:min(100, archive.capacity)] # best fitness values
        logger.debug(f'best {min(ys):.3f} worst {max(ys):.3f} ' + 
                 f'mean {np.mean(ys):.3f} stdev {np.std(ys):.3f} time {dtime(t0)} s')    

def minimize_parallel_(archive, fitness, bounds, workers, opt_params, max_evals):
    sg = SeedSequence()
    rgs = [Generator(MT19937(s)) for s in sg.spawn(workers)]
    evals = mp.RawValue(ct.c_long, 0)
    proc=[Process(target=run_minimize_,
            args=(archive, fitness, bounds, rgs[p],
                  opt_params, p, workers, evals, max_evals)) for p in range(workers)]
    [p.start() for p in proc]
    [p.join() for p in proc]
                    
def run_minimize_(archive, fitness, bounds, rg, opt_params, p, workers, evals, max_evals):  
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        if isinstance(opt_params, (list, tuple, np.ndarray)):
            default_workers = int(workers/2) if len(opt_params) > 1 else workers
            for params in opt_params: # call MAP-Elites
                if 'elites' == params.get('solver'):
                    elites_workers = params.get('workers', default_workers) 
                    if p < elites_workers:
                        run_map_elites_(archive, fitness, bounds, rg, evals, max_evals, params)
                        return
        while evals.value < max_evals: # call solvers in loop     
            best_x = None
            if isinstance(opt_params, (list, tuple, np.ndarray)):
                for params in opt_params: # call in sequence
                    if 'elites' == params.get('solver'):
                        continue # ignore in loop
                    if best_x is None:
                        # selecting a niche elite is no improvement over random x0
                        x0 = None#, _, _ = archive.random_xs_one(select_n, rg)
                        best_x = minimize_(archive, fitness, bounds, rg, evals, max_evals, params, 
                                           x0 = x0)
                    else:
                        best_x = minimize_(archive, fitness, bounds, rg, evals, max_evals, params, x0 = best_x)
            else:        
                minimize_(archive, fitness, bounds, rg, evals, max_evals, opt_params) 

from fcmaes.mapelites import variation_,  iso_dd_
                
def run_map_elites_(archive, fitness, bounds, rg, evals, max_evals, opt_params = {}):    
    popsize = opt_params.get('popsize', 32)  
    use_sbx = opt_params.get('use_sbx', True)     
    dis_c = opt_params.get('dis_c', 20)   
    dis_m = opt_params.get('dis_m', 20)  
    iso_sigma = opt_params.get('iso_sigma', 0.01)
    line_sigma = opt_params.get('line_sigma', 0.2)
    select_n = archive.capacity
    while evals.value < max_evals:              
        if use_sbx:
            pop = archive.random_xs(select_n, popsize, rg)
            xs = variation_(pop, bounds.lb, bounds.ub, rg, dis_c, dis_m)
        else:
            x1 = archive.random_xs(select_n, popsize, rg)
            x2 = archive.random_xs(select_n, popsize, rg)
            xs = iso_dd_(x1, x2, bounds.lb, bounds.ub, rg, iso_sigma, line_sigma)    
        yds = [fitness(x) for x in xs]
        evals.value += popsize
        descs = np.array([yd[1] for yd in yds])
        niches = archive.index_of_niches(descs)
        for i in range(len(yds)):
            archive.set(niches[i], yds[i], xs[i])
        archive.argsort()   
        select_n = archive.get_occupied()  

def minimize_(archive, fitness, bounds, rg, evals, max_evals, opt_params, x0 = None): 
    if 'BITE_CPP' == opt_params.get('solver'):
        return run_bite_(archive, fitness, bounds, rg, evals, max_evals, opt_params, x0 = None)
    else:
        es = get_solver_(bounds, opt_params, rg, x0) 
        stall_criterion = opt_params.get('stall_criterion', 20)
        max_evals_iter = opt_params.get('max_evals', 50000)
        max_iters = int(max_evals_iter/es.popsize)
        old_ys = None
        last_improve = 0
        best_x = None
        best_y = np.inf
        for iter in range(max_iters):
            xs = es.ask()
            ys, real_ys = update_archive(archive, xs, fitness)
            evals.value += es.popsize
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
            if stop != 0 or evals.value >= max_evals:
                break 
            old_ys = np.sort(ys)
        return best_x # real best solution

from fcmaes import cmaes, cmaescpp, crfmnescpp, pgpecpp, decpp, crfmnes, de, bitecpp

def run_bite_(archive, fitness, bounds, rg, evals, max_evals, opt_params, x0 = None):  
    # BiteOpt doesn't support ask/tell, so we have to "patch" fitness. Note that Voronoi 
    # tesselation is more expensive if called for single behavior vectors and not for batches. 
    
    def fit(x: Callable[[ArrayLike], float]):
        if evals.value >= max_evals:
            return np.inf
        evals.value += 1
        ys, _ = update_archive(archive, [x], fitness)
        return ys[0]
    
    max_evals_iter = opt_params.get('max_evals', 50000)       
    stall_criterion = opt_params.get('stall_criterion', 20)   
    popsize = opt_params.get('popsize', 0) 
    ret = bitecpp.minimize(fit, bounds, x0 = x0, M = 1, 
                           stall_criterion = stall_criterion,
                           max_evaluations = max_evals_iter, rg = rg)
    return ret.x   

def get_solver_(bounds, opt_params, rg, x0 = None):
    dim = len(bounds.lb)
    popsize = opt_params.get('popsize', 31) 
    #sigma = opt_params.get('sigma',rg.uniform(0.03, 0.3)**2)
    sigma = opt_params.get('sigma',rg.uniform(0.1, 0.5)**2)
    #sigma = opt_params.get('sigma',rg.uniform(0.2, 0.5)**2)
    #sigma = opt_params.get('sigma',rg.uniform(0.1, 0.5))
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
            
