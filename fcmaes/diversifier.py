# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

""" Numpy based implementation of an diversifying wrapper / parallel retry mechanism. 

Uses the archive from CVT MAP-Elites (https://arxiv.org/abs/1610.05729)
and generalizes ideas from CMA-ME (https://arxiv.org/pdf/1912.02400.pdf)
to other wrapped algorithms. 

Both the parallel retry and the archive based modification of the fitness 
function enhance the diversification of the optimization result.
The resulting archive may be stored and can be used to continue the
optimization later.   

Requires a ND-fitness function returning both an fitness value and a
behavior vector used to determine the corresponding archive niche using
Voronoi tesselation. 

Returns an archive of niche-elites containg also for each niche statistics 
about the associated solutions.     
"""

import numpy as np
from numpy.random import Generator, MT19937, SeedSequence
from multiprocessing import Process
from fcmaes.optimizer import logger, dtime
import multiprocessing as mp
import ctypes as ct
from time import perf_counter
from fcmaes.mapelites import Archive
import threadpoolctl

def minimize(fitness, bounds, 
            desc_bounds, 
            niche_num = 4000, 
            samples_per_niche = 10, 
            retries = None, 
            workers = mp.cpu_count(), 
            archive = None, 
            opt_params = {}, 
            logger = logger()):
    
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
    fitness : callable
        The objective function to be minimized. Returns a fitness value and a behavior vector. 
            ``fitness(x) -> float, ndarray``
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
    opt_params : dictionary, optional 
        Parameters selecting and configuring the wrapped solver.
        'solver' - supported are 'CMA','CMA_CPP','CRMFNES','CRMFNES_CPP','DE','DE_CPP','PGPE'
                    default is 'CMA_CPP'
        'popsize' - population size, default = 32
        'sigma' -  initial distribution sigma, default = rg.uniform(0.03, 0.3)**2)
        'mean' - initial distribution mean, default=rg.uniform(bounds.lb, bounds.ub)) 
        'max_evals' - maximal number of evaluations per run, default = 50000
        'stall_criterion' - how many iterations without progress allowed, default = 50 iterations      
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
    desc_dim = len(desc_bounds.lb) 
    if archive is None: 
        archive = Archive(dim, desc_dim, niche_num)
        archive.init_niches(desc_bounds, samples_per_niche)       
    t0 = perf_counter()   
    fitness.archive = archive
    count = mp.RawValue(ct.c_long, 0)
    #archive.argsort() # sort archive to select the best_n
    minimize_parallel_(archive, fitness, bounds, workers, 
                       opt_params, count, retries)
    if not logger is None:
        ys = np.sort(archive.get_ys())[:100] # best 100 fitness values
        logger.info(f'best 100, best {min(ys):.3f} worst {max(ys):.3f} ' + 
                 f'mean {np.mean(ys):.3f} stdev {np.std(ys):.3f} time {dtime(t0)} s')
    return archive

def minimize_parallel_(archive, fitness, bounds, workers, 
                         opt_params, count, retries):
    sg = SeedSequence()
    rgs = [Generator(MT19937(s)) for s in sg.spawn(workers)]
    proc=[Process(target=run_minimize_,
            args=(archive, fitness, bounds, rgs[p],
                  opt_params, count, retries)) for p in range(workers)]
    [p.start() for p in proc]
    [p.join() for p in proc]
                    
def run_minimize_(archive, fitness, bounds, rg, opt_params, count, retries): 
    
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        while count.value < retries:
            count.value += 1               
            minimize_(archive, fitness, bounds, rg, opt_params)    

def minimize_(archive, fitness, bounds, rg, opt_params):  
    es = get_solver(bounds, opt_params, rg)  
    max_evals = opt_params.get('max_evals', 50000)
    stall_criterion = opt_params.get('stall_criterion', 50)
    old_ys = None
    last_improve = 0
    max_iters = int(max_evals/es.popsize)
    for iter in range(max_iters):
        xs = es.ask()
        ys = update_archive_(archive, xs, fitness)
        if not old_ys is None:
            if (np.sort(ys) < old_ys).any():
                last_improve = iter          
        if last_improve + stall_criterion < iter:
            break
        if es.tell(ys) != 0:
            break 
        old_ys = np.sort(ys)

def update_archive_(archive, xs, fitness):
    # evaluate population, update archive and determine ranking for cma-es
    popsize = len(xs) 
    yds = [fitness(x) for x in xs]
    
    # descs = np.array([yd[1] for yd in yds])
    # niches = archive.index_of_niches(descs)
    # ys = np.array(np.fromiter((yd[0] for yd in yds), dtype=float))
    # for i in range(popsize):
    #     archive.set(niches[i], yds[i], xs[i])   
    # return ys

    descs = np.array([yd[1] for yd in yds])
    niches = archive.index_of_niches(descs)
    ys = np.array(np.fromiter((yd[0] for yd in yds), dtype=float))
    oldys = np.array(np.fromiter((archive.get_y(niches[i]) for i in range(popsize)), dtype=float))
    is_inf = (oldys == np.inf)
    oldys[is_inf] = np.amax(ys)+1
    
    diff = ys - oldys
    neg = np.argwhere(diff < 0)
    if len(neg) > 0:
        neg = neg.reshape((len(neg)))
        for i in neg:
            archive.set(niches[i], yds[i], xs[i])
    return diff

from fcmaes import cmaes, cmaescpp, crfmnescpp, pgpecpp, decpp, crfmnes, de

def get_solver(bounds, opt_params, rg):
    dim = len(bounds.lb)
    popsize = opt_params.get('popsize', 32) 
    sigma = opt_params.get('sigma',rg.uniform(0.03, 0.3)**2)
    mean = opt_params.get('mean', rg.uniform(bounds.lb, bounds.ub))
    name = opt_params.get('solver', 'CMA_CPP')
    if name == 'CMA':
        return cmaes.Cmaes(bounds, x0 = mean,
                          popsize = popsize, input_sigma = sigma, rg = rg)
    elif name == 'CMA_CPP':
        return cmaescpp.ACMA_C(dim, bounds, x0 = mean,
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
    
class wrapper(object):
    """Fitness function wrapper for multi processing logging."""

    def __init__(self, fit, desc_dim, logger=logger()):
        self.fit = fit
        self.evals = mp.RawValue(ct.c_int, 0) 
        self.best_y = mp.RawValue(ct.c_double, np.inf) 
        self.t0 = perf_counter()
        self.desc_dim = desc_dim
        self.logger = logger

    def __call__(self, x):
        try:
            if np.isnan(x).any():
                return np.inf, np.zeros(self.desc_dim)
            self.evals.value += 1
            y, desc = self.fit(x)
            if np.isnan(y) or np.isnan(desc).any():
                return np.inf, np.zeros(self.desc_dim)
            y0 = y if np.isscalar(y) else sum(y)
            if y0 < self.best_y.value:
                self.best_y.value = y0
                if not self.logger is None:
                    occ = self.archive.get_occupied() if hasattr(self, 'archive') else 0
                    self.logger.info(
                        f'{dtime(self.t0)}  {occ} {self.evals.value:.3f} {self.evals.value/(1E-9 + dtime(self.t0)):.3f} {self.best_y.value} {list(x)}')
            return y, desc
        except Exception as ex:
            print(str(ex))  
            return np.inf, np.zeros(self.desc_dim)
        
