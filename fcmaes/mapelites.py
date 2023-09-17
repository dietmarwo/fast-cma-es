# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.
from __future__ import annotations

""" Numpy based implementation of CVT MAP-Elites including CMA-ES emitter and CMA-ES drilldown. 

See https://arxiv.org/abs/1610.05729 and https://arxiv.org/pdf/1912.02400.pdf

MAP-Elites implementations differ in the following details:

1) Initialisation of the behavior space:

a) Generated from some solution distribution by applying the fitness function to determine their behavior.
b) Generated from uniform samples of the behavior space. 

We use b) because random solutions may cover only parts of the behavior space. Some parts may only be reachable 
by optimization. Another reason: Fitness computations may be expensive. Therefore we don't compute fitness
values for the initial solution population.     

2) Initialization of the niches: 

a) Generated from some solution distribution.
b) Generated from uniform samples of the solution space. These solutions are never evaluated but serve as
initial population for SBX or Iso+LineDD. Their associated fitness value is set to math.inf (infinity).

We use b) because this way we: 
- Avoid computing fitness values for the initial population.
- Enhance the diversity of initial solutions emitted by SBX or Iso+LineDD.

3) Iso+LineDD (https://arxiv.org/pdf/1804.03906) is implemented but doesn't work well with extremely ragged solution
landscapes. Therefore SBX+mutation is the default setting.

4) SBX (Simulated binary crossover) is taken from mode.py and simplified. It is combined with mutation.
Both spread factors - for crossover and mutation - are randomized for each application. 

5) Candidates for CMA-ES are sampled with a bias to better niches. As for SBX only a subset of the archive is
used, the worst niches are ignored. 

6) There is a CMA-ES drill down for specific niches - in this mode all solutions outside the niche
are rejected. Restricted solution box bounds are used derived from statistics maintained by the archive
during the addition of new solution candidates. 

7) The QD-archive uses shared memory to reduce inter-process communication overhead.
"""

import numpy as np
from numpy.random import Generator, MT19937, SeedSequence
from multiprocessing import Process
import multiprocessing as mp
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
from scipy.optimize import Bounds
from pathlib import Path
from fcmaes.optimizer import dtime
from fcmaes import cmaescpp
from numpy.random import default_rng
import ctypes as ct
from time import perf_counter
import threadpoolctl
from numba import njit
from fcmaes.evaluator import is_debug_active
from loguru import logger

from typing import Optional, Callable, Tuple, Dict
from numpy.typing import ArrayLike

rng = default_rng()

def optimize_map_elites(qd_fitness: Callable[[ArrayLike], Tuple[float, np.ndarray]], 
                        bounds: Bounds, 
                        qd_bounds: Bounds, 
                        niche_num: Optional[int] = 4000, 
                        samples_per_niche: Optional[int] = 20, 
                        workers: Optional[int] = mp.cpu_count(), 
                        iterations: Optional[int] = 100, 
                        archive: Optional[Archive] = None, 
                        me_params: Optional[Dict] = {}, 
                        cma_params: Optional[Dict] = {}, 
                        use_stats: Optional[bool] = False,
                        ) -> Archive:
    
    """Application of CVT-Map Elites with additional CMA-ES emmitter.
     
    Parameters
    ----------
    qd_fitness : callable
        The objective function to be minimized.
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
    max_evaluations : int, optional
        Forced termination after ``max_evaluations`` function evaluations.
    workers : int, optional
        Number of spawned parallel worker processes.
    iterations : int, optional
        Number of MAP-elites iterations.
    archive : Archive, optional
        If defined MAP-elites is continued for this archive.
    me_params : dictionary, optional 
        Parameters for MAP-elites.
    cma_params : dictionary, optional 
        Parameters for the CMA-ES emitter.
    use_stats : bool, optional 
        If True, archive accumulates statistics of the solutions
        
    Returns
    -------
    archive : Archive
        Resulting archive of niches. Can be stored for later continuation of MAP-elites."""

    dim = len(bounds.lb) 
    if archive is None: 
        archive = Archive(dim, qd_bounds, niche_num, use_stats)
        archive.init_niches(samples_per_niche)
        # initialize archive with random values
        archive.set_xs(rng.uniform(bounds.lb, bounds.ub, (niche_num, dim))) 
    t0 = perf_counter() 
    qd_fitness.archive = archive # attach archive for logging  
    for iter in range(iterations):
        archive.argsort() # sort archive to select the best_n
        optimize_map_elites_(archive, qd_fitness, bounds, workers,
                    me_params, cma_params)
        if is_debug_active():
            ys = np.sort(archive.get_ys())[:100] # best 100 fitness values
            logger.debug(f'best 100 iter {iter} best {min(ys):.3f} worst {max(ys):.3f} ' + 
                     f'mean {np.mean(ys):.3f} stdev {np.std(ys):.3f} time {dtime(t0)} s')
    return archive

def empty_archive(dim: int, 
                  qd_bounds: Bounds, 
                  niche_num: int, 
                  samples_per_niche: int, 
                  use_stats: Optional[bool] = False) -> Archive:
    
    """Creates an empty archive.
     
    Parameters
    ----------
    archive: Archive
    qd_bounds : `Bounds`
        Bounds on behavior descriptors. Instance of the `scipy.Bounds` class.        
    niche_num : int, optional
        Number of niches.
    samples_per_niche : int, optional
        Number of samples used for niche computation.             
    use_stats : bool, optional 
        If True, archive accumulates statistics of the solutions
                 
    Returns
    -------
    archive : Archive
        Empty archive of niches."""

    archive = Archive(dim, qd_bounds, niche_num, use_stats)
    archive.init_niches(samples_per_niche)
    return archive

def set_KDTree(archive: Archive,
                        centers:Optional[np.ndarray] = None, 
                        niche_num: Optional[int]  = None, 
                        qd_bounds: Optional[Bounds] = None, 
                        samples_per_niche: Optional[int] = 100):   
    
    """Returns a function deciding niche membership.
     
    Parameters
    ----------
    archive: Archive
    centers : ndarray, shape (n,m), optional
        If defined, these behavior vectors are used as niche centers
    niche_num : int, optional
        Number of niches. Required if centers is None.
    qd_bounds : `Bounds`
        Bounds on behavior descriptors. Instance of the `scipy.Bounds` class.
        Required if centers is None.
    samples_per_niche : int, optional
        Number of samples used for niche computation. 
        If samples_per_niche > 0 cvt-clustering is used, else grid-clustering is used.        
         
    Returns
    -------
    index_of_niches : callable
        Maps an array of description vectors to their corresponding niche indices.
    centers : ndarray, shape (n,m)
        behavior vectors used as niche centers."""

    if centers is None: # cache centers 
        centers = get_centers_(niche_num, len(qd_bounds.lb), samples_per_niche)
    archive.kdt = KDTree(centers, leaf_size=30, metric='euclidean')  
    archive.centers = centers         

def load_archive(name: str, 
                 bounds: Bounds, 
                 qd_bounds: Bounds, 
                 niche_num: Optional[int] = 10000,
                 use_stats: Optional[bool] = False, 
                 ) -> Archive:
    
    """Loads an archive from disk.

    Parameters
    ----------
    name: string
        Name of the archive.
    bounds : `Bounds`
        Bounds on variables. Instance of the `scipy.Bounds` class.
    qd_bounds : `Bounds`
        Bounds on behavior descriptors. Instance of the `scipy.Bounds` class.        
    niche_num : int, optional
        Number of niches.
    use_stats : bool, optional 
        If True, archive accumulates statistics of the solutions
        
    Returns
    -------
    archive : Archive
        Archive of niches. Can be used for continuation of MAP-elites."""
     
    dim = len(bounds.lb)
    archive = Archive(dim, qd_bounds, niche_num, name, use_stats)
    archive.load(name)
    return archive

def optimize_map_elites_(archive, fitness, bounds, workers,
                         me_params, cma_params):
    sg = SeedSequence()
    rgs = [Generator(MT19937(s)) for s in sg.spawn(workers)]
    proc=[Process(target=run_map_elites_,
            args=(archive, fitness, bounds, rgs[p],
                  me_params, cma_params)) for p in range(workers)]
    [p.start() for p in proc]
    [p.join() for p in proc]
          
def run_map_elites_(archive, fitness, bounds, rg, 
                    me_params, cma_params): 
    
    generations = me_params.get('generations', 10) 
    chunk_size = me_params.get('chunk_size', 20)   
    use_sbx = me_params.get('use_sbx', True)     
    dis_c = me_params.get('dis_c', 20)   
    dis_m = me_params.get('dis_m', 20)  
    iso_sigma = me_params.get('iso_sigma', 0.02)
    line_sigma = me_params.get('line_sigma', 0.2)
    cma_generations = cma_params.get('cma_generations', 20)
    select_n = archive.capacity
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        for _ in range(generations):                
            if use_sbx:
                pop = archive.random_xs(select_n, chunk_size, rg)
                xs = variation_(pop, bounds.lb, bounds.ub, rg, dis_c, dis_m)
            else:
                x1 = archive.random_xs(select_n, chunk_size, rg)
                x2 = archive.random_xs(select_n, chunk_size, rg)
                xs = iso_dd_(x1, x2, bounds.lb, bounds.ub, rg, iso_sigma, line_sigma)    
            yds = [fitness(x) for x in xs]
            descs = np.array([yd[1] for yd in yds])
            niches = archive.index_of_niches(descs)
            for i in range(len(yds)):
                archive.set(niches[i], yds[i], xs[i]) 
            archive.argsort()   
            select_n = archive.get_occupied()            
    
        for _ in range(cma_generations):                
            optimize_cma_(archive, fitness, bounds, rg, cma_params)    

def optimize_cma_(archive, fitness, bounds, rg, cma_params):
    select_n = cma_params.get('best_n', 100)
    x0, y, iter = archive.random_xs_one(select_n, rg)
    sigma = cma_params.get('sigma',rg.uniform(0.03, 0.3)**2)
    popsize = cma_params.get('popsize', 31) 
    es = cmaescpp.ACMA_C(archive.dim, bounds, x0 = x0,  
                         popsize = popsize, input_sigma = sigma, rg = rg)
    maxiters = cma_params.get('maxiters', 100)
    stall_criterion = cma_params.get('stall_criterion', 5)
    old_ys = None
    last_improve = 0
    for iter in range(maxiters):
        xs = es.ask()
        improvement, ys = update_archive(archive, xs, fitness)
        if iter > 0:
            if (np.sort(ys) < old_ys).any():
                last_improve = iter          
        if last_improve + stall_criterion < iter:
            # no improvement
            break
        if es.tell(improvement) != 0:
            break 
        old_ys = np.sort(ys)
        
def update_archive(archive: Archive, xs: np.ndarray, 
                   fitness: Optional[Callable[[ArrayLike], Tuple[float, np.ndarray]]] = None,
                   yds: Optional[ArrayLike] = None):
    # evaluate population, update archive and determine ranking
    popsize = len(xs)
    if yds is None: 
        yds = [fitness(x) for x in xs]
    descs = np.array([yd[1] for yd in yds])
    niches = archive.index_of_niches(descs)
    # real values
    ys = np.fromiter((yd[0] for yd in yds), dtype=float)
    oldys = np.fromiter((archive.get_y(niches[i]) for i in range(popsize)), dtype=float)
    improvement = ys - oldys
    neg = np.argwhere(improvement < 0)
    if len(neg) > 0:
        neg = neg.reshape((len(neg)))
        # update archive for all real improvements
        for i in neg:
            archive.set(niches[i], yds[i], xs[i])
        # prioritize empty niches
        empty = (improvement == -np.inf) # these need to be sorted according to fitness
        occupied = np.logical_not(empty)
        min_valid = np.amin(improvement[occupied]) if sum(occupied) > 0 else 0
        norm_ys = ys[empty] - np.amax(ys) - 1E-9
        improvement[empty] = min_valid + norm_ys
    # return both improvement compared to archive elites  and real fitness
    return improvement, ys

@njit()
def get_grid_indices(ds, capacity, lb, ub):
    rdim = int(capacity ** (1/ds.shape[1]) + 0.5)
    ds_norm = (ds - lb) / (ub - lb)
    indices = np.empty(len(ds), dtype=np.int32)   
    for i, d in enumerate(ds_norm):
        index = 0
        f = 1
        for di in d:
            index += f * int(rdim*di)
            f *= rdim
        indices[i] = max(0, min(capacity-1, int(index)))    
    return indices
        
class Archive(object):
    """Multi-processing map elites archive. 
        Stores decision vectors, fitness values , 
        description vectors, niche centers and statistics for the x-values"""
       
    def __init__(self, 
                 dim: int,
                 qd_bounds: Bounds,
                 capacity: int,    
                 name: Optional[str] = "",
                 use_stats = False             
                ):    
        """Creates an empty archive."""
        self.dim = dim
        self.qd_dim = len(qd_bounds.lb)
        self.qd_bounds = Bounds(np.array(qd_bounds.lb), np.array(qd_bounds.ub))
        self.desc_lb = self.qd_bounds.lb
        self.desc_scale = self.qd_bounds.ub - self.qd_bounds.lb
        self.capacity = capacity
        self.name = name
        self.cs = None
        self.lock = mp.Lock()
        self.use_stats = use_stats
        self.reset()
    
    def reset(self):
        """Resets all submitted solutions but keeps the niche centers."""
        self.xs = mp.RawArray(ct.c_double, self.capacity * self.dim)
        self.ds = mp.RawArray(ct.c_double, self.capacity * self.qd_dim)
        self.ys = mp.RawArray(ct.c_double, self.capacity)
        self.counts = mp.RawArray(ct.c_long, self.capacity) # count
        self.occupied = mp.RawValue(ct.c_long, 0)
        self.stats = mp.RawArray(ct.c_double, self.capacity * self.dim * 4 if self.use_stats else 0)
        for i in range(self.capacity):
            self.counts[i] = 0
            self.set_y(i, np.inf)  
            self.set_d(i, np.full(self.qd_dim, np.inf))
            if self.stats:
                self.set_stat(i, 0, np.zeros(self.dim)) # mean
                self.set_stat(i, 1, np.zeros(self.dim)) # qmean
                self.set_stat(i, 2, np.full(self.dim, np.inf)) # min
                self.set_stat(i, 3, np.full(self.dim, -np.inf)) # max
         
    def init_niches(self, samples_per_niche: int = 10): 
        """Computes the niche centers using KMeans and builds the KDTree for niche determination."""
        # If samples_per_niche > 0 cvt-clustering is used, else grid-clustering is used.   
        self.cvt_clustering = samples_per_niche > 0
        if self.cvt_clustering:
            set_KDTree(self, None, self.capacity, self.qd_bounds, samples_per_niche)
            self.cs = mp.RawArray(ct.c_double, self.capacity * self.qd_dim)
            self.set_cs(self.centers)
    
    def get_occupied_data(self):
        ys = self.get_ys()
        occupied = (ys < np.inf)
        return ys[occupied], self.get_ds()[occupied], self.get_xs()[occupied]        
   
    def join(self, archive: Archive):    
        ys, ds, xs = archive.get_occupied_data()
        niches = archive.index_of_niches(ds)
        yds = np.array([(y, d) for y, d in zip(ys, ds)])
        for i in range(len(ys)):
            archive.set(niches[i], yds[i], xs[i]) 
        archive.argsort()   

    def fname(self, name): 
        """Archive file name."""
        return f'arch.{name}.{self.capacity}.{self.dim}.{self.qd_dim}'
           
    def save(self, name: str):
        """Saves the archive to disc.""" 
        np.savez_compressed(self.fname(name), 
                            xs=self.get_xs(), 
                            ds=self.get_ds(), 
                            ys=self.get_ys(), 
                            cs=self.get_cs() if self.cvt_clustering else np.empty(0),
                            stats=self.get_stats(),
                            counts=self.get_counts()
                            )

    def load(self, name: str):
        """Loads the archive from disc."""   
        self.cs = mp.RawArray(ct.c_double, self.capacity * self.qd_dim)
        with np.load(self.fname(name) + '.npz') as data:
            self.cvt_clustering = len(data['cs']) > 0
            xs = data['xs']
            ds = data['ds']
            self.set_xs(xs)
            self.set_ds(ds)
            self.set_ys(data['ys'])
            if self.cvt_clustering:
                self.set_cs(data['cs'])
            self.counts[:] = data['counts']
            stats = data['stats']
            if len(stats) == len(self.stats):
                self.set_stats(stats)
        self.occupied.value = np.count_nonzero(self.get_ys() < np.inf)
        self.dim = xs.shape[1]
        self.qd_dim = ds.shape[1]
        self.capacity = xs.shape[0]
        if self.cvt_clustering:
            set_KDTree(self, self.get_cs(), None, None, None)
      
    def index_of_niches(self, ds):
        if hasattr(self, "kdt"): # use k-means clusters
            return self.kdt.query(self.encode_d(ds), k=1, sort_results=False)[1].T[0] 
        else: # use grid based clustering
            return get_grid_indices(ds, self.capacity, self.qd_bounds.lb, self.qd_bounds.ub)
        
    def in_niche_filter(self, 
                        fit: Callable[[ArrayLike], float], 
                        index: int):
        """Creates a fitness function wrapper rejecting out of niche arguments."""
        return in_niche_filter(fit, index, self.index_of_niches)
                                                               
    def set(self, 
            i: int, 
            yd: np.ndarray, 
            x: np.ndarray):
        """Adds a solution to the archive if it improves the corresponding niche.
        Updates solution.""" 
        self.update_stats(i, x)
        y, d = yd
        # register improvement
        yold = self.get_y(i)
        if y < yold:
            if yold == np.inf: # not yet occupied
                self.occupied.value += 1
            self.set_y(i, y)
            self.set_x(i, x)
            self.set_d(i, d)
    
    def update_stats(self, 
                     i: int, 
                     x: np.ndarray):
        """Updates solution statistics."""
        with self.lock:
            self.counts[i] += 1
        count = self.counts[i]
        if self.use_stats:  
            mean = self.get_x_mean(i)
            diff = x - mean      
            self.set_stat(i, 0, mean + diff * (1./count)) # mean
            self.set_stat(i, 1, self.get_stat(i, 1) + np.multiply(diff,diff) * ((count-1)/count)) # qmean                  
            self.set_stat(i, 2, np.minimum(x, self.get_stat(i, 2))) # min
            self.set_stat(i, 3, np.maximum(x, self.get_stat(i, 3))) # max
 
    def get_occupied(self) -> int:
        return self.occupied.value
    
    def get_count(self, i: int) -> int:
        return self.counts[i]

    def get_counts(self) -> np.ndarray:
        return np.array(self.counts[:])
 
    def get_x_mean(self, i: int) -> np.ndarray:
        return self.get_stat(i, 0)

    def get_x_stdev(self, i: int) -> np.ndarray:
        count = self.get_count(i)
        if count == 0:
            return np.zeros(self.dim)
        else:
            qmean = np.array(self.get_stat(i, 1))
            return np.sqrt(qmean * (1./count))

    def get_x_min(self, i: int) -> np.ndarray:
        return self.get_stat(i, 2)

    def get_x_max(self, i: int) -> np.ndarray:
        return self.get_stat(i, 3)
           
    def get_x(self, i: int) -> np.ndarray:
        return self.xs[i*self.dim:(i+1)*self.dim]

    def get_xs(self) -> np.ndarray:
        return np.array([self.get_x(i) for i in range(self.capacity)])

    def set_x(self, i: int, x: ArrayLike):
        self.xs[i*self.dim:(i+1)*self.dim] = x[:]
    
    def set_xs(self, xs: ArrayLike):
        for i in range(len(xs)):
            self.set_x(i, xs[i])
            
    def encode_d(self, d):
        return (d - self.desc_lb) / self.desc_scale

    def decode_d(self, d):
        return (d * self.desc_scale) + self.desc_lb 

    def get_d(self, i: int) -> float:
        return self.ds[i*self.qd_dim:(i+1)*self.qd_dim]

    def get_ds(self) -> np.ndarray:
        return np.array([self.get_d(i) for i in range(self.capacity)])
    
    def set_d(self, i: int, d: float):
        self.ds[i*self.qd_dim:(i+1)*self.qd_dim] = d[:]
 
    def set_ds(self, ds: ArrayLike):
        for i in range(len(ds)):
            self.set_d(i, ds[i])
  
    def get_y(self, i: int) -> float:
        return self.ys[i]
        
    def get_ys(self) -> np.ndarray:
        return np.array(self.ys[:])
    
    def get_qd_score(self) -> float:
        ys = self.get_ys()
        occupied = (ys != np.inf)
        ys = ys[occupied]
        if len(ys) == 0:
            return 0
        min_y = np.amin(ys)
        if min_y > 0: # if all y > 0 use sum of reciprocal
            return np.sum(np.reciprocal(ys, where = ys!=0)) 
        else: # else use only the negative ones
            neg = (ys < 0)
            ys = ys[neg]
            return np.sum(-ys) 
         
    def set_y(self, i: int, y: float):
        self.ys[i] = y    

    def set_ys(self, ys: ArrayLike):
        for i in range(len(ys)):
            self.set_y(i, ys[i])
            
    def get_c(self, i: int) -> float:
        return self.cs[i*self.qd_dim:(i+1)*self.qd_dim]
        
    def get_cs(self) -> np.ndarray:
        return np.array([self.get_c(i) for i in range(self.capacity)])

    def get_cs_decoded(self) -> np.ndarray:
        return self.decode_d(np.array([self.get_c(i) for i in range(self.capacity)]))
           
    def set_c(self, i: int, c: float):
        self.cs[i*self.qd_dim:(i+1)*self.qd_dim] = c[:]

    def set_cs(self, cs: ArrayLike):
        for i in range(len(cs)):
            self.set_c(i, cs[i])

    def get_stat(self, i: int, j: int) -> float:
        p = 4*i+j
        return self.stats[p*self.dim:(p+1)*self.dim]

    def get_stats(self) ->  np.ndarray:
        return np.array(self.stats[:])
           
    def set_stat(self, i: int, j: int, stat: ArrayLike):
        p = 4*i+j
        self.stats[p*self.dim:(p+1)*self.dim] = stat[:]

    def set_stats(self, stats: ArrayLike):
        self.stats[:] = stats[:]

    def random_xs(self, best_n: int, chunk_size: int, rg: Generator) -> np.ndarray:
        selection = rg.integers(0, best_n, chunk_size)
        if best_n < self.capacity: 
            selection = np.fromiter((self.si[i] for i in selection), dtype=int)
        return self.get_xs()[selection]
    
    def random_xs_one(self, best_n: int, rg: Generator) -> Tuple[np.ndarray, float, int]:
        i = int(rg.random()*best_n)
        return self.get_x(i), self.get_y(i),  i
        
    def argsort(self) -> np.ndarray:
        """Sorts the archive according to its niche values.""" 
        self.si = np.argsort(self.get_ys())
        return self.si
            
    def dump(self, n: Optional[int] = None):
        """Dumps the archive content.""" 
        if n is None:
            n = self.capacity
        ys = self.get_ys()
        si = np.argsort(ys)
        for i in range(n):
            print(si[i], ys[si[i]], self.get_d(si[i]), self.get_x(si[i]))    
    
    def info(self) -> str:        
        occ = self.get_occupied()
        score = self.get_qd_score()
        best_y = np.amin(self.get_ys())
        count = np.sum(self.get_counts())
        return f'{occ} {score:.3f} {best_y:.3f} {count}'

class wrapper(object):
    """Fitness function wrapper for multi processing logging."""

    def __init__(self, 
                 fit:Callable[[ArrayLike], Tuple[float, np.ndarray]], 
                 qd_dim: int, 
                 interval: Optional[int] = 1000000,
                 save_interval: Optional[int] = 1E20):
        
        self.fit = fit
        self.evals = mp.RawValue(ct.c_int, 0) 
        self.best_y = mp.RawValue(ct.c_double, np.inf) 
        self.t0 = perf_counter()
        self.qd_dim = qd_dim
        self.interval = interval
        self.save_interval = save_interval
        self.lock = mp.Lock()
        
    def __call__(self, x: ArrayLike):
        try:
            if np.isnan(x).any():
                return np.inf, np.zeros(self.qd_dim)
            with self.lock:
                self.evals.value += 1
            log = self.evals.value % self.interval == 0
            save = self.evals.value % self.save_interval == 0
            y, desc = self.fit(x)
            if np.isnan(y) or np.isnan(desc).any():
                return np.inf, np.zeros(self.qd_dim)
            y0 = y if np.isscalar(y) else sum(y)
            if y0 < self.best_y.value:
                self.best_y.value = y0
                log = True 
            if log:
                archinfo = self.archive.info() if hasattr(self, 'archive') else ''
                logger.info(
                    f'{dtime(self.t0)} {archinfo} {self.evals.value:.0f} {self.evals.value/(1E-9 + dtime(self.t0)):.0f} {self.best_y.value:.3f} {list(x)}')            
            if save and hasattr(self, 'archive'):
                self.archive.save(f'{self.evals.value}')
            return y, desc
        except Exception as ex:
            print(str(ex))  
            return np.inf, np.zeros(self.qd_dim)
        
class in_niche_filter(object):
    """Fitness function wrapper rejecting out of niche arguments."""
    
    def __init__(self, 
                 fit:Callable[[ArrayLike], Tuple[float, np.ndarray]], 
                 index: int, 
                 index_of_niches: Callable[[ArrayLike], np.ndarray]):
        self.fit = fit
        self.index_of_niches = index_of_niches
        self.index = index

    def __call__(self, x: ArrayLike) -> float:
        y, desc = self.fit(x)
        if self.index_of_niches([desc])[0] == self.index:
            return y
        else:
            return np.inf

def variation_(pop, lower, upper, rg, dis_c = 20, dis_m = 20):
    """Generate offspring individuals using SBX (Simulated Binary Crossover) and mutation."""
    dis_c *= 0.5 + 0.5*rg.random() # vary spread factors randomly 
    dis_m *= 0.5 + 0.5*rg.random() 
    pop = pop[:(len(pop) // 2) * 2][:]
    (n, d) = np.shape(pop)
    parent_1 = pop[:n // 2, :]
    parent_2 = pop[n // 2:, :]
    beta = np.zeros((n // 2, d))
    mu = rg.random((n // 2, d))
    beta[mu <= 0.5] = np.power(2 * mu[mu <= 0.5], 1 / (dis_c + 1))
    beta[mu > 0.5] = np.power(2 * mu[mu > 0.5], -1 / (dis_c + 1))
    beta = beta * ((-1)** rg.integers(2, size=(n // 2, d)))
    beta[rg.random((n // 2, d)) < 0.5] = 1
    parent_mean = (parent_1 + parent_2) / 2
    parent_diff = (parent_1 - parent_2) / 2
    offspring = np.vstack((parent_mean + beta * parent_diff, parent_mean - beta * parent_diff))
    site = rg.random((n, d)) < 1.0 / d
    mu = rg.random((n, d))
    temp = site & (mu <= 0.5)
    lower, upper = np.tile(lower, (n, 1)), np.tile(upper, (n, 1))
    norm = (offspring[temp] - lower[temp]) / (upper[temp] - lower[temp])
    offspring[temp] += (upper[temp] - lower[temp]) * \
                           (np.power(2. * mu[temp] + (1. - 2. * mu[temp]) * np.power(np.abs(1. - norm), dis_m + 1.),
                                     1. / (dis_m + 1)) - 1.)
    temp = site & (mu > 0.5)
    norm = (upper[temp] - offspring[temp]) / (upper[temp] - lower[temp])
    offspring[temp] += (upper[temp] - lower[temp]) * \
                           (1. - np.power(
                               2. * (1. - mu[temp]) + 2. * (mu[temp] - 0.5) * np.power(np.abs(1. - norm), dis_m + 1.),
                               1. / (dis_m + 1.)))
    return np.clip(offspring, lower, upper)

def iso_dd_(x1, x2, lower, upper, rg, iso_sigma = 0.01, line_sigma = 0.2):
    """Generate offspring individuals using Iso+Line."""
    a = rg.normal(0, iso_sigma, x1.shape) 
    b = rg.normal(0, line_sigma, x2.shape) 
    z = x1 + a + np.multiply(b, (x1 - x2))
    return np.clip(z, lower, upper)

def get_centers_(niche_num, dim, samples_per_niche):
        p = Path('voronoi_cache')
        p.mkdir(exist_ok=True)
        fname = f'centers_{niche_num}_{dim}_{samples_per_niche}.npz'
        files = p.glob(fname)
        for file in files: # if cached just load
            with np.load(file) as data:
                return data['cs']
        else:
            descs = rng.uniform(0, 1, (niche_num*samples_per_niche, dim))
            # Applies KMeans to the random samples determine the centers of each niche."""
            k_means = KMeans(init='k-means++', n_clusters=niche_num, n_init=1, verbose=1)
            k_means.fit(descs)
            centers = k_means.cluster_centers_
            np.savez_compressed(f'voronoi_cache/centers_{niche_num}_{dim}_{samples_per_niche}', cs=centers)
            return centers
