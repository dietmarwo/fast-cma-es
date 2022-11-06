# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

""" Implementation of CVT MAP-Elites including CMA emitter
"""

import numpy as np
from numpy.random import Generator, MT19937, SeedSequence
from multiprocessing import Process
import multiprocessing as mp
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
import math
from pathlib import Path
from fcmaes.optimizer import dtime, logger
from fcmaes import cmaescpp
from numpy.random import default_rng
import ctypes as ct
from time import perf_counter

rng = default_rng()

def optimize_map_elites(fitness, bounds, desc_bounds, 
                niche_num = 4000, samples_per_niche = 100, workers = 24, 
                iterations = 100, min_capacity = 0.2, capacity_reduce = 0.9, 
                archive = None, me_params = {}, cma_params = {}, logger = logger()):
    dim = len(bounds.lb)
    desc_dim = len(desc_bounds.lb) 
    if archive is None: 
        archive = Archive(dim, desc_dim, niche_num)
        archive.init_niches(desc_bounds, samples_per_niche)
        # initialize archive with random values
        archive.set_xs(rng.uniform(bounds.lb, bounds.ub, (niche_num, dim)))        
    t0 = perf_counter()      
    for iter in range(iterations):
        archive.argsort() # sort archive to select the best_n
        optimize_map_elites_(archive, fitness, bounds, workers, 
                     best_n, chunk_size, generations, use_sbx, dis_c, dis_m, cma_params, iter)
        if not logger is None:
            ys = np.sort(archive.get_ys())[:100] # best 100 fitness values
            log.info(f'best 100 level{i} num {best_n} best {min(ys):.3f} worst {max(ys):.3f} ' + 
                     f'mean {np.mean(ys):.3f} stdev {np.std(ys):.3f} time {dtime(t0)} s')
        if best_n * capacity_reduce > archive.capacity * min_capacity :
            best_n = int(capacity_reduce*best_n)
    return archive

def get_index_of_niches(centers = None, niche_num = None, desc_bounds = None, samples_per_niche = 100):   
    """Returns a function deciding niche membership."""
    if centers is None:
        dim = len(desc_bounds.lb)
        descs = rng.uniform(desc_bounds.lb, desc_bounds.ub, (niche_num*samples_per_niche,dim))
        # Applies KMeans to the random samples determine the centers of each niche."""
        k_means = KMeans(init='k-means++', n_clusters=niche_num, n_init=1, verbose=1)
        k_means.fit(descs)
        centers = k_means.cluster_centers_
    kdt = KDTree(centers, leaf_size=30, metric='euclidean')     
    # Uses the KDtree to determine the niche indexes.
    def index_of_niches(ds):
        return kdt.query(ds, k=1, sort_results=False)[1].T[0] 
         
    return index_of_niches, centers

def load_archive(name, bounds, desc_bounds, niche_num):
    dim = len(bounds.lb)
    desc_dim = len(desc_bounds.lb) 
    archive = Archive(dim, desc_dim, niche_num)
    archive.load(name)
    return archive

def optimize_map_elites_(archive, fitness, bounds, workers, me_params, cma_params, iter):
    sg = SeedSequence()
    rgs = [Generator(MT19937(s)) for s in sg.spawn(workers)]
    proc=[Process(target=run_map_elites_,
            args=(archive, fitness, bounds, rgs[p], 
                  me_params, cma_params, iter)) for p in range(workers)]
    [p.start() for p in proc]
    [p.join() for p in proc]
          
def run_map_elites_(archive, fitness, bounds, rg, me_params, cma_params, iter): 
    
    generations = me_params.get('generations', 10) 
    best_n = me_params.get('best_n', archive.capacity)   
    chunk_size = me_params.get('chunk_size', 20)   
    use_sbx = me_params.get('use_sbx', True)     
    dis_c = me_params.get('dis_c', 20)   
    dis_m = me_params.get('dis_m', 20)  
    iso_sigma = me_params.get('iso_sigma', 0.01)
    line_sigma = me_params.get('line_sigma', 0.2)
    cma_generations = cma_params.get('cma_generations', 20)
       
    for _ in range(generations):                
        if use_sbx:
            pop = archive.random_xs(best_n, chunk_size, rg)
            xs = variation(pop, bounds.lb, bounds.ub, rg, dis_c, dis_m)
        else:
            x1 = archive.random_xs(best_n, chunk_size, rg)
            x2 = archive.random_xs(best_n, chunk_size, rg)
            xs = iso_dd(x1, x2, bounds.lb, bounds.ub, rg, iso_sigma, line_sigma)    
        yds = [fitness(x) for x in xs]
        descs = np.array([yd[1] for yd in yds])
        niches = archive.index_of_niches(descs)
        for i in range(len(yds)):
            archive.set(niches[i], yds[i], xs[i])    
    if iter > 0:    
        for _ in range(cma_generations):                
            optimize_cma_(archive, fitness, bounds, rg, cma_params)    

def optimize_cma_(archive, fitness, bounds, rg, cma_params):
    x0, y, i = archive.random_xs_one(cma_params.get('best_n', 100), rg)
    sigma = cma_params.get('sigma',rg.uniform(0.03, 0.3)**2)
    popsize = cma_params.get('popsize', 31) 
    es = cmaescpp.ACMA_C(archive.dim, bounds, x0 = x0,  
                         popsize = popsize, input_sigma = sigma, rg = rg)
    maxiters = cma_params.get('maxiters', 100)
    miniters = cma_params.get('miniters', 20)
    old_ys = None
    for i in range(maxiters):
        xs = es.ask()
        ys, stop = update_archive_cma_(archive, xs, fitness)
        if old_ys is None or (np.sort(ys) < np.sort(old_ys)).any(): # any improvement?
            stop = False
        if stop and i > miniters:
            break
        if es.tell(ys) != 0:
            break 
        old_ys = ys

def update_archive_cma_(archive, xs, fitness):
    # evaluate population, update archive and determine ranking for cma-es
    popsize = len(xs)
    yds = [fitness(x) for x in xs]
    descs = np.array([yd[1] for yd in yds])
    niches = archive.index_of_niches(descs)
    ys = np.array(np.fromiter((yd[0] for yd in yds), dtype=float))
    oldys = np.array(np.fromiter((archive.get_y(niches[i]) for i in range(popsize)), dtype=float))
    diff = ys - oldys
    neg = np.argwhere(diff < 0)
    if len(neg) > 0:
        neg = neg.reshape((len(neg)))
        for i in neg:
            archive.set(niches[i], yds[i], xs[i])
        return diff, False
    else: 
        return diff, True
        
class Archive(object):
    """Multi-processing map elites archive. 
        Stores decision vectors, fitness values , 
        description vectors, niche centers and statistics for the x-values"""
       
    def __init__(self, 
                 dim,
                 desc_dim,
                 capacity,                 
                ):    
        self.dim = dim
        self.desc_dim = desc_dim
        self.capacity = capacity
        self.cs = None
        self.index_of_niches = None
        self.reset()
    
    def reset(self):
        self.xs = mp.RawArray(ct.c_double, self.capacity * self.dim)
        self.ds = mp.RawArray(ct.c_double, self.capacity * self.desc_dim)
        self.ys = mp.RawArray(ct.c_double, self.capacity)
        self.counts = mp.RawArray(ct.c_long, self.capacity) # count
        self.stats = mp.RawArray(ct.c_double, self.capacity * self.dim * 4)
        
        for i in range(self.capacity):
            self.counts[i] = 0
            self.set_y(i, np.inf)  
            self.set_d(i, np.full(self.desc_dim, np.inf))
            self.set_stat(i, 0, np.zeros(self.dim)) # mean
            self.set_stat(i, 1, np.zeros(self.dim)) # qmean
            self.set_stat(i, 2, np.full(self.dim, np.inf)) # min
            self.set_stat(i, 3, np.full(self.dim, -np.inf)) # max
         
    def init_niches(self, desc_bounds, samples_per_niche = 10):  
        self.index_of_niches, centers = get_index_of_niches(None, self.capacity, desc_bounds, samples_per_niche)
        self.cs = mp.RawArray(ct.c_double, self.capacity * self.desc_dim)
        self.set_cs(centers)
    
    def fname(self, name):
        return f'arch.{name}.{self.capacity}.{self.dim}.{self.desc_dim}'
           
    def save(self, name):
        np.savez_compressed(self.fname(name), 
                            xs=self.get_xs(), 
                            ds=self.get_ds(), 
                            ys=self.get_ys(), 
                            cs=self.get_cs(),
                            stats=self.get_stats(),
                            counts=self.get_counts()
                            )

    def load(self, name):  
        self.cs = mp.RawArray(ct.c_double, self.capacity * self.desc_dim)
        with np.load(self.fname(name) + '.npz') as data:
            xs = data['xs']
            ds = data['ds']
            self.set_xs(xs)
            self.set_ds(ds)
            self.set_ys(data['ys'])
            self.set_cs(data['cs'])
            self.set_stats(data['stats'])
            self.counts[:] = data['counts']
            
        self.dim = xs.shape[1]
        self.desc_dim = ds.shape[1]
        self.capacity = xs.shape[0]
        self.index_of_niches, _ = get_index_of_niches(self.get_cs(), None, None, None)
        
    def in_niche_filter(self, fit, index):
        return in_niche_filter(fit, index, self.index_of_niches)
                                                               
    def set(self, i, yd, x):
        self.update_stats(i, x)
        y, d = yd
        # register improvement
        if y < self.get_y(i):
            self.set_y(i, y)
            self.set_x(i, x)
            self.set_d(i, d)
    
    def update_stats(self, i, x):
        count = self.counts[i] + 1
        mean = self.get_x_mean(i)
        diff = x - mean        
        self.set_stat(i, 0, mean + diff * (1./count)) # mean
        self.set_stat(i, 1, self.get_stat(i, 1) + np.multiply(diff,diff) * ((count-1)/count)) # qmean              
        self.counts[i] = count # count      
        self.set_stat(i, 2, np.minimum(x, self.get_stat(i, 2))) # min
        self.set_stat(i, 3, np.maximum(x, self.get_stat(i, 3))) # max
    
    def get_count(self, i):
        return self.counts[i]

    def get_counts(self):
        return np.array(self.counts[:])
 
    def get_x_mean(self, i):
        return self.get_stat(i, 0)

    def get_x_stdev(self, i):
        count = self.get_count(i)
        if count == 0:
            return np.zeros(self.dim)
        else:
            qmean = np.array(self.get_stat(i, 1))
            return np.sqrt(qmean * (1./count))

    def get_x_min(self, i):
        return self.get_stat(i, 2)

    def get_x_max(self, i):
        return self.get_stat(i, 3)
           
    def get_x(self, i):
        return self.xs[i*self.dim:(i+1)*self.dim]

    def get_xs(self):
        return np.array([self.get_x(i) for i in range(self.capacity)])

    def set_x(self, i, x):
        self.xs[i*self.dim:(i+1)*self.dim] = x[:]
    
    def set_xs(self, xs):
        for i in range(len(xs)):
            self.set_x(i, xs[i])

    def get_d(self, i):
        return self.ds[i*self.desc_dim:(i+1)*self.desc_dim]

    def get_ds(self):
        return np.array([self.get_d(i) for i in range(self.capacity)])
    
    def set_d(self, i, d):
        self.ds[i*self.desc_dim:(i+1)*self.desc_dim] = d[:]
 
    def set_ds(self, ds):
        for i in range(len(ds)):
            self.set_d(i, ds[i])
  
    def get_y(self, i):
        return self.ys[i]
        
    def get_ys(self):
        return np.array(self.ys[:])
           
    def set_y(self, i, y):
        self.ys[i] = y    

    def set_ys(self, ys):
        for i in range(len(ys)):
            self.set_y(i, ys[i])

    def get_c(self, i):
        return self.cs[i*self.desc_dim:(i+1)*self.desc_dim]
        
    def get_cs(self):
        return np.array([self.get_c(i) for i in range(self.capacity)])
           
    def set_c(self, i, c):
        self.cs[i*self.desc_dim:(i+1)*self.desc_dim] = c[:]

    def set_cs(self, cs):
        for i in range(len(cs)):
            self.set_c(i, cs[i])

    def get_stat(self, i, j):
        p = 4*i+j
        return self.stats[p*self.dim:(p+1)*self.dim]

    def get_stats(self):
        return np.array(self.stats[:])
           
    def set_stat(self, i, j, stat):
        p = 4*i+j
        self.stats[p*self.dim:(p+1)*self.dim] = stat[:]

    def set_stats(self, stats):
        self.stats[:] = stats[:]

    def random_xs(self, best_n, chunk_size, rg):
        selection = rg.integers(0, best_n, chunk_size)
        if best_n < self.capacity: 
            selection = np.array(np.fromiter((self.si[i] for i in selection), dtype=int))
        return self.get_xs()[selection]
    
    def random_xs_one(self, best_n, rg):
        r = rg.random()
        i = int(r*r*best_n)
        return self.get_x(i), self.get_y(i),  i
        
    def argsort(self):
        self.si = np.argsort(self.get_ys())
        return self.si
            
    def dump(self, n = None):
        if n is None:
            n = self.capacity
        ys = self.get_ys()
        si = np.argsort(ys)
        for i in range(n):
            print(si[i], ys[si[i]], self.get_x(si[i]))         

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
                    self.logger.info(
                        f'{dtime(self.t0)}  {self.evals.value:.3f} {self.evals.value/(1E-9 + dtime(self.t0)):.3f} {self.best_y.value} {list(x)}')
            return y, desc
        except Exception as ex:
            print(str(ex))  
            return np.inf, np.zeros(self.desc_dim)
        
class in_niche_filter(object):
    """Fitness function wrapper rejecting out of niche arguments."""
    
    def __init__(self, fit, index, index_of_niches):
        self.fit = fit
        self.index_of_niches = index_of_niches
        self.index = index

    def __call__(self, x):
        y, desc = self.fit(x)
        if self.index_of_niches([desc])[0] == self.index:
            return y
        else:
            return np.inf

# SBX (Simulated Binary Crossover + Mutation)
def variation(pop, lower, upper, rg, dis_c = 20, dis_m = 20):
    """Generate offspring individuals"""
    dis_c *= 0.5 + 0.5*rg.random() # vary parameters randomly 
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
    offspring = np.vstack(((parent_1 + parent_2) / 2 + beta * (parent_1 - parent_2) / 2,
                               (parent_1 + parent_2) / 2 - beta * (parent_1 - parent_2) / 2))
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

# Iso+Line
def iso_dd(x1, x2, lower, upper, rg, iso_sigma = 0.01, line_sigma = 0.2):
    a = rg.normal(0, iso_sigma, x1.shape) 
    b = rg.normal(0, line_sigma, x2.shape) 
    z = x1 + a + np.multiply(b, (x1 - x2))
    return np.clip(z, lower, upper)
