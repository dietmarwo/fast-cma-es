# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# parallel optimization retry of a list of problems. 

import numpy as np
import _pickle as cPickle
import bz2
import multiprocessing as mp
from scipy.optimize import OptimizeResult, Bounds
from fcmaes.optimizer import de_cma, eprint, Optimizer
from fcmaes import advretry

from fcmaes.evaluator import is_debug_active
from loguru import logger
from typing import Optional, Callable, Tuple, List
from numpy.typing import ArrayLike

def minimize(problems: ArrayLike, 
             ids: Optional[ArrayLike] = None, 
             retries_inc: Optional[int] = min(256, 8*mp.cpu_count()), 
             num_retries: Optional[int] = 10000,
             keep: Optional[float] = 0.7, 
             optimizer: Optional[Optimizer] = de_cma(1500), 
             datafile = None) -> List:
      
    """Minimization of a list of optimization problems by first applying parallel retry
    to filter the best ones and then applying coordinated retry to evaluate these further. 
    Can replace mixed integer optimization if the integer variables are narrowly bound. 
    In this case all combinations of these integer values can be enumerated to generate a
    list of problem instances each representing one combination. See for instance 
    https://www.esa.int/gsp/ACT/projects/gtop/tandem where there is a problem instance for each
    planet sequence.
     
    Parameters
    ----------
    
    problems: list
        list of objects providing name, fun and bounds attributes like fcmaes.astro.Astrofun

    ids:  list, optional
        list of objects corresponding to the list of problems used in logging to identify the 
        problem variant currently logged. If None, the index of the problem 
        variant is used instead.

    retries_inc:  int, optional
        number of coordinated retries applied in the problem filter for each problem 
        in each iteration.
    
    num_retries:  int, optional
        number of coordinated retries applied in the problem filter for the winner problem.
 
    keep:  float, optional
        rate of the problems kept after each iteration. 100*(1 - keep) % will be deleted. 
                        
    optimizer: optimizer.Optimizer, optional
        optimizer to use for the problem filter.
        
    datafile, optional
        file to persist / retrieve the internal state of the optimizations. 
     
    Returns
    -------
    dictionary( optimizer -> ret): scipy.OptimizeResult
        The optimization result is represented as an ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, 
        ``fun`` the best function value, ``nfev`` the number of function evaluations,
        ``success`` a Boolean flag indicating if the optimizer exited successfully. """

    solver = multiretry()
    n = len(problems)
        
    for i in range(n):    
        id = str(i+1) if ids is None else ids[i]   
        solver.add(problem_stats(problems[i], id, i, retries_inc, num_retries))
    
    if not datafile is None:
        solver.load(datafile)
        
    while solver.size() > 1:    
        solver.retry(optimizer)
        to_remove = int(round((1.0 - keep) * solver.size()))
        if to_remove == 0 and keep < 1.0:
            to_remove = 1
        solver.remove_worst(to_remove)
        solver.dump()
        if not datafile is None:
            solver.save(datafile)
            
    idx = solver.values_all().argsort()
    return list(np.asarray(solver.all_stats)[idx])
        
class problem_stats:

    def __init__(self, prob, id, index, retries_inc = 64, num_retries = 10000):
        self.store = advretry.Store(prob.fun, prob.bounds, num_retries=num_retries)
        self.prob = prob
        self.name = prob.name
        self.fun = prob.fun
        self.retries_inc = retries_inc
        self.value = 0
        self.id = id
        self.index = index
        self.ret = None
        self.store.num_retries = self.retries_inc

    def retry(self, optimizer):
        self.store.num_retries += self.retries_inc
        self.ret = advretry.retry(self.store, optimizer.minimize)
        self.value = self.store.get_y_best()
 
class multiretry:
    
    def __init__(self):
        self.problem_stats = []
        self.all_stats = []
    
    def add(self, stats):
        self.problem_stats.append(stats)
        self.all_stats.append(stats)
    
    def retry(self, optimizer):
        for ps in self.problem_stats:
            if is_debug_active():
                logger.debug("problem " + ps.prob.name + ' ' + str(ps.id))
            ps.retry(optimizer)
    
    def values(self):
        return np.fromiter((ps.value for ps in self.problem_stats), dtype=float)
     
    def remove_worst(self, n = 1):
        idx = self.values().argsort()
        self.problem_stats = list(np.asarray(self.problem_stats)[idx])
        for _ in range(n):
            self.problem_stats.pop(-1)

    def size(self):
        return len(self.problem_stats)
                    
    def dump(self):
        if is_debug_active():
            for i in range(self.size()):
                ps = self.problem_stats[i]
                logger.debug(str(ps.id) + ' ' + str(ps.value))
                
    def dump_all(self):
        if is_debug_active():
            idx = self.values_all().argsort()
            self.all_stats = list(np.asarray(self.all_stats)[idx])
            for i in range(len(self.all_stats)):
                ps = self.all_stats[i]
                logger.debug(str(ps.id) + ' ' + str(ps.value))

    def values_all(self):
        return np.fromiter((ps.value for ps in self.all_stats), dtype=float)
 
    def result(self):
        idx = self.values_all().argsort()
        self.all_stats = list(np.asarray(self.all_stats)[idx])
        ret = []
        for i in range(len(self.all_stats)):
            problem = self.all_stats[i].prob
            store = self.all_stats[i].store
            ret.append([problem, 
                        OptimizeResult(x=store.get_x_best(), fun=store.get_y_best(), 
                          nfev=store.get_count_evals(), success=True)])
            
    # persist all stats
    def save(self, name):
        try:
            with bz2.BZ2File(name + '.pbz2', 'w') as f: 
                cPickle.dump(self.get_data(), f)
        except Exception as ex:
            eprint('error writing data file ' + name + '.pbz2 ' + str(ex))

    def load(self, name):
        try:
            data = cPickle.load(bz2.BZ2File(name + '.pbz2', 'rb'))
            self.set_data(data)
        except Exception as ex:
            eprint('error reading data file ' + name + '.pbz2 ' + str(ex))
  
    def get_data(self):
        data = []
        for stats in self.all_stats:            
            data.append(stats.store.get_data())
        return data
        
    def set_data(self, data):
        for i in range(len(data)):
            self.all_stats[i].store.set_data(data[i])
        
    
