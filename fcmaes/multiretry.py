# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# parallel optimization retry of a list of problems. 

import numpy as np
import _pickle as cPickle
import bz2
import multiprocessing as mp
from scipy.optimize import OptimizeResult
from fcmaes.optimizer import logger, de_cma, eprint
from fcmaes import advretry

def minimize(problems, ids=None, num_retries = min(256, 8*mp.cpu_count()), 
             keep = 0.7, optimizer = de_cma(1500), logger = None, datafile = None):
      
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
    
    num_retries:  int, optional
        number of coordinated retries applied in the problem filter for each problem 
        in each iteration.
 
    keep:  float, optional
        rate of the problems kept after each iteration. 100*(1 - keep) % will be deleted. 
                        
    optimizer: optimizer.Optimizer, optional
        optimizer to use for the problem filter.
        
    logger, optional
        logger for log output. If None, logging
        is switched off. Default is a logger which logs both to stdout and
        appends to a file.  
        
    datafile, optional
        file to persist / retrieve the internal state of the optimizations. 
     
    Returns
    -------
    dictionary( optimizer -> ret): scipy.OptimizeResult
        The optimization result is represented as an ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, 
        ``fun`` the best function value, ``nfev`` the number of function evaluations,
        ``success`` a Boolean flag indicating if the optimizer exited successfully. """

    solver = multiretry(logger)
    n = len(problems)
        
    for i in range(n):    
        id = str(i+1) if ids is None else ids[i]   
        solver.add(problem_stats(problems[i], id, i, num_retries, logger))
    
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

    def __init__(self, prob, id, index, num_retries = 64, logger = None):
        self.store = advretry.Store(prob.bounds, logger = logger, num_retries=num_retries)
        self.prob = prob
        self.name = prob.name
        self.fun = prob.fun
        self.num_retries = num_retries
        self.retries = 0
        self.value = 0
        self.id = id
        self.index = index
        self.ret = None

    def retry(self, optimizer):
        self.retries += self.num_retries
        self.ret = advretry.retry(self.fun, self.store, optimizer.minimize)
        self.value = self.store.get_y_best()
 
class multiretry:
    
    def __init__(self, logger = None):
        self.problem_stats = []
        self.all_stats = []
        self.logger = logger
    
    def add(self, stats):
        self.problem_stats.append(stats)
        self.all_stats.append(stats)
    
    def retry(self, optimizer):
        for ps in self.problem_stats:
            if not self.logger is None:
                self.logger.info("problem " + ps.prob.name + ' ' + str(ps.id))
            ps.retry(optimizer)
    
    def values(self):
        return np.array([ps.value for ps in self.problem_stats])
     
    def remove_worst(self, n = 1):
        idx = self.values().argsort()
        self.problem_stats = list(np.asarray(self.problem_stats)[idx])
        for _ in range(n):
            self.problem_stats.pop(-1)

    def size(self):
        return len(self.problem_stats)
                    
    def dump(self):
        if not self.logger is None:
            for i in range(self.size()):
                ps = self.problem_stats[i]
                self.logger.info(str(ps.id) + ' ' + str(ps.value))
                
    def dump_all(self):
        if not self.logger is None:
            idx = self.values_all().argsort()
            self.all_stats = list(np.asarray(self.all_stats)[idx])
            for i in range(len(self.all_stats)):
                ps = self.all_stats[i]
                self.logger.info(str(ps.id) + ' ' + str(ps.value))

    def values_all(self):
        return np.array([ps.value for ps in self.all_stats])
 
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
        
    