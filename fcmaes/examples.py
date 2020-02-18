'''
Created on Feb 9, 2020

@author: Dietmar Wolz
'''

import time
    
from fcmaes.astro import MessFull, Messenger, Gtoc1, Cassini1
from fcmaes.optimizer import Optimizer, dtime, logger
from fcmaes.retry import Store, retry, minimize
from fcmaes.testfun import RastriginMean
from fcmaes import cmaes

def messengerFullLoop():    
    while True:    
        problem = MessFull()
        logger.info(problem.name + ' cmaes c++')
        ret = minimize(problem.fun, bounds=problem.bounds, num_retries = 40000, 
                       max_evaluations = 50000, useCpp = True)
            
        
def parallel_execution_example(dim, n):
    maxEval = 10000
    popsize = 32
    testfun = RastriginMean(dim, n)
    sdevs = [1]*dim
    
    t0 = time.perf_counter()
  
    ret = cmaes.minimize(testfun.fun, 
                testfun.bounds, max_evaluations=maxEval, popsize=popsize, 
                input_sigma=sdevs, is_parallel=True)         
    print(ret.fun, dtime(t0))
  
    t0 = time.perf_counter()
    ret = cmaes.minimize(testfun.fun, 
                testfun.bounds, max_evaluations=maxEval, popsize=popsize, 
                input_sigma=sdevs, is_parallel=False)         
    print(ret.fun, dtime(t0))
     

def test_all(max_evals = 50000, num_retries = 2000, num = 20):

    problem = Cassini1()
    logger.info(problem.name + ' cmaes c++')
    for i in range(num):
        ret = minimize(problem.fun, bounds=problem.bounds, num_retries = num_retries, 
                       max_evaluations = max_evals, useCpp = True)
 
    problem = Messenger()
    logger.info(problem.name + ' cmaes c++')
    for i in range(num):
        ret = minimize(problem.fun, bounds=problem.bounds, num_retries = num_retries, 
                       max_evaluations = max_evals, useCpp = True)
 
    problem = Gtoc1()
    logger.info(problem.name + ' cmaes c++')
    for i in range(num):
        ret = minimize(problem.fun, bounds=problem.bounds, num_retries = num_retries, 
                       max_evaluations = max_evals, useCpp = True)
 
    problem = MessFull()
    logger.info(problem.name + ' cmaes c++')
    for i in range(num):
        ret = minimize(problem.fun, bounds=problem.bounds, num_retries = num_retries, 
                       max_evaluations = max_evals, useCpp = True)

    problem = Cassini1()
    logger.info(problem.name + ' cmaes python')
    for i in range(num):
        ret = minimize(problem.fun, bounds=problem.bounds, num_retries = num_retries, 
                       max_evaluations = max_evals, useCpp = True)
 
    problem = Messenger()
    logger.info(problem.name + ' cmaes python')
    for i in range(num):
        ret = minimize(problem.fun, bounds=problem.bounds, num_retries = num_retries, 
                       max_evaluations = max_evals, useCpp = False)

    problem = Gtoc1()
    logger.info(problem.name + ' cmaes python')
    for i in range(num):
        ret = minimize(problem.fun, bounds=problem.bounds, num_retries = num_retries, 
                       max_evaluations = max_evals, useCpp = False)

    problem = MessFull()
    logger.info(problem.name + ' cmaes python')
    for i in range(num):
        ret = minimize(problem.fun, bounds=problem.bounds, num_retries = num_retries, 
                       max_evaluations = max_evals, useCpp = False)
         

    problem = Cassini1()
    logger.info(problem.name + ' dual annealing')
    for i in range(num):
        store = Store(problem.bounds, max_evals)
        optimizer  = Optimizer(store, 0)
        ret = retry(problem.fun, store, optimizer.dual_annealing, num_retries)
      
    problem = Messenger()
    logger.info(problem.name + ' dual annealing')
    for i in range(num):
        store = Store(problem.bounds, max_evals)
        optimizer  = Optimizer(store, 0)
        ret = retry(problem.fun, store, optimizer.dual_annealing, num_retries)
  
    problem = Gtoc1()
    logger.info(problem.name + ' dual annealing')
    for i in range(num):
        store = Store(problem.bounds, max_evals)
        optimizer  = Optimizer(store, 0)
        ret = retry(problem.fun, store, optimizer.dual_annealing, num_retries)
   
    problem = MessFull()
    logger.info(problem.name + ' dual annealing')
    for i in range(num):
        store = Store(problem.bounds, max_evals)
        optimizer  = Optimizer(store, 0)
        ret = retry(problem.fun, store, optimizer.dual_annealing, num_retries)
      
 
    problem = Cassini1()
    logger.info(problem.name + ' differential evolution')
    for i in range(num):
        store = Store(problem.bounds, max_evals)
        optimizer  = Optimizer(store, 0)
        ret = retry(problem.fun, store, optimizer.differential_evolution, num_retries)
 
    problem = Messenger()
    logger.info(problem.name + ' differential evolution')
    for i in range(num):
        store = Store(problem.bounds, max_evals)
        optimizer  = Optimizer(store, 0)
        ret = retry(problem.fun, store, optimizer.differential_evolution, num_retries)
 
    problem = Gtoc1()
    logger.info(problem.name + ' differential evolution')
    for i in range(num):
        store = Store(problem.bounds, max_evals)
        optimizer  = Optimizer(store, 0)
        ret = retry(problem.fun, store, optimizer.differential_evolution, num_retries)
 
    problem = MessFull()
    logger.info(problem.name + ' differential evolution')
    for i in range(num):
        store = Store(problem.bounds, max_evals)
        optimizer  = Optimizer(store, 0)
        ret = retry(problem.fun, store, optimizer.differential_evolution, num_retries)

def main():

    test_all()
    #parallel_execution_example(27, 1000)

if __name__ == '__main__':
    main()