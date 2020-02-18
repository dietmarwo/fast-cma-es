'''
Created on Feb 9, 2020

@author: Dietmar Wolz
'''

from fcmaes.astro import MessFull, Messenger, Gtoc1, Cassini1
from fcmaes.optimizer import  Optimizer, logger
from fcmaes.advretry import Store, retry, minimize

def messengerFullLoop():    
    while True:    
        problem = MessFull()
        logger.info(problem.name + ' cmaes c++')
        ret = minimize(problem.fun, bounds=problem.bounds, num_retries = 40000, 
                       max_evaluations = 50000, value_limit = 10.0, useCpp = True)

def test_all(num_retries = 4000, num = 20):
     
    problem = Cassini1()
    logger.info(problem.name + ' cmaes c++')
    for i in range(num):
        ret = minimize(problem.fun, bounds=problem.bounds, num_retries = num_retries, 
                       useCpp = True, value_limit = 12.0)
 
    problem = Messenger()
    logger.info(problem.name + ' cmaes c++')
    for i in range(num):
        ret = minimize(problem.fun, bounds=problem.bounds, num_retries = num_retries, 
                       useCpp = True, value_limit = 10.0)
 
    problem = Gtoc1()
    logger.info(problem.name + ' cmaes c++')
    for i in range(num):
        ret = minimize(problem.fun, bounds=problem.bounds, num_retries = num_retries, 
                       useCpp = True, value_limit = -1000000)
 
    problem = MessFull()
    logger.info(problem.name + ' cmaes c++')
    for i in range(num):
        ret = minimize(problem.fun, bounds=problem.bounds, num_retries = num_retries, 
                       useCpp = True, value_limit = 10.0)


    problem = Cassini1()
    logger.info(problem.name + ' cmaes python')
    for i in range(num):
        ret = minimize(problem.fun, bounds=problem.bounds, num_retries = num_retries, 
                       useCpp = False, value_limit = 12.0)
 
    problem = Messenger()
    logger.info(problem.name + ' cmaes python')
    for i in range(num):
        ret = minimize(problem.fun, bounds=problem.bounds, num_retries = num_retries, 
                       useCpp = False, value_limit = 10.0)

    problem = Gtoc1()
    logger.info(problem.name + ' cmaes python')
    for i in range(num):
        ret = minimize(problem.fun, bounds=problem.bounds, num_retries = num_retries, 
                       useCpp = False, value_limit = -1000000)

    problem = MessFull()
    logger.info(problem.name + ' cmaes python')
    for i in range(num):
        ret = minimize(problem.fun, bounds=problem.bounds, num_retries = num_retries, 
                       useCpp = False, value_limit = 10.0)
        

    problem = Cassini1()
    logger.info(problem.name + ' dual annealing')
    for i in range(num):
        store = Store(problem.bounds)
        optimizer  = Optimizer(store, 0)
        ret = retry(problem.fun, store, optimizer.dual_annealing, num_retries, 20.0)

    problem = Messenger()
    logger.info(problem.name + ' dual annealing')
    for i in range(num):
        store = Store(problem.bounds)
        optimizer  = Optimizer(store, 0)
        ret = retry(problem.fun, store, optimizer.dual_annealing, num_retries, 20.0)

    problem = Gtoc1()
    logger.info(problem.name + ' dual annealing')
    for i in range(num):
        store = Store(problem.bounds)
        optimizer  = Optimizer(store, 0)
        ret = retry(problem.fun, store, optimizer.dual_annealing, num_retries, -200000)

    problem = MessFull()
    logger.info(problem.name + ' dual annealing')
    for i in range(num):
        store = Store(problem.bounds)
        optimizer  = Optimizer(store, 0)
        ret = retry(problem.fun, store, optimizer.dual_annealing, num_retries, 20.0)
   
        
    problem = Cassini1()
    logger.info(problem.name + ' differential evolution')
    for i in range(num):
        store = Store(problem.bounds)
        optimizer  = Optimizer(store, 0)
        ret = retry(problem.fun, store, optimizer.differential_evolution, num_retries, 20.0)

    problem = Messenger()
    logger.info(problem.name + ' differential evolution')
    for i in range(num):
        store = Store(problem.bounds)
        optimizer  = Optimizer(store, 0)
        ret = retry(problem.fun, store, optimizer.differential_evolution, num_retries, 20.0)

    problem = Gtoc1()
    logger.info(problem.name + ' differential evolution')
    for i in range(num):
        store = Store(problem.bounds)
        optimizer  = Optimizer(store, 0)
        ret = retry(problem.fun, store, optimizer.differential_evolution, num_retries, -200000)

    problem = MessFull()
    logger.info(problem.name + ' differential evolution')
    for i in range(num):
        store = Store(problem.bounds)
        optimizer  = Optimizer(store, 0)
        ret = retry(problem.fun, store, optimizer.differential_evolution, num_retries, 20.0)


def main():
    test_all()
    #messengerFullLoop()

if __name__ == '__main__':
    main()