# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

import time
    
from fcmaes.astro import MessFull, Messenger, Gtoc1, Cassini1, Cassini2, Rosetta
from fcmaes.optimizer import Optimizer, dtime, logger
from fcmaes.retry import Store, retry, minimize
from fcmaes.testfun import RastriginMean
from fcmaes import cmaes

def messengerFullLoop():    
    while True:    
        problem = MessFull()
        logger().info(problem.name + ' cmaes c++')
        ret = minimize(problem.fun, bounds=problem.bounds, num_retries = 40000, 
                max_evaluations = 50000, logger = logger(), useCpp = True)
            
        
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
    
    # test C++ version
    _test_problem(Cassini1(), max_evals, num_retries, num, useCpp = True) 
    _test_problem(Cassini2(), max_evals, num_retries, num, useCpp = True) 
    _test_problem(Rosetta(), max_evals, num_retries, num, useCpp = True) 
    _test_problem(Messenger(), max_evals, num_retries, num, useCpp = True) 
    _test_problem(Gtoc1(), max_evals, num_retries, num, useCpp = True) 
    _test_problem(MessFull(), max_evals, num_retries, num, useCpp = True) 

    # test python version
    _test_problem(Cassini1(), max_evals, num_retries, num) 
    _test_problem(Cassini2(), max_evals, num_retries, num) 
    _test_problem(Rosetta(), max_evals, num_retries, num) 
    _test_problem(Messenger(), max_evals, num_retries, num) 
    _test_problem(Gtoc1(), max_evals, num_retries, num) 
    _test_problem(MessFull(), max_evals, num_retries, num) 

    # test dual annealing
    _test_optimizer("dual_annealing", Cassini1(), max_evals, num_retries, num) 
    _test_optimizer("dual_annealing", Cassini2(), max_evals, num_retries, num) 
    _test_optimizer("dual_annealing", Rosetta(), max_evals, num_retries, num) 
    _test_optimizer("dual_annealing", Messenger(), max_evals, num_retries, num) 
    _test_optimizer("dual_annealing", Gtoc1(), max_evals, num_retries, num) 
    _test_optimizer("dual_annealing", MessFull(), num_retries, num) 

    # test differential evolution
    _test_optimizer("differential_evolution", Cassini1(), max_evals, num_retries, num) 
    _test_optimizer("differential_evolution", Cassini2(), max_evals, num_retries, num) 
    _test_optimizer("differential_evolution", Rosetta(), max_evals, num_retries, num) 
    _test_optimizer("differential_evolution", Messenger(), max_evals, num_retries, num) 
    _test_optimizer("differential_evolution", Gtoc1(), max_evals, num_retries, num) 
    _test_optimizer("differential_evolution", MessFull(), max_evals, num_retries, num) 
     
def _test_problem(problem, max_evals = 50000, num_retries = 2000, num = 20, 
                  log = logger(), useCpp = False):
    log.info(problem.name + ' cmaes ' + ('c++' if useCpp else 'python'))
    for i in range(num):
        ret = minimize(problem.fun, bounds=problem.bounds, num_retries = num_retries, 
                       max_evaluations = max_evals, useCpp = useCpp, logger = log)
        
def _test_optimizer(opt_name, problem, max_evals = 50000, num_retries = 2000, 
                    num = 20, log = logger()):
    log.info(problem.name + ' ' + opt_name)
    for i in range(num):
        store = Store(problem.bounds, max_evals, logger = log)
        optimizer = Optimizer(store, 0)
        method = getattr(optimizer, opt_name)
        ret = retry(problem.fun, store, method, num_retries)

def main():
    test_all()
    #_test_optimizer("dual_annealing", Cassini1(), 50000, 200, 1) 
    #_test_problem(Cassini1(), 50000, 200, 1) 
    #parallel_execution_example(27, 1000)

if __name__ == '__main__':
    main()