# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

from fcmaes.astro import MessFull, Messenger, Cassini2, Rosetta, Gtoc1, Cassini1
from fcmaes.optimizer import  Optimizer, logger
from fcmaes.advretry import Store, retry, minimize

def messengerFullLoop():    
    while True:    
        problem = MessFull()
        logger().info(problem.name + ' cmaes c++')
        ret = minimize(problem.fun, bounds=problem.bounds, num_retries = 40000, 
            max_evaluations = 50000, value_limit = 10.0, logger = logger(), 
            useCpp = True)
               
def test_all(num_retries = 4000, num = 20):
    
    # test C++ version
    _test_problem(Cassini1(), num_retries, num, useCpp = True) 
    _test_problem(Cassini2(), num_retries, num, useCpp = True) 
    _test_problem(Rosetta(), num_retries, num, useCpp = True) 
    _test_problem(Messenger(), num_retries, num, useCpp = True) 
    _test_problem(Gtoc1(), num_retries, num, value_limit = -1000000, useCpp = True) 
    _test_problem(MessFull(), num_retries, num, value_limit = 10.0, useCpp = True) 

    # test python version
    _test_problem(Cassini1(), num_retries, num) 
    _test_problem(Cassini2(), num_retries, num) 
    _test_problem(Rosetta(), num_retries, num) 
    _test_problem(Messenger(), num_retries, num) 
    _test_problem(Gtoc1(), num_retries, num, value_limit = -1000000) 
    _test_problem(MessFull(), num_retries, num, value_limit = 10.0) 

    # test dual annealing
    _test_optimizer("dual_annealing", Cassini1(), num_retries, num) 
    _test_optimizer("dual_annealing", Cassini2(), num_retries, num) 
    _test_optimizer("dual_annealing", Rosetta(), num_retries, num) 
    _test_optimizer("dual_annealing", Messenger(), num_retries, num) 
    _test_optimizer("dual_annealing", Gtoc1(), num_retries, num, value_limit = -200000) 
    _test_optimizer("dual_annealing", MessFull(), num_retries, num) 

    # test differential evolution
    _test_optimizer("differential_evolution", Cassini1(), num_retries, num) 
    _test_optimizer("differential_evolution", Cassini2(), num_retries, num) 
    _test_optimizer("differential_evolution", Rosetta(), num_retries, num) 
    _test_optimizer("differential_evolution", Messenger(), num_retries, num) 
    _test_optimizer("differential_evolution", Gtoc1(), num_retries, num, value_limit = -200000) 
    _test_optimizer("differential_evolution", MessFull(), num_retries, num) 

def _test_problem(problem, num_retries = 4000, num = 20, value_limit = 12.0, 
                  log = logger(), useCpp = False):
    log.info(problem.name + ' cmaes ' + ('c++' if useCpp else 'python'))
    for i in range(num):
        ret = minimize(problem.fun, bounds=problem.bounds, num_retries = num_retries, 
                       useCpp = useCpp, logger = log, value_limit = value_limit)
        
def _test_optimizer(opt_name, problem, num_retries = 4000, num = 20, value_limit = 20.0, 
                   log = logger()):
    log.info(problem.name + ' ' + opt_name)
    for i in range(num):
        store = Store(problem.bounds, logger = log)
        optimizer = Optimizer(store, 0)
        method = getattr(optimizer, opt_name)
        ret = retry(problem.fun, store, method, num_retries, value_limit = value_limit)

def main():
    test_all()
    #_test_optimizer("dual_annealing", Cassini1(), 500, 1) 
    #_test_problem(Cassini1(), 500, 1) 
    #messengerFullLoop()

if __name__ == '__main__':
    main()