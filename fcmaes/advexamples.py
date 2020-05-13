# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

from fcmaes.astro import MessFull, Messenger, Cassini2, Rosetta, Gtoc1, Cassini1
from fcmaes.optimizer import logger, de_cma, da_cma, Cma_cpp, De_cpp, Da_cpp, Dual_annealing, Differential_evolution
from fcmaes.advretry import minimize

def messengerFullLoop():    
    while True:    
        problem = MessFull()
        logger().info(problem.name + ' de + cmaes c++')
        ret = minimize(problem.fun, bounds=problem.bounds, num_retries = 60000, 
            value_limit = 10.0, logger = logger())
               
def test_all(min_evals = 1000, num_retries = 10000, num = 10):

    _test_optimizer(de_cma(min_evals), Cassini1(), num_retries, num) 
    _test_optimizer(de_cma(min_evals), Cassini2(), num_retries, num) 
    _test_optimizer(de_cma(min_evals), Rosetta(), num_retries, num) 
    _test_optimizer(de_cma(min_evals), Messenger(), num_retries, num) 
    _test_optimizer(de_cma(min_evals), Gtoc1(), num_retries, num) 
    _test_optimizer(de_cma(min_evals), MessFull(), num_retries, num) 
  
    _test_optimizer(da_cma(min_evals), Cassini1(), num_retries, num) 
    _test_optimizer(da_cma(min_evals), Cassini2(), num_retries, num) 
    _test_optimizer(da_cma(min_evals), Rosetta(), num_retries, num) 
    _test_optimizer(da_cma(min_evals), Messenger(), num_retries, num) 
    _test_optimizer(da_cma(min_evals), Gtoc1(), num_retries, num) 
    _test_optimizer(da_cma(min_evals), MessFull(), num_retries, num) 
  
    _test_optimizer(Cma_cpp(min_evals), Cassini1(), num_retries, num) 
    _test_optimizer(Cma_cpp(min_evals), Cassini2(), num_retries, num) 
    _test_optimizer(Cma_cpp(min_evals), Rosetta(), num_retries, num) 
    _test_optimizer(Cma_cpp(min_evals), Messenger(), num_retries, num) 
    _test_optimizer(Cma_cpp(min_evals), Gtoc1(), num_retries, num) 
    _test_optimizer(Cma_cpp(min_evals), MessFull(), num_retries, num) 
  
    _test_optimizer(De_cpp(min_evals), Cassini1(), num_retries, num) 
    _test_optimizer(De_cpp(min_evals), Cassini2(), num_retries, num) 
    _test_optimizer(De_cpp(min_evals), Rosetta(), num_retries, num) 
    _test_optimizer(De_cpp(min_evals), Messenger(), num_retries, num) 
    _test_optimizer(De_cpp(min_evals), Gtoc1(), num_retries, num) 
    _test_optimizer(De_cpp(min_evals), MessFull(), num_retries, num) 

    _test_optimizer(Da_cpp(min_evals), Cassini1(), num_retries, num) 
    _test_optimizer(Da_cpp(min_evals), Cassini2(), num_retries, num) 
    _test_optimizer(Da_cpp(min_evals), Rosetta(), num_retries, num) 
    _test_optimizer(Da_cpp(min_evals), Messenger(), num_retries, num) 
    _test_optimizer(Da_cpp(min_evals), Gtoc1(), num_retries, num) 
    _test_optimizer(Da_cpp(min_evals), MessFull(), num_retries, num) 
    
    _test_optimizer(Dual_annealing(min_evals), Cassini1(), num_retries, num) 
    _test_optimizer(Dual_annealing(min_evals), Cassini2(), num_retries, num) 
    _test_optimizer(Dual_annealing(min_evals), Rosetta(), num_retries, num) 
    _test_optimizer(Dual_annealing(min_evals), Messenger(), num_retries, num) 
    _test_optimizer(Dual_annealing(min_evals), Gtoc1(), num_retries, num) 
    _test_optimizer(Dual_annealing(min_evals), MessFull(), num_retries, num) 
    
    _test_optimizer(Differential_evolution(min_evals), Cassini1(), num_retries, num) 
    _test_optimizer(Differential_evolution(min_evals), Cassini2(), num_retries, num) 
    _test_optimizer(Differential_evolution(min_evals), Rosetta(), num_retries, num) 
    _test_optimizer(Differential_evolution(min_evals), Messenger(), num_retries, num) 
    _test_optimizer(Differential_evolution(min_evals), Gtoc1(), num_retries, num) 
    _test_optimizer(Differential_evolution(min_evals), MessFull(), num_retries, num) 

def _test_optimizer(opt, problem, num_retries = 10000, num = 1, value_limit = 20.0, log = logger()):
    log.info(problem.name + ' ' + opt.name)
    for i in range(num):
        ret = minimize(problem.fun, problem.bounds, value_limit, num_retries, log, optimizer=opt)

def main():
    test_all()
    #messengerFullLoop()

if __name__ == '__main__':
    main()