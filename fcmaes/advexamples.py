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
        ret = minimize(problem.fun, bounds=problem.bounds, num_retries = 40000, 
            value_limit = 10.0, logger = logger())
               
def test_all(num_retries = 5000, num = 10):
    
    _test_optimizer(de_cma(2000), Cassini1(), num_retries, num) 
    _test_optimizer(de_cma(2000), Cassini2(), num_retries, num) 
    _test_optimizer(de_cma(2000), Rosetta(), num_retries, num) 
    _test_optimizer(de_cma(2000), Messenger(), num_retries, num) 
    _test_optimizer(de_cma(2000), Gtoc1(), num_retries, num) 
    _test_optimizer(de_cma(2000), MessFull(), num_retries, num) 
  
    _test_optimizer(da_cma(2000), Cassini1(), num_retries, num) 
    _test_optimizer(da_cma(2000), Cassini2(), num_retries, num) 
    _test_optimizer(da_cma(2000), Rosetta(), num_retries, num) 
    _test_optimizer(da_cma(2000), Messenger(), num_retries, num) 
    _test_optimizer(da_cma(2000), Gtoc1(), num_retries, num) 
    _test_optimizer(da_cma(2000), MessFull(), num_retries, num) 
  
    _test_optimizer(Cma_cpp(2000), Cassini1(), num_retries, num) 
    _test_optimizer(Cma_cpp(2000), Cassini2(), num_retries, num) 
    _test_optimizer(Cma_cpp(2000), Rosetta(), num_retries, num) 
    _test_optimizer(Cma_cpp(2000), Messenger(), num_retries, num) 
    _test_optimizer(Cma_cpp(2000), Gtoc1(), num_retries, num) 
    _test_optimizer(Cma_cpp(2000), MessFull(), num_retries, num) 
  
    _test_optimizer(De_cpp(2000), Cassini1(), num_retries, num) 
    _test_optimizer(De_cpp(2000), Cassini2(), num_retries, num) 
    _test_optimizer(De_cpp(2000), Rosetta(), num_retries, num) 
    _test_optimizer(De_cpp(2000), Messenger(), num_retries, num) 
    _test_optimizer(De_cpp(2000), Gtoc1(), num_retries, num) 
    _test_optimizer(De_cpp(2000), MessFull(), num_retries, num) 

    _test_optimizer(Da_cpp(2000), Cassini1(), num_retries, num) 
    _test_optimizer(Da_cpp(2000), Cassini2(), num_retries, num) 
    _test_optimizer(Da_cpp(2000), Rosetta(), num_retries, num) 
    _test_optimizer(Da_cpp(2000), Messenger(), num_retries, num) 
    _test_optimizer(Da_cpp(2000), Gtoc1(), num_retries, num) 
    _test_optimizer(Da_cpp(2000), MessFull(), num_retries, num) 
    
    _test_optimizer(Dual_annealing(2000), Cassini1(), num_retries, num) 
    _test_optimizer(Dual_annealing(2000), Cassini2(), num_retries, num) 
    _test_optimizer(Dual_annealing(2000), Rosetta(), num_retries, num) 
    _test_optimizer(Dual_annealing(2000), Messenger(), num_retries, num) 
    _test_optimizer(Dual_annealing(2000), Gtoc1(), num_retries, num) 
    _test_optimizer(Dual_annealing(2000), MessFull(), num_retries, num) 
    
    _test_optimizer(Differential_evolution(2000), Cassini1(), num_retries, num) 
    _test_optimizer(Differential_evolution(2000), Cassini2(), num_retries, num) 
    _test_optimizer(Differential_evolution(2000), Rosetta(), num_retries, num) 
    _test_optimizer(Differential_evolution(2000), Messenger(), num_retries, num) 
    _test_optimizer(Differential_evolution(2000), Gtoc1(), num_retries, num) 
    _test_optimizer(Differential_evolution(2000), MessFull(), num_retries, num) 

def _test_optimizer(opt, problem, num_retries = 4000, num = 1, value_limit = 20.0, log = logger()):
    log.info(problem.name + ' ' + opt.name)
    for i in range(num):
        ret = minimize(problem.fun, problem.bounds, value_limit, num_retries, log, optimizer=opt)

def main():
    test_all()
    #messengerFullLoop()

if __name__ == '__main__':
    main()