# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

import math
from fcmaes.astro import MessFull, Messenger, Gtoc1, Cassini1, Cassini2, Rosetta
from fcmaes.optimizer import logger, de_cma, da_cma, Cma_cpp, De_cpp, Da_cpp, Dual_annealing, Differential_evolution
from fcmaes.retry import minimize
            
def test_all(max_evals = 50000, num_retries = 2000, num = 10):
           
    _test_optimizer(de_cma(max_evals), Cassini1(), num_retries, num) 
    _test_optimizer(de_cma(max_evals), Cassini2(), num_retries, num) 
    _test_optimizer(de_cma(max_evals), Rosetta(), num_retries, num) 
    _test_optimizer(de_cma(max_evals), Messenger(), num_retries, num) 
    _test_optimizer(de_cma(max_evals), Gtoc1(), num_retries, num) 
    _test_optimizer(de_cma(max_evals), MessFull(), num_retries, num) 
  
    _test_optimizer(da_cma(max_evals), Cassini1(), num_retries, num) 
    _test_optimizer(da_cma(max_evals), Cassini2(), num_retries, num) 
    _test_optimizer(da_cma(max_evals), Rosetta(), num_retries, num) 
    _test_optimizer(da_cma(max_evals), Messenger(), num_retries, num) 
    _test_optimizer(da_cma(max_evals), Gtoc1(), num_retries, num) 
    _test_optimizer(da_cma(max_evals), MessFull(), num_retries, num) 
  
    _test_optimizer(Cma_cpp(max_evals), Cassini1(), num_retries, num) 
    _test_optimizer(Cma_cpp(max_evals), Cassini2(), num_retries, num) 
    _test_optimizer(Cma_cpp(max_evals), Rosetta(), num_retries, num) 
    _test_optimizer(Cma_cpp(max_evals), Messenger(), num_retries, num) 
    _test_optimizer(Cma_cpp(max_evals), Gtoc1(), num_retries, num) 
    _test_optimizer(Cma_cpp(max_evals), MessFull(), num_retries, num) 
  
    _test_optimizer(De_cpp(max_evals), Cassini1(), num_retries, num) 
    _test_optimizer(De_cpp(max_evals), Cassini2(), num_retries, num) 
    _test_optimizer(De_cpp(max_evals), Rosetta(), num_retries, num) 
    _test_optimizer(De_cpp(max_evals), Messenger(), num_retries, num) 
    _test_optimizer(De_cpp(max_evals), Gtoc1(), num_retries, num) 
    _test_optimizer(De_cpp(max_evals), MessFull(), num_retries, num) 
 
    _test_optimizer(Da_cpp(max_evals), Cassini1(), num_retries, num) 
    _test_optimizer(Da_cpp(max_evals), Cassini2(), num_retries, num) 
    _test_optimizer(Da_cpp(max_evals), Rosetta(), num_retries, num) 
    _test_optimizer(Da_cpp(max_evals), Messenger(), num_retries, num) 
    _test_optimizer(Da_cpp(max_evals), Gtoc1(), num_retries, num) 
    _test_optimizer(Da_cpp(max_evals), MessFull(), num_retries, num) 
 
    _test_optimizer(Dual_annealing(max_evals), Cassini1(), num_retries, num) 
    _test_optimizer(Dual_annealing(max_evals), Cassini2(), num_retries, num) 
    _test_optimizer(Dual_annealing(max_evals), Rosetta(), num_retries, num) 
    _test_optimizer(Dual_annealing(max_evals), Messenger(), num_retries, num) 
    _test_optimizer(Dual_annealing(max_evals), Gtoc1(), num_retries, num) 
    _test_optimizer(Dual_annealing(max_evals), MessFull(), num_retries, num) 
 
    _test_optimizer(Differential_evolution(max_evals), Cassini1(), num_retries, num) 
    _test_optimizer(Differential_evolution(max_evals), Cassini2(), num_retries, num) 
    _test_optimizer(Differential_evolution(max_evals), Rosetta(), num_retries, num) 
    _test_optimizer(Differential_evolution(max_evals), Messenger(), num_retries, num) 
    _test_optimizer(Differential_evolution(max_evals), Gtoc1(), num_retries, num) 
    _test_optimizer(Differential_evolution(max_evals), MessFull(), num_retries, num) 
     
def _test_optimizer(opt, problem, num_retries = 32, num = 10):
    log = logger()
    log.info(problem.name + ' ' + opt.name)
    for _ in range(num):
        ret = minimize(problem.fun, problem.bounds, math.inf, num_retries, log, optimizer=opt)

def main():
    test_all()

if __name__ == '__main__':
    main()