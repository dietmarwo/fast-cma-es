# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

import math
from fcmaes.astro import MessFull, Messenger, Gtoc1, Cassini1, Cassini2, Rosetta, Tandem
from fcmaes.optimizer import logger, de_cma, da_cma, Cma_cpp, De_cpp, Da_cpp, Dual_annealing, Differential_evolution
from fcmaes.retry import minimize
            
problems = [Cassini1(), Cassini2(), Rosetta(), Tandem(5), Messenger(), Gtoc1(), MessFull()]

max_evals = 50000

algos = [de_cma(max_evals), da_cma(max_evals), Cma_cpp(max_evals), De_cpp(max_evals), 
         Da_cpp(max_evals), Dual_annealing(max_evals), Differential_evolution(max_evals)]
               
def test_all(num_retries = 10000, num = 10):
    for problem in problems:
        for algo in algos:
            _test_optimizer(algo, problem, num_retries, num) 
      
def _test_optimizer(opt, problem, num_retries = 32, num = 10):
    log = logger()
    log.info(problem.name + ' ' + opt.name)
    for _ in range(num):
        ret = minimize(problem.fun, problem.bounds, math.inf, num_retries, log, optimizer=opt)

def main():
    test_all()

if __name__ == '__main__':
    main()