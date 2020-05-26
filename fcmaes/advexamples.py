# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

from fcmaes.astro import MessFull, Messenger, Cassini2, Rosetta, Gtoc1, Cassini1, Tandem
from fcmaes.optimizer import logger, de_cma, da_cma, Cma_cpp, De_cpp, Da_cpp, Dual_annealing, Differential_evolution
from fcmaes.advretry import minimize

def messengerFullLoop():    
    while True:    
        problem = MessFull()
        logger().info(problem.name + ' de + cmaes c++')
        ret = minimize(problem.fun, bounds=problem.bounds, num_retries = 60000, 
            value_limit = 10.0, logger = logger())
        
problems = [Cassini1(), Cassini2(), Rosetta(), Tandem(5), Messenger(), Gtoc1(), MessFull()]

min_evals = 1500

algos = [de_cma(min_evals), da_cma(min_evals), Cma_cpp(min_evals), De_cpp(min_evals), 
         Da_cpp(min_evals), Dual_annealing(min_evals), Differential_evolution(min_evals)]
               
def test_all(num_retries = 10000, num = 10):
    for problem in problems:
        for algo in algos:
            _test_optimizer(algo, problem, num_retries, num) 

def _test_optimizer(opt, problem, num_retries = 10000, num = 1, value_limit = 20.0, log = logger()):
    log.info(problem.name + ' ' + opt.name)
    for i in range(num):
        ret = minimize(problem.fun, problem.bounds, value_limit, num_retries, log, optimizer=opt)

def main():
    test_all()
    #messengerFullLoop()

if __name__ == '__main__':
    main()