# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# Examples for fcmaes coordinated retry from https://www.esa.int/gsp/ACT/projects/gtop/
# Used to generate the results in https://github.com/dietmarwo/fast-cma-es/blob/master/Results.adoc

from fcmaes.astro import MessFull, Messenger, Cassini2, Rosetta, Gtoc1, Cassini1, Tandem, Sagas, Cassini1minlp
from fcmaes.optimizer import logger, de_cma, de_cma_py, de2_cma, da_cma, Cma_cpp, De_cpp, Da_cpp, Hh_cpp, Dual_annealing, Differential_evolution
from fcmaes.advretry import minimize

def messengerFullLoop():    
    while True:    
        problem = MessFull()
        logger().info(problem.name + ' de + cmaes c++')
        minimize(problem.fun, bounds=problem.bounds, num_retries = 50000, 
            value_limit = 10.0, logger = logger(), optimizer=de_cma(1500))
        
problems = [Cassini1(), Cassini2(), Rosetta(), Tandem(5), Messenger(), Gtoc1(), MessFull(), Sagas(), Cassini1minlp()]

min_evals = 1500

algos = [ de_cma(min_evals), de2_cma(min_evals), da_cma(min_evals), Cma_cpp(min_evals), De_cpp(min_evals), Hh_cpp(min_evals),
         Da_cpp(min_evals), Dual_annealing(min_evals), Differential_evolution(min_evals), de_cma_py(min_evals)]
               
def test_all(num_retries = 10000, num = 10):
    for problem in problems:
        for algo in algos:
            _test_optimizer(algo, problem, num_retries, num, value_limit = 1E99) 

def _test_optimizer(opt, problem, num_retries = 10000, num = 1, value_limit = 100.0, log = logger()):
    log.info(problem.name + ' ' + opt.name)
    for i in range(num):
        ret = minimize(problem.fun, problem.bounds, value_limit, num_retries, log, optimizer=opt)

def main():
    test_all()
    #messengerFullLoop()

if __name__ == '__main__':
    main()
    