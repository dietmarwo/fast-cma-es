# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# Examples for fcmaes coordinated retry from https://www.esa.int/gsp/ACT/projects/gtop/
# Used to generate the results in https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/Results.adoc

# Tested using https://docs.conda.io/en/main/miniconda.html on Linux Mint 21.2

from fcmaes.astro import MessFull, Messenger, Cassini2, Rosetta, Gtoc1, Cassini1, Tandem, Sagas, Cassini1minlp
from fcmaes.optimizer import de_cma, de_cma_py, da_cma, Cma_cpp, De_cpp, Da_cpp, Dual_annealing, Differential_evolution
from fcmaes import advretry

import sys 
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")
logger.add("log_{time}.txt", format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")

def messengerFullLoop(opt, num = 1,):    
    for i in range(num):
        problem = MessFull()
        logger.info(problem.name + ' ' + opt.name)
        name = str(i+1) + ' ' + problem.name if num > 1 else problem.name
        advretry.minimize_plot(name, opt, problem.fun, 
                               problem.bounds, 12.0, 12.0, 50000)
        
problems = [Cassini1(), Cassini2(), Rosetta(), Tandem(5), Messenger(), Gtoc1(), MessFull(), Sagas(), Cassini1minlp()]

min_evals = 1500

algos = [ de_cma(min_evals), da_cma(min_evals), Cma_cpp(min_evals), De_cpp(min_evals), 
         Da_cpp(min_evals), Dual_annealing(min_evals), Differential_evolution(min_evals), de_cma_py(min_evals)]
               
def test_all(num_retries = 1000, num = 1):
    for problem in problems:
        for algo in algos:
            _test_optimizer(algo, problem, num_retries, num, value_limit = 1E99) 

def _test_optimizer(opt, problem, num_retries = 10000, num = 1, value_limit = 100.0):
    logger.info(problem.name + ' ' + opt.name)
    for i in range(num):
        name = str(i+1) + ' ' + problem.name if num > 1 else problem.name
        advretry.minimize_plot(name, opt, problem.fun, 
                               problem.bounds, value_limit, 10.0, num_retries)

def main():
    test_all()
    #messengerFullLoop(de_cma(min_evals))

if __name__ == '__main__':
    main()
    