# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# Examples for fcmaes parallel retry from https://www.esa.int/gsp/ACT/projects/gtop/
# Used to generate the results in https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/Results.adoc

# Tested using https://docs.conda.io/en/main/miniconda.html on Linux Mint 21.2

import math
import numpy as np
from fcmaes.astro import MessFull, Messenger, Gtoc1, Cassini1, Cassini2, Rosetta, Tandem, Sagas, Cassini1minlp
from fcmaes.optimizer import De_python, De_ask_tell, de_cma, da_cma, Cma_cpp, Cma_python, De_cpp, Da_cpp, Bite_cpp, Crfmnes, Crfmnes_cpp, Pgpe_cpp, de_crfmnes, crfmnes_bite, Dual_annealing, Differential_evolution
from fcmaes import retry

import sys 
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")
logger.add("log_{time}.txt", format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")

problems = [Cassini1(), Cassini2(), Rosetta(), Tandem(5), Messenger(), Gtoc1(), MessFull(), Sagas(), Cassini1minlp()]

max_evals = 50000

algos = [ de_cma(max_evals), crfmnes_bite(max_evals), de_crfmnes(max_evals), da_cma(max_evals), Cma_cpp(max_evals), De_cpp(max_evals),
          Da_cpp(max_evals), Bite_cpp(max_evals), Dual_annealing(max_evals), Differential_evolution(max_evals)]

#algos = [ Cma_python(max_evals)]
              
def test_all(num_retries = 320, num = 1):
    for problem in problems:
        for algo in algos:
            _test_optimizer(algo, problem, num_retries, num) 
      
def _test_optimizer(opt, problem, num_retries = 32, num = 1):
    logger.info(problem.name + ' ' + opt.name)
    for i in range(num):
        name = str(i+1) + ' ' + problem.name if num > 1 else problem.name
        retry.minimize_plot(name, opt, problem.fun, problem.bounds, 
                            np.inf, 10.0, num_retries)

def main():
        
    test_all()

if __name__ == '__main__':
    main()
    
