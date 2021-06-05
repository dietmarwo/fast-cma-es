# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# Examples for fcmaes parallel retry from https://www.esa.int/gsp/ACT/projects/gtop/
# Used to generate the results in https://github.com/dietmarwo/fast-cma-es/blob/master/Results.adoc

import math
from fcmaes.astro import MessFull, Messenger, Gtoc1, Cassini1, Cassini2, Rosetta, Tandem, Sagas, Cassini1minlp
from fcmaes.optimizer import logger, De_python, De_ask_tell, de_cma, de2_cma, da_cma, Cma_cpp, De_cpp, Da_cpp, Csma_cpp, Bite_cpp, Hh_cpp, Dual_annealing, Differential_evolution
from fcmaes import retry

import numpy as np
from scipy.optimize import Bounds

problems = [Cassini1(), Cassini2(), Rosetta(), Tandem(5), Messenger(), Gtoc1(), MessFull(), Sagas(), Cassini1minlp()]

max_evals = 50000

algos = [ de_cma(max_evals), de2_cma(max_evals), da_cma(max_evals), Cma_cpp(max_evals), De_cpp(max_evals), Hh_cpp(max_evals),
         Da_cpp(max_evals), Bite_cpp(max_evals), Csma_cpp(max_evals), Dual_annealing(max_evals), Differential_evolution(max_evals)]
               
def test_all(num_retries = 10000, num = 1):
    for problem in problems:
        for algo in algos:
            _test_optimizer(algo, problem, num_retries, num) 
      
def _test_optimizer(opt, problem, num_retries = 32, num = 1):
    log = logger()
    log.info(problem.name + ' ' + opt.name)
    for i in range(num):
        name = str(i+1) + ' ' + problem.name if num > 1 else problem.name
        retry.minimize_plot(name, opt, problem.fun, problem.bounds, 
                            math.inf, 10.0, num_retries, logger=log)

def main():
        
    test_all()

if __name__ == '__main__':
    main()
    
