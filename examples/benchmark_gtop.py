# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# Test for fcmaes coordinated retry applied to https://www.esa.int/gsp/ACT/projects/gtop/
# Generates the log files used to produce the tables in the README. 

# Tested using https://docs.conda.io/en/main/miniconda.html on Linux Mint 21.2

from fcmaes.astro import Messenger, Cassini2, Rosetta, Gtoc1, Cassini1, Sagas, Tandem, MessFull
from fcmaes.optimizer import de_cma
from fcmaes.advretry import minimize

import sys 
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}")
logger.add("log_{time}.txt")

def _test_optimizer(opt, problem, num_retries = 10000, num = 1, value_limit = 100.0, 
                    stop_val = -1E99):
    logger.info("Testing coordinated retry " + opt.name +  ' ' + problem.name )
    for _ in range(num):
        ret = minimize(problem.fun, problem.bounds, value_limit, num_retries,
                       optimizer=opt, stop_fitness = stop_val)

def main():
    numRuns = 100
    min_evals = 1500
    
    _test_optimizer(de_cma(min_evals), MessFull(), num_retries = 50000, num = numRuns, 
                    value_limit = 12.0, stop_val = 1.960)
    
    _test_optimizer(de_cma(min_evals), Gtoc1(), num_retries = 10000, num = numRuns, 
                    value_limit = -300000.0, stop_val = -1581949)
    _test_optimizer(de_cma(min_evals), Cassini1(), num_retries = 4000, num = numRuns, 
                    value_limit = 20.0, stop_val = 4.93075)
    _test_optimizer(de_cma(min_evals), Cassini2(), num_retries = 6000, num = numRuns, 
                    value_limit = 20.0, stop_val = 8.38305)
    _test_optimizer(de_cma(min_evals), Messenger(), num_retries = 8000, num = numRuns, 
                    value_limit = 20.0, stop_val = 8.72)
    _test_optimizer(de_cma(min_evals), Rosetta(), num_retries = 4000, num = numRuns, 
                    value_limit = 20.0, stop_val = 1.34335)
    _test_optimizer(de_cma(min_evals), Sagas(), num_retries = 4000, num = numRuns, 
                    value_limit = 100.0, stop_val = 18.188)
    _test_optimizer(de_cma(min_evals), Tandem(5), num_retries = 20000, num = numRuns, 
                    value_limit = -300.0, stop_val = -1500)
    _test_optimizer(de_cma(min_evals), MessFull(), num_retries = 50000, num = numRuns, 
                    value_limit = 12.0, stop_val = 1.960)
 
if __name__ == '__main__':
    main()
    