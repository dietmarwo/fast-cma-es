# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# See http://www.midaco-solver.com/data/pub/CEC2019_Schlueter_Munetomo.pdf for a description of the
# MINLP problem solved here. 
# Used to generate the results in https://github.com/dietmarwo/fast-cma-es/blob/master/MINLP.adoc

from fcmaes.astro import Cassini1minlp
from fcmaes.optimizer import logger, de_cma
from fcmaes.advretry import minimize

def _test_optimizer(opt, problem, num_retries = 100000, num = 100, value_limit = 100.0, log = logger()):
    log.info(problem.name + ' ' + opt.name)
    for i in range(num):
        ret = minimize(problem.fun, problem.bounds, value_limit, num_retries, log, optimizer=opt)

def main():
    _test_optimizer(de_cma(1500), Cassini1minlp()) 

if __name__ == '__main__':
    main()
    