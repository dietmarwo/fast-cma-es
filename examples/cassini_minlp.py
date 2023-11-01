# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# See http://www.midaco-solver.com/data/pub/CEC2019_Schlueter_Munetomo.pdf for a description of the
# MINLP problem solved here. 
# Used to generate the results in https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/MINLP.adoc

# Tested using https://docs.conda.io/en/main/miniconda.html on Linux Mint 21.2

from fcmaes.astro import Cassini1minlp, Cassini1multi, cassini1minlp
from fcmaes.optimizer import de_cma
from fcmaes import advretry, multiretry

import sys 
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")
logger.add("log_{time}.txt", format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")

# minlp approach, planet sequence is additional arguments
def test_optimizer(opt, problem, num_retries = 120000, num = 100, value_limit = 10.0):
    logger.info(problem.name + ' ' + opt.name)
    for _ in range(num):
        ret = advretry.minimize(problem.fun, problem.bounds, value_limit, 
                                num_retries, optimizer=opt)

def sequences():
    for p1 in range(2,4):
        for p2 in range(2,4):
            for p3 in range(2,4):
                for p4 in range(3,6):
                    yield[p1,p2,p3,p4]

# simultaneous optimization 
def test_multiretry(retries_inc = 128, 
             keep = 0.7, optimizer = de_cma(1500), repeat = 50):
    problems = []
    ids = []
    for seq in sequences():
        problems.append(Cassini1minlp(planets = seq))
        ids.append(str(seq))

    problem_stats = multiretry.minimize(problems, ids, retries_inc, retries_inc*repeat,
                                         keep, optimizer)
    ps = problem_stats[0]
    for _ in range(repeat):
        logger.info("problem " + ps.prob.name + ' ' + str(ps.id))
        ps.retry(optimizer)

def main():
#    test_optimizer(de_cma(1500), Cassini1minlp()) 
    test_multiretry(repeat = 50)
    
if __name__ == '__main__':
    main()
    
