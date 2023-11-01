# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# Tandem MINLP problem. 
# Used to generate the results in https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/MINLP.adoc

# Tested using https://docs.conda.io/en/main/miniconda.html on Linux Mint 21.

from fcmaes.astro import Tandem_minlp, Tandem
from fcmaes.optimizer import de_cma
from fcmaes import advretry, multiretry

import sys 
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")
logger.add("log_{time}.txt", format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")

# minlp approach, planet sequence is additional arguments
def test_optimizer(opt, problem, num_retries = 120000, num = 100, 
                    value_limit = -10.0):
    logger.info(problem.name + ' ' + opt.name)
    for _ in range(num):
        ret = advretry.minimize(problem.fun, problem.bounds, value_limit, 
                                num_retries, optimizer=opt)

# simultaneous optimization        
def test_multiretry(num_retries = 512, 
             keep = 0.7, optimizer = de_cma(1500), repeat = 50):
    seqs = Tandem(0).seqs
    n = len(seqs)
    problems = [Tandem(i) for i in range(n)]
    ids = [str(seqs[i]) for i in range(n)]
    for _ in range(100):
        problem_stats = multiretry.minimize(problems, ids, num_retries, keep, optimizer)
        ps = problem_stats[0]
        for _ in range(repeat):
            logger.info("problem " + ps.prob.name + ' ' + str(ps.id))
            ps.retry(optimizer)
            
def main():
    test_optimizer(de_cma(1500), Tandem_minlp()) 
    #test_multiretry(repeat = 50)
    
if __name__ == '__main__':
    main()
    