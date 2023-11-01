# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# Tested using https://docs.conda.io/en/main/miniconda.html on Linux Mint 21.2

import math
from fcmaes import multiretry
from fcmaes.astro import cassini2multi
from fcmaes.optimizer import de_cma, logger
from scipy.optimize import Bounds

import sys 
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")
logger.add("log_{time}.txt", format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")

def cassini1(x):
    y = cassini2multi(x)
    return y[0]

class CassiniMulti(object):
    """ see https://www.esa.int/gsp/ACT/projects/gtop/cassini2/ """
    
    def __init__(self, planets = [2,2,3,5]):    
        self.name = "Cassini2"
        self.planets = planets
        self.fun = self.cassini2
        self.bounds = Bounds([-1000,3,0,0,100,100,30,400,800,0.01,0.01,0.01,0.01,0.01,1.05,1.05,1.15,1.7, -math.pi, -math.pi, -math.pi, -math.pi],
                [0,5,1,1,400,500,300,1600,2200,0.9,0.9,0.9,0.9,0.9,6,6,6.5,291,math.pi,  math.pi,  math.pi,  math.pi])  

    def cassini2(self, x):
        return cassini1(list(x) + self.planets)

def sequences():
    for p1 in range(2,4):
        for p2 in range(2,4):
            for p3 in range(2,4):
                for p4 in range(3,6):
                    yield[p1,p2,p3,p4]

# simultaneous optimization 
def check_multiretry(retries_inc = 100, 
             keep = 0.7, optimizer = de_cma(1500), repeat = 100):
    problems = []
    ids = []
    for seq in sequences():
        problems.append(CassiniMulti(planets = seq))
        ids.append(str(seq))
        
    problem_stats = multiretry.minimize(problems, ids, retries_inc, 
                                        retries_inc*repeat, keep, optimizer)
    ps = problem_stats[0]
    for _ in range(repeat):
        logger.info("problem " + ps.prob.name + ' ' + str(ps.id))
        ps.retry(optimizer) 

def main():
    check_multiretry()
        
if __name__ == '__main__':
    main()
    