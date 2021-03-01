# requires pykep PR branch for 
# https://github.com/esa/pykep/pull/127
# requires fcmaes version >= 1.2.13, type 'pip install fcmaes --upgrade'

import time

from fcmaes import multiretry
from fcmaes.optimizer import logger, de_cma, single_objective
from pykep import epoch
from pykep.planet import jpl_lp
from pykep.trajopt.gym._solar_orbiter import _solar_orbiter_udp

import numpy as np
import pygmo as pg

#log to file and stdout
logger('solarorbiter_multi.log')

# Other imports
tmin = epoch(time.time() / (24*3600) - 30*365 -7 + 2/24 - 2*365)
tmax = epoch(time.time() / (24*3600) - 30*365 -7 + 2/24 + 2*365)

def names(seq):
    return " ".join((p.name) for p in seq)

def compute_solar_orbiter():
     
    earth = jpl_lp("earth")
    venus = jpl_lp("venus")
    
    # all sequences of length 8    
    def sequences():
        for p2 in [earth,venus]:
            for p3 in [earth,venus]:
                for p4 in [earth,venus]:
                    for p5 in [earth,venus]:
                        for p6 in [earth,venus]:
                            for p7 in [earth,venus]:
                                yield[earth,p2,p3,p4,p5,p6,p7,venus]
    seqs = [s for s in sequences()]  

    solar_orbiters = [_solar_orbiter_udp([tmin, tmax], seq=seq) for seq in seqs]
    
    # Include delta v, mass and sun distance constraints
    probs = [pg.unconstrain(solar_orbiter,method="weighted",weights=[1.0, 10.0, 100, 100]) 
            for solar_orbiter in solar_orbiters]
    
    fprobs = [single_objective(pg.problem(prob)) for prob in probs]         

    logger().info('solar orbiter' + ' de -> cmaes c++ smart retry')    
    ids = [names(seq) for seq in seqs]
    optimizer = de_cma(1500)
    problem_stats = multiretry.minimize(fprobs, ids, 256, 0.7, optimizer, logger())
    values = np.array([ps.value for ps in problem_stats])
    ps = problem_stats[values.argsort()[0]] # focus on the best one
    logger().info("continue to optimize best sequence " + ps.prob.name + ' ' + str(ps.id))
    for _ in range(20):
        ps.retry(optimizer)
    
if __name__ == '__main__':
    compute_solar_orbiter()
    pass