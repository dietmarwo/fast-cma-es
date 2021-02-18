# requires pykep PR branch for 
# https://github.com/esa/pykep/pull/127

from pykep.trajopt import mga_1dsm, launchers
from pykep.planet import jpl_lp
from pykep import epoch
from pykep.core import lambert_problem, propagate_lagrangian, fb_prop
from pykep import DAY2SEC, DAY2YEAR, AU, RAD2DEG, ic2par
from pykep.trajopt.gym._solar_orbiter import _solar_orbiter_udp

# Other imports
import numpy as np
from numpy.linalg import norm
from numpy import sign
from math import acos, asin, cos, exp, log, pi, sin
import matplotlib.pyplot as plt
from copy import deepcopy
import pygmo as pg
import time

tmin = epoch(time.time() / (24*3600) - 30*365 -7 + 2/24 - 2*365)
tmax = epoch(time.time() / (24*3600) - 30*365 -7 + 2/24 + 2*365)


def names(seq):
    return " ".join((p.name) for p in seq)

def compute_solar_orbiter():
     
    earth = jpl_lp("earth")
    venus = jpl_lp("venus")
    
    seq1=[earth, venus, venus, earth, venus, venus, venus, venus, venus]
    seq2=[earth, venus, venus, earth, venus, earth, venus, venus, venus]
    seq3=[earth, venus, venus, venus, venus, earth, venus, venus, venus]
    seq4=[earth, venus, venus, venus, earth, earth, venus, venus, venus]
    seq5=[earth, venus, venus, venus, earth, venus, venus, venus, venus]
    seq6=[earth, earth, venus, venus, earth, venus, venus, venus, venus]
    seq7=[earth, earth, venus, earth, earth, venus, venus, venus, venus]
    seq8=[earth, earth, venus, earth, venus, venus, earth, venus, venus]
    
    seqs = [seq1,seq2,seq3,seq4,seq5,seq6,seq7,seq8]  
     
    solar_orbiters = [_solar_orbiter_udp([tmin, tmax], seq=seq) for seq in seqs]
    
    # Include delta v, mass and sun distance constraints
    probs = [pg.unconstrain(
        pg.problem(solar_orbiter),method="weighted",weights=[1.0, 10.0, 100, 100]) 
            for solar_orbiter in solar_orbiters]
    
    from fcmaes.advretry import minimize
    from fcmaes.optimizer import logger, de_cma, single_objective
    from fcmaes import multiretry
   
    fprobs = [single_objective(pg.problem(prob)) for prob in probs]
    
    #we replace pagmo pg.sade by fcmaes smart retry:
     
    logger().info('solar orbiter' + ' de -> cmaes c++ smart retry')
#     ret = minimize(fprobs[0].fun, bounds=fprob[0].bounds, num_retries = 6000, 
#         value_limit = 100000, logger = logger(), optimizer=de_cma(1500))
    
    ids = [names(seq) for seq in seqs]

    logger().info('solar orbiter' + ' de -> cmaes c++ smart retry')    
    ids = [names(seq) for seq in seqs]
    optimizer = de_cma(1500)
    problem_stats = multiretry.minimize(fprobs, ids, 256, 0.9, optimizer, logger())
    ps = problem_stats[0] # focus on the best one
    for _ in range(6):
        logger.info("problem " + ps.prob.name + ' ' + str(ps.id))
        ps.retry(optimizer)
    
    pop_champion_x = ps.ret.x
    
    solar_orbiter.pretty(pop_champion_x)
    solar_orbiter.plot(pop_champion_x)
    
    # Plot solar distance in AE
    timeframe = range(1,5*365)
    earth = jpl_lp("earth")
    venus = jpl_lp("venus")
    
    distances = []
    edistances = []
    vdistances = []
    
    for i in timeframe:
        epoch = pop_champion_x[0]+i
        pos, vel = solar_orbiter.eph(pop_champion_x, epoch)
        epos, evel = earth.eph(epoch)
        vpos, vvel = venus.eph(epoch)
        distances.append(norm(pos) / AU)
        edistances.append(norm(epos) / AU)
        vdistances.append(norm(vpos) / AU)
    
    fig, ax = plt.subplots()
    ax.plot(list(timeframe), distances, label="Solar Orbiter")
    ax.plot(list(timeframe), edistances, label="Earth")
    ax.plot(list(timeframe), vdistances, label="Venus")
    ax.set_xlabel("Days")
    ax.set_ylabel("AU")
    ax.set_title("Distance to Sun")
    ax.legend()
    
    # Plot inclination and distance
    inclinations = []
    for i in timeframe:
        epoch = pop_champion_x[0]+i
        pos, _ = solar_orbiter.eph(pop_champion_x, epoch)
        inclination = sign(pos[2])*acos(norm(pos[:2]) / norm(pos))
        inclinations.append(inclination)
    
    color = 'tab:red'
    fig2, ax2 = plt.subplots()
    ax2.plot(list(timeframe), inclinations, color=color)
    ax2.set_ylabel("Inclination", color=color)
    ax2.set_xlabel("Days")
    ax.set_title("Distance and Inclination")
    
    ax3 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax3.set_ylabel('AU', color=color)
    ax3.plot(list(timeframe), distances, color=color)
    ax3.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    
if __name__ == '__main__':
    compute_solar_orbiter()
    pass