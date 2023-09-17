# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# This code is derived from https://github.com/esa/pykep/pull/127 
# originally developed by Moritz v. Looz @mlooz . 
# It was modified following suggestions from Waldemar Martens @MartensWaldemar_gitlab

# Solar orbiter is quite a challenge for state of the art optimizers, but
# good solutions fulfilling the requirements can be found and an example is
# shown in check_good_solution() . At 
# https://gist.github.com/dietmarwo/86f24e1b9a702e18615b767e226e883f you may find good solutions
# for this and two other solo models.  
#
# See https://www.esa.int/Science_Exploration/Space_Science/Solar_Orbiter

# Requires pykep which needs python 3.8, 
# Create an python 3.8 environment:

# mamba create -n env38 python=3.8
# conda activate env38

# Install dependencies:
# pip install pykep 

# Tested using https://docs.conda.io/en/main/miniconda.html on Linux Mint 21.

import math
from math import cos, pi, sin, sqrt
from fcmaes import retry, advretry, modecpp, mode
from fcmaes.optimizer import de_cma, single_objective, de, Bite_cpp
import numpy as np
import matplotlib.pyplot as plt
from pykep import RAD2DEG, AU

from solo_mgar_udp import solo_mgar_udp

import sys 
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}")
logger.add("log_{time}.txt")

def read_solutions(fname):
    ys = []
    xs = [] 
    with open(fname) as csvfile:
        lines = csvfile.readlines()
        for line in lines:
            row = line.split(' ')
            if len(row) < 12:
                continue
            ys.append(float(row[0]))
            x = []
            i = -1
            while(True):
                xi = row[i]
                while not xi[-1].isdigit():
                    xi = xi[:-1]
                if not (xi[0].isdigit() or xi[0] == '-'):
                    xi = xi[1:]
                    x.insert(0, float(xi))
                    break
                x.insert(0, float(xi))
                i -= 1
            xs.append(x)
    return ys, xs
 

def verify(ys, xs):
    for i in range(len(ys)):
        solo_mgar = solo_mgar_udp([7000, 8000])  
        y0 = ys[i]
        x = xs[i]
        if len(x) != 10:
            continue
        lambert_legs = []
        resonances = []
        solo_mgar._compute_dvs(x, lambert_legs, resonances)
        resos = [reso._resonance for reso in resonances]
        # assert resos0 ==  resos
        y = solo_mgar.fitness(x)[0]
        print(y0, y, y0-y)
        assert abs(y0 - y < 0.23)

def check_good_solution(x):
    import pygmo as pg
    solo_mgar = solo_mgar_udp([7000, 8000])  
    prob = pg.problem(solo_mgar)
    print (str(prob.fitness(x))) 
    solo_mgar.pretty(x)
    solo_mgar.plot(x)
    solo_mgar.plot_distance_and_flybys(x)

def print_good_solutions(xs):
    from functools import reduce
    
    for i in range(len(xs)):
        solo_mgar = solo_mgar_udp([7000, 8000])  
        lambert_legs = []
        resos = []
        x = xs[i]
        rvt_outs, rvt_ins, rvt_pls, _, _ = solo_mgar._compute_dvs(x, lambert_legs, resos)    
        #rvt_outs = [rvt.rotate(solo_mgar._rotation_axis, solo_mgar._theta) for rvt in rvt_outs]
        rvt_out = rvt_outs[-1].rotate(solo_mgar._rotation_axis, solo_mgar._theta) # rotate
        a, e, incl, _, _, _ = rvt_out.kepler()
        # orbit should be as polar as possible, but we do not care about prograde/retrograde
        corrected_inclination = abs(abs(incl) % pi - pi / 2) * RAD2DEG
        final_perhelion = a * (1 - e) / AU

        y = solo_mgar.fitness(x)
        resos = [str(resos[i]._resonance) for i in range(len(resos))]
        resos = reduce((lambda x, y: x + ',' + y), resos)
        print (str(i) + ' ' + str(incl*RAD2DEG) + ' ' + str(final_perhelion) + ' [' + str(y[0]), ', [' + resos + '], ' + str(x) + '],')
     
def optimize():   
    from scipy.optimize import Bounds
    solo_mgar = solo_mgar_udp([7000, 8000])  
    
    def fun(x):
        try:
            return solo_mgar.fitness(x)[0]
        except Exception as ex:
            return 1E99

    def mofun(x):
        try:
            return solo_mgar.mo_fitness(x)
        except Exception as ex:
            return np.array([1E99]*6)

    bounds = solo_mgar.get_bounds()
        
    # logger.info('solar orbiter' + ' de -> cmaes c++ smart retry')
    # ret = advretry.minimize(fun, bounds=Bounds(bounds[0], bounds[1]), num_retries = 60000, 
    #     optimizer=de_cma(2000))
        #optimizer=bite_cma(2000))
    
    logger.info('solar orbiter' + ' BiteOpt parallel retry')    
    ret = retry.minimize(fun, bounds=Bounds(bounds[0], bounds[1]), num_retries = 32000, 
                         optimizer=Bite_cpp(200000, M=6))
    
    # x, y = modecpp.retry(mode.wrapper(mofun, 3, interval = 1000000000), 3, 3,
    #           Bounds(bounds[0], bounds[1]), popsize = 128, 
    #           max_evaluations = 300000, 
    #           nsga_update=False, num_retries = 32000,
    #           workers=32)
    return ret

def archipelago():   
    import pygmo as pg 
    udp = solo_mgar_udp([7000, 8000])  
    #uda = pg.sga(gen = 6000)
    uda = pg.sade(memory=True,variant=1,gen=6000)
    # instantiate an unconnected archipelago
    for _ in range(1000):
        archi = pg.archipelago(t = pg.topologies.unconnected())
        for _ in range(32):
            alg = pg.algorithm(uda)
            #alg.set_verbosity(1)    
            prob = pg.problem(udp)
            pop = pg.population(prob, 20)    
            isl = pg.island(algo=alg, pop=pop)
            archi.push_back(isl)      
        archi.evolve()
        archi.wait_check()

def optimize_pagmo():  
    import pygmo as pg  
    solo_mgar = solo_mgar_udp([7000, 8000])  
    for i in range(6000):
        prob = pg.problem(solo_mgar)   
        pop = pg.population(prob=prob, size=32)
        alg = pg.algorithm(pg.sade(memory=True,gen=1))
        pop = alg.evolve(pop)
        print(i, pop.champion_f, solo_mgar.fitness(pop.champion_x))

if __name__ == '__main__':
    optimize()
    #archipelago()
    ys, xs = read_solutions('data/solo_results.txt')
    #print_good_solutions(xs) 
    #verify(ys, xs)
    check_good_solution(xs[0])  
    plt.show()
   

    pass
